from typing import Tuple, Callable, List, Optional
import time
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils.pdp_functions import *
from algs.large_neighborhood_search import repair
from algs.greedy_search import two_opt_improve
from typing import Optional, Union
from utils.functions import *
from tqdm import tqdm


def  multi_start_large_neighborhood_search(
    depots: torch.Tensor,                 # (B, D, 2)
    requests: torch.Tensor,               # (B, R, 2) or your request shape
    tour_init: torch.Tensor,              # (B, L) initial routes (e.g., tour_pomo_2opt)
    *,
    starting_nodes: int,                  # args.starting_nodes (beam used for parallel repairs)
    beta: int = 3,                        # keep top-β candidates per pruning step
    alpha: float = 1.0,                   # temperature for softmax over -counts (larger => stronger penalty)
    max_iters: int = 200,                 # iteration cap if time_cap is None
    k_max: int = 3,
    is_hill_climb: bool = False,
    obj: str = 'revenue',
    seed: int = 1234,
    time_cap: Optional[float] = None,     # seconds; if set, loop stops once exceeded
    compute_obj_fn: Callable = None,         # (routes, revenues_b) -> (K,) or (K,1)
    destroy_method: str = 'softmax',
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      best_routes: (B, S, L) the best single route per batch after LNS
      best_revs:   (B,)   revenue of those routes
    """
    if device is None:
        device = depots.device

    B = requests.size(0)
    R = requests.size(1)
    D = depots.size(1)
    N_total = D + 2 * R
    pad_val = D - 1  # typical "end depot" padding
    seed_everything(seed)

    # ---- preprocess once ----
    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)
    loads_flat = loads.squeeze(-1)  # (B, N_total)

    # track destroyed pickups per instance
    destroyed: List[set] = [set() for _ in range(B)]

    def destroy_keep_shape(
        routes: torch.Tensor,                          # (L,) or (B,L) long tensor
        *,
        R: Optional[int] = None,                       # #requests; required if pick_ids is used
        pad_value: Optional[int] = 0,                  # padding token to keep length
        pick_ids: Optional[Union[torch.Tensor, list]] = None,  # pickups to remove; 1D or 2D ((B,k))
        pickup_idx: Optional[int] = None,              # single pickup
        delivery_idx: Optional[int] = None            # paired delivery (or pickup_idx+R)
    ) -> torch.Tensor:
        """
        Remove requests from route(s) and right-pad to keep original length.

        Removal modes (choose ONE):
        1) pick_ids is provided: remove all pickups in pick_ids and their paired deliveries (+R).
            - pick_ids can be 1D (global for all rows) or 2D (B,k) per row.
        2) pickup_idx and delivery_idx are provided: remove this single pair.

        Returns a tensor with the same shape as `routes`.
        """
        assert routes.dim() in (1, 2), f"routes must be 1D or 2D, got {routes.shape}"
        dev, dt = routes.device, routes.dtype

        # ---- build `to_remove` ----
        if pick_ids is not None:
            if R is None:
                raise ValueError("R must be provided when using pick_ids.")

            pick_ids = torch.as_tensor(pick_ids, device=dev, dtype=dt)
            if pick_ids.dim() == 1:
                # global removal set for all rows
                to_remove = torch.cat([pick_ids, pick_ids + R])            # (2k,)
            elif pick_ids.dim() == 2:
                # per-row removal set
                to_remove = torch.cat([pick_ids, pick_ids + R], dim=1)     # (B, 2k)
            else:
                raise ValueError(f"pick_ids must be 1D or 2D, got {pick_ids.shape}")

        else:
            # single pair mode
            if pickup_idx is None or delivery_idx is None:
                raise ValueError("Provide either pick_ids or (pickup_idx, delivery_idx).")
            to_remove = torch.tensor([pickup_idx, delivery_idx], device=dev, dtype=dt)  # (2,)

        # ---- remove & pad (1D) ----
        if routes.dim() == 1:
            if to_remove.dim() == 2:
                # if per-row sets were passed but we have a single route, use the first row
                to_remove = to_remove[0]
            keep = ~torch.isin(routes, to_remove)
            kept = routes[keep]
            out = routes.new_full((routes.size(0),), pad_value)
            out[:kept.numel()] = kept
            return out

        # ---- remove & pad (2D) ----
        B, L = routes.shape
        out = routes.new_full((B, L), pad_value)

        if to_remove.dim() == 1:
            # same removal set for all rows (broadcast)
            keep = ~torch.isin(routes, to_remove)                      # (B,L) bool
            for b in range(B):
                kept = routes[b][keep[b]]
                out[b, :kept.numel()] = kept
            return out

        # per-row removal sets: to_remove shape (B, K)
        # build per-row membership: (B,L,1) == (B,1,K) -> (B,L,K) -> any(-1)
        eq = routes.unsqueeze(-1) == to_remove.unsqueeze(1)            # (B,L,K)
        rem = eq.any(dim=-1)                                           # (B,L)
        keep = ~rem
        for b in range(B):
            kept = routes[b][keep[b]]
            out[b, :kept.numel()] = kept
        return out

    def dedup_routes_by_pickups(routes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Keep first occurrence per unique set of pickups.
        routes: (K, L)
        Returns (unique_routes, keep_idx)
        """
        assert routes.dim() == 2, f"routes must be (K,L), got {tuple(routes.shape)}"
        K_, L_ = routes.shape
        dev = routes.device

        is_pick = (routes >= D) & (routes < D + R)
        pick_idx = torch.where(is_pick, routes - D, torch.full_like(routes, -1))

        pick_pres = torch.zeros(K_, R, dtype=torch.bool, device=dev)
        row_ids = torch.arange(K_, device=dev).unsqueeze(1).expand(K_, L_).reshape(-1)
        pi = pick_idx.reshape(-1)
        m = pi >= 0
        pick_pres[row_ids[m], pi[m]] = True

        sig_unique, inv = torch.unique(pick_pres, dim=0, return_inverse=True)
        keep_idx = torch.stack([(inv == g).nonzero(as_tuple=False)[0, 0] for g in range(sig_unique.size(0))]).to(dev)
        return routes[keep_idx], keep_idx

    # -------- main per-batch loop --------
    best_routes_out: List[torch.Tensor] = []
    best_revs_out: List[torch.Tensor] = []

    best_routes_record = []
    best_revs_record = []

    for b in tqdm(range(B)):
        seed_everything(seed)
        # start with a (beam=starting_nodes) stack: (1, L)
        route0 = tour_init[b].clone()
        best_route = route0.clone()
        beam = best_route.size(0)

        out = repeat_interleave(
            beam,
            positions=positions[b:b+1],
            revenues=revenues[b:b+1],
            T_max=T_max[b],
        )    
        (
            positions_b, revenues_b, T_max_b
        ) = (
            out["positions"], out["revenues"], out["T_max"]
        )

        best_rev = compute_obj(best_route, positions_b, revenues_b, T_max_b, beam, obj).view(-1)  # (1,)

        k = min(int(starting_nodes), best_route.size(0))
        vals, idx = torch.topk(best_rev.view(-1), k=k, largest=True, sorted=True)

        best_route = best_route.index_select(0, idx)   # (k, L)
        best_rev   = vals                              # (k,)
        
        start_clock = time.time()
        iters = 0
        # print("-----------------------------------------------------------------")
        while True:
            # print(iters, time.time() - start_clock)

            if (time_cap is None and iters >= max_iters) or (time_cap is not None and (time.time() - start_clock) >= time_cap):
                break
            iters += 1

            old_route = best_route.clone()

            # ----- choose a pickup to destroy (from current beam) -----
            cust_mask = (best_route >= D) & (best_route < D + R)  # (beam, L) -> mask
            customers = best_route[cust_mask]                             # flattened list of pickups
            if customers.numel() == 0:
                break

            uniq, counts = torch.unique(customers, return_counts=True)
            if destroyed[b]:
                keep_mask = torch.tensor([int(u.item()) not in destroyed[b] for u in uniq],
                                         device=uniq.device, dtype=torch.bool)
                uniq, counts = uniq[keep_mask], counts[keep_mask]
            if uniq.numel() == 0:
                break
            
            
            if 'softmax' in destroy_method:
                # ----- destroy + repair (apply to all rows in the beam) -----
                # larger counts -> larger prob
                probs = torch.softmax(alpha * counts.float(), dim=0)
                picked = int(torch.multinomial(probs, 1).item())
                destroy_idx = int(uniq[picked].item())
                destroyed[b].add(destroy_idx)

                new_route = destroy_keep_shape(best_route, pickup_idx=destroy_idx, delivery_idx=destroy_idx + R, pad_value=D-1)  # (beam, L)

                # smaller counts -> larger prob
                probs = torch.softmax(- alpha * counts.float(), dim=0)
                if destroy_method == 'softmax_fix':
                    k = min(k_max, probs.size(-1))      # choose how many to remove this iter 
                else:
                    k = min(iters-1, k_max)           # choose how many to remove this iter
                    k = min(k, probs.size(-1))        # choose how many to remove this iter 
                if k > 0:
                    # sample k distinct requests according to probs
                    idx = torch.multinomial(probs, num_samples=k, replacement=False)  # (k,)
                    picked = uniq[idx]    
                    # remember them if you track history
                    destroyed[b].update(picked.tolist())

                    new_route = destroy_keep_shape(new_route, pick_ids=picked, R=R, pad_value=D-1)  # (beam, L)

            elif 'random' in destroy_method:
                if destroy_method == 'random_fix':
                    k = min(k_max, uniq.size(-1))      # choose how many to remove this iter 
                else:
                    k = min(iters-1, k_max)           # choose how many to remove this iter
                    k = min(k, uniq.size(-1))        # choose how many to remove this iter 
                
                k += 1
                picked = uniq[torch.randperm(len(uniq))[:k]]
                destroyed[b].update(picked.tolist())
                new_route = destroy_keep_shape(best_route, pick_ids=picked, R=R, pad_value=D-1)  # (beam, L)

            # best_routes_record.append(new_route)

            beam_now = best_route.size(0)

            out = repeat_interleave(
                beam_now,
                positions=positions[b:b+1],
                revenues=revenues[b:b+1],
                T_max=T_max[b],
            )    
            (
                positions_b, revenues_b, T_max_b
            ) = (
                out["positions"], out["revenues"], out["T_max"]
            )

            out = repeat_interleave(
                new_route.size(0),
                positions=positions[b:b+1],
                revenues=revenues[b:b+1],
                T_max=T_max[b],
            )    
            (
                positions_r, revenues_r, T_max_r
            ) = (
                out["positions"], out["revenues"], out["T_max"]
            )

            if not is_hill_climb:
            
                # evaluate revenues row-wise
                revs_b = compute_obj(new_route, positions_r, revenues_r, T_max_r, new_route.size(0), obj).view(-1)
                mask = revs_b > best_rev  # (beam,)
                # update routes & revenues where improved
                if mask.any():
                    best_route = best_route.clone()
                    best_route[mask] = new_route[mask]
                    best_rev[mask] = revs_b[mask]

                # ----- multi-start repair from each row -----
                # build a repaired set for each row 's' in current beam
                repaired_list = []
                for s in range(beam_now):
                    repaired = repair(
                        best_route[s],                 # 1D route
                        positions[b:b+1], loads_flat[b], revenues[b:b+1],
                        Q_max[b], T_max[b],
                        R, D, device, obj, False
                    )
                    if not torch.is_tensor(repaired):
                        repaired = torch.tensor(repaired, dtype=torch.long, device=device)
                    repaired_list.append(repaired)

                # stack & pad to original L
                repaired_stack = pad_sequence(repaired_list, batch_first=True, padding_value=pad_val)  # (beam_now, L')
                if repaired_stack.size(1) < route0.size(1):
                    repaired_stack = F.pad(repaired_stack, (0, route0.size(1) - repaired_stack.size(1)), value=pad_val)

                # re-evaluate; update where better
                revs_rep = compute_obj(repaired_stack, positions_b, revenues_b, T_max_b, beam_now, obj).view(-1)
                mask_rep = revs_rep > best_rev
                if mask_rep.any():
                    best_route = best_route.clone()
                    best_route[mask_rep] = repaired_stack[mask_rep]
                    best_rev[mask_rep] = revs_rep[mask_rep]
            
            else:
                while True:
                    
                    revs_b = compute_obj(new_route, positions_r, revenues_r, T_max_r, new_route.size(0), obj).view(-1)
                    br = best_rev.view(-1)

                    # where the new score is better
                    mask = revs_b > br                     # shape: (beam,)

                    # update the scores (pick per position)
                    best_rev = torch.where(mask, revs_b, br)

                    # update the routes: replace rows where mask==True
                    # (mask indexes the first dimension)
                    best_route = best_route.clone()
                    best_route[mask] = new_route[mask]

                    # try repairs from multiple starting nodes
                    next_route = []
                    for s in range(beam_now):
                        repaired = repair(
                            new_route[s],
                            positions[b:b+1], loads_flat[b], revenues[b:b+1],
                            Q_max[b], T_max[b],
                            R, D, device, obj, False
                        )
                        next_route.append(
                            repaired if torch.is_tensor(repaired) else torch.tensor(repaired, dtype=torch.long, device=device)
                        )

                    # pad all repaired routes to same length
                    pad_val = D - 1
                    next_route = pad_sequence(next_route, batch_first=True, padding_value=pad_val)
                    if next_route.size(1) < route0.size(1):
                        next_route = F.pad(next_route, (0, N_total - next_route.size(1)), value=pad_val)

                    if torch.equal(next_route, new_route):
                        break
                    new_route = next_route.clone()

            # ----- deduplicate by pickup sets & prune to top-β -----
            route_all = torch.cat([old_route, best_route], dim=0)
            best_route, _ = dedup_routes_by_pickups(route_all)

            out = repeat_interleave(
                best_route.size(0),
                positions=positions[b:b+1],
                revenues=revenues[b:b+1],
                T_max=T_max[b],
            )    
            (
                positions_b, revenues_b, T_max_b
            ) = (
                out["positions"], out["revenues"], out["T_max"]
            )
            best_rev = compute_obj(best_route, positions_b, revenues_b, T_max_b, best_route.size(0), obj).view(-1)

            k = min(int(beta), best_route.size(0))
            vals, idx = torch.topk(best_rev.view(-1), k=k, largest=True, sorted=True)
            best_route = best_route.index_select(0, idx)   # (k, L)
            best_rev   = vals                              # (k,)

            # ----- optional 2-opt on current beam -----
            beam_now = best_route.size(0)

            out = repeat_interleave(
                beam_now,
                depots=depots[b:b+1],
                requests=requests[b:b+1],
                positions=positions[b:b+1],
                revenues=revenues[b:b+1],
                T_max=T_max[b],
            )    
            (
                depots_b, requests_b, positions_b, revenues_b, T_max_b
            ) = (
                out["depots"], out["requests"], out["positions"], out["revenues"], out["T_max"]
            )

            route_list, _ = two_opt_improve(
                best_route.tolist(),
                depots_b,
                requests_b
            )
            
            seqs = [torch.tensor(t, dtype=torch.long, device=device) for t in route_list]
            best_route = pad_sequence(seqs, batch_first=True, padding_value=pad_val)
            if best_route.size(1) < route0.size(1):
                best_route = F.pad(best_route, (0, route0.size(1) - best_route.size(1)), value=pad_val)

            # best_routes_record.append(best_route)
            
        
        # ----- hill climbing (apply to all rows in the beam) -----
        # if not is_hill_climb:
        
        new_route = best_route.clone()
        while True:
            revs_b = compute_obj(new_route, positions_b, revenues_b, T_max_b, beam_now, obj).view(-1)
            br = best_rev.view(-1)

            # where the new score is better
            mask = revs_b > br                     # shape: (beam,)

            # update the scores (pick per position)
            best_rev = torch.where(mask, revs_b, br)

            # update the routes: replace rows where mask==True
            # (mask indexes the first dimension)
            best_route = best_route.clone()
            best_route[mask] = new_route[mask]

            # try repairs from multiple starting nodes
            next_route = []
            for s in range(beam_now):
                repaired = repair(
                    new_route[s],
                    positions[b:b+1], loads_flat[b], revenues[b:b+1],
                    Q_max[b], T_max[b],
                    R, D, device, obj, False
                )
                next_route.append(
                    repaired if torch.is_tensor(repaired) else torch.tensor(repaired, dtype=torch.long, device=device)
                )

            # pad all repaired routes to same length
            pad_val = D - 1
            next_route = pad_sequence(next_route, batch_first=True, padding_value=pad_val)
            if next_route.size(1) < route0.size(1):
                next_route = F.pad(next_route, (0, N_total - next_route.size(1)), value=pad_val)

            if torch.equal(next_route, new_route):
                break
            new_route = next_route.clone()
        
        best_routes_record.append(best_route)
        best_revs_record.append(best_rev)
            
        # pick the single best row for this batch
        final_idx = best_rev.view(-1).argmax().item()
        best_routes_out.append(best_route[final_idx])
        best_revs_out.append(best_rev[final_idx])
    

    all_routes_record = {
        "route": best_routes_record,   # list[Tensor with shape [k_i, 42]]
        "obj":   best_revs_record
    }

    # return torch.stack(best_routes_out, dim=0), torch.stack(best_revs_out, dim=0)
    return torch.stack(best_routes_out, dim=0), all_routes_record
    # return best_routes_record, _
