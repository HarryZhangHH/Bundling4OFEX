import numpy as np
import networkx as nx
import torch
import copy
import math

from utils.pdp_functions import *

# def greedy_pdp(depots, requests):
#     """
#     Greedy PDP that uses your build_pdp_graph & mask_pdp unchanged.
    
#     Inpicts
#     ------
#       depots    : (1,2,4) tensor  [[x,y,Q_max,T_max] for start/end]
#       requests  : (1,R,6) tensor  [[px,py,dx,dy,load,revenue]]
#     Returns
#     -------
#       route     : list of node-indices visited (0=start,1=end,2.. pickups,2+R.. deliveries)
#       total_T   : float total distance
#     """
#     # unpack sizes
#     device   = depots.device
#     R        = requests.size(1)
#     N        = 2 + 2*R

#     # state (batch size = 1 assumed)
#     mask     = torch.zeros(1, N, dtype=torch.bool, device=device)
#     vis_node= torch.zeros_like(mask)
#     undel    = torch.zeros(1, R, dtype=torch.bool, device=device)
#     current  = torch.zeros(1, 1, dtype=torch.long, device=device)  # start=0
#     Q        = torch.zeros(1, 1,device=device)
#     T        = torch.zeros(1, 1,device=device)
#     route    = [0]

#     # pre-extract Q_max, T_max, end coords as numpy
#     Q_max = torch.tensor(depots[0,0,2].item(),device=device).view(1,1,1)
#     T_max = torch.tensor(depots[0,0,3].item(),device=device).view(1,1,1)
#     # convert depot coords to numpy for build_pdp_graph:
#     end_xy_np = depots[0,1,:2].cpic().numpy()

#     while True:
#         # 1) update mask
#         mask, vis_node = mask_pdp(
#             mask, vis_node,
#             depots, requests,
#             current, undel, Q, T, bsz=1
#         )
#         print(mask)

#         # 2) list feasible nodes
#         feasible = (~mask[0]).nonzero(as_tuple=True)[0].cpic().tolist()
#         print(feasible)
#         if not feasible:
#             break

#         # 3) current (x,y) in numpy
#         # build positions tensor once
#         coords_depots     = depots[:, :2, :2]       # (1,2,2)
#         coords_pickups    = requests[:, :, :2]      # (1,R,2)
#         coords_deliveries = requests[:, :, 2:4]     # (1,R,2)
#         positions  = torch.cat([coords_depots, coords_pickups, coords_deliveries], dim=1)
#         pos_xy     = positions[0].cpic().numpy()         # (N,2)
#         current_i  = current.item()
#         current_xy = pos_xy[current_i]

#         best_cost = float('inf')
#         best_revenue = -float('inf')
#         best_i    = None

#         # 4) evaluate each candidate
#         for i in feasible:
#             dist = T.clone()
#             # only allow end=1 if all deliveries done
#             if i==1 and undel.any():
#                 continue

#             # a) direct leg cost
#             next_xy = pos_xy[i]
#             dist += np.hypot(current_xy[0]-next_xy[0], current_xy[1]-next_xy[1])

#             # b) build small graph for the future leg:
#             #    from next_xy → all pending deliveries (in undel) → end_xy_np
#             #    we pass only the coords of still-pending deliveries
#             undel_idxs = undel[0].nonzero(as_tuple=True)[0].cpic().tolist()
#             # if we just picked up a new pickup, add its paired delivery
#             if 2 <= i < 2+R:
#                 j = i-2
#                 if not undel[0,j]:
#                     undel_idxs.append(j)
#             undel_coords = np.stack([requests[0,j,2:4].cpic().numpy() for j in undel_idxs], axis=0) \
#                            if undel_idxs else np.zeros((0,2))
#             # note: build_pdp_graph expects: (current,end,undel_coords)
#             G = build_pdp_graph(next_xy, end_xy_np, undel_coords)

#             # c) shortest‐path cost from node 0→1 in G
#             if i != 1:
#                 try:
#                     dist += nx.dijkstra_path_length(G, source=0, target=1, weight='weight')
#                 except nx.NetworkXNoPath:
#                     print("error")
#                     continue  # infeasible future 
                
#             revenue = dist.item()
#             if 2 <= i < 2+R:
#                 revenue += 1/2 * requests[0,i-2,5]
#             elif i >= 2+R:
#                 revenue += 1/2 * requests[0,i-2-R,5]
            
#             if revenue > best_revenue:
#                 best_cost, best_revenue, best_i = dist.item(), revenue, i

#         print(best_i)

#         # 5) if none feasible, bail out
#         if best_i is None:
#             break

#         # 6) commit to best_i
#         # update T
#         # but T currently is shape (1,1,1) so:
#         T -= best_cost

#         # update Q or undel and vis_node
#         if 2 <= best_i < 2+R:
#             # pickup
#             j = best_i - 2
#             load = requests[0,j,4].item()
#             undel[0,j] = True
#         elif best_i >= 2+R:
#             # delivery
#             j = best_i - (2+R)
#             load = - requests[0,j,4].item()
#             undel[0,j] = False
#         else: 
#             load = 0
         
#         Q += load
#         vis_node[0,best_i] = True
#         mask[0,best_i] = True

#         current = torch.tensor([best_i], device=device)
#         route.append(best_i)

#         # stop if we reached the end depot
#         if best_i == 1:
#             print(best_i)
#             break
#     return route, T.item()


import torch
import random
import numpy as np


def construct_cpdp(depots, requests, method="greedy", precise=True, seed=1234):
    """
    Unified batched constructive heuristic for CPDP.

    Supported methods
    -----------------
      - "greedy"  : choose feasible node with max (revenue - distance)
      - "random"  : randomly choose a feasible non-end node
      - "nearest" : choose nearest feasible non-end node

    Inputs
    ------
      depots    : (B,2,4) tensor  [[x,y,Q_max,T_max] for start,end]
      requests  : (B,R,6) tensor  [[px,py,dx,dy,load,revenue]]

    Returns
    -------
      routes  : list of length B, each a Python list of node-indices visited
      L       : (B,) float tensor of total travelled distance per example
    """
    assert method in {"greedy", "random", "nearest"}, f"Unknown method: {method}"

    seed_everything(seed)
    device = depots.device
    B, R_req, D = requests.size(0), requests.size(1), depots.size(1)
    N = D + 2 * R_req
    zero_to_bsz = torch.arange(B, device=device)
    end_idx = D - 1

    # State
    mask     = torch.zeros(B, N, dtype=torch.bool, device=device)
    vis_node = torch.zeros_like(mask)
    undel    = torch.zeros(B, R_req, dtype=torch.bool, device=device)
    current  = torch.zeros(B, dtype=torch.long, device=device)   # start depot index = 0
    Q        = torch.zeros(B, 1, device=device)
    T        = torch.zeros(B, 1, device=device)

    # Static data
    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)

    dist_matrix = torch.stack(
        [torch.cdist(positions[b], positions[b]) for b in range(B)],
        dim=0
    )  # (B,N,N)

    routes = [[0] for _ in range(B)]

    while True:
        # 1) update feasibility mask
        mask, vis_node, path = mask_cpdp(
            mask, vis_node,
            depots, requests,
            current.view(B, 1),
            undel,
            Q.view(B, 1),
            T.view(B, 1),
            positions, Q_max, T_max,
            dist_matrix,
            precise
        )

        # 2) current-to-all distances
        dists = dist_matrix[zero_to_bsz, current, :]   # (B,N)

        # default choice: end depot
        best_i = torch.full((B,), end_idx, dtype=torch.long, device=device)

        if method == "greedy":
            profits = revenues.view(B, N) - dists
            profits_masked = profits.clone()

            # mask infeasible nodes
            profits_masked = profits_masked.masked_fill(mask, -float('inf'))

            # do not choose end depot early if there are feasible non-end nodes
            profits_no_end = profits_masked.clone()
            profits_no_end[:, end_idx] = -float('inf')

            best_non_end_val, best_non_end_i = profits_no_end.max(dim=1)

            has_non_end = ~torch.isinf(best_non_end_val)
            best_i[has_non_end] = best_non_end_i[has_non_end]
            best_i[~has_non_end] = end_idx

        elif method == "nearest":
            dists_masked = dists.clone()
            dists_masked = dists_masked.masked_fill(mask, float('inf'))

            # do not choose end depot early if there are feasible non-end nodes
            dists_no_end = dists_masked.clone()
            dists_no_end[:, end_idx] = float('inf')

            best_non_end_val, best_non_end_i = dists_no_end.min(dim=1)

            has_non_end = ~torch.isinf(best_non_end_val)
            best_i[has_non_end] = best_non_end_i[has_non_end]
            best_i[~has_non_end] = end_idx

        elif method == "random":
            for b in range(B):
                if current[b].item() == end_idx:
                    best_i[b] = end_idx
                    continue

                feasible = (~mask[b]).nonzero(as_tuple=False).squeeze(-1)

                if feasible.numel() == 0:
                    best_i[b] = end_idx
                    continue

                feasible_non_end = feasible[feasible != end_idx]

                if feasible_non_end.numel() > 0:
                    rand_pos = torch.randint(
                        low=0,
                        high=feasible_non_end.numel(),
                        size=(1,),
                        device=device
                    ).item()
                    best_i[b] = feasible_non_end[rand_pos]
                else:
                    best_i[b] = end_idx

        # keep samples already at end depot fixed there
        best_i[current == end_idx] = end_idx

        # 3) transition cost
        best_cost = dist_matrix[zero_to_bsz, current, best_i]  # (B,)

        # 4) update load
        Q += loads[zero_to_bsz, best_i].view(B, 1)

        # 5a) pickups
        pic_mask = (best_i >= D) & (best_i < D + R_req)
        if pic_mask.any():
            pic_idx = best_i[pic_mask] - D
            undel[pic_mask, pic_idx] = True

        # 5b) deliveries
        del_mask = (best_i >= D + R_req)
        if del_mask.any():
            del_idx = best_i[del_mask] - D - R_req
            undel[del_mask, del_idx] = False

        # 5c) record visited
        vis_node[zero_to_bsz, best_i] = True

        # 6) append routes
        for b in range(B):
            routes[b].append(best_i[b].item())

        # 7) update time and current node
        T[:, 0] += best_cost
        current = best_i

        # 8) terminate when all are at end depot
        if (current == end_idx).all():
            break

    L = compute_route_length(routes, positions)
    return routes, L


def multi_start_greedy_cpdp(depots, requests, starting_nodes=10, precise=True, seed=1234):
    """
    Batched greedy PDP over B examples simultaneously.
    
    Inpicts
    ------
      depots    : (B,2,4) tensor  [[x,y,Q_max,T_max] for start,end]
      requests  : (B,R,6)   tensor  [[px,py,dx,dy,load,revenue]]
    
    Returns
    -------
      routes  : list of length B, each a Python list of node-indices visited
      T_total : (B,) float tensor of total travelled distance per example
    """
    seed_everything(seed)
    
    device = depots.device
    B, R, D = requests.size(0), requests.size(1), depots.size(1)
    N = D + 2*R
    zero_to_bsz = torch.arange(B, device=device)

    # State
    mask     = torch.zeros(B, N, dtype=torch.bool, device=device)
    vis_node = torch.zeros_like(mask)
    undel    = torch.zeros(B, R, dtype=torch.bool, device=device)
    current  = torch.zeros(B, dtype=torch.long, device=device)      # shape (B,)
    Q        = torch.zeros(B, 1, device=device)                     # we won’t really use Q_max here
    T        = torch.zeros(B, 1, device=device)                     # accumulated distance

    # Pre-build static data
    # positions[b] is (N,2): depot0, depot1, pickups, deliveries
    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)

    dist_matrix = torch.stack([torch.cdist(positions[b], positions[b]) 
                                for b in range(B)], dim=0)  # (bsz,N,N)
    
    # 1) batch‐mask update
    mask, vis_node, path = mask_cpdp(
        mask, vis_node,
        depots, requests,
        current.view(B,1),   # mask_pdp expects (B,1)
        undel,
        Q.view(B,1),       # shapes must match your mask_pdp signature
        T.view(B,1),
        positions, Q_max, T_max,
        dist_matrix,
        precise
    )

    # 2) compicte direct costs from current→all nodes
    #    gather current‐coords:
    cur_xy  = positions[zero_to_bsz, current]   # (B,2)
    deltas  = positions - cur_xy.unsqueeze(1)   # (B,N,2)
    dists   = deltas.norm(dim=2)                # (B,N)
    profits = revenues.view(B,N) - dists        # (B,N)
    
    # 3) mask out infeasible by setting cost→+inf
    profits_masked = profits.masked_fill(mask, -float('inf'))

    # multi-start
    valid = ~mask
    num_valid = valid.sum(dim=1)         # (B,)
    
    assert (num_valid != 0).all(), f"Batch rows {torch.nonzero(num_valid == 0, as_tuple=False).flatten().tolist()} have NO valid moves!"

    # clamp so we never ask for more than *every* row has
    E_eff = torch.minimum(num_valid, torch.full_like(num_valid, starting_nodes))
    E_min = int(E_eff.min().item())      # the smallest valid count across the batch

    # get the “core” top-k from E_min
    _, topk_idx = profits_masked.topk(E_min, dim=1)  # shape (B, E_min)
    
    if E_min >= starting_nodes:
        final_idx = topk_idx.reshape(-1)
    else:
        # now pad out to (B, E)
        final_idx  = torch.empty((B, starting_nodes), dtype=torch.long, device=device)

        for b in range(B):
            ev = int(E_eff[b].item())           # how many *this* row really has
            _, topk_idx = profits_masked[b].topk(ev)  # (B,N)
            # copy the “real” ones
            final_idx[b, :ev]  = topk_idx
            # if we still need more, sample *with replacement* from the remaining valid actions
            if ev < starting_nodes:
                rem = starting_nodes - ev
                allowed = torch.nonzero(valid[b], as_tuple=False).squeeze(1)
                # sample rem times from `allowed`
                extra = allowed[torch.randint(0, allowed.numel(), (rem,), device=device)]
                final_idx[b, ev:]  = extra
                # and grab their true log-prob

        final_idx = final_idx.reshape(-1)
    
    current = final_idx.clone()
    
    routes = [torch.zeros(B*starting_nodes, dtype=torch.int, device=device), current]

    out = repeat_interleave(
        starting_nodes,
        depots=depots,
        requests=requests,
        positions=positions,
        revenues=revenues,
        loads=loads,
        dist_matrix=dist_matrix,
        dists=dists,
        Q_max=Q_max,
        T_max=T_max,
        Q=Q,
        T=T,
        mask=mask,
        vis_node=vis_node,
        undel=undel,
    )    
    (
        depots, requests, positions, revenues, loads,
        dist_matrix, dists, Q_max, T_max, Q_flat, T_flat,
        mask, vis_node, undel
    ) = (
        out["depots"], out["requests"], out["positions"], out["revenues"], out["loads"],
        out["dist_matrix"], out["dists"], out["Q_max"], out["T_max"], out["Q"], out["T"],
        out["mask"], out["vis_node"], out["undel"]
    )
    
    b_e         = B * starting_nodes
    zero_to_b_e = torch.arange(b_e, device=device) # [0,1,...,bsz-1]
    
    distance = dists[zero_to_b_e, current] # (B,)

    while True:
        # 5) commit updates **vectorized**
        Q_flat += loads[zero_to_b_e, current]

        # 5a) pickups
        pic_mask = (current >= D) & (current < D+R)       # (B,)
        if pic_mask.any():
            pic_idx = current[pic_mask] - D               # which request
            # mark them undelivered
            undel[pic_mask, pic_idx] = True  

        # 5b) deliveries
        del_mask = (current >= D+R)
        if del_mask.any():
            del_idx = current[del_mask] - D - R
            undel[del_mask, del_idx] = False

        # 5c) record that we’ve visited best_i
        vis_node[zero_to_b_e, current] = True

        # 7) update total distance & current position
        T_flat[:,0] += distance

        # 8) if every batch chose “end” (or was inf), we’re done
        if (current == D-1).all():
            break

        # 1) batch‐mask update
        mask, vis_node, path = mask_cpdp(
            mask, vis_node,
            depots, requests,
            current.view(B*starting_nodes,1),   # mask_pdp expects (B,1)
            undel,
            Q_flat.view(B*starting_nodes,1),       # shapes must match your mask_pdp signature
            T_flat.view(B*starting_nodes,1),
            positions, Q_max, T_max,
            dist_matrix,
            precise
        )

        # 2) compicte direct costs from current→all nodes
        #    gather current‐coords:
        cur_xy  = positions[zero_to_b_e, current]   # (B,2)
        deltas  = positions - cur_xy.unsqueeze(1)   # (B,N,2)
        dists   = deltas.norm(dim=2)                # (B,N)
        profits = revenues.view(B*starting_nodes,N) - dists        # (B,N)
        
        # 3) mask out infeasible by setting cost→+inf
        profits_masked = profits.masked_fill(mask, -float('inf'))
       
        # 4) pick the nearest feasible node per batch
        best_profit, best_i = profits_masked.max(dim=1) # each is (B,)
        distance = dists[zero_to_b_e, best_i] # (B,)

        current = best_i.clone()
        routes.append(current) 
    
    routes = torch.stack(routes,dim=1) # size(col_index)=(bsz, nb_nodes)
    # print(routes)
    # L = compute_route_length(routes, positions)
    R = compute_collected_revenue(routes, revenues)

    max_idxs = R.view(B, starting_nodes).argmax(dim=1)
    routes = routes.view(B, starting_nodes, -1)         # (B, E, L)
    best_route = routes[zero_to_bsz, max_idxs]

    # print(L, R)
    # return a list of Python routes plus final T
    return best_route.tolist(), routes

def two_opt_improve(init_routes, depots, requests):
    """
        Apply two_opt_improve independently to each batch element.
        init_routes: list of length B
        depots:      Tensor     (B, 2, 4)
        requests:    Tensor     (B, R, 6)
        Returns:
        best_routes: LongTensor (B, L_best)  # may pad if lengths differ
        best_lens:   FloatTensor (B,)
    """
    B, nb_requests = requests.size(0), requests.size(1)
    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)
    lens = compute_route_length(init_routes, positions).view(B,1)
    
    for b in range(B):
        improved = True
        best_route = init_routes[b]
        best_len   = lens[b]

        # remove all duplicates end node 1
        best_route = [
            x
            for i, x in enumerate(best_route)
            if not (x == 1 and i != best_route.index(1))
        ]
        L = len(best_route)

        pos, loa, Q_m, T_m = positions[b].view(1,-1,2), loads[b].view(1,-1,1), Q_max[b].view(1,1), T_max[b].view(1,1)

        while improved:
            improved = False
            # don't touch endpoints 0 and N-1
            for i in range(1, L-2):
                for j in range(i+1, L-1):
                    candidate = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    # must start at 0 and end at 1
                    if candidate[0]!=0 or candidate[-1]!=1:
                        continue

                    try:
                        check_route_feasibility_batch([candidate], pos, loa, Q_m, T_m, nb_requests)
                    except AssertionError:
                        continue  # infeasible route

                    cand_len = compute_route_length([candidate], pos)
                    if cand_len + 1e-6 < best_len:
                        best_route, best_len = candidate, cand_len
                        improved = True
                        break
                if improved:
                    break
        init_routes[b] = best_route
        lens[b]        = best_len
    
    return init_routes, lens
