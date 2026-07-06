from typing import Tuple, Callable, List, Optional, Union, Dict
import time
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.pdp_functions import *
from algs.large_neighborhood_search import repair
from algs.greedy_search import two_opt_improve
from utils.functions import *


def multi_start_large_neighborhood_search(
    depots: torch.Tensor,                 # (B, D, 2)
    requests: torch.Tensor,               # (B, R, ...)
    tour_init: torch.Tensor,              # (B, beam, L) or (B, L)
    *,
    starting_nodes: int,
    beta: int = 3,
    alpha: float = 1.0,
    max_iters: int = 200,
    k_max: int = 5,
    is_hill_climb: bool = False,
    obj: str = "revenue",
    seed: int = 1234,
    time_cap: Optional[float] = None,
    device: Optional[torch.device] = None,

    # ------------------------------------------------------------------
    # Unified destroy configuration.
    # Either use destroy_method as a compact string,
    # or set destroy_score / destroy_size / freq_mode separately.
    # ------------------------------------------------------------------
    destroy_method: Optional[str] = None,
    destroy_score: str = "softmax",       # "softmax", "regret", "random", "softmax_regret"
    destroy_size: str = "coverage",       # "coverage", "fix", "increase"
    freq_mode: str = "mid",               # "mid", "high", "low", "mix"

    # ------------------------------------------------------------------
    # Coverage-aware parameters.
    # For destroy_size="coverage":
    # select requests until at least target_coverage fraction of beam rows
    # have at least min_destroy_per_route removed requests.
    # ------------------------------------------------------------------
    target_coverage: float = 1.0,
    min_destroy_per_route: int = 1,

    # ------------------------------------------------------------------
    # Score-combination parameters.
    # ------------------------------------------------------------------
    regret_lambda: float = 1.0,
    freq_lambda: float = 1.0,
    random_lambda: float = 1.0,
    coverage_lambda: float = 0,
    freq_target: float = 0.6,

    # Whether to avoid destroying the same request twice within one instance.
    avoid_repeated_destroy: bool = True,
) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
    """
    Multi-start LNS for m1-PDSTSP.

    Destroy strategy is controlled by:

        destroy_score:
            "softmax"         -> frequency-based score
            "regret"          -> cheap regret proxy
            "random"          -> random score
            "softmax_regret"  -> regret score + frequency score

        destroy_size:
            "coverage"        -> adaptive destroy size:
                                  keep sampling until enough trajectories each
                                  have at least min_destroy_per_route removed requests,
                                  or until k_max is reached.
            "fix"             -> select exactly k_max requests
            "increase"        -> select min(iteration, k_max) requests

        freq_mode:
            "high"            -> prefer high-frequency requests
            "low"             -> prefer low-frequency requests
            "mid"             -> prefer requests close to freq_target
            "mix"             -> first selected request is high-frequency biased,
                                  remaining selected requests are low-frequency biased

    Compact examples:
        "softmax_coverage_mid"
        "regret_fix_low"
        "random_increase_high"
        "softmax+regret_coverage_mix"
        "softmax_regret_coverage_mix"

    Returns:
        best_routes: (B, L)
        all_routes_record:
            {
                "route": list[Tensor], each (beam_i, L),
                "obj":   list[Tensor], each (beam_i,)
            }
    """

    # ------------------------------------------------------------------
    # Parse destroy configuration
    # ------------------------------------------------------------------

    def parse_destroy_config(
        method: Optional[str],
        score: str,
        size: str,
        mode: str,
    ) -> Tuple[str, str, str]:
        allowed_scores = {"softmax", "regret", "random", "softmax_regret"}
        allowed_sizes = {"coverage", "fix", "increase"}
        allowed_modes = {"mid", "high", "low", "mix"}

        if method is None:
            score = score.lower().replace("+", "_")
            score = "softmax_regret" if score in {"softmax_regret", "softmax+regret"} else score
            size = size.lower()
            mode = mode.lower()

            if score not in allowed_scores:
                raise ValueError(f"destroy_score must be one of {allowed_scores}, got {score}.")
            if size not in allowed_sizes:
                raise ValueError(f"destroy_size must be one of {allowed_sizes}, got {size}.")
            if mode not in allowed_modes:
                raise ValueError(f"freq_mode must be one of {allowed_modes}, got {mode}.")

            return score, size, mode

        m = method.lower().replace("+", "_plus_")
        parts = m.split("_")

        parsed_mode = mode.lower()
        if parts[-1] in allowed_modes:
            parsed_mode = parts.pop()

        parsed_size = size.lower()
        if parts[-1] in allowed_sizes:
            parsed_size = parts.pop()

        parsed_score = "_".join(parts)
        if parsed_score in {"softmax_plus_regret", "softmax_regret"}:
            parsed_score = "softmax_regret"

        if parsed_score not in allowed_scores:
            raise ValueError(
                f"Cannot parse destroy score from destroy_method={method}. "
                f"Expected one of {allowed_scores}."
            )
        if parsed_size not in allowed_sizes:
            raise ValueError(
                f"Cannot parse destroy size from destroy_method={method}. "
                f"Expected one of {allowed_sizes}."
            )
        if parsed_mode not in allowed_modes:
            raise ValueError(
                f"Cannot parse freq mode from destroy_method={method}. "
                f"Expected one of {allowed_modes}."
            )

        return parsed_score, parsed_size, parsed_mode

    destroy_score, destroy_size, freq_mode = parse_destroy_config(
        destroy_method,
        destroy_score,
        destroy_size,
        freq_mode,
    )

    # ------------------------------------------------------------------
    # Basic setup
    # ------------------------------------------------------------------

    if device is None:
        device = depots.device

    B = requests.size(0)
    R = requests.size(1)
    D = depots.size(1)

    pad_val = D - 1

    seed_everything(seed)

    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)
    loads_flat = loads.squeeze(-1)

    destroyed: List[set] = [set() for _ in range(B)]

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def repeat_instance(b: int, n: int, include_depots_requests: bool = False):
        kwargs = dict(
            positions=positions[b:b + 1],
            revenues=revenues[b:b + 1],
            T_max=T_max[b],
        )

        if include_depots_requests:
            kwargs.update(
                depots=depots[b:b + 1],
                requests=requests[b:b + 1],
            )

        return repeat_interleave(n, **kwargs)

    def eval_routes(routes: torch.Tensor, b: int) -> torch.Tensor:
        out = repeat_instance(b, routes.size(0), include_depots_requests=False)

        return compute_obj(
            routes,
            out["positions"],
            out["revenues"],
            out["T_max"],
            routes.size(0),
            obj,
        ).view(-1)

    def as_long_tensor(x) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.long)
        return torch.tensor(x, dtype=torch.long, device=device)

    def pad_to_length(routes: torch.Tensor, target_len: int) -> torch.Tensor:
        if routes.size(1) < target_len:
            routes = F.pad(routes, (0, target_len - routes.size(1)), value=pad_val)
        return routes

    # ------------------------------------------------------------------
    # Route/request helpers
    # ------------------------------------------------------------------

    def compute_request_presence(routes: torch.Tensor) -> torch.Tensor:
        """
        routes: (beam, L)

        Returns:
            presence: (beam, R)
            presence[s, r] = True iff pickup node D+r appears in route s.
        """
        beam_size, route_len = routes.shape
        dev = routes.device

        is_pick = (routes >= D) & (routes < D + R)
        req_ids = torch.where(is_pick, routes - D, torch.full_like(routes, -1))

        presence = torch.zeros(beam_size, R, dtype=torch.bool, device=dev)

        row_ids = (
            torch.arange(beam_size, device=dev)
            .unsqueeze(1)
            .expand(beam_size, route_len)
            .reshape(-1)
        )

        flat_req_ids = req_ids.reshape(-1)
        valid = flat_req_ids >= 0

        presence[row_ids[valid], flat_req_ids[valid]] = True

        return presence

    def normalize_score(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        Normalize valid entries only.
        Invalid entries are assigned a very negative value.
        """
        out = torch.full_like(x, -1e9)

        if valid.any():
            valid_x = x[valid]
            std = valid_x.std(unbiased=False).clamp_min(1e-6)
            out[valid] = (valid_x - valid_x.mean()) / std

        return out

    def destroyed_to_request_ids(destroyed_set: set, dev: torch.device) -> torch.Tensor:
        """
        Convert stored destroyed ids to raw request ids r in [0, R).

        Supports both:
            D+r pickup node ids
            raw request ids r
        """
        req_ids = []

        for x in destroyed_set:
            xi = int(x)

            if D <= xi < D + R:
                req_ids.append(xi - D)
            elif 0 <= xi < R:
                req_ids.append(xi)

        if len(req_ids) == 0:
            return torch.empty(0, dtype=torch.long, device=dev)

        return torch.tensor(req_ids, dtype=torch.long, device=dev).unique()

    def destroy_keep_shape(
        routes: torch.Tensor,
        *,
        pick_ids: Optional[Union[torch.Tensor, List[int]]] = None,
        pickup_idx: Optional[int] = None,
        delivery_idx: Optional[int] = None,
        pad_value: int = pad_val,
    ) -> torch.Tensor:
        """
        Remove pickup-delivery pairs and right-pad to keep original route length.

        pick_ids should be pickup NODE ids, i.e. D+r.
        Corresponding delivery node ids are D+r+R.
        """
        assert routes.dim() in (1, 2), f"routes must be 1D or 2D, got {routes.shape}"

        dev = routes.device
        dtype = routes.dtype

        if pick_ids is not None:
            pick_ids = torch.as_tensor(pick_ids, device=dev, dtype=dtype)

            if pick_ids.dim() == 1:
                to_remove = torch.cat([pick_ids, pick_ids + R], dim=0)
            elif pick_ids.dim() == 2:
                to_remove = torch.cat([pick_ids, pick_ids + R], dim=1)
            else:
                raise ValueError(f"pick_ids must be 1D or 2D, got {pick_ids.shape}")

        else:
            if pickup_idx is None or delivery_idx is None:
                raise ValueError("Provide either pick_ids or pickup_idx/delivery_idx.")

            to_remove = torch.tensor([pickup_idx, delivery_idx], device=dev, dtype=dtype)

        if routes.dim() == 1:
            if to_remove.dim() == 2:
                to_remove = to_remove[0]

            keep = ~torch.isin(routes, to_remove)
            kept = routes[keep]

            out = routes.new_full((routes.size(0),), pad_value)
            out[:kept.numel()] = kept
            return out

        batch_size, route_len = routes.shape
        out = routes.new_full((batch_size, route_len), pad_value)

        if to_remove.dim() == 1:
            keep = ~torch.isin(routes, to_remove)
        else:
            keep = ~(routes.unsqueeze(-1) == to_remove.unsqueeze(1)).any(dim=-1)

        for row in range(batch_size):
            kept = routes[row][keep[row]]
            out[row, :kept.numel()] = kept

        return out

    def dedup_routes_by_pickups(routes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Keep first occurrence for each unique pickup set.
        """
        assert routes.dim() == 2, f"routes must be (K,L), got {tuple(routes.shape)}"

        K, L = routes.shape
        dev = routes.device

        is_pick = (routes >= D) & (routes < D + R)
        req_ids = torch.where(is_pick, routes - D, torch.full_like(routes, -1))

        pickup_presence = torch.zeros(K, R, dtype=torch.bool, device=dev)

        row_ids = (
            torch.arange(K, device=dev)
            .unsqueeze(1)
            .expand(K, L)
            .reshape(-1)
        )

        flat_req_ids = req_ids.reshape(-1)
        valid = flat_req_ids >= 0

        pickup_presence[row_ids[valid], flat_req_ids[valid]] = True

        _, inv = torch.unique(pickup_presence, dim=0, return_inverse=True)

        keep_idx = torch.stack([
            (inv == group).nonzero(as_tuple=False)[0, 0]
            for group in range(inv.max().item() + 1)
        ]).to(dev)

        return routes[keep_idx], keep_idx

    # ------------------------------------------------------------------
    # Destroy scoring
    # ------------------------------------------------------------------

    def request_revenue_score(
        presence: torch.Tensor,
        revenues_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lower-revenue requests get higher destroy score.
        """
        dev = presence.device
        freq = presence.float().mean(dim=0)
        valid = freq > 0

        rev = revenues_b[0]
        if rev.dim() > 1:
            rev = rev.squeeze(-1)

        pickup_node_ids = torch.arange(D, D + R, device=dev)
        req_revenue = rev[pickup_node_ids].float()

        return normalize_score(-req_revenue, valid)

    def frequency_score_for_mode(
        freq: torch.Tensor,
        mode: str,
        step: int,
    ) -> torch.Tensor:
        """
        Frequency preference.

        mode:
            high -> prefer high-frequency requests
            low  -> prefer low-frequency requests
            mid  -> prefer requests close to freq_target
            mix  -> first request high-frequency, remaining requests low-frequency
        """
        valid = freq > 0

        if mode == "high":
            raw = freq
        elif mode == "low":
            raw = -freq
        elif mode == "mid":
            raw = -torch.abs(freq - freq_target)
        elif mode == "mix":
            raw = freq if step == 0 else -freq
        else:
            raise ValueError(f"Unknown freq_mode: {mode}")

        return normalize_score(raw, valid)

    def regret_score(
        presence: torch.Tensor,
        revenues_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cheap regret proxy.

        Larger score means more removable.

        Components:
            lower request revenue -> more removable
            lower frequency       -> less stable -> more removable
        """
        freq = presence.float().mean(dim=0)
        valid = freq > 0

        low_rev = request_revenue_score(presence, revenues_b)
        instability = normalize_score(1.0 - freq, valid)

        return low_rev + instability

    def destroy_base_score(
        score_type: str,
        mode: str,
        step: int,
        presence: torch.Tensor,
        revenues_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build request-level base score.

        Larger score means more likely to destroy.
        """
        dev = presence.device
        freq = presence.float().mean(dim=0)
        valid = freq > 0

        freq_score = frequency_score_for_mode(freq, mode, step)
        reg_score = regret_score(presence, revenues_b)

        if score_type == "softmax":
            score = freq_lambda * freq_score

        elif score_type == "regret":
            score = regret_lambda * reg_score

        elif score_type == "random":
            rand = normalize_score(torch.rand(R, device=dev), valid)
            score = random_lambda * rand

        elif score_type == "softmax_regret":
            score = regret_lambda * reg_score + freq_lambda * freq_score

        else:
            raise ValueError(f"Unknown destroy_score: {score_type}")

        return torch.where(valid, score, torch.full_like(score, -1e9))

    # ------------------------------------------------------------------
    # Destroy-set selection
    # ------------------------------------------------------------------

    def select_destroy_set(
        routes: torch.Tensor,
        revenues_b: torch.Tensor,
        destroyed_set: set,
        *,
        iter_idx: int,
    ) -> torch.Tensor:
        """
        Select pickup node ids D+r according to:

            destroy_score ∈ {softmax, regret, random, softmax_regret}
            destroy_size  ∈ {coverage, fix, increase}
            freq_mode     ∈ {mid, high, low, mix}

        New coverage rule:
            If destroy_size == "coverage", adaptively sample requests until
            at least target_coverage fraction of trajectories each have at least
            min_destroy_per_route selected requests, or k_max is reached.

        Returns:
            picked_node_ids: 1D tensor containing pickup node ids.
        """
        dev = routes.device
        beam_size = routes.size(0)

        presence = compute_request_presence(routes)       # (beam, R)
        freq = presence.float().mean(dim=0)               # (R,)

        available = freq > 0

        if avoid_repeated_destroy and destroyed_set:
            old_req_ids = destroyed_to_request_ids(destroyed_set, dev)
            if old_req_ids.numel() > 0:
                available[old_req_ids] = False

        candidate_ids = torch.where(available)[0]

        if candidate_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=dev)

        if destroy_size == "coverage":
            k_limit = min(k_max, int(candidate_ids.numel()))
            min_global_selected = k_limit
            k_limit = int(candidate_ids.numel())

        elif destroy_size == "fix":
            k_limit = min(k_max, int(candidate_ids.numel()))
            min_global_selected = k_limit

        elif destroy_size == "increase":
            k_limit = min(max(1, iter_idx), k_max, int(candidate_ids.numel()))
            min_global_selected = k_limit

        else:
            raise ValueError(f"Unknown destroy_size: {destroy_size}")

        if k_limit <= 0:
            return torch.empty(0, dtype=torch.long, device=dev)

        selected: List[int] = []
        selected_mask = torch.zeros(R, dtype=torch.bool, device=dev)

        # removed_count_per_route[s] =
        # how many selected requests are contained in trajectory s.
        removed_count_per_route = torch.zeros(
            beam_size,
            dtype=torch.long,
            device=dev,
        )

        for step in range(k_limit):
            base = destroy_base_score(
                score_type=destroy_score,
                mode=freq_mode,
                step=step,
                presence=presence,
                revenues_b=revenues_b,
            )

            if destroy_size == "coverage":
                # ------------------------------------------------------
                # Row-wise coverage gain:
                # Prefer requests that appear in trajectories that still
                # have fewer than min_destroy_per_route selected removals.
                # ------------------------------------------------------
                under_destroyed = removed_count_per_route < min_destroy_per_route

                coverage_gain = under_destroyed.float() @ presence.float()
                coverage_gain = coverage_gain / max(1, beam_size)

                total_score = base + coverage_lambda * coverage_gain

            else:
                total_score = base

            total_score = torch.where(
                available & (~selected_mask),
                total_score,
                torch.full_like(total_score, -1e9),
            )

            if torch.all(total_score <= -1e8):
                break

            candidate_score = total_score[candidate_ids]
            probs = torch.softmax(alpha * candidate_score, dim=0)

            local_idx = torch.multinomial(probs, num_samples=1).item()
            req_id = int(candidate_ids[local_idx].item())

            selected.append(req_id)
            selected_mask[req_id] = True

            # Update row-wise destroy count.
            removed_count_per_route += presence[:, req_id].long()

            if destroy_size == "coverage":
                satisfied_ratio = (
                    removed_count_per_route >= min_destroy_per_route
                ).float().mean().item()

                if (
                    len(selected) >= min_global_selected
                    and
                    satisfied_ratio >= target_coverage
                ):
                    break

            else:
                if len(selected) >= min_global_selected:
                    break

        if len(selected) == 0:
            return torch.empty(0, dtype=torch.long, device=dev)

        selected_req_ids = torch.tensor(selected, dtype=torch.long, device=dev)

        return selected_req_ids + D

    def make_destroyed_route(
        best_route: torch.Tensor,
        b: int,
        iter_idx: int,
        revenues_b: torch.Tensor,
    ) -> torch.Tensor:
        picked = select_destroy_set(
            routes=best_route,
            revenues_b=revenues_b,
            destroyed_set=destroyed[b],
            iter_idx=iter_idx,
        )

        if picked.numel() == 0:
            return best_route

        destroyed[b].update([int(x) for x in picked.tolist()])

        return destroy_keep_shape(
            best_route,
            pick_ids=picked,
            pad_value=pad_val,
        )

    # ------------------------------------------------------------------
    # Repair and 2-opt helpers
    # ------------------------------------------------------------------

    def repair_beam(routes: torch.Tensor, b: int, target_len: int) -> torch.Tensor:
        repaired_list = []

        for s in range(routes.size(0)):
            repaired = repair(
                routes[s],
                positions[b:b + 1],
                loads_flat[b],
                revenues[b:b + 1],
                Q_max[b],
                T_max[b],
                R,
                D,
                device,
                obj,
                False,
            )
            repaired_list.append(as_long_tensor(repaired))

        repaired_stack = pad_sequence(
            repaired_list,
            batch_first=True,
            padding_value=pad_val,
        )

        return pad_to_length(repaired_stack, target_len)

    def apply_two_opt(routes: torch.Tensor, b: int, target_len: int) -> torch.Tensor:
        out = repeat_instance(b, routes.size(0), include_depots_requests=True)

        route_list, _ = two_opt_improve(
            routes.tolist(),
            out["depots"],
            out["requests"],
        )

        seqs = [as_long_tensor(t) for t in route_list]

        improved = pad_sequence(
            seqs,
            batch_first=True,
            padding_value=pad_val,
        )

        return pad_to_length(improved, target_len)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    best_routes_out: List[torch.Tensor] = []
    best_revs_out: List[torch.Tensor] = []

    best_routes_record: List[torch.Tensor] = []
    best_revs_record: List[torch.Tensor] = []

    for b in tqdm(range(B)):
        seed_everything(seed)

        route0 = tour_init[b].clone()

        if route0.dim() == 1:
            route0 = route0.unsqueeze(0)

        target_len = route0.size(1)

        best_route = route0.clone()
        best_rev = eval_routes(best_route, b)

        # Keep top starting_nodes.
        k = min(int(starting_nodes), best_route.size(0))
        vals, idx = torch.topk(best_rev, k=k, largest=True, sorted=True)

        best_route = best_route.index_select(0, idx)
        best_rev = vals

        start_clock = time.time()
        iters = 0

        while True:
            if time_cap is None:
                if iters >= max_iters:
                    break
            else:
                if time.time() - start_clock >= time_cap:
                    break

            iters += 1
            old_route = best_route.clone()

            out_current = repeat_instance(
                b,
                best_route.size(0),
                include_depots_requests=False,
            )
            revenues_b_current = out_current["revenues"]

            new_route = make_destroyed_route(
                best_route=best_route,
                b=b,
                iter_idx=iters,
                revenues_b=revenues_b_current,
            )

            if torch.equal(new_route, best_route):
                break

            if not is_hill_climb:
                # Evaluate destroyed route directly.
                new_rev = eval_routes(new_route, b)
                improve_mask = new_rev > best_rev

                if improve_mask.any():
                    best_route = best_route.clone()
                    best_route[improve_mask] = new_route[improve_mask]
                    best_rev[improve_mask] = new_rev[improve_mask]

                # Repair current best beam.
                repaired = repair_beam(best_route, b, target_len)
                repaired_rev = eval_routes(repaired, b)

                improve_mask = repaired_rev > best_rev

                if improve_mask.any():
                    best_route = best_route.clone()
                    best_route[improve_mask] = repaired[improve_mask]
                    best_rev[improve_mask] = repaired_rev[improve_mask]

            else:
                # Iterative hill-climb repair from destroyed route.
                while True:
                    new_rev = eval_routes(new_route, b)
                    improve_mask = new_rev > best_rev

                    if improve_mask.any():
                        best_route = best_route.clone()
                        best_route[improve_mask] = new_route[improve_mask]
                        best_rev[improve_mask] = new_rev[improve_mask]

                    next_route = repair_beam(new_route, b, target_len)

                    if torch.equal(next_route, new_route):
                        break

                    new_route = next_route.clone()

            # Deduplicate and prune.
            route_all = torch.cat([old_route, best_route], dim=0)
            best_route, _ = dedup_routes_by_pickups(route_all)
            best_rev = eval_routes(best_route, b)

            k = min(int(beta), best_route.size(0))
            vals, idx = torch.topk(best_rev, k=k, largest=True, sorted=True)

            best_route = best_route.index_select(0, idx)
            best_rev = vals

            # 2-opt improvement.
            best_route = apply_two_opt(best_route, b, target_len)
            best_rev = eval_routes(best_route, b)

        # Final hill climbing after LNS loop.
        new_route = best_route.clone()

        while True:
            new_rev = eval_routes(new_route, b)
            improve_mask = new_rev > best_rev

            if improve_mask.any():
                best_route = best_route.clone()
                best_route[improve_mask] = new_route[improve_mask]
                best_rev[improve_mask] = new_rev[improve_mask]

            next_route = repair_beam(new_route, b, target_len)

            if torch.equal(next_route, new_route):
                break

            new_route = next_route.clone()

        best_routes_record.append(best_route)
        best_revs_record.append(best_rev)

        final_idx = best_rev.argmax().item()

        best_routes_out.append(best_route[final_idx])
        best_revs_out.append(best_rev[final_idx])

    all_routes_record = {
        "route": best_routes_record,
        "obj": best_revs_record,
    }

    return torch.stack(best_routes_out, dim=0), all_routes_record