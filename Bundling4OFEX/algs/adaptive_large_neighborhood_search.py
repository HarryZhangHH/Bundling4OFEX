import time
import math
import random
from typing import List, Callable, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from utils.pdp_functions import preprocess_data, compute_obj
from utils.functions import seed_everything
from algs.large_neighborhood_search import *

def regret_repair(
    route,
    positions,
    loads,
    revenues,
    Q_max,
    T_max,
    R,
    D,
    device,
    obj="revenue",
    regret_k=2,
    max_passes=50,
):
    """
    Regret-k insertion repair for selective capacitated PDP.

    For each unserved request, evaluate feasible pickup-delivery insertion positions.
    Then choose the request with the largest regret value.

    regret_k=2 means standard regret-2 insertion.
    """

    # ------------------------------------------------------------
    # 1. Clean route
    # ------------------------------------------------------------
    if torch.is_tensor(route):
        full_route = route.detach().cpu().tolist()
    else:
        full_route = list(route)

    full_route = [int(x) for x in full_route]

    end_idx = D - 1

    try:
        first_end = full_route.index(end_idx)
        route = full_route[:first_end + 1]
    except ValueError:
        route = full_route[:]

    if len(route) == 0:
        route = [0, end_idx]
    elif route[-1] != end_idx:
        route = route + [end_idx]

    q_max_scalar = Q_max.view(-1)[0]
    t_max_scalar = T_max.view(-1)[0]

    passes = 0

    # ------------------------------------------------------------
    # 2. Iteratively insert one request at a time
    # ------------------------------------------------------------
    while passes < max_passes:
        passes += 1

        base_obj = compute_obj(
            route,
            positions,
            revenues,
            T_max,
            1,
            obj,
        ).item()

        # Capacity profile of current route
        picks = torch.tensor(route, device=device, dtype=torch.long)
        loads_along = loads[picks]
        cum_load = torch.cumsum(loads_along, dim=0)
        cap_left = q_max_scalar - cum_load

        best_request = None
        best_candidate_route = None
        best_regret_score = -float("inf")
        best_candidate_obj = -float("inf")

        # --------------------------------------------------------
        # 3. Check every unserved request
        # --------------------------------------------------------
        for re in range(R):
            pickup_idx = D + re
            delivery_idx = D + R + re

            if pickup_idx in route:
                continue

            this_load = loads[pickup_idx]
            gap_diff = (cap_left - this_load).detach().cpu().numpy()
            gaps = find_capacity_gap(gap_diff)

            candidate_routes = []
            candidate_objs = []

            for g_start, g_end in gaps:
                sub = route[g_start + 1:g_end]
                new_paths = insert_nodes(sub, pickup_idx, delivery_idx)

                for path in new_paths:
                    candidate = route[:g_start + 1] + path + route[g_end:]

                    dist = compute_route_length(candidate, positions)

                    if dist.item() <= t_max_scalar.item():
                        candidate_obj = compute_obj(
                            candidate,
                            positions,
                            revenues,
                            T_max,
                            1,
                            obj,
                        ).item()

                        candidate_routes.append(candidate)
                        candidate_objs.append(candidate_obj)

            if len(candidate_routes) == 0:
                continue

            # Sort feasible insertions for this request by objective
            order = np.argsort(candidate_objs)[::-1]
            sorted_objs = [candidate_objs[i] for i in order]

            best_obj_for_req = sorted_objs[0]
            best_route_for_req = candidate_routes[order[0]]

            if len(sorted_objs) >= regret_k:
                kth_obj = sorted_objs[regret_k - 1]
            elif len(sorted_objs) >= 2:
                kth_obj = sorted_objs[-1]
            else:
                # Only one feasible insertion position.
                # Give it high priority because it may disappear later.
                kth_obj = base_obj

            regret_score = best_obj_for_req - kth_obj

            # Tie-breaker: if same regret, choose better objective
            if (
                regret_score > best_regret_score
                or (
                    regret_score == best_regret_score
                    and best_obj_for_req > best_candidate_obj
                )
            ):
                best_regret_score = regret_score
                best_candidate_obj = best_obj_for_req
                best_candidate_route = best_route_for_req
                best_request = re

        # --------------------------------------------------------
        # 4. Stop if no feasible insertion remains
        # --------------------------------------------------------
        if best_candidate_route is None:
            break

        # If the selected insertion does not improve objective, stop.
        # For pure revenue this usually improves; for profit it may not.
        if best_candidate_obj <= base_obj:
            break

        route = best_candidate_route

    assert route[-1] == end_idx, route

    return route

def _clean_route(route, end_idx: int) -> List[int]:
    """
    Convert tensor/list route to a Python list and cut after the first end depot.
    """
    if torch.is_tensor(route):
        route = route.detach().cpu().tolist()

    route = [int(x) for x in route]

    if end_idx in route:
        route = route[:route.index(end_idx) + 1]

    return route


def _served_pickups(route: List[int], D: int, R: int) -> List[int]:
    """
    Return pickup node ids currently served in the route.
    Pickup nodes are D, ..., D+R-1.
    """
    return [x for x in route if D <= x < D + R]


def _remove_pickups(route: List[int], pickups: List[int], R: int) -> List[int]:
    """
    Remove pickup-delivery pairs from route.
    If pickup is p, delivery is p + R.
    """
    remove = set()
    for p in pickups:
        remove.add(p)
        remove.add(p + R)

    return [x for x in route if x not in remove]


def _roulette(weights: np.ndarray) -> int:
    """
    Select operator index according to positive weights.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-12)
    probs = weights / weights.sum()
    return int(np.random.choice(len(weights), p=probs))


def _adaptive_update(weights, scores, counts, reaction_factor):
    """
    ALNS weight update:
        w_i <- (1-rho) w_i + rho * average_score_i
    """
    for i in range(len(weights)):
        if counts[i] > 0:
            avg_score = scores[i] / counts[i]
            weights[i] = (1.0 - reaction_factor) * weights[i] + reaction_factor * avg_score

    scores[:] = 0.0
    counts[:] = 0.0
    return weights


def adaptive_large_neighborhood_search(
    routes,
    depots: torch.Tensor,
    requests: torch.Tensor,
    *,
    iteration: int = 200,
    time_cap: Optional[float] = None,   # seconds per instance
    k_min: int = 1,
    k_max: int = 5,
    obj: str = "revenue",
    seed: int = 1234,
    use_sa_acceptance: bool = True,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.995,
    segment_length: int = 25,
    reaction_factor: float = 0.2,
    score_global_best: float = 8.0,
    score_improvement: float = 4.0,
    score_accepted: float = 1.0,
    max_repair_passes: int = 50,
):
    """
    Adaptive Large Neighborhood Search for your selective capacitated PDP.

    Input:
        routes:
            Initial routes, either tensor shape (B, L) or list of B routes.

        depots:
            Tensor (B, D, 4) or compatible with preprocess_data.

        requests:
            Tensor (B, R, 6).

    Output:
        update_routes:
            List of best routes, one route per instance.

        logs:
            Per-instance information about best objective and final operator weights.
    """

    seed_everything(seed)

    device = depots.device
    B, R, D = requests.size(0), requests.size(1), depots.size(1)

    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)
    loads_flat = loads.squeeze(-1)

    end_idx = D - 1

    update_routes = []
    logs = []

    # ------------------------------------------------------------
    # Main loop over instances
    # ------------------------------------------------------------
    for b in tqdm(range(B)):
        seed_everything(seed + b)

        # ------------------------------------------------------------
        # Per-instance runtime limit
        # ------------------------------------------------------------
        start_clock = time.time()

        def time_exceeded() -> bool:
            return time_cap is not None and (time.time() - start_clock) >= time_cap

        def eval_route(route: List[int]) -> float:
            return compute_obj(
                route,
                positions[b:b + 1],
                revenues[b:b + 1],
                T_max[b],
                1,
                obj,
            ).item()

        def repair_fixed_point(route: List[int], stochastic: bool) -> List[int]:
            old_route = None
            new_route = route
            passes = 0

            while old_route != new_route and passes < max_repair_passes:
                if time_exceeded():
                    break

                old_route = new_route

                new_route = repair(
                    old_route,
                    positions[b:b + 1],
                    loads_flat[b],
                    revenues[b:b + 1],
                    Q_max[b],
                    T_max[b],
                    R,
                    D,
                    device,
                    obj,
                    stochastic,
                )

                new_route = _clean_route(new_route, end_idx)
                passes += 1

            return new_route

        def repair_one_pass(route: List[int], stochastic: bool) -> List[int]:
            if time_exceeded():
                return route

            new_route = repair(
                route,
                positions[b:b + 1],
                loads_flat[b],
                revenues[b:b + 1],
                Q_max[b],
                T_max[b],
                R,
                D,
                device,
                obj,
                stochastic,
            )

            return _clean_route(new_route, end_idx)

        def regret_repair_op(route: List[int]) -> List[int]:
            return regret_repair(
                route,
                positions[b:b + 1],
                loads_flat[b],
                revenues[b:b + 1],
                Q_max[b],
                T_max[b],
                R,
                D,
                device,
                obj=obj,
                regret_k=2,
                max_passes=max_repair_passes,
            )

        # ------------------------------------------------------------
        # Destroy operators
        # ------------------------------------------------------------
        def random_removal(route: List[int], k: int) -> List[int]:
            pickups = _served_pickups(route, D, R)
            if len(pickups) == 0:
                return route

            chosen = random.sample(pickups, min(k, len(pickups)))
            return _remove_pickups(route, chosen, R)

        def worst_contribution_removal(route: List[int], k: int) -> List[int]:
            pickups = _served_pickups(route, D, R)
            if len(pickups) == 0:
                return route

            scored = []

            for p in pickups:
                if time_exceeded():
                    break

                partial = _remove_pickups(route, [p], R)
                val = eval_route(partial)
                scored.append((val, p))

            if len(scored) == 0:
                return route

            scored.sort(reverse=True)
            chosen = [p for _, p in scored[:min(k, len(scored))]]
            return _remove_pickups(route, chosen, R)

        def shaw_removal(route: List[int], k: int) -> List[int]:
            pickups = _served_pickups(route, D, R)
            if len(pickups) == 0:
                return route

            seed_pickup = random.choice(pickups)
            seed_req = seed_pickup - D
            seed_pd = requests[b, seed_req, :4]

            related = []

            for p in pickups:
                if time_exceeded():
                    break

                req_id = p - D
                pd = requests[b, req_id, :4]

                pickup_dist = torch.norm(seed_pd[:2] - pd[:2]).item()
                delivery_dist = torch.norm(seed_pd[2:4] - pd[2:4]).item()
                revenue_dist = abs(
                    requests[b, seed_req, 5].item() - requests[b, req_id, 5].item()
                )

                relatedness = pickup_dist + delivery_dist + 0.1 * revenue_dist
                related.append((relatedness, p))

            if len(related) == 0:
                return route

            related.sort(key=lambda x: x[0])
            chosen = [p for _, p in related[:min(k, len(related))]]
            return _remove_pickups(route, chosen, R)

        def route_segment_removal(route: List[int], k: int) -> List[int]:
            pickups_in_order = [x for x in route if D <= x < D + R]

            if len(pickups_in_order) == 0:
                return route

            k_eff = min(k, len(pickups_in_order))
            start = random.randint(0, len(pickups_in_order) - k_eff)
            chosen = pickups_in_order[start:start + k_eff]

            return _remove_pickups(route, chosen, R)

        destroy_ops = [
            ("random_removal", random_removal),
            ("worst_contribution_removal", worst_contribution_removal),
            ("shaw_removal", shaw_removal),
            ("route_segment_removal", route_segment_removal),
        ]

        # ------------------------------------------------------------
        # Repair operators
        # ------------------------------------------------------------
        def greedy_repair(route: List[int]) -> List[int]:
            return repair_fixed_point(route, stochastic=False)

        def stochastic_repair(route: List[int]) -> List[int]:
            return repair_fixed_point(route, stochastic=True)

        def one_pass_repair(route: List[int]) -> List[int]:
            return repair_one_pass(route, stochastic=False)

        repair_ops = [
            ("greedy_repair", greedy_repair),
            ("stochastic_repair", stochastic_repair),
            ("one_pass_repair", one_pass_repair),
            ("regret_repair", regret_repair_op),
        ]

        destroy_weights = np.ones(len(destroy_ops), dtype=np.float64)
        repair_weights = np.ones(len(repair_ops), dtype=np.float64)

        destroy_scores = np.zeros(len(destroy_ops), dtype=np.float64)
        repair_scores = np.zeros(len(repair_ops), dtype=np.float64)

        destroy_counts = np.zeros(len(destroy_ops), dtype=np.float64)
        repair_counts = np.zeros(len(repair_ops), dtype=np.float64)

        # ------------------------------------------------------------
        # Initial route
        # ------------------------------------------------------------
        if torch.is_tensor(routes):
            current_route = _clean_route(routes[b], end_idx)
        else:
            current_route = _clean_route(routes[b], end_idx)

        if not time_exceeded():
            current_route = repair_fixed_point(current_route, stochastic=False)

        current_value = eval_route(current_route)

        best_route = current_route.copy()
        best_value = current_value

        temperature = initial_temperature
        iters = 0
        stopped_by_time = False

        # ------------------------------------------------------------
        # ALNS loop
        # ------------------------------------------------------------
        while True:
            if time_exceeded():
                stopped_by_time = True
                break

            if time_cap is None and iters >= iteration:
                break

            iters += 1

            pickups = _served_pickups(current_route, D, R)
            if len(pickups) == 0:
                break

            k = random.randint(k_min, min(k_max, len(pickups)))

            d_idx = _roulette(destroy_weights)
            r_idx = _roulette(repair_weights)

            destroy_name, destroy_op = destroy_ops[d_idx]
            repair_name, repair_op = repair_ops[r_idx]

            # ---------------- destroy ----------------
            partial_route = destroy_op(current_route.copy(), k)

            if time_exceeded():
                stopped_by_time = True
                break

            # ---------------- repair ----------------
            candidate_route = repair_op(partial_route)

            if time_exceeded():
                stopped_by_time = True
                break

            candidate_value = eval_route(candidate_route)

            delta = candidate_value - current_value

            accepted = False
            new_global_best = False
            improved_current = False

            if delta >= 0:
                accepted = True
                improved_current = True
            elif use_sa_acceptance:
                accept_prob = math.exp(delta / max(temperature, 1e-12))
                if random.random() < accept_prob:
                    accepted = True

            if accepted:
                current_route = candidate_route
                current_value = candidate_value

                if candidate_value > best_value:
                    best_route = candidate_route.copy()
                    best_value = candidate_value
                    new_global_best = True

            # ---------------- operator reward ----------------
            reward = 0.0

            if new_global_best:
                reward = score_global_best
            elif improved_current:
                reward = score_improvement
            elif accepted:
                reward = score_accepted

            destroy_scores[d_idx] += reward
            repair_scores[r_idx] += reward

            destroy_counts[d_idx] += 1
            repair_counts[r_idx] += 1

            # ---------------- adaptive update ----------------
            if iters % segment_length == 0:
                destroy_weights = _adaptive_update(
                    destroy_weights,
                    destroy_scores,
                    destroy_counts,
                    reaction_factor,
                )

                repair_weights = _adaptive_update(
                    repair_weights,
                    repair_scores,
                    repair_counts,
                    reaction_factor,
                )

            temperature *= cooling_rate

        # ------------------------------------------------------------
        # Final cleanup only if time remains
        # ------------------------------------------------------------
        if not time_exceeded():
            best_route = repair_fixed_point(best_route, stochastic=False)
            best_value = eval_route(best_route)

        update_routes.append(best_route)

        logs.append({
            "best_obj": best_value,
            "destroy_weights": {
                name: float(w) for (name, _), w in zip(destroy_ops, destroy_weights)
            },
            "repair_weights": {
                name: float(w) for (name, _), w in zip(repair_ops, repair_weights)
            },
            "iterations": iters,
            "runtime": time.time() - start_clock,
            "stopped_by_time": stopped_by_time,
        })

    return update_routes, logs