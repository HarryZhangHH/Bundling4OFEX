import time
import math
import torch
import numpy as np
from utils.pdp_functions import *

def find_capacity_gap(gap_diff: np.ndarray) -> list[tuple[int,int]]:
    """
    Given a 1D array `gap_diff`, returns all (start, end) intervals
    (inclusive) where gap_diff[i] >= 0.  As soon as gap_diff[i] < 0,
    that “closes” the current interval.
    """
    gaps: list[tuple[int,int]] = []
    start = None

    for i, v in enumerate(gap_diff):
        if v >= 0 and start is None:
            start = i
        if v < 0 and start is not None:
            gaps.append((start, i))
            start = None

    # If we were in a gap at the very end, close it here
    if start is not None:
        gaps.append((start, len(gap_diff)-1))
    return gaps

def insert_nodes(path, pickup, delivery):
    path = list(path)
    new_paths = []
    for i in range(len(path)+1):
        for j in range(i, len(path)+1):
            new_paths.append(path[:i] + [pickup] + path[i:j] + [delivery] + path[j:])
    return new_paths

def calculate_prob(rev_list):
    if len(rev_list) == 1:
        return [1]
    
    rev_array = np.array(rev_list)
    max_rev   = np.max(rev_array)
    delta_rev = max_rev - rev_array

    prob = delta_rev / (np.sum(delta_rev))
    return prob

def repair(route, positions, loads, revenues, Q_max, T_max, R, D, device, obj='revenue', stochastic=False):
    # 1a) turn the padded route into a Python list and cut at first end‐depot
    full_route = list(route)   # length = L
    try:
        first_end = full_route.index(D-1)
        route = full_route[:first_end+1]
    except ValueError:
        # if for some reason end_idx never appears, just keep full length
        route = full_route[:]

    #  2) compute capacity_left for that route b
    #     we need loads_along[b, t] for t=0..len(route)-1
    picks = torch.tensor(route, device=device, dtype=torch.long)  # (Lb,)
    loads_along = loads[picks]    # (Lb,)
    cum_load = torch.cumsum(loads_along, dim=0)              # (Lb,)
    cap_left = Q_max[0] - cum_load                        # (Lb,)

    # 3) now attempt each possible pickup “re” in [ D .. D+R-1 ]
    valid_route, rev_list = [full_route], [compute_obj(full_route, positions, revenues, T_max, 1, obj).item()]
    for re in range(R):
        pickup_idx  = D + re         # the pickup node for request re
        deliver_idx = D + R + re     # its paired delivery

        if pickup_idx in route:
            continue

        # build gap_diff = cap_left - loads[b, pickup_idx]
        #   shape = (Lb,)
        this_load = loads[pickup_idx]            # scalar
        gap_diff = (cap_left - this_load).cpu().numpy() # turn to numpy for your old find_capacity_gap

        # find all (start, end) intervals in that 1D array
        #   each is an inclusive interval of indices [s..e] where gap_diff[i] >= 0
        gaps = find_capacity_gap(gap_diff)

        # for each gap, try to insert (pickup_idx → deliver_idx) in all possible ways
        for (g_start, g_end) in gaps:
            # original code: new_paths = insert_nodes(route[g_start+1 : g_end], re+D, re+R+D)
            # here route[g_start+1 : g_end] is the sub‐list of nodes between those indices
            sub = route[g_start+1 : g_end]  # a Python slice
            new_paths = insert_nodes(sub, pickup_idx, deliver_idx)

            for path in new_paths:
                # splice “path” back into the old route:
                candidate = route[: g_start+1 ] + path + route[g_end : ]

                # now compute its length, but remember that candidate is still “short”
                dist = compute_route_length(candidate, positions)  # returns a 1‐element tensor
                if dist.item() <= T_max[0].item():
                    valid_route.append(candidate)
                    rev_list.append(compute_obj(candidate, positions, revenues, T_max, 1, obj).item())

    # 4) pick the best (or sample if stochastic)
    if len(valid_route) > 0:
        idx_choice = (np.random.choice(len(valid_route), p=calculate_prob(rev_list)) if stochastic else int(np.argmax(rev_list)))
        chosen = valid_route[idx_choice]
    else:
        chosen = route  # no improvement found

    assert chosen[-1] == D-1, chosen
    return chosen

def hill_climb(routes, depots, requests, stochastic=False, obj='revenue', seed=1234):
    """
    routes: (B, L) long tensor of node‐indices 0..(N−1),  
            but after the first time we hit `end_idx` it is
            “padding” (all subsequent positions are also end_idx).
    We want to treat each tour as “real” up to the first end_idx, then ignore the padding.
    """
    seed_everything(seed)
    device = depots.device
    B, R, D = requests.size(0), requests.size(1), depots.size(1)
    N = D + 2*R   # total number of nodes
    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)

    # 1) loads_flat[b, i] = loads[b, i, 0]
    loads_flat = loads.squeeze(-1)      # (B, N)

    update_routes = []
    for b in range(B):
        seed_everything(seed)
        
        new_route = routes[b]
        old_route = []
        while old_route != new_route:
            old_route = new_route
            new_route = repair(old_route, positions[b:b+1], loads_flat[b], revenues[b:b+1], Q_max[b], T_max[b], R, D, device, obj, stochastic)

        update_routes.append(old_route)

    return update_routes

def destory(route, pickup_idx, delivery_idx):
    route.remove(pickup_idx)
    route.remove(delivery_idx)
    return route

def large_negihborhood_search(routes, depots, requests, iteration=1, running_time=None, stochastic=False, obj='revenue', seed=1234):
    """
    routes: (B, L) long tensor of node‐indices 0..(N−1),  
            but after the first time we hit `end_idx` it is
            “padding” (all subsequent positions are also end_idx).
    We want to treat each tour as “real” up to the first end_idx, then ignore the padding.
    """
    seed_everything(seed)

    device = depots.device
    B, R, D = requests.size(0), requests.size(1), depots.size(1)
    N = D + 2*R   # total number of nodes
    zero_to_bsz = torch.arange(B, device=device)
    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)

    # 1) loads_flat[b, i] = loads[b, i, 0]
    loads_flat = loads.squeeze(-1)      # (B, N)
    update_routes = []
    for b in range(B):
        seed_everything(seed)

        best_route = routes[b] 
        best_rev   = compute_obj(routes[b], positions[b:b+1], revenues[b:b+1], T_max[b], 1, obj).item()

        start_clock = time.time()
        iters = 0

        last_best_route = best_route.copy()

        # keep going until we hit either the iteration cap or the time cap
        while True:
            # 1) stop if we've done enough iterations
            if running_time is None and iters >= iteration:
                break
            # 2) stop if we've run out of time
            if running_time is not None and (time.time() - start_clock) >= running_time:
                break

            iters += 1

            in_path_requests = [r for r in best_route if r < D+R and r > D-1]

            if len(in_path_requests) == 0:
                break

            best_route_init = best_route.copy()

            for r in in_path_requests:
                if iters > 1 and running_time is not None and (time.time() - start_clock) >= running_time:
                    break

                new_route = destory(best_route_init.copy(), r, r+R)
                old_route = []
                while old_route != new_route:
                    old_route = new_route
                    new_route = repair(old_route, positions[b:b+1], loads_flat[b], revenues[b:b+1], Q_max[b], T_max[b], R, D, device, obj, stochastic)

                new_rev = compute_obj(new_route, positions[b:b+1], revenues[b:b+1], T_max[b], 1, obj).item()
                if new_rev > best_rev:
                    best_route = new_route
                    best_rev   = new_rev

            if best_route == last_best_route:
                break
            last_best_route = best_route.copy()
            
        update_routes.append(best_route)
    return update_routes


def k_large_negihborhood_search(routes, depots, requests, k=1, iteration=200,
                                running_time=None, stochastic=False, obj='revenue',
                                patience=None, random_destroy=True, seed=1234):
    """
    LNS with no-improvement early stopping.
    patience: max consecutive non-improving iterations allowed before stopping.
              If None, defaults to max(100, iteration//10) when iteration is set,
              or 500 when running_time is used.
    """
    seed_everything(seed)
    device = depots.device
    B, R, D = requests.size(0), requests.size(1), depots.size(1)
    N = D + 2*R
    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)
    loads_flat = loads.squeeze(-1)

    if patience is None:
        patience = max(10, iteration // 10) if running_time is None else 10

    update_routes = []
    for b in range(B):
        seed_everything(seed)

        best_route = routes[b]
        best_rev = compute_obj(best_route, positions[b:b+1], revenues[b:b+1], T_max[b], 1, obj).item()

        start_clock = time.time()
        iters = 0
        no_improve = 0

        while True:
            # hard stops
            if running_time is None and iters >= iteration: break
            if running_time is not None and (time.time() - start_clock) >= running_time: break
            if no_improve >= patience: break  # <<< no-improvement early stop

            iters += 1

            # requests currently on the route: indices in [D, D+R)
            in_path_requests = [r for r in best_route if D <= r < D+R]
            request_num = len(in_path_requests)
            if request_num == 0:
                break

            # choose destroy size
            if k >= 1:
                max_destroy = min(request_num, int(k))
                if random_destroy:
                    destroy_num = np.random.randint(1, max_destroy + 1)
                else:
                    destroy_num = max_destroy
            elif 0 < k < 1:
                destroy_num = min(request_num, math.ceil(request_num * k))
            else:
                destroy_num = 1  # fallback

            # pick which request-IDs to destroy (sample from in-path requests!)
            chosen = random.sample(in_path_requests, destroy_num)

            # cumulative destroy: start from the current route and apply all destroys
            current = best_route.copy() if isinstance(best_route, list) else list(best_route)
            for r in chosen:
                current = destory(current, r, r + R)  # remove pickup r and its delivery r+R

            # greedy/repair to a fixed point
            old_route = None
            new_route = current
            while old_route != new_route:
                old_route = new_route
                new_route = repair(
                    old_route, positions[b:b+1], loads_flat[b], revenues[b:b+1],
                    Q_max[b], T_max[b], R, D, device, obj, stochastic
                )

            # evaluate
            new_rev = compute_obj(new_route, positions[b:b+1], revenues[b:b+1], T_max[b], 1, obj).item()

            if new_rev > best_rev:
                best_route = new_route
                best_rev = new_rev
                no_improve = 0          # <<< reset on improvement
            else:
                no_improve += 1         # <<< count non-improvement

        update_routes.append(best_route)

    return update_routes
