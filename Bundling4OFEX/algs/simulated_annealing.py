import time
import random
import torch
import numpy as np
from tqdm import tqdm
from utils.pdp_functions import *
from algs.large_neighborhood_search import repair, destory

def is_feasible(feasibility_cache, route, positions, loads, Q_max, T_max, R):
    key = tuple(route)
    if key in feasibility_cache:
        return feasibility_cache[key]
    try:
        check_route_feasibility(route.copy(), positions, loads, Q_max, T_max, R, thresh=0.01)
        feasibility_cache[key] = True
    except AssertionError:
        feasibility_cache[key] = False
    return feasibility_cache[key]

def random_destory(route, R, D):
    "Destory a request randomly"
    delivery_requests = [r for r in route if r < D+R and r > D-1]
    if delivery_requests:
        request = random.choice(delivery_requests)
        destory(route, request, request+R)
    return route

# def swap(route, positions, loads, Q_max, T_max, R, feasibility_cache):
#     "Swap nodes at positions i and j with each other"
#     feasible_swap = []
#     for i in range(1, len(route)-1):
#         # if route[i] < D+R and route[i] > D-1:
#         for j in range(i+1, len(route)-1):
#             new_route = route.copy()
#             new_route[i], new_route[j] = route[j], route[i]
#             if is_feasible(feasibility_cache, new_route, positions, loads, Q_max, T_max, R):
#                 feasible_swap.append(new_route.copy())
#     if feasible_swap:
#         route = random.choice(feasible_swap)
#     return route

def swap(route, positions, loads, Q_max, T_max, R, feasibility_cache):
    if len(route) <= 2:
        return route
    for _ in range(len(route)*2):
        i, j = random.randrange(1, len(route)-1), random.randrange(1, len(route)-1)
        new_route = route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]
        if is_feasible(feasibility_cache, new_route, positions, loads, Q_max, T_max, R):
            return new_route
    return route

def replace(route, positions, loads, Q_max, T_max, R, D, feasibility_cache):
    "Replace a request with another"
    delivery_requests = [r for r in route if r < D+R and r > D-1]
    undelive_requests = [r for r in range(D, int(D+R)) if r not in route]
    
    # feasible_replace = []
    # 2. Shuffle in-place
    random.shuffle(delivery_requests)
    random.shuffle(undelive_requests)

    for re in delivery_requests:
        new_route = route.copy()
        index_i = new_route.index(re)
        index_j = new_route.index(re+R)
        assert index_i < index_j, f"node order of request {re} in the route is wrong! {(index_i, index_j)}"
        for un_re in undelive_requests:
            new_route[index_i] = un_re
            new_route[index_j] = un_re+R
            if is_feasible(feasibility_cache, new_route, positions, loads, Q_max, T_max, R):
                return new_route
    #             feasible_replace.append(new_route.copy())
    # if feasible_replace:
    #     route = random.choice(feasible_replace)
    return route

def get_neighbors(route, positions, loads, revenues, Q_max, T_max, R, D, device, feasibility_cache, obj, stochastic):
    """Returns neighbor of your solution."""
    weights = [0.2, 0.4, 0.2, 0.2]
    func = random.choices([0,1,2,3], weights)[0]
    if func == 0:
        route = random_destory(route, R, D)
    elif func == 1:
        route = repair(route, positions, loads, revenues, Q_max, T_max, R, D, device, obj, stochastic)
    elif func == 2:
        route = replace(route, positions, loads, Q_max, T_max, R, D, feasibility_cache)
    elif func == 3:
        route = swap(route, positions, loads, Q_max, T_max, R, feasibility_cache)
    return route

def simulated_annealing(routes, depots, requests, initial_temperature=10, cooling_rate=0.95, stopping_temperature=0.01, iterations_per_temperature=20, running_time=None, stochastic=False, obj='revenue', seed=1234):
    """Peforms simulated annealing to find a solution"""
    seed_everything(seed)
    
    device = depots.device
    B, R, D = requests.size(0), requests.size(1), depots.size(1)
    N = D + 2*R   # total number of nodes
    zero_to_bsz = torch.arange(B, device=device)
    positions, loads, revenues, Q_max, T_max = preprocess_data(depots, requests)

    loads_flat = loads.squeeze(-1)      # (B, N)

    update_routes = []
    for b in tqdm(range(B)):
        best_route = routes[b]
        best_value = compute_collected_revenue(routes[b], revenues[b:b+1]).item()

        current_route = routes[b] 
        current_value = compute_collected_revenue(routes[b], revenues[b:b+1]).item()

        # Set the initial temperature
        temperature = initial_temperature
        feasibility_cache = {}

        # mark time budget
        start_time = time.time()

        while temperature > stopping_temperature:

            for _ in range(iterations_per_temperature):
        
                # Generate a new candidate solution by making a small modification
                new_route = get_neighbors(current_route.copy(), positions[b:b+1], loads_flat[b], revenues[b:b+1], Q_max[b], T_max[b], R, D, device, feasibility_cache, obj, stochastic)
                new_value = compute_collected_revenue(new_route, revenues[b:b+1]).item()

                # Calculate the change in the objective function
                delta_value = new_value - current_value
                # If the new solution is better, accept it
                if delta_value >= 0:
                    current_route = new_route
                    current_value = new_value
                # If the new solution is worse, accept it with a probability
                else:
                    acceptance_probability = np.exp(delta_value / temperature)
                    if np.random.rand() < acceptance_probability:
                        current_route = new_route
                        current_value = new_value
                    
                if new_value - best_value > 0:
                    best_route = new_route
                    best_value = new_value
                        
            # Decrease the temperature according to the cooling rate
            temperature *= cooling_rate

            # if we have a time budget and it’s exceeded, stop immediately
            if running_time is not None and (time.time() - start_time) >= running_time:
                break

        # print(temperature)

        old_route = []
        while old_route != best_route:
            old_route = best_route
            best_route = repair(old_route, positions[b:b+1], loads_flat[b], revenues[b:b+1], Q_max[b], T_max[b], R, D, device, obj, stochastic)

        update_routes.append(best_route)
        # print(f"Final Route: {current_solution}, Objective: {current_value}")
    return update_routes, temperature