import torch
import pickle
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.functions import seed_everything
from typing import Dict, Tuple

def compute_route_length(routes, positions):
    """
    Input
        routes: size (B, L)
        positions: size (B, N, 2)
    """
    B = positions.size(0)
    with torch.no_grad():
        routes = torch.tensor(routes, dtype=torch.int, device=positions.device).view(B, -1)
        # gather the (x,y) coords along each route: (bsz, L, 2)
        batch_idx = torch.arange(B, device=positions.device).unsqueeze(1)
        pos_on_route = positions[batch_idx, routes]         # (bsz, L, 2) or (bsz, L, 4)
        if positions.size(2) == 2:
            length = pos_on_route.diff(dim=1).norm(dim=2).sum(dim=1)
        elif positions.size(2) == 4:
            heads = pos_on_route[..., 0:2]   # shape (B, L, 2)
            tails = pos_on_route[..., 2:4]   # shape (B, L, 2)

            # 1) distance within each segment: ‖headᵣ – tailᵣ‖₂
            dist_head_tail = (heads - tails).norm(dim=2)      # (B, L)

            # 2) distance from tailᵣ to headᵣ₊₁, for r=0..L-2
            dist_tail_nexthead = (tails[:, :-1] - heads[:, 1:]).norm(dim=2)  # (B, L-1)

            # if you meant to sum over ALL r, you can either
            #   • sum them separately and add
            #   • or pad one of them with a 0
            length = (dist_head_tail.sum(dim=1) + dist_tail_nexthead.sum(dim=1))  # shape (B,)

            # length = pos_on_route.diff(dim=1).norm(dim=2).sum(dim=1)
    return length.squeeze()

def compute_collected_revenue(routes, revenues):
    """
    Input
        routes: size (B, L)
        revenues: size (B, N, 1)
    """
    B = revenues.size(0)
    with torch.no_grad():
        routes = torch.tensor(routes, dtype=torch.int, device=revenues.device).view(B, -1)
        batch_idx = torch.arange(B, device=revenues.device).unsqueeze(1)
        rev_on_route = revenues[batch_idx, routes]         # (bsz, L, 1)
        collected_revenue = rev_on_route.sum(dim=1)
    return collected_revenue.squeeze()

def compute_obj(route, positions, revenues, T_max, bsz, obj):
    R       = compute_collected_revenue(route, revenues).view(bsz)
    if obj == 'revenue':
        return R
    
    L       = compute_route_length(route, positions).view(bsz) # size(L_train)=(bsz)
    if obj == 'profit':
        return R - L
    
    if obj == 'ratio':
        return R + 1e-6 /L
        

def compute_loss(route, positions, revenues, T_max, bsz, penalty_factor, multi_obj):
    # get the lengths of the tours
    L       = compute_route_length(route, positions).view(bsz) # size(L_train)=(bsz)
    R       = compute_collected_revenue(route, revenues).view(bsz)
    if multi_obj:
        penalty = torch.clip(L - T_max.view(bsz), min=0)
        loss    = penalty * penalty_factor - R
    else:
        loss = - R
    return L, R, loss

# def compute_obj(route, positions, revenues, T_max, bsz, penalty_factor, multi_obj):
#     L, R, loss = compute_loss(route, positions, revenues, T_max, bsz, penalty_factor, multi_obj)
#     ratio = R / L
#     profit = R - L
#     return L, R, loss

def collate_pdp(instances, device='cpu', c=None, seed=None):
    """
    instances: list of length bsz of dicts with keys 'depots' and 'requests'
    returns: depots, requests, current, undelivered, Q, T, done_mask, mask
    """
    seed_everything(seed=seed) if seed is not None else None
    bsz = len(instances)
    R   = len(instances[0]['requests'])
    D   = len(instances[0]['depots'])
    
    # 1) requests_np: (bsz, R, 6)
    requests_np = np.zeros((bsz,R,6), dtype=np.float32)
    for b, inst in enumerate(instances):
        for j, req in enumerate(inst['requests']):
            requests_np[b,j,0:2] = req['pickup']
            requests_np[b,j,2:4] = req['delivery']
            requests_np[b,j,4]   = req['load']
            requests_np[b,j,5]   = req['revenue']

    # 2) depots_np: (bsz, D, 4)
    depots_np = np.zeros((bsz,D,4), dtype=np.float32)
    for b, inst in enumerate(instances):
        # depot coords
        depots_np[b,0,:2] = inst['depots'][0]
        depots_np[b,D-1,:2] = inst['depots'][D-1]

        # x1, y1 = inst['depots'][0]
        # x2, y2 = inst['depots'][D-1]
        T_max  = inst['T']
        Q_max  = inst['Q']
        # T_max = max(np.random.randint(3,5) * np.hypot(x1 - x2, y1 - y2), 2) * int(R / 10)
        # if np.max(requests_np[b,:,4]) == np.min(requests_np[b,:,4]):
        #     Q_max = 1s
        # else:
        #     Q_max = np.random.randint(2*np.max(requests_np[b,:,4])-2, 3*np.max(requests_np[b,:,4]))

        # limits
        if isinstance(c, list): 
            depots_np[b,:,2] = int(c[0])
            depots_np[b,:,3] = c[1]
        elif c == 0 or c == -1:
            depots_np[b,:,2] = Q_max[c]
            depots_np[b,:,3] = T_max[c]
        elif c == 'fix':
            depots_np[b,:,2] = 10 
            depots_np[b,:,3] = 4
        else:
            depots_np[b,:,2] = np.random.choice(Q_max)
            depots_np[b,:,3] = np.random.choice(T_max)
    
    # 3) to torch
    depots   = torch.from_numpy(depots_np).to(device)
    requests = torch.from_numpy(requests_np).to(device)
    
    return depots, requests

def normalize_features(depots, requests, r_max):
    assert depots.size(2) == 4
    assert requests.size(2) == 6
    
    B = depots.size(0)
    Q_max = depots[:, 0, 2]
    R_max = torch.tensor(r_max, dtype=torch.float, device= depots.device).expand(B)

    new_depots = torch.cat([depots[:, :, :2], depots[:, :, 3:4]], dim=2)
    pickups = requests.clone()
    pickups[:, :, 4] = requests[:, :, 4] / Q_max.view(B,1)
    pickups[:, :, 5] = requests[:, :, 5] / R_max.view(B,1)
    deliveries = pickups[:, :, 2:].clone()
    deliveries[:, :, 2] = -deliveries[:, :, 2]
    return new_depots, pickups, deliveries

def preprocess_data(depots, requests):
    device = depots.device
    D = depots.shape[1]
    R = requests.shape[1]
    B = requests.shape[0]
    if depots.shape[2] == 4:
        Q_max = depots[:, 0, 2].view(B,1)
        T_max = depots[:, 0, 3].view(B,1)
    elif depots.shape[2] == 3:
        Q_max = torch.ones(B, 1, dtype=torch.long, device=device)
        T_max = depots[:, 0, 2].view(B,1)
    coords_depots     = depots[:, :D, :2]             # (B,D,2) start/end coords
    coords_pickups    = requests[:, :, :2]            # (B,R,2)
    coords_deliveries = requests[:, :, 2:4]           # (B,R,2)
    positions         = torch.cat([coords_depots,
                                coords_pickups,
                                coords_deliveries], dim=1)  # (B,N,2)
    
    load_depots   = torch.zeros(B, D, 1, dtype=torch.long, device=device)
    load_requests = requests[:, :, 4]
    loads = torch.cat([
        load_depots,
        load_requests.view(B,R,1),
        - load_requests.view(B,R,1)
    ], dim=1)

    revenue_depots     = torch.zeros(B, D, 1, dtype=torch.long, device=device)
    revenue_pickups    = requests[:, :, 5]   # (B,R,1)
    revenue_deliveries = torch.zeros(B, R, 1, device=device)
    revenues = torch.cat([
        revenue_depots,
        revenue_pickups.view(B,R,1),
        revenue_deliveries
    ], dim=1)                                     # --> (B, N, 1)

    return positions, loads, revenues, Q_max, T_max

def preprocess_data_pdp(depots, requests):
    device = depots.device
    D = depots.shape[1]
    R = requests.shape[1]
    B = requests.shape[0]

    T_max = depots[:, 0, -1].view(B,1)

    coords_depots   = torch.cat([depots[:, :D, :2].clone(), depots[:, :D, :2].clone()], dim=2)            # (B,D,2) start/end coords
    coords_requests = requests[:, :, :4]            # (B,R,2)
    positions       = torch.cat([coords_depots,
                                coords_requests], dim=1)  # (B,R+D,4)

    revenue_depots     = torch.zeros(B, D, 1, dtype=torch.long, device=device)
    revenue_pickups    = requests[:, :, -1]   # (B,R,1)
    revenues = torch.cat([
        revenue_depots,
        revenue_pickups.view(B,R,1),
    ], dim=1)                                     # --> (B, N, 1)

    return positions, revenues, T_max

def mask_pdp(mask, depots, requests, current, T, positions, T_max):
    """
    Inputs : 
      mask: size (bsz, nb_nodes)
      depots: size (bsz, nb_depots, 3) Euclidian coordinates of the depots
      requests: size (bsz, nb_requests, 5) Euclidian coordinates (x,y) of the pickup nodes and paired delivery nodes, load, revenue
      T: size (bsz, 1, ) Current distance
      current: size (bsz, 1) Current node index
    """
    device = depots.device
    bsz = requests.shape[0] # batch size
    nb_depots = depots.shape[1]
    nb_requests = requests.shape[1]
    nb_nodes = nb_requests + nb_depots
    zero_to_bsz = torch.arange(bsz, device=device)
    # mask = torch.zeros(bsz, nb_nodes, device=device).bool() # False
    current = current.view(bsz,)   # flatten if needed
    # positions, _, T_max = preprocess_data(depots, requests)

    # 0) mark “just visited” nodes
    mask[zero_to_bsz,current] = True
    
    at_start = (current == 0)
    mask[at_start, nb_depots-1] = True

    # 1) Once you at the end, you mask all nodes:
    at_end = (current == nb_depots-1)
    mask[at_end, nb_depots:] = True

    mask[~at_start, nb_depots-1] = False

    total_dists = 0
    # 2) Build a (bsz, nb_nodes, 2) tensor of all node coordinates
    cur_xy = positions[zero_to_bsz, current, 2:].unsqueeze(1) # (bsz,1,2)
    pic_xy = positions[:, nb_depots:, :2]                     # (bsz,nb_requests,2)
    del_xy = positions[:, nb_depots:, 2:]                     # (bsz,nb_requests,2)
    end_xy = positions[:, nb_depots-1, :2].unsqueeze(1)       # (bsz,1,2) 
    
    # compute the three segment‐lengths in parallel:
    d_cur_pic = torch.norm(cur_xy - pic_xy, dim=2)       # (bsz,nb_requests)
    d_pic_del = torch.norm(pic_xy - del_xy, dim=2)       # (bsz,nb_requests)
    d_del_end = torch.norm(del_xy - end_xy, dim=2)       # (bsz,nb_requests)

    # distances current -> every node-> end
    total_dists = d_cur_pic + d_pic_del + d_del_end              # (bsz,nb_requests)

    # 3) for each pickup‐candidate j in each batch, check T+dist<=T_max
    #    first we’ll do it for all pickups with direct cost,
    #    then override with graph‐based values in a small loop.
    T_flat = T.view(bsz,1)                                        # (bsz,1)
    exceed_T = (T_flat + total_dists) > T_max             # (bsz,nb_requests)
    mask[:, nb_depots:nb_depots+nb_requests] |= exceed_T

    return mask

def build_pdp_graph(current, end, undeliveries):
    """
    depot:    np.array shape (2,4), rows are [x,y] for start and end
    requests: np.array shape (n,2), rows are [dx,dy]
    
    Returns
    -------
    G: nx.DiGraph with nodes 0..2+2n-1 and weight attr on every edge
    """
    # 1) gather the x,y positions in order
    coords = []
    coords.append(current[:2].tolist())       # node 0 = start
    coords.append(end[:2].tolist())       # node 1 = end
    # assert not undeliveries.nelement() == 0
    for j in range(len(undeliveries)):
        coords.append(undeliveries[j, :2].tolist())
    N = len(coords)  # = 2 + 2n
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    
    # 2) add all directed edges (u->v) with Euclidean weight
    for u in range(N-1):
        x1,y1 = coords[u]
        for v in range(1,N-1):
            if u == v:
                continue
            x2,y2 = coords[v]
            w = np.hypot(x1-x2, y1-y2)
            G.add_edge(u, v, weight=w)
    return G

def nearest_neighbor_tour(pts: torch.Tensor, start: int = 0, end: int = 1):
    """
    Greedy nearest-neighbor tour using PyTorch tensor operations.
    pts: (m+2, 2) tensor of [start] + intermediates + [end]
    Returns a 1D LongTensor of node indices for the tour.
    """
    # Compute full pairwise distance matrix
    D = torch.cdist(pts, pts, p=2)  # (m+2, m+2)
    m = pts.size(0)
    unvis = set(range(2, m))  # only the intermediates
    tour = [start]
    
    # Greedy NN loop (still sequential)
    while unvis:
        last = tour[-1]
        idxs = torch.tensor(list(unvis), device=pts.device, dtype=torch.long)
        dists = D[last, idxs]
        # pick nearest unvisited
        argmin = torch.argmin(dists).item()
        nxt = idxs[argmin].item()
        tour.append(nxt)
        unvis.remove(nxt)
        
    tour.append(end)
    return torch.tensor(tour, device=pts.device, dtype=torch.long)


def two_opt_fixed_end(pts: torch.Tensor, tour: torch.Tensor):
    """
    2-opt local search with fixed endpoints using PyTorch tensor operations.
    pts:  (m+2, 2) tensor of coordinates
    tour: LongTensor of initial tour indices (should start with 0 and end with len(pts)-1)
    Returns the improved tour and its length.
    """
    # Precompute distance matrix
    D = torch.cdist(pts, pts, p=2)
    
    def tour_length(t: torch.Tensor) -> float:
        # Sum distances along consecutive edges
        return D[t[:-1], t[1:]].sum().item()
    
    best_tour = tour.clone()
    best_len = tour_length(best_tour)
    L = tour.size(0)
    improved = True
    
    while improved:
        improved = False
        # Only consider swaps on the interior (keep endpoints fixed)
        for i in range(1, L-2):
            for j in range(i+1, L-1):
                # Reverse segment [i:j)
                new_t = torch.cat([
                    best_tour[:i],
                    best_tour[i:j].flip(0),
                    best_tour[j:]
                ], dim=0)
                new_len = tour_length(new_t)
                if new_len + 1e-8 < best_len:
                    best_tour = new_t
                    best_len = new_len
                    improved = True
        # Loop until no improvement
    
    assert best_tour[0].item() == 0 and best_tour[-1].item() == tour[-1].item()
    return best_tour, best_len

def one_tree_bound(current, undeliveries, dist_matrix, B, N, nb_depots):
    # positions: (B, N, 2)
    # D:         (B, N, N) precomputed distances via torch.cdist

    root_idx = current       # (B,)  index of current node
    end_idx  = torch.full_like(root_idx, nb_depots-1)  # index 1 = end depot
    undel_mask = undeliveries  # (B, R) boolean mask of pending deliveries

    # 1) Gather the full set U = {deliveries still pending} ∪ {end depot}
    #    We'll build a (B, U, 2) tensor of their indices:
    batch_idxs = torch.arange(B, device=dist_matrix.device)
    # first collect delivery‐indices per batch
    del_idxs = torch.nonzero(undel_mask, as_tuple=False)  # (K,2) rows=(b,j)
    # then add the end‐depot index
    end_rows = torch.stack([batch_idxs, torch.full((B,), nb_depots-1, device=dist_matrix.device)], dim=1)
    all_rows = torch.cat([del_idxs, end_rows], dim=0)
    # U_b[i] = batch index; U_j[i] = a node within that batch that still needs visiting (or the end depot)
    U_b, U_j = all_rows[:,0], all_rows[:,1]  # (K+B,)

    # 2) Two smallest edges *from each u∈U to any other node*:
    #    we mask out self‐distances by setting D[b,u,u]=+∞ first
    D_masked = dist_matrix.clone()
    idx = torch.arange(N, device=dist_matrix.device)
    D_masked[:, idx, idx] = float('inf')

    # Build boolean mask for U_b per batch
    U_mask = torch.zeros((B, N), device=dist_matrix.device, dtype=torch.bool)
    U_mask[U_b, U_j] = True

    # Build a (B,N,N) mask that’s True only for edges (i→j) with both i,j in U_b
    pair_mask = U_mask.unsqueeze(2) & U_mask.unsqueeze(1)   # (B, N, N)

    # Now mask out any edge leaving U_b
    D_masked.masked_fill_(~pair_mask, float('inf'))

    #    Now get the two smallest entries per u via topk(k=2, largest=False)
    #    but we need to do it *per row* in D_masked[b]
    #    we'll gather the U rows, then do topk along dim=1:
    Du = D_masked[U_b, U_j]                # shape (K+B, N)
    two_mins, _ = torch.topk(Du, k=2, dim=1, largest=False)  # (K+B,2)
    sum_two_mins = two_mins.sum(dim=1).view(-1)               # (K+B,)

    # zero the MST‐term for singleton or empty U_b
    counts      = torch.bincount(U_b, minlength=B)      # how many u’s per batch
    valid       = counts >= 2                           # must have at least two nodes
    sum_two_mins = sum_two_mins * valid[U_b].to(sum_two_mins.dtype) * 0.5

    # # 3) Now group‐sum by batch to get ∑(min1+min2) per batch
    one_tree_approx = torch.zeros(B, device=dist_matrix.device)
    one_tree_approx.index_add_(0, U_b, sum_two_mins)  # adds each u’s sum into its batch

    # # 4) Finally add the shortest connection from root to U and from U to end
    # #    These are simply mins over Du but at columns root_idx and end_idx:
    # min_root_to_U = dist_matrix[batch_idxs, root_idx, U_j].view(-1).view(-1)
    # min_U_to_end  = dist_matrix[U_b, U_j, end_idx[U_b]].view(-1)
    # # accumulate those into per‐batch via scatter_add
    # one_tree_approx.index_add_(0, U_b, min_root_to_U)
    # one_tree_approx.index_add_(0, U_b, min_U_to_end)

    # U_mask[:, nb_depots-1] = True

    # 2) confirm dist_matrix’s shape matches B,N,N

    # # 4a) root→U min
    # d_root_all = dist_matrix[batch_idxs, root_idx, :]    # (B, N)
    # d_root_all.masked_fill_(~U_mask, float('inf'))
    # min_root_to_U = d_root_all.min(dim=1).values       # (B,)

    # # 4b) U→end min
    # end_idx = nb_depots - 1
    # d_end_all = dist_matrix[:, :, end_idx]             # (B, N)
    # d_end_all.masked_fill_(~U_mask, float('inf'))
    # min_U_to_end = d_end_all.min(dim=1).values         # (B,)
    # # After U_mask creation
    # print("U_mask.sum per batch:", U_mask.sum(dim=1))  
    # # Expect every entry ≥ 1

    # # View a few rows of d_root_all BEFORE masking
    # print("d_root_all[0,:5] =", dist_matrix[0, root_idx[0], :5].cpu().tolist())

    # # View the mask for batch 0:
    # print("U_mask[0,:5]    =", U_mask[0, :5].cpu().tolist())

    # # After masking
    # print("d_root_all[0,:5] after mask =", d_root_all[0, :5].cpu().tolist())

    # # 5) finalize
    # one_tree_approx += min_U_to_end

    # `one_tree_approx[b]` is now your batch‐wise 1‐tree lower‐bound approximation.
    return one_tree_approx

def mask_cpdp(mask, vis_nodes, depots, requests, current, undeliveries, Q, T, positions, Q_max, T_max, dist_matrix, precise=False):
    """
    Inputs : 
      mask: size (bsz, nb_nodes)
      vis_nodes: size (bsz, nb_nodes) Only mask the requests have done 
      depots: size (bsz, nb_depots, 4) Euclidian coordinates of the depots
      requests: size (bsz, nb_requests, 6) Euclidian coordinates (x,y) of the pickup nodes and paired delivery nodes, load, revenue
      undeliveries: size (bsz, nb_requests)
      Q: size (bsz, 1, ) Current loads
      T: size (bsz, 1, ) Current distance
      current: size (bsz, 1) Current node index
    """
    device = depots.device
    bsz = requests.shape[0] # batch size
    nb_depots = depots.shape[1]
    nb_requests = requests.shape[1]
    nb_nodes = 2*nb_requests + nb_depots
    zero_to_bsz = torch.arange(bsz, device=device)
    # mask = torch.zeros(bsz, nb_nodes, device=device).bool() # False
    current = current.view(bsz,)   # flatten if needed
    # positions, _, _, Q_max, T_max = preprocess_data(depots, requests)

    # 0) mark “just visited” nodes
    mask[zero_to_bsz,current]      = True
    vis_nodes[zero_to_bsz,current] = True
    mask[:, nb_depots:nb_depots+nb_requests] = vis_nodes[:, nb_depots:nb_depots+nb_requests].clone()

    # 1) allow “end” only if no undeliveries remain
    mask[:,nb_depots-1] = undeliveries.any(dim=1)

    # 2) If at the start depot, mark it visited and block all deliveries
    at_start = (current == 0)
    mask[at_start, 0] = True
    mask[at_start, nb_depots-1] = True
    # block all delivery nodes until their pickup is done
    # deliveries occupy indices [nb_depots+nb_requests  :  nb_depots+2*nb_requests)
    mask[at_start, nb_depots+nb_requests : nb_depots+2*nb_requests] = True

    # 3) Once you visit a pickup, you would unmask its paired delivery:
    at_pickup = (current < nb_depots+nb_requests) & (current >= nb_depots)  # (bsz,)
    # at_delivery = (current >= nb_depots+nb_requests).all(dim=-1)  # (bsz,)
    deliveries = current[at_pickup] + nb_requests
    mask[zero_to_bsz[at_pickup], deliveries] = False
    # print(mask)

    # # 4) sanity check vs. undeliveries  # (bsz, nb_requests)
    # mask_deliveries = mask[:, nb_depots+nb_requests : nb_depots+2*nb_requests]
    # assert (torch.logical_not(mask_deliveries) == undeliveries).all(), f"Some undeliveries deliveries are (incorrectly) masked! {torch.logical_not(mask_deliveries)},\n {undeliveries}"

    # 5) Once you at the end, you mask all nodes:
    at_end = (current == nb_depots-1)
    mask[at_end, nb_depots:] = True

    # 6) Capacity chec, mask pickups that would exceed Q_max
    loads = requests[..., 4]               # (bsz,nb_requests,1)
    # For each pickup j, its node‐index = 2 + j
    Q_flat   = Q.view(bsz,1)               # (bsz,1)
    exceed_Q = (Q_flat + loads) > Q_max    # (bsz,nb_requests)
    # pickups occupy indices 2..2+R-1
    mask[:, nb_depots:nb_depots+nb_requests] |= exceed_Q

    # for j in range(nb_requests):
    #     idx = 2 + j
    #     # Check if Q + load_j > Q_max
    #     exceed_Q = (Q.view(bsz) + loads[:, j] > Q_max).view(bsz)  # (bsz,)
    #     mask[:, idx] |= exceed_Q
    path_dict = {}

    # 7) Build a (bsz, nb_nodes, 2) tensor of all node coordinates
    if not precise:
        # valid_idxs = (~mask.clone()).nonzero(as_tuple=False)
        j = torch.arange(nb_requests, device=dist_matrix.device).unsqueeze(0)  # (1,R)
        pick_idx = nb_depots + j                                     # (1,R)
        del_idx  = nb_depots + nb_requests + j                       # (1,R)
        end_idx  = nb_depots - 1                                   # scalar (usually 1)
        end_idx = torch.full((bsz, nb_requests), end_idx, dtype=torch.long, device=mask.device)

        # batch index 0..B-1
        b_idx    = zero_to_bsz.unsqueeze(1).expand(bsz, nb_requests)   
        # the “current” node for each batch, repeated over P
        cur_idx  = current.unsqueeze(1).expand(bsz, nb_requests)    
        # the list of pickup node‐indices, repeated downwards
        pick_idx = pick_idx.squeeze(1).expand(bsz, nb_requests)  
        del_idx  = del_idx.squeeze(1).expand(bsz, nb_requests)   

        # 1) current → pickups
        d_cur_pic = dist_matrix[b_idx, cur_idx, pick_idx]            # (B,R)
        # 2) pickups → deliveries
        d_pic_del = dist_matrix[b_idx, pick_idx, del_idx]            # (B,R)
        # 3) deliveries → end
        d_del_end = dist_matrix[b_idx, del_idx, end_idx]     # (B,R)

        # assert (abs(d_cur_pic_ - d_cur_pic) < 1e-4).all(), f'{d_cur_pic_} {d_cur_pic}'
        # assert (abs(d_pic_del_ - d_pic_del) < 1e-4).all()
        # assert (abs(d_del_end_ - d_del_end) < 1e-4).all()

        # distances current -> every node-> end
        total_dists  = d_cur_pic + d_pic_del + d_del_end              # (bsz,nb_requests)
                
        # total_dists_ = one_tree_bound(current, undeliveries, dist_matrix.clone(), bsz, nb_nodes, nb_depots)

        # # 4) total
        # total_dists += total_dists_.view(bsz,1)       # (B,R)
        
        # 8) for each pickup‐candidate j in each batch, check T+dist<=T_max
        #    first we’ll do it for all pickups with direct cost,
        #    then override with graph‐based values in a small loop.
        T_flat = T.view(bsz,1)                                        # (bsz,1)
        unfeasible_pickup = (T_flat + total_dists) > T_max             # (bsz,nb_requests)
        mask[:, nb_depots:nb_depots+nb_requests] |= unfeasible_pickup
    
    else:
        cur_xy     = positions[zero_to_bsz, current].unsqueeze(1)    # (bsz,1,2)
        pickup_xy  = positions[:, nb_depots:nb_depots+nb_requests, :]# (bsz,nb_requests,2)
        deliver_xy = positions[:, nb_depots+nb_requests:, :]         # (bsz,nb_requests,2)
        end_xy     = positions[:, nb_depots-1].unsqueeze(1)          # (bsz,1,2) 
        
        # compute the three segment‐lengths in parallel:
        d_cur_pic = torch.norm(cur_xy     - pickup_xy,  dim=2)       # (bsz,nb_requests)
        d_pic_del = torch.norm(pickup_xy  - deliver_xy, dim=2)       # (bsz,nb_requests)
        d_del_end = torch.norm(deliver_xy - end_xy,     dim=2)       # (bsz,nb_requests)

        # # distances current -> every node-> end
        total_dists = d_cur_pic + d_pic_del + d_del_end              # (bsz,nb_requests)

        # 8) for each pickup‐candidate j in each batch, check T+dist<=T_max
        #    first we’ll do it for all pickups with direct cost,
        #    then override with graph‐based values in a small loop.
        T_flat = T.view(bsz,1)                                        # (bsz,1)
        unfeasible_pickup = (T_flat + total_dists) > T_max             # (bsz,nb_requests)
        mask[:, nb_depots:nb_depots+nb_requests] |= unfeasible_pickup

        # 9) Distance from current_node to each candidate
        #    current_node: (bsz,1,2) → broadcast to (bsz,nb_nodes,2)
        for b in range(bsz):
            path_dict[b] = {}

            for j in range(nb_requests):
                idx = nb_depots + j
                # I need to write a shortest path function to check whether it is still feasible to a request
                # the shortest path between undelivery nodes, a new pickup node and paired delivery node
                # note that the pickup node must be visited before the delivery noode

                if vis_nodes[b,idx]:
                    continue   # skip already visited
                if unfeasible_pickup[b,j]:
                    # no hope even by direct lower‐bound → still mask
                    mask[b, idx] = True
                    continue
                
                # Build and solve
                # mask_expanded = undeliveries.unsqueeze(-1).expand(-1, -1, 2)  # (bsz, nb_requests, 2)
                # # undel_coords  = del_coords[mask_expanded].view(bsz, -1, 2)
                # undel_coords = [
                #     deliver_xy[b, undeliveries[b]]   # shape = (k_b, 2), where k_b varies by batch
                #     for b in range(bsz)
                # ]
                undel_idxs = undeliveries[b].nonzero().view(-1).tolist()

                if len(undel_idxs) == 0:
                    # no remaining undelivered requests
                    continue
                if (nb_depots <= idx < nb_depots+nb_requests) and not undeliveries[b, j]:
                    undel_idxs.append(j)
                
                # how many undeliveries?
                nb_undel = len(undel_idxs)

                # full_graph_idx[t] gives you the node-index in the original PDP graph
                full_graph_idx = [0]*(nb_undel + nb_depots)
                full_graph_idx[0] = idx
                for i, val in enumerate(undel_idxs):
                    full_graph_idx[nb_depots + i] = nb_depots + nb_requests + val
                full_graph_idx[nb_depots-1] = nb_depots-1

                # build an array pts = [next_xy] + [end_xy] + undel_coords
                pts = torch.cat([
                    positions[b,idx,:2].view(1,2),    # (1,2)
                    depots[b,nb_depots-1,:2].view(1,2),       # (1,2)
                    positions[b,nb_depots+nb_requests:][undel_idxs]               # (k_b,2)
                ], dim=0)                        # (k_b+2, 2)

                # run Nearest Neighborhood + 2-opt
                init = nearest_neighbor_tour(pts, start=0, end=1)
                tour, dist = two_opt_fixed_end(pts, init)

                path_dict[b][idx] = [full_graph_idx[k] for k in tour]

                # # Suppose G is your complete (Di)Graph, and each edge (u,v) has a numeric attribute 'weight'.
                # G = build_pdp_graph(
                #       positions[b,idx,:2].cpu().numpy(),
                #       depots[b,1,:2].cpu().numpy(),
                #       undel_coords)
                # try:
                #     dist = nx.dijkstra_path_length(G, source=0, target=1,
                #                                     weight='weight')
                #     # 2. Path + length:
                #     path = nx.dijkstra_path(G, source=0, target=1, weight='weight')
                #     if path != [0,1]:
                #         print(path)

                # except nx.NetworkXNoPath:
                #     print("error")
                #     mask[b,idx] = True
                #     continue

                # undel_coords_batch = undel_coords[b].view(-1, 2)
            
                # G = build_pdp_graph(positions[b,idx,:2].view(2), depots[b,1,:2].view(2), undel_coords_batch)

                # 4) compute the extra leg distance in torch
                leg = torch.norm(cur_xy[b, 0] - positions[b, idx, :2], p=2)    # torch scalar
                total_dist = leg + dist                                      # torch scalar

                # 5) mask if time budget violated
                mask[b, idx] |= (T[b, 0] + total_dist) > T_max[b, 0]
        
    return mask, vis_nodes, path_dict

def check_route_feasibility_batch(routes, positions, loads, Q_max, T_max, nb_requests, thresh=0.5):
    """
    Inputs : 
      route is a list of len (bsz)
      positions of size (bsz, nb_nodes, 2) Euclidian coordinates of the depots
      loads of size (bsz, nb_nodes, 1) 
    """
    
    for b in range(len(routes)):
        route = routes[b]
        Q = 0

        # remove all duplicates end node 1
        route = [
            x
            for i, x in enumerate(route)
            if not (x == 1 and i != route.index(1))
        ]

        # check once visit
        assert len(route) == len(set(route)), f"some nodes are visited more than once in route: {route}"

        for idx, r in enumerate(route):
            if r >= 2 and r < 2 + nb_requests:
                assert r + nb_requests in route[idx+1:], f"paired delivery nodes of {r} are not visited afterwards in route {route}"
            elif r >= 2 + nb_requests:
                assert r - nb_requests in route[:idx], f"paired pickup nodes of {r} are not visited before in route {route}"

            Q += loads[b,r]
            assert Q <= Q_max[b], f"capacity constraint is not satisfied in route {route} with q is {Q} and Q is {Q_max[b]}"

        assert Q == 0, f"not all pickuped requests are delivered in route {route}"
        
        route_length = compute_route_length(route, positions[b].view(1, -1, 2))
        assert route_length <= T_max[b] + thresh, f"max length constraint is not satisfied in route {route} with length is {route_length} and T is {T_max[b]}"

def check_route_feasibility(route, positions, loads, Q_max, T_max, nb_requests, thresh=0.5):
    """
    Inputs : 
      route is a list of len 
      positions of size (nb_nodes, 2) Euclidian coordinates of the depots
      loads of size (nb_nodes, 1) 
    """
    Q = 0

    # remove all duplicates end node 1
    route = [
        x
        for i, x in enumerate(route)
        if not (x == 1 and i != route.index(1))
    ]

    # check once visit
    assert len(route) == len(set(route)), f"some nodes are visited more than once in route: {route}"

    for idx, r in enumerate(route):
        if r >= 2 and r < 2 + nb_requests:
            assert r + nb_requests in route[idx+1:], f"paired delivery nodes of {r} are not visited afterwards in route {route}"
        elif r >= 2 + nb_requests:
            assert r - nb_requests in route[:idx], f"paired pickup nodes of {r} are not visited before in route {route}"

        Q += loads[r]
        assert Q <= Q_max, f"capacity constraint is not satisfied in route {route} with q is {Q} and Q is {Q_max}"

    assert Q == 0, f"not all pickuped requests are delivered in route {route}"
    
    route_length = compute_route_length(route, positions.view(1, -1, 2))
    assert route_length <= T_max + thresh, f"max length constraint is not satisfied in route {route} with length is {route_length} and T is {T_max}"

def pad_routes(routes, value):
    # 1) Find the length of the longest sub‐list
    max_len = max(len(sub) for sub in routes)

    # 2) For each sub‐list, append 1’s until its length == max_len
    padded = []
    for sub in routes:
        pad_size = max_len - len(sub)
        if pad_size > 0:
            padded.append(sub + [value] * pad_size)
        else:
            padded.append(sub[:])  # already that length
    return padded

def pad_tours_to_fixed_length(tours: list[torch.Tensor],
                              fixed_len: int,
                              pad_value: int) -> torch.Tensor:
    """
    Given a list of 1D LongTensors (each of shape [seq_len]),
    pad or truncate each one so that every output row has length fixed_len,
    then stack into a (batch_size, fixed_len) LongTensor.
    
    tours:       list of LongTensor, each shape (seq_i,)
    fixed_len:   desired sequence length for all tours
    pad_value:   index to use for padding
    """
    padded = []
    for seq in tours:
        seq = seq.long()
        if seq.size(0) < fixed_len:
            # pad on right
            pad_sz = fixed_len - seq.size(0)
            pad   = seq.new_full((pad_sz,), pad_value)
            seq   = torch.cat([seq, pad], dim=0)
        else:
            # truncate on right
            seq = seq[:fixed_len]
        padded.append(seq)
    # now every seq is exactly [fixed_len], stack into [batch, fixed_len]
    return padded

def repeat_interleave(B: int, **tensors: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Repeat-interleave any subset of tensors along dim=0 by B."""
    return {name: t.repeat_interleave(B, dim=0) for name, t in tensors.items()}

def plot_pdp_route(depots, requests, route, coord_range=1.0):
    """
    depot:    np.array (2,4) rows [x,y, Qmax, Tmax]
    requests: np.array (R,6) rows [px,py, dx,dy, load, revenue]
    route:    list of ints in [0 .. 2+2R-1], 0=start, 1=end, 2..=pickups, >=2+R=deliveries
    coord_range: max axis limit (for [0,1] data use 1.0)
    """
    R, D = requests.shape[0], depots.shape[0]
    # build coords array
    N = D + 2*R
    coords = np.zeros((N,2))
    coords[0]         = depots[0,:2]           # start
    coords[D-1]       = depots[D-1,:2]           # end
    coords[D:D+R]     = requests[:, :2]       # pickups
    coords[D+R:D+2*R] = requests[:, 2:4]   # deliveries

    cmap = mpl.colormaps['tab10']
    colors = [cmap(i % 10) for i in range(R)]

    # plot
    fig, ax = plt.subplots(figsize=(8,8))
    # plot all nodes lightly
    # ax.scatter(coords[:,0], coords[:,1], c='lightgray', s=100, zorder=1)
    # annotate pickups/deliveries
    for j in range(R):
        col = colors[j]
        p = coords[D+j]
        d = coords[D+R+j]
        ax.scatter(*p, marker='s', c=[col], s=200, edgecolors='black', zorder=2)
        ax.scatter(*d, marker='^', c=[col], s=200, edgecolors='black', zorder=2)
        ax.text(p[0], p[1]-0.02*coord_range, f"P{j}", ha='center', va='top')
        ax.text(d[0], d[1]+0.02*coord_range, f"D{j}", ha='center', va='bottom')
    # plot start/end
    ax.scatter(*coords[0], marker='*', c='black', s=200, label='Start', zorder=3)
    ax.scatter(*coords[1], marker='X', c='black', s=200, label='End', zorder=3)

    # plot route lines
    for a,b in zip(route, route[1:]):
        x1,y1 = coords[a]
        x2,y2 = coords[b]
        ax.arrow(x1, y1, x2-x1, y2-y1,
                 length_includes_head=True,
                 head_width=0.01*coord_range,
                 head_length=0.02*coord_range,
                 fc='C0', ec='C0', linewidth=2, zorder=4)

    ax.set_xlim(0, coord_range)
    ax.set_ylim(0, coord_range)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()