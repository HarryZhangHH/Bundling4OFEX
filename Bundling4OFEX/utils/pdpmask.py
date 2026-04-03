import torch
import networkx as nx
import numpy as np

def build_pdp_graph(current, end, undeliveries):
    """
    depot:    np.array shape (2,4), rows are [x,y,Q_max,T_max] for start and end
    requests: np.array shape (n,6), rows are [px,py, dx,dy, load, revenue]
    
    Returns
    -------
    G: nx.DiGraph with nodes 0..2+2n-1 and weight attr on every edge
    """
    # 1) gather the x,y positions in order
    coords = []
    coords.append(current[:2].tolist())       # node 0 = start
    coords.append(end[:2].tolist())       # node 1 = end
    for j in range(len(undeliveries)):
        coords.append(undeliveries[j, 2:4].tolist())
    
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

def pdp_masking(mask, depots, requests, current, undeliveries, Q, T, bsz):
    """
    Inputs : 
      mask of size (bsz, nb_nodes) 
      depots of size (bsz, 2, 4) Euclidian coordinates of the depots
      requests of size (bsz, nb_requests, 6) Euclidian coordinates (x,y) of the pickup nodes and paired delivery nodes, load, revenue
      undeliveries of size (bsz, ...)
      Q of size (bsz, 1, 1) Current loads
      T of size (bsz, 1, 1) Current distance
      current of size (bsz, ) Current node index
      bsz of size 1, batch size
    """
    device = depots.device
    nb_requests = requests.shape[1]
    nb_nodes = 2*requests.shape[1] + depots.shape[1]
    zero_to_bsz = torch.arange(bsz, device=device)
    # mask = torch.zeros(bsz, nb_nodes, device=device).bool() # False
    current = current.view(bsz,)   # flatten if needed
    mask[:,current] = True
    # Once you visit a pickup, you would unmask its paired delivery:
    at_pickup   = (current < 2 + nb_requests and current > 2).all(dim=-1)  # (bsz,)
    # at_delivery = (current >= 2 + nb_requests).all(dim=-1)  # (bsz,)
    mask[at_pickup, current+nb_requests] = True

    # 1) Slice out just the delivery‐node columns from mask

    mask_deliveries = mask[:, 2 + nb_requests : 2 + 2*nb_requests] # (bsz, nb_requests)
    assert not mask_deliveries[undeliveries].any(), "Some undeliveries deliveries are (incorrectly) masked!"

    Q_max = depots[zero_to_bsz, 0, 2]
    T_max = depots[zero_to_bsz, 0, 3]

    # 1) If at the start depot, mark it visited and block all deliveries
    at_start = (current == 0)
    mask[at_start, 0] = True
    # block all delivery nodes until their pickup is done
    # deliveries occupy indices [2 + nb_requests  :  2 + 2*nb_requests)
    mask[:, 2 + nb_requests : 2 + 2*nb_requests] = True

    # 2) Mask pickups that would exceed Q_max
    loads = requests[:, :, 4]                        # (bsz,n)
    # For each pickup j, its node‐index = 2 + j
    for j in range(nb_requests):
        idx = 2 + j
        # Check if Q + load_j > Q_max
        exceed_Q = (Q + loads[:, j] > Q_max).squeeze(-1)  # (bsz,)
        mask[:, idx] |= exceed_Q

    # 3) Build a (bsz, nb_nodes, 2) tensor of all node coordinates
    coords_depots     = depots[:, :2, :2]              # (bsz,2,2) start/end coords
    coords_pickups    = requests[:, :, :2]            # (bsz,n,2)
    coords_deliveries = requests[:, :, 2:4]           # (bsz,n,2)
    positions         = torch.cat([coords_depots,
                                   coords_pickups,
                                   coords_deliveries], dim=1)  # (bsz,nb_nodes,2)

    # lookup current node coordinates
    cur_coords = positions[zero_to_bsz, current]  # (bsz,2)
    undel_coords = positions[zero_to_bsz, undeliveries+nb_requests+2]

    # 4) Distance from current_node to each candidate
    #    current_node: (bsz,1,2) → broadcast to (bsz,nb_nodes,2)
    # if at pickup
    for b in range(bsz):
        for j in range(nb_requests):
            idx = 2 + j
            # I need to write a shortest path function to check whether it is still feasible to a request
            # the shortest path between undelivery nodes, a new pickup node and paired delivery node
            # note that the pickup node must be visited before the delivery noode

            # Suppose G is your complete (Di)Graph, and each edge (u,v) has a numeric attribute 'weight'.
            G = build_pdp_graph(j, depots[b,1], undel_coords[b])

            # 1. Length only:
            dist = nx.dijkstra_path_length(G, source=0, target=1, weight='weight')
            x1,y1 = cur_coords[b]
            x2,y2 = positions[j]

            dist += np.hypot(x1 - x2, y1 - y2)
            if dist > T_max:
                mask[b, idx] = True
    return mask
    
# def path_length_coords(coords, path):
#     """
#     coords : array‐like of shape (N, 2), coords[i] = (x_i, y_i)
#     path   : list of node‐indices [n0, n1, ..., nk]
#     """
#     total = 0.0
#     for u, v in zip(path, path[1:]):
#         x1, y1 = coords[u]
#         x2, y2 = coords[v]
#         total += np.hypot(x1 - x2, y1 - y2)
#     return total