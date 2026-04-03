import torch
import pickle
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.functions import seed_everything

def mask_op(mask, depots, requests, current, T, positions, T_max):
    """
    Inputs : 
      mask: size (bsz, nb_nodes)
      depots: size (bsz, nb_depots, 3) Euclidian coordinates of the depots
      requests: size (bsz, nb_requests, 3) Euclidian coordinates (x,y) of the pickup nodes and paired delivery nodes, load, revenue
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

    # 2) Build a (bsz, nb_nodes, 2) tensor of all node coordinates
    cur_xy  = positions[zero_to_bsz, current].unsqueeze(1)    # (bsz,1,2)
    node_xy = positions[:, nb_depots:nb_depots+nb_requests, :]# (bsz,nb_requests,2)
    end_xy  = positions[:, nb_depots-1].unsqueeze(1)          # (bsz,1,2) 
    
    # compute the three segment‐lengths in parallel:
    d_cur_nod = torch.norm(cur_xy  - node_xy, dim=2)       # (bsz,nb_requests)
    d_nod_end = torch.norm(node_xy - end_xy,  dim=2)       # (bsz,nb_requests)

    # distances current -> every node-> end
    total_dists = d_cur_nod + d_nod_end              # (bsz,nb_requests)

    # 3) for each pickup‐candidate j in each batch, check T+dist<=T_max
    #    first we’ll do it for all pickups with direct cost,
    #    then override with graph‐based values in a small loop.
    T_flat = T.view(bsz,1)                                        # (bsz,1)
    exceed_T = (T_flat + total_dists) > T_max             # (bsz,nb_requests)
    mask[:, nb_depots:nb_depots+nb_requests] |= exceed_T

    return mask

def normalize_features_op(depots, requests):
    assert depots.size(2) == 4
    assert requests.size(2) == 6
    
    # B = depots.size(0)
    # R_max = torch.tensor(1.42 * 1, dtype=torch.float, device= depots.device).expand(B)

    # requests[:, :, 5] = requests[:, :, 5] / R_max.view(B,1)
    
    new_depots = torch.cat([depots[:, :, :2], depots[:, :, 3:4]], dim=2)
    new_requests = torch.cat([requests[:, :, :2], requests[:, :, 5:6]], dim=2)

    return new_depots, new_requests

def preprocess_data_op(depots, requests):
    device = depots.device
    D = depots.shape[1]
    R = requests.shape[1]
    B = requests.shape[0]
    
    T_max = depots[:, 0, -1].view(B,1)
        
    coords_depots    = depots[:, :D, :2]             # (B,D,2) start/end coords
    coords_requests  = requests[:, :, :2]            # (B,R,2)
    positions        = torch.cat([coords_depots,
                                   coords_requests], dim=1)  # (B,N,2)
    
    revenue_depots   = torch.zeros(B, D, 1, dtype=torch.long, device=device)
    revenue_requests = requests[:, :, -1]   # (B,R,1)
    revenues = torch.cat([
        revenue_depots,
        revenue_requests.view(B,R,1)
    ], dim=1)                                     # --> (B, N, 1)

    return positions, revenues, T_max