import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from torch.nn.utils.rnn import pad_sequence
from utils.pdp_functions import *

def plot_tsp(x_coord, x_path, T, revenues, plot_dist_pair=False):
    """
    Helper function to plot TSP tours.
    """

    # pytorch detach
    x_coord = x_coord.detach().cpu()
    x_path = x_path.detach().cpu()
    
    # compute TSP lengths
    length_tsp = compute_route_length(x_path, x_coord)
    
    # preparation  
    x_coord = np.array(x_coord)
    x_path = np.array(x_path)
    nb_nodes = x_coord.shape[1]
    tour_size = x_path.shape[1]
    G = nx.from_numpy_array(np.zeros((nb_nodes,nb_nodes)))
    colors = ['r'] + ['g'] + ['b'] * (nb_nodes-2)  # Green for 0th node, blue for others
    batch_size = x_coord.shape[0]
    max_nb_plots = 3**2 # max number of TSP plots, x^2 for x rows and x cols 
    nb_plots = batch_size if batch_size<max_nb_plots else max_nb_plots 

    nb_rows = 1; nb_cols = nb_plots
    f = plt.figure(figsize=(3*nb_cols+3, 3)) # figure size  

    
    # loop over TSPs
    for i in range(nb_plots):
        x_coord_i = x_coord[i]
        pos_i = dict(zip(range(len(x_coord_i)), x_coord_i.tolist()))
        if plot_dist_pair: # Compute pairwise distances matrix for better visualization
            dist_pair_i = squareform(pdist(x_coord_i, metric='euclidean')) 
            G = nx.from_numpy_array(dist_pair_i)
        x_path_i = x_path[i] 
        length_tsp_i = length_tsp[i]
        nodes_pair_tsp_i = []
        nodes_pair_tsp_j = []
        for r in range(tour_size-1): # compute consecutive nodes in the solution
            if x_path_i[r] + int((nb_nodes-2)/2) == x_path_i[r+1]:
                nodes_pair_tsp_i.append((x_path_i[r], x_path_i[r+1]))
            else:
                nodes_pair_tsp_j.append((x_path_i[r], x_path_i[r+1]))

        # nodes_pair_tsp_i.append((x_path_i[nb_nodes-1], x_path_i[0]))
        
        subf = f.add_subplot(nb_rows,nb_cols,i+1)
        nx.draw_networkx_nodes(G, pos_i, node_color=colors, node_size=20)
        nx.draw_networkx_edges(G, pos_i, edgelist=nodes_pair_tsp_i, alpha=1, width=1, edge_color='r')
        nx.draw_networkx_edges(G, pos_i, edgelist=nodes_pair_tsp_j, alpha=1, width=1, edge_color='orange')
        if plot_dist_pair:
            nx.draw_networkx_edges(G, pos_i, alpha=0.3, width=0.5)
        subf.set_title('L: ' + str(length_tsp_i.item())[:5] + '/' + str(T[i].item())[:5] + '  R:' + str(revenues[i].item())[:5])
    
    plt.show()
    return subf

def transform_tour(tours, nb_nodes):
    try:
        tours = tours.tolist()  # now a List[List[int]] of shape B×L
    except AttributeError:
        None

    final_tour = [
        # for each row, we “flat‐map” each x ↦ [x] or [x,x+nb_nodes]
        [y
        for x in row
        for y in ( [x, x + nb_nodes] if x > 1 else [x] )
        ]
        for row in tours
    ]

    # turn each into a 1-D LongTensor
    tensors = [torch.tensor(row, dtype=torch.long) for row in final_tour]

    # pad shorter ones with, say, `-1` up to the max length
    final_tour_tensor = pad_sequence(tensors, batch_first=True, padding_value=1)
    # shape is now (B, max_len), with padding where row was shorter
    return final_tour_tensor