###################
# Libs
###################

import torch
import torch.nn as nn
import time
import argparse

import os
import datetime

from typing import Dict, Tuple

import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# visualization 
# from IPython.display import set_matplotlib_formats, clear_output
# set_matplotlib_formats('png2x','pdf')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try: 
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform
    # from concorde.tsp import TSPSolver # !pip install -e pyconcorde
except:
    pass
import warnings

from utils.pdp_functions import *
warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_detect_anomaly(True)


###################
# Network definition
# Notation : 
#            bsz : batch size
#            nb_nodes : number of nodes/cities
#            dim_emb : embedding/hidden dimension
#            nb_heads : nb of attention heads
#            dim_ff : feed-forward dimension
#            nb_layers : number of encoder/decoder layers
###################
class Transformer_encoder_net(nn.Module):
    """
    Encoder network based on self-attention transformer
    Inputs :  
      h of size      (bsz, nb_nodes, dim_emb)    batch of input cities
    Outputs :  
      h of size      (bsz, nb_nodes, dim_emb)    batch of encoded cities
      score of size  (bsz, nb_nodes, nb_nodes) batch of attention scores
    """
    def __init__(self, nb_layers, dim_emb, nb_heads, dim_ff, batchnorm):
        super(Transformer_encoder_net, self).__init__()
        assert dim_emb == nb_heads* (dim_emb//nb_heads) # check if dim_emb is divisible by nb_heads
        self.MHA_layers = nn.ModuleList( [nn.MultiheadAttention(dim_emb, nb_heads) for _ in range(nb_layers)] )
        self.linear1_layers = nn.ModuleList( [nn.Linear(dim_emb, dim_ff) for _ in range(nb_layers)] )
        self.linear2_layers = nn.ModuleList( [nn.Linear(dim_ff, dim_emb) for _ in range(nb_layers)] )   
        if batchnorm:
            self.norm1_layers = nn.ModuleList( [nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)] )
            self.norm2_layers = nn.ModuleList( [nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)] )
        else:
            self.norm1_layers = nn.ModuleList( [nn.LayerNorm(dim_emb) for _ in range(nb_layers)] )
            self.norm2_layers = nn.ModuleList( [nn.LayerNorm(dim_emb) for _ in range(nb_layers)] )
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.batchnorm = batchnorm
        
    def forward(self, h):      
        # PyTorch nn.MultiheadAttention requires input size (seq_len, bsz, dim_emb) 
        h = h.transpose(0,1) # size(h)=(nb_nodes, bsz, dim_emb)  
        score = None
        # L layers
        for i in range(self.nb_layers):
            h_rc = h # residual connection, size(h_rc)=(nb_nodes, bsz, dim_emb)
            h, score = self.MHA_layers[i](h, h, h) # size(h)=(nb_nodes, bsz, dim_emb), size(score)=(bsz, nb_nodes, nb_nodes)
            # add residual connection
            h = h_rc + h # size(h)=(nb_nodes, bsz, dim_emb)
            if self.batchnorm:
                # Pytorch nn.BatchNorm1d requires input size (bsz, dim, seq_len)
                h = h.permute(1,2,0).contiguous() # size(h)=(bsz, dim_emb, nb_nodes)
                h = self.norm1_layers[i](h)       # size(h)=(bsz, dim_emb, nb_nodes)
                h = h.permute(2,0,1).contiguous() # size(h)=(nb_nodes, bsz, dim_emb)
            else:
                h = self.norm1_layers[i](h)       # size(h)=(nb_nodes, bsz, dim_emb) 
            # feedforward
            h_rc = h # residual connection
            h = self.linear2_layers[i](torch.relu(self.linear1_layers[i](h)))
            h = h_rc + h # size(h)=(nb_nodes, bsz, dim_emb)
            if self.batchnorm:
                h = h.permute(1,2,0).contiguous() # size(h)=(bsz, dim_emb, nb_nodes)
                h = self.norm2_layers[i](h)       # size(h)=(bsz, dim_emb, nb_nodes)
                h = h.permute(2,0,1).contiguous() # size(h)=(nb_nodes, bsz, dim_emb)
            else:
                h = self.norm2_layers[i](h) # size(h)=(nb_nodes, bsz, dim_emb)
        # Transpose h
        h = h.transpose(0,1) # size(h)=(bsz, nb_nodes, dim_emb)
        return h, score
    

def myMHA(Q, K, V, nb_heads, mask=None, clip_value=None):
    """
    Compute multi-head attention (MHA) given a query Q, key K, value V and attention mask :
      h = Concat_{k=1}^nb_heads softmax(Q_k^T.K_k).V_k 
    Note : We did not use nn.MultiheadAttention to avoid re-computing all linear transformations at each call.
    Inputs : Q of size (bsz, dim_emb, 1)                batch of queries
             K of size (bsz, dim_emb, nb_nodes)       batch of keys
             V of size (bsz, dim_emb, nb_nodes)       batch of values
             mask of size (bsz, nb_nodes)             batch of masks of visited cities
             clip_value is a scalar 
    Outputs : attn_output of size (bsz, 1, dim_emb)     batch of attention vectors
              attn_weights of size (bsz, 1, nb_nodes) batch of attention weights
    """
    bsz, nb_nodes, emd_dim = K.size() #  dim_emb must be divisable by nb_heads
    if nb_heads>1:
        # PyTorch view requires contiguous dimensions for correct reshaping
        Q = Q.transpose(1,2).contiguous() # size(Q)=(bsz, dim_emb, 1)
        Q = Q.view(bsz*nb_heads, emd_dim//nb_heads, 1) # size(Q)=(bsz*nb_heads, dim_emb//nb_heads, 1)
        Q = Q.transpose(1,2).contiguous() # size(Q)=(bsz*nb_heads, 1, dim_emb//nb_heads)
        K = K.transpose(1,2).contiguous() # size(K)=(bsz, dim_emb, nb_nodes)
        K = K.view(bsz*nb_heads, emd_dim//nb_heads, nb_nodes) # size(K)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes)
        K = K.transpose(1,2).contiguous() # size(K)=(bsz*nb_heads, nb_nodes, dim_emb//nb_heads)
        V = V.transpose(1,2).contiguous() # size(V)=(bsz, dim_emb, nb_nodes)
        V = V.view(bsz*nb_heads, emd_dim//nb_heads, nb_nodes) # size(V)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes)
        V = V.transpose(1,2).contiguous() # size(V)=(bsz*nb_heads, nb_nodes, dim_emb//nb_heads)
    attn_weights = torch.bmm(Q, K.transpose(1,2))/ Q.size(-1)**0.5 # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes)
    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)
    if mask is not None:
        mask = mask.clone()
        if nb_heads>1:
            mask = torch.repeat_interleave(mask, repeats=nb_heads, dim=0) # size(mask)=(bsz*nb_heads, nb_nodes)
        assert not mask.all(dim=1).any(), "Error — some samples have no legal next action!"
        #attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-inf')) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-1e9')) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes)
    # attn_weights = torch.softmax(10 * torch.tanh(attn_weights) + attn_weights, dim=-1)
    attn_weights = torch.softmax(attn_weights, dim=-1) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes)
    attn_output = torch.bmm(attn_weights, V) # size(attn_output)=(bsz*nb_heads, 1, dim_emb//nb_heads)
    if nb_heads>1:
        attn_output = attn_output.transpose(1,2).contiguous() # size(attn_output)=(bsz*nb_heads, dim_emb//nb_heads, 1)
        attn_output = attn_output.view(bsz, emd_dim, 1) # size(attn_output)=(bsz, dim_emb, 1)
        attn_output = attn_output.transpose(1,2).contiguous() # size(attn_output)=(bsz, 1, dim_emb)
        attn_weights = attn_weights.view(bsz, nb_heads, 1, nb_nodes) # size(attn_weights)=(bsz, nb_heads, 1, nb_nodes)
        attn_weights = attn_weights.mean(dim=1) # mean over the heads, size(attn_weights)=(bsz, 1, nb_nodes)
    return attn_output, attn_weights
    
    
class AutoRegressiveDecoderLayer(nn.Module):
    """
    Single decoder layer based on self-attention and query-attention
    Inputs :  
      h_t of size      (bsz, 1, dim_emb)          batch of input queries
      K_att of size    (bsz, nb_nodes, dim_emb) batch of query-attention keys
      V_att of size    (bsz, nb_nodes, dim_emb) batch of query-attention values
      mask of size     (bsz, nb_nodes)          batch of masks of visited cities
    Output :  
      h_t of size (bsz, nb_nodes)               batch of transformed queries
    """
    def __init__(self, dim_emb, nb_heads):
        super(AutoRegressiveDecoderLayer, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.Wq_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wk_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wv_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_att = nn.Linear(dim_emb, dim_emb)
        self.Wq_att = nn.Linear(dim_emb, dim_emb)
        self.W1_MLP = nn.Linear(dim_emb, dim_emb)
        self.W2_MLP = nn.Linear(dim_emb, dim_emb)
        self.BN_selfatt = nn.LayerNorm(dim_emb)
        self.BN_att = nn.LayerNorm(dim_emb)
        self.BN_MLP = nn.LayerNorm(dim_emb)
        self.K_sa = None
        self.V_sa = None

    def reset_selfatt_keys_values(self):
        self.K_sa = None
        self.V_sa = None
        
    def forward(self, h_t, K_att, V_att, mask):
        bsz = h_t.size(0)
        h_t = h_t.view(bsz,1,self.dim_emb) # size(h_t)=(bsz, 1, dim_emb)
        # embed the query for self-attention
        q_sa = self.Wq_selfatt(h_t) # size(q_sa)=(bsz, 1, dim_emb)
        k_sa = self.Wk_selfatt(h_t) # size(k_sa)=(bsz, 1, dim_emb)
        v_sa = self.Wv_selfatt(h_t) # size(v_sa)=(bsz, 1, dim_emb)
        # concatenate the new self-attention key and value to the previous keys and values
        if self.K_sa is None:
            self.K_sa = k_sa # size(self.K_sa)=(bsz, 1, dim_emb)
            self.V_sa = v_sa # size(self.V_sa)=(bsz, 1, dim_emb)
        else:
            self.K_sa = torch.cat([self.K_sa, k_sa], dim=1)
            self.V_sa = torch.cat([self.V_sa, v_sa], dim=1)
        # compute self-attention between nodes in the partial tour
        h_t = h_t + self.W0_selfatt( myMHA(q_sa, self.K_sa, self.V_sa, self.nb_heads)[0] ) # size(h_t)=(bsz, 1, dim_emb)
        h_t = self.BN_selfatt(h_t.squeeze()) # size(h_t)=(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb) # size(h_t)=(bsz, 1, dim_emb)
        # compute attention between self-attention nodes and encoding nodes in the partial tour (translation process)
        q_a = self.Wq_att(h_t) # size(q_a)=(bsz, 1, dim_emb)
        h_t = h_t + self.W0_att( myMHA(q_a, K_att, V_att, self.nb_heads, mask)[0] ) # size(h_t)=(bsz, 1, dim_emb)
        h_t = self.BN_att(h_t.squeeze()) # size(h_t)=(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb) # size(h_t)=(bsz, 1, dim_emb)
        # MLP
        h_t = h_t + self.W2_MLP(torch.relu(self.W1_MLP(h_t)))
        h_t = self.BN_MLP(h_t.squeeze(1)) # size(h_t)=(bsz, dim_emb)
        return h_t
    
# class SimpleDecoderLayer(nn.Module):
#     """
#     Single decoder layer based on self-attention and query-attention
#     Inputs :  
#       h_t of size      (bsz, 1, dim_emb)          batch of input queries
#       K_att of size    (bsz, nb_nodes, dim_emb) batch of query-attention keys
#       V_att of size    (bsz, nb_nodes, dim_emb) batch of query-attention values
#       mask of size     (bsz, nb_nodes)          batch of masks of visited cities
#     Output :  
#       h_t of size (bsz, nb_nodes)               batch of transformed queries
#     """
#     def __init__(self, dim_emb, nb_heads):
#         super(SimpleDecoderLayer, self).__init__()
#         self.dim_emb = dim_emb
#         self.nb_heads = nb_heads
#         self.Wq_att = nn.Linear(dim_emb, dim_emb)
#         self.W0_att = nn.Linear(dim_emb, dim_emb)

#     def reset_selfatt_keys_values(self):
#         self.K_sa = None
#         self.V_sa = None
        
#     def forward(self, h_t, K_att, V_att, mask):
#         bsz = h_t.size(0)
#         h_t = h_t.view(bsz,1,self.dim_emb) # size(h_t)=(bsz, 1, dim_emb)

#         q_a = self.Wq_att(h_t) # size(q_a)=(bsz, 1, dim_emb)
#         h_t = h_t + self.W0_att( myMHA(q_a, K_att, V_att, self.nb_heads, mask)[0] ) # size(h_t)=(bsz, 1, dim_emb)

#         return h_t
        
class Transformer_decoder_net(nn.Module): 
    """
    Decoder network based on self-attention and query-attention transformers
    Inputs :  
      h_t of size      (bsz, 1, dim_emb)                            batch of input queries
      K_att of size    (bsz, nb_nodes, dim_emb*nb_layers_decoder) batch of query-attention keys for all decoding layers
      V_att of size    (bsz, nb_nodes, dim_emb*nb_layers_decoder) batch of query-attention values for all decoding layers
      mask of size     (bsz, nb_nodes)                            batch of masks of visited cities
    Output :  
      prob_next_node of size (bsz, nb_nodes)                      batch of probabilities of next node
    """
    def __init__(self, dim_emb, nb_heads, nb_layers_decoder):
        super(Transformer_decoder_net, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.nb_layers_decoder = nb_layers_decoder
        # self.decoder_layers = nn.ModuleList( [AutoRegressiveDecoderLayer(dim_emb, nb_heads) for _ in range(nb_layers_decoder-1)] )
        self.W0_att = nn.Linear(dim_emb, dim_emb)
        self.Wq_final = nn.Linear(dim_emb, dim_emb)
        
    # # Reset to None self-attention keys and values when decoding starts 
    # def reset_selfatt_keys_values(self): 
    #     for l in range(self.nb_layers_decoder-1):
    #         self.decoder_layers[l].reset_selfatt_keys_values()
            
    def forward(self, query, K_att, V_att, mask):
        for l in range(self.nb_layers_decoder):
            K_att_l = K_att[:,:,l*self.dim_emb:(l+1)*self.dim_emb].contiguous()  # size(K_att_l)=(bsz, nb_nodes, dim_emb)
            V_att_l = V_att[:,:,l*self.dim_emb:(l+1)*self.dim_emb].contiguous()  # size(V_att_l)=(bsz, nb_nodes, dim_emb)
            if l<self.nb_layers_decoder-1: # decoder layers with multiple heads (intermediate layers)
                # h_t = self.decoder_layers[l](h_t, K_att_l, V_att_l, mask)
                query = self.W0_att( myMHA(query, K_att_l, V_att_l, self.nb_heads, mask)[0] )
            else: # decoder layers with single head (final layer)
                q_final = self.Wq_final(query)
                bsz = query.size(0)
                q_final = q_final.view(bsz, 1, self.dim_emb)
                attn_weights = myMHA(q_final, K_att_l, V_att_l, 1, mask, 10)[1] 
        prob_next_node = attn_weights.squeeze(1) 
        return prob_next_node


# def generate_positional_encoding(d_model, max_len):
#     """
#     Create standard transformer PEs.
#     Inputs :  
#       d_model is a scalar correspoding to the hidden dimension
#       max_len is the maximum length of the sequence
#     Output :  
#       pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
#     """
#     pe = torch.zeros(max_len, d_model)
#     position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#     pe[:,0::2] = torch.sin(position * div_term)
#     pe[:,1::2] = torch.cos(position * div_term)
#     return pe
    
    
class CPDP_AM_net(nn.Module): 
    """
    The TSP network is composed of two steps :
      Step 1. Encoder step : Take a set of 2D points representing a fully connected graph 
                             and encode the set with self-transformer.
      Step 2. Decoder step : Build the TSP tour recursively/autoregressively, 
                             i.e. one node at a time, with a self-transformer and query-transformer. 
    Inputs : 
      x of size (bsz, nb_nodes, dim_emb) Euclidian coordinates of the nodes/cities
      deterministic is a boolean : If True the salesman will chose the city with highest probability. 
                                   If False the salesman will chose the city with Bernouilli sampling.
    Outputs : 
      tours of size (bsz, nb_nodes) : batch of tours, i.e. sequences of ordered cities 
                                      tours[b,t] contains the idx of the city visited at step t in batch b
      sumLogProbOfActions of size (bsz,) : batch of sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
    """
    
    def __init__(self, dim_input_nodes, dim_emb, dim_ff, step_context_dim, nb_layers_encoder, nb_layers_decoder, nb_heads,
                 batchnorm=True):
        super(CPDP_AM_net, self).__init__()
        
        self.dim_emb = dim_emb
        
        # input embedding layer
        if type(dim_input_nodes) is list:
            self.input_emb_depot    = nn.Linear(dim_input_nodes[0], dim_emb)
            self.input_emb_pickup   = nn.Linear(dim_input_nodes[1], dim_emb)
            self.input_emb_delivery = nn.Linear(dim_input_nodes[2], dim_emb)
        
        # encoder layer
        self.encoder = Transformer_encoder_net(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batchnorm)
        
        # vector to start decoding 
        # self.start_placeholder = nn.Parameter(torch.randn(dim_emb))
        
        # decoder layer
        self.decoder = Transformer_decoder_net(dim_emb, nb_heads, nb_layers_decoder)
        self.WK_att_decoder = nn.Linear(dim_emb, nb_layers_decoder* dim_emb) 
        self.WV_att_decoder = nn.Linear(dim_emb, nb_layers_decoder* dim_emb) 
        # self.PE = generate_positional_encoding(dim_emb, max_len_PE)        
        self.project_fixed_context = nn.Linear(dim_emb, dim_emb, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, dim_emb, bias=False)
        
    def forward(self, depots, pickups, deliveries, deterministic=False, precise=False):
        """ loads and revenues are already normalized """

        # some parameters
        device      = pickups.device
        bsz         = depots.size(0)
        nb_requests = pickups.size(1)
        nb_depots   = depots.size(1)
        nb_nodes    = 2*nb_requests + nb_depots
        zero_to_bsz = torch.arange(bsz, device=device) # [0,1,...,bsz-1]
        
        # State
        mask        = torch.zeros(bsz, nb_nodes, dtype=torch.bool, device=device)
        vis_node    = torch.zeros_like(mask)
        undel       = torch.zeros(bsz, nb_requests, dtype=torch.bool, device=device)
        current     = torch.zeros(bsz, dtype=torch.long, device=device)      # shape (B,)
        Q           = torch.zeros(bsz, 1, device=device)                     # we won’t really use Q_max here
        T           = torch.zeros(bsz, 1, device=device)                     # accumulated distance

        positions, loads, _, Q_max, T_max = preprocess_data(depots, pickups)

        dist_matrix = torch.stack([torch.cdist(positions[b], positions[b]) 
                                for b in range(bsz)], dim=0)  # (bsz,N,N)

        # input embedding layer
        h_depot    = self.input_emb_depot(depots) # size(h)=(bsz, 2, 4)
        h_pickup   = self.input_emb_pickup(pickups) # size(h)=(bsz, nb_nodes, 6)
        h_delivery = self.input_emb_delivery(deliveries) # size(h)=(bsz, nb_nodes, 4)
        
        # # concat the nodes and the input placeholder that starts the decoding
        # h = torch.cat([h, self.start_placeholder.repeat(bsz, 1, 1)], dim=1) # size(start_placeholder)=(bsz, nb_nodes, dim_emb)
        
        h = torch.cat([h_depot, h_pickup, h_delivery], dim=1)
        # encoder layer
        h_encoder, _ = self.encoder(h) # size(h)=(bsz, nb_nodes, dim_emb)

        graph_embed = h_encoder.mean(1)
        # fixed context = (bsz, 1, dim_emb) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours = [torch.zeros(bsz, dtype=torch.int, device=device)]

        # list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
        sumLogProbOfActions = []

        # key and value for decoder    
        K_att_decoder = self.WK_att_decoder(h_encoder) # size(K_att)=(bsz, nb_nodes, dim_emb*nb_layers_decoder)
        V_att_decoder = self.WV_att_decoder(h_encoder) # size(V_att)=(bsz, nb_nodes, dim_emb*nb_layers_decoder)
        
        # input placeholder that starts the decoding
        # self.PE = self.PE.to(x.device)
        # idx_start_placeholder = torch.Tensor([nb_nodes]).long().repeat(bsz).to(device)
        # h_start = h_encoder[zero_to_bsz, idx_start_placeholder, :] 
        # + self.PE[0].repeat(bsz,1) # size(h_start)=(bsz, dim_emb)
        h_start = h_encoder[zero_to_bsz, 0, :] 
        h_end   = h_encoder[zero_to_bsz, nb_depots-1, :] 
        
        # initialize mask of visited cities
        # mask_visited_nodes = torch.zeros(bsz, nb_nodes, device=device).bool() # False
        # mask_visited_nodes[zero_to_bsz, idx_start_placeholder] = True
        
        # clear key and val stored in the decoder
        # self.decoder.reset_selfatt_keys_values()
        # construct tour recursively
        h_t = h_start
        while True:
            with torch.no_grad():
                # 1) batch‐mask update
                mask, vis_node, _ = mask_cpdp(
                    mask, vis_node,
                    depots, pickups,
                    current.view(bsz,1),   # mask_pdp expects (B,1)
                    undel,
                    Q.detach().clone().view(bsz,1),       # shapes must match your mask_pdp signature
                    T.detach().clone().view(bsz,1),
                    positions, Q_max, T_max, dist_matrix, 
                    precise
                )

            # Embedding of previous node + remaining capacity            
            query = fixed_context + self.project_step_context(torch.cat(
                [h_t.view(bsz,1,self.dim_emb), h_end.view(bsz,1,self.dim_emb), 
                 Q_max.view(bsz,1,1)-Q.view(bsz,1,1), 
                 T_max.view(bsz,1,1)-T.view(bsz,1,1)
                 ], -1))
            
            # compute probability over the next node in the tour
            prob_next_node = self.decoder(query, K_att_decoder, V_att_decoder, mask) # size(prob_next_node)=(bsz, nb_nodes)
            
            # choose node with highest probability or sample with Bernouilli 
            if deterministic:
                idx = torch.argmax(prob_next_node, dim=1) # size(query)=(bsz,)
            else:
                idx = Categorical(prob_next_node).sample() # size(query)=(bsz,)

            # compute logprobs of the action items in the list sumLogProbOfActions   
            ProbOfChoices = prob_next_node[zero_to_bsz, idx] 
            sumLogProbOfActions.append( torch.log(ProbOfChoices) )  # size(query)=(bsz,)

            # update embedding of the current visited node
            h_t = h_encoder[zero_to_bsz, idx, :] # size(h_start)=(bsz, dim_emb)
            # h_t = h_t + self.PE[t+1].expand(bsz, self.dim_emb)
            
            # update tour
            tours.append(idx)

            Q += loads[zero_to_bsz, idx]

            # cur_xy = positions[zero_to_bsz, current]   # (B,2)
            # deltas = positions - cur_xy.unsqueeze(1)   # (B,N,2)
            # dists  = deltas.norm(dim=2)                # (B,N)
            # cost   = dists[zero_to_bsz, idx] 
            prev_pos = positions[zero_to_bsz, current]   # (B,2)
            next_pos = positions[zero_to_bsz, idx]       # (B,2)
            cost     = (next_pos - prev_pos).norm(dim=1)  # (B,)
            T[:, 0] += cost

            # update masks and constraints
            with torch.no_grad():
                
                # 7a) pickups
                is_pick = (idx >= nb_depots) & (idx < nb_depots+nb_requests)       # (B,)
                if is_pick.any():
                    pic_idx = idx[is_pick] - nb_depots               # which request
                    # mark them undelivered
                    undel[is_pick, pic_idx] = True  

                # 7b) deliveries
                is_delv = (idx >= nb_depots+nb_requests)
                if is_delv.any():
                    del_idx = idx[is_delv] - nb_depots - nb_requests
                    undel[is_delv, del_idx] = False

                # 7c) record that we’ve visited idx
                vis_node[zero_to_bsz, idx] = True
            
            current = idx.clone()

            if (current == nb_depots-1).all():
                break
        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        sumLogProbOfActions = torch.stack(sumLogProbOfActions,dim=1).sum(dim=1) # size(sumLogProbOfActions)=(bsz,)

        # convert the list of nodes into a tensor of shape (bsz,num_cities)
        tours = torch.stack(tours,dim=1) # size(col_index)=(bsz, nb_nodes)
        tours = F.pad(tours, (0, nb_nodes - tours.size(1)), value=nb_depots-1)
        
        return tours, sumLogProbOfActions

class CPDP_AM_net_beam(CPDP_AM_net):
    def forward(self,
                depots, pickups, deliveries,
                B=5,
                deterministic=False,
                beamsearch=False,
                precise=True):
        
        # initialize beam containers
        if not beamsearch:
            return super().forward(depots, pickups, deliveries, deterministic, precise)
        
        device      = pickups.device
        bsz         = depots.size(0)
        nb_requests = pickups.size(1)
        nb_depots   = depots.size(1)
        nb_nodes    = 2*nb_requests + nb_depots
        zero_to_bsz = torch.arange(bsz, device=device) # [0,1,...,bsz-1]
        b_B         = bsz*B
        zero_to_b_B = torch.arange(b_B, device=device) # [0,1,...,bsz-1]

        # -- shared encoding and contexts (same for all beams) --
        h_depot    = self.input_emb_depot(depots)
        h_pickup   = self.input_emb_pickup(pickups)
        h_delivery = self.input_emb_delivery(deliveries)
        h = torch.cat([h_depot, h_pickup, h_delivery], dim=1)
        h_encoder, _ = self.encoder(h)
        fixed_context = self.project_fixed_context(h_encoder.mean(1))[:, None, :]
        K_att_decoder = self.WK_att_decoder(h_encoder)
        V_att_decoder = self.WV_att_decoder(h_encoder)

        # beam tensors shape: (bsz, B, ...)
        # sum_logp = torch.zeros(bsz, B, device=device)
        # tours = torch.zeros(bsz, B, nb_nodes, dtype=torch.long, device=device)
        
        mask        = torch.zeros(bsz, nb_nodes, dtype=torch.bool, device=device)
        vis_node    = torch.zeros_like(mask)
        undel       = torch.zeros(bsz, nb_requests, dtype=torch.bool, device=device)
        current     = torch.zeros(bsz, dtype=torch.long, device=device)      # shape (B,)
        Q           = torch.zeros(bsz, 1, device=device)                     # we won’t really use Q_max here
        T           = torch.zeros(bsz, 1, device=device)                     # accumulated distance

        positions, loads, revenues, Q_max, T_max = preprocess_data(depots, pickups)

        dist_matrix = torch.stack([torch.cdist(positions[b], positions[b]) 
                                for b in range(bsz)], dim=0)  # (bsz,N,N)

        # time step t = 1: start at each depot in beam
        t = 1
        # use depot 0 as start for all beams
        h_t = h_encoder[zero_to_bsz, 0, :]
        h_end = h_encoder[zero_to_bsz, nb_depots-1, :]

        with torch.no_grad():
            # 1) batch‐mask update
            mask, vis_node, _ = mask_cpdp(
                mask, vis_node,
                depots, pickups,
                current.view(bsz,1),   # mask_pdp expects (B,1)
                undel,
                Q.detach().clone().view(bsz,1),       # shapes must match your mask_pdp signature
                T.detach().clone().view(bsz,1),
                positions, Q_max, T_max, dist_matrix, 
                precise
            )
        # Embedding of previous node + remaining capacity            
        query = fixed_context + self.project_step_context(torch.cat(
            [h_t.view(bsz,1,self.dim_emb), h_end.view(bsz,1,self.dim_emb), 
                Q_max.view(bsz,1,1)-Q.view(bsz,1,1), 
                T_max.view(bsz,1,1)-T.view(bsz,1,1)
                ], -1))

        # compute probability over the next node in the tour
        prob_next_node = self.decoder(query, K_att_decoder, V_att_decoder, mask) # size(prob_next_node)=(bsz, nb_nodes)
        # score_t = torch.log(prob_next_node) # size(score_t)=(bsz, nb_nodes+1) for t=0 
        # sum_scores = score_t # size(score_t)=(bsz, nb_nodes+1)
        # choose nodes with top-B sumScores 
        top_val, top_idx = sample_actions(prob_next_node, mask, B)  # size(top_idx)=(bsz, B)

        # top_val, top_idx = torch.topk(sum_scores, B, dim=1) # size(sumScores)=(bsz, B_t0)

        # update sum_t score_{t} for all beams
        sum_scores = top_val # size(sumScores)=(bsz, B_t0) 

        # zero_to_B = torch.arange(B, device=device) # [0,1,...,B_t0-1]
        tours = torch.zeros(bsz, B, nb_nodes, device=device).long() # size(tours)=(bsz, B_t0, nb_nodes)
        tours[:,:,t] = top_idx # size(tours)=(bsz, B_t0, nb_nodes)

        K_att_decoder = K_att_decoder.repeat_interleave(B, dim=0) # size(K_att_decoder)=(bsz*B_t0, nb_nodes+1, dim_emb*nb_layers_decoder)
        V_att_decoder = V_att_decoder.repeat_interleave(B, dim=0) # size(V_att_decoder)=(bsz*B_t0, nb_nodes+1, dim_emb*nb_layers_decoder)
        fixed_context = fixed_context.expand(bsz, B, -1).reshape(b_B, 1, self.dim_emb)    
        h_end         = h_end.unsqueeze(1).expand(bsz, B, -1).reshape(b_B, 1, self.dim_emb)    
                
        out = repeat_interleave(
            B,
            depots=depots,
            pickups=pickups,
            positions=positions,
            revenues=revenues,
            loads=loads,
            dist_matrix=dist_matrix,
            Q_max=Q_max,
            T_max=T_max,
            Q=Q,
            T=T,
            mask=mask,
            vis_node=vis_node,
            undel=undel,
        )    
        (
            depots, pickups, positions, revenues, loads,
            dist_matrix, Q_max, T_max, Q_flat, T_flat,
            mask, vis_node, undel
        ) = (
            out["depots"], out["pickups"], out["positions"], out["revenues"], out["loads"],
            out["dist_matrix"], out["Q_max"], out["T_max"], out["Q"], out["T"],
            out["mask"], out["vis_node"], out["undel"]
        )

        final_idx   = top_idx.reshape(b_B)
        idx_in_beams = top_idx
        
        # iterative beam search
        for t in range(2, nb_nodes):
            current   = tours[:,:,t-2].reshape(b_B)

            def is_terminal(current):
                return (current == (nb_depots - 1)).all()
            
            if is_terminal(current):
                tours[:,:,t] = nb_depots-1
                continue

            Q_flat  += loads[zero_to_b_B, final_idx]
            prev_pos = positions[zero_to_b_B, current]    # (B*E,2)
            next_pos = positions[zero_to_b_B, final_idx]  # (B*E,2)
            cost     = (next_pos - prev_pos).norm(dim=1)  # (B*E,)
            T_flat[:, 0] += cost

            # print(T_flat.reshape(bsz, B))
            # L       = compute_route_length(tours[:,:,:t], positions)
            # print(L.reshape(bsz, B))

            # update masks and constraints
            with torch.no_grad():
                
                # 7a) pickups
                is_pick = (final_idx >= nb_depots) & (final_idx < nb_depots+nb_requests)       # (B,)
                if is_pick.any():
                    pic_idx = final_idx[is_pick] - nb_depots               # which request
                    # mark them undelivered
                    undel[is_pick, pic_idx] = True  

                # 7b) deliveries
                is_delv = (final_idx >= nb_depots+nb_requests)
                if is_delv.any():
                    del_idx = final_idx[is_delv] - nb_depots - nb_requests
                    undel[is_delv, del_idx] = False

                # 7c) record that we’ve visited idx
                vis_node[zero_to_b_B, final_idx] = True

            current = final_idx.clone()

            with torch.no_grad():
                # 1) batch‐mask update
                mask, vis_node, _ = mask_cpdp(
                    mask, vis_node,
                    depots, pickups,
                    current.view(b_B,1),   # mask_pdp expects (B,1)
                    undel,
                    Q_flat.detach().clone().view(b_B,1),       # shapes must match your mask_pdp signature
                    T_flat.detach().clone().view(b_B,1),
                    positions, Q_max, T_max, dist_matrix, 
                    precise
                )
            
            # update embedding of the current visited node
            idx_in_exp = idx_in_beams.unsqueeze(-1).expand(-1, -1, self.dim_emb)  # (bsz, B, dim_emb)
            h_t = torch.gather(h_encoder, dim=1, index=idx_in_exp)                # (bsz, B, dim_emb)
            h_t = h_t.view(b_B, self.dim_emb)
            mask = mask.view(b_B, nb_nodes)
            query = fixed_context.reshape(b_B,1,self.dim_emb) + self.project_step_context(
                torch.cat([h_t.reshape(b_B,1,self.dim_emb),
                           h_end.reshape(b_B,1,self.dim_emb),
                           Q_max.view(b_B,1,1)-Q_flat.view(b_B,1,1),
                           T_max.view(b_B,1,1)-T_flat.view(b_B,1,1)],
                          dim=-1)
            )
            prob_next_node = self.decoder(query, K_att_decoder, V_att_decoder, mask) # size(prob_next_node)=(bsz.B_t0, nb_nodes+1) 
            prob_next_node = prob_next_node.view(bsz, B, nb_nodes) # size(prob_next_node)=(bsz, B_t0, nb_nodes+1) 
            # mask_visited_nodes = mask_visited_nodes.view(bsz, B_t0, nb_nodes+1)
            # h_t = h_t.view(bsz, B, self.dim_emb) 
            # compute score_t + sum_t score_{t-1} for all beams
            score_t = torch.log(prob_next_node) # size(score_t)=(bsz, B, nb_nodes+1) 
            # score_t = torch.log(prob_next_node.clamp_min(1e-12))
            sum_scores = score_t + sum_scores.unsqueeze(2) # size(score_t)=(bsz, B, nb_nodes+1)
            sum_scores_flatten = sum_scores.view(bsz, -1) # size(sumScores_next_node)=(bsz, B.(nb_nodes+1))
            # choose nodes with top-B sumScores 
            top_val, top_idx = torch.topk(sum_scores_flatten, B, dim=1)
            idx_top_beams = top_idx // nb_nodes # size(idx_beam_topB)=(bsz, B)
            idx_in_beams = top_idx - idx_top_beams* nb_nodes # size(idx_in_beams)=(bsz, B)
            # update sum_t score_{t} for all beams
            sum_scores = top_val

            # update beam tours with visited nodes
            tours_tmp = tours.clone()

            mask_tmp = mask.clone().reshape(bsz, B, nb_nodes)
            undel_tmp = undel.clone().reshape(bsz, B, nb_requests)
            vis_node_tmp = vis_node.clone().reshape(bsz, B, nb_nodes)

            Q_flat_tmp = Q_flat.clone().reshape(bsz, B, 1)
            T_flat_tmp = T_flat.clone().reshape(bsz, B, 1)

            # Expand to pick nb_nodes entries per beam
            idx_exp = idx_top_beams.unsqueeze(-1).expand(bsz, B, nb_nodes)  # (bsz, B, nb_nodes)
            idx_exp_req = idx_top_beams.unsqueeze(-1).expand(bsz, B, nb_requests)  # (bsz, B, nb_nodes)

            # Gather along dim=1 (the beam dimension)
            tours = torch.gather(tours_tmp, 1, idx_exp)
            tours[:, :, t] = idx_in_beams

            mask = torch.gather(mask_tmp, 1, idx_exp).reshape(b_B, nb_nodes)
            undel = torch.gather(undel_tmp, 1, idx_exp_req).reshape(b_B, nb_requests)
            vis_node = torch.gather(vis_node_tmp, 1, idx_exp).reshape(b_B, nb_nodes)

            # For Q_flat and T_flat (shape (bsz, B, 1)):
            idx_exp_flat = idx_top_beams.unsqueeze(-1)  # (bsz, B, 1)
            Q_flat = torch.gather(Q_flat_tmp, 1, idx_exp_flat).reshape(b_B, 1)
            T_flat = torch.gather(T_flat_tmp, 1, idx_exp_flat).reshape(b_B, 1)

            # current = tours[:,:,t-1].reshape(b_B)
            # current = torch.gather(current.reshape(bsz,B), 1, idx_top_beams).reshape(b_B)  # (bsz, B)
            final_idx = idx_in_beams.reshape(b_B)

            # L_train, R_train, obj_train = compute_loss(tours, positions, revenues, T_max, b_B, 1, True)
            # print(L_train)

            # if (final_idx == nb_depots-1).all():
            #     break

            # # update embedding of the current visited node
            # h_t = torch.zeros(bsz, B, self.dim_emb, device=device) # size(tours)=(bsz, B_t0, dim_emb)
            # for b in range(bsz):
            #     h_t[b, :, :] = h_encoder[b, idx_in_beams[b], :] # size(h_t)=(bsz, B, dim_emb)

        # after loop: select best beam per batch
        # best_beam = torch.argmax(sum_logp, dim=1)
        # final_tours = tours[torch.arange(bsz), best_beam]
        # final_scores = sum_logp[torch.arange(bsz), best_beam]
        return tours, sum_scores


def sample_actions(prob_next_node: torch.Tensor,
                    mask: torch.BoolTensor,
                    E: int):
    """
    Given:
      - prob_next_node: (B, N) unnormalized or softmax'd scores over next-node
      - mask           : (B, N) True=forbidden, False=allowed
      - E              : number of POMO episodes (starting_nodes)
    Returns:
      - idx_flat  : (B * E,) long tensor of the chosen node indices
      - logp_flat : (B * E,) float tensor of log-probs for each choice
    """
    B, N = prob_next_node.size()
    device = prob_next_node.device

    valid     = ~mask                     # (B, N)
    num_valid = valid.sum(dim=1)          # (B,)
    assert (num_valid > 0).all(), \
        f"Rows with no valid moves: {torch.nonzero(num_valid==0).flatten().tolist()}"

    # how many we can take per row
    E_eff = torch.minimum(num_valid, torch.full_like(num_valid, E))
    E_min = int(E_eff.min().item())

    if E_min >= E:
        # take the “core” top‐E_min for every row
        topk_vals, topk_idx = prob_next_node.topk(E_min, dim=1)  # both (B, E_min)
        topk_logp = torch.log(topk_vals)                        # (B, E_min)
    else:
        # we need to pad each row up to E
        topk_idx  = torch.empty((B, E), dtype=torch.long,   device=device)
        topk_logp = torch.zeros((B, E), dtype=prob_next_node.dtype, device=device)

        for b in range(B):
            ev = int(E_eff[b].item())       # real valid count
            topk_vals_b, topk_idx_b = prob_next_node[b].topk(ev)  # (B,N)
            topk_logp_b = torch.log(topk_vals_b)                # (B, starting_nodes)
            # copy the top-ev
            topk_idx[b, :ev]  = topk_idx_b
            topk_logp[b, :ev] = topk_logp_b
            if ev < E:
                # sample (with replacement) from remaining valid
                allowed = torch.nonzero(valid[b], as_tuple=False).squeeze(1)
                rem = E - ev
                extra = allowed[torch.randint(0, allowed.size(0), (rem,), device=device)]
                topk_idx[b, ev:]  = extra
                topk_logp[b, ev:] = torch.log(prob_next_node[b, extra])

    return topk_logp, topk_idx


class CPDP_POMO_net(nn.Module): 
    """
    The TSP network is composed of two steps :
      Step 1. Encoder step : Take a set of 2D points representing a fully connected graph 
                             and encode the set with self-transformer.
      Step 2. Decoder step : Build the TSP tour recursively/autoregressively, 
                             i.e. one node at a time, with a self-transformer and query-transformer. 
    Inputs : 
      x of size (bsz, nb_nodes, dim_emb) Euclidian coordinates of the nodes/cities
      deterministic is a boolean : If True the salesman will chose the city with highest probability. 
                                   If False the salesman will chose the city with Bernouilli sampling.
    Outputs : 
      tours of size (bsz, nb_nodes) : batch of tours, i.e. sequences of ordered cities 
                                      tours[b,t] contains the idx of the city visited at step t in batch b
      sumLogProbOfActions of size (bsz,) : batch of sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
    """
    
    def __init__(self, dim_input_nodes, dim_emb, dim_ff, step_context_dim, nb_layers_encoder, nb_layers_decoder, nb_heads,
                 batchnorm=True):
        super(CPDP_POMO_net, self).__init__()
        
        self.dim_emb = dim_emb
        
        # input embedding layer
        if type(dim_input_nodes) is list:
            self.input_emb_depot    = nn.Linear(dim_input_nodes[0], dim_emb)
            self.input_emb_pickup   = nn.Linear(dim_input_nodes[1], dim_emb)
            self.input_emb_delivery = nn.Linear(dim_input_nodes[2], dim_emb)
        
        # encoder layer
        self.encoder = Transformer_encoder_net(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batchnorm)
        
        # decoder layer
        self.decoder = Transformer_decoder_net(dim_emb, nb_heads, nb_layers_decoder)
        self.WK_att_decoder = nn.Linear(dim_emb, nb_layers_decoder* dim_emb) 
        self.WV_att_decoder = nn.Linear(dim_emb, nb_layers_decoder* dim_emb) 
        # self.PE = generate_positional_encoding(dim_emb, max_len_PE)        
        self.project_fixed_context = nn.Linear(dim_emb, dim_emb, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, dim_emb, bias=False)
        
    def forward(self, depots, pickups, deliveries, starting_nodes, deterministic=False, precise=False):
        """ loads and revenues are already normalized """

        # some parameters
        device      = pickups.device
        bsz         = depots.size(0)
        nb_requests = pickups.size(1)
        nb_depots   = depots.size(1)
        nb_nodes    = 2*nb_requests + nb_depots
        zero_to_bsz = torch.arange(bsz, device=device) # [0,1,...,bsz-1]
        
        # State
        mask        = torch.zeros(bsz, nb_nodes, dtype=torch.bool, device=device)
        vis_node    = torch.zeros_like(mask)
        undel       = torch.zeros(bsz, nb_requests, dtype=torch.bool, device=device)
        current     = torch.zeros(bsz, dtype=torch.long, device=device)      # shape (B,)
        Q           = torch.zeros(bsz, 1, device=device)                     # we won’t really use Q_max here
        T           = torch.zeros(bsz, 1, device=device)                     # accumulated distance

        positions, loads, revenues, Q_max, T_max = preprocess_data(depots, pickups)

        dist_matrix = torch.stack([torch.cdist(positions[b], positions[b]) 
                                for b in range(bsz)], dim=0)  # (bsz,N,N)

        # input embedding layer
        h_depot    = self.input_emb_depot(depots) # size(h)=(bsz, 2, 4)
        h_pickup   = self.input_emb_pickup(pickups) # size(h)=(bsz, nb_nodes, 6)
        h_delivery = self.input_emb_delivery(deliveries) # size(h)=(bsz, nb_nodes, 4)
        
        h = torch.cat([h_depot, h_pickup, h_delivery], dim=1)
        # encoder layer
        h_encoder, _ = self.encoder(h) # size(h)=(bsz, nb_nodes, dim_emb)

        graph_embed = h_encoder.mean(1)
        # fixed context = (bsz, 1, dim_emb) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours = [torch.zeros(bsz*starting_nodes, dtype=torch.int, device=device)]

        # list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
        sumLogProbOfActions = []

        # key and value for decoder    
        K_att_decoder = self.WK_att_decoder(h_encoder) # size(K_att)=(bsz, nb_nodes, dim_emb*nb_layers_decoder)
        V_att_decoder = self.WV_att_decoder(h_encoder) # size(V_att)=(bsz, nb_nodes, dim_emb*nb_layers_decoder)
        
        h_start = h_encoder[zero_to_bsz, 0, :] 
        h_end   = h_encoder[zero_to_bsz, nb_depots-1, :] 

        # POMO
        h_t = h_start
        
        with torch.no_grad():
            # 1) batch‐mask update
            mask, vis_node, _ = mask_cpdp(
                mask, vis_node,
                depots, pickups,
                current.view(bsz,1),   # mask_pdp expects (B,1)
                undel,
                Q.detach().clone().view(bsz,1),       # shapes must match your mask_pdp signature
                T.detach().clone().view(bsz,1),
                positions, Q_max, T_max, dist_matrix, 
                precise
            )
             
        assert (~mask).any(1).all(), "Some rows are fully masked out!"

        # Embedding of previous node + remaining capacity            
        query = fixed_context + self.project_step_context(torch.cat(
            [h_t.view(bsz,1,self.dim_emb), h_end.view(bsz,1,self.dim_emb), 
                Q_max.view(bsz,1,1)-Q.view(bsz,1,1), 
                T_max.view(bsz,1,1)-T.view(bsz,1,1)
                ], -1))

        # compute probability over the next node in the tour
        prob_next_node = self.decoder(query, K_att_decoder, V_att_decoder, mask) # size(prob_next_node)=(bsz, nb_nodes)

        # POMO
        final_logp, final_idx = sample_actions(prob_next_node, mask, starting_nodes)
        final_logp = final_logp.reshape(-1)
        final_idx = final_idx.reshape(-1)

        # topk_vals, topk_idx = prob_next_node.topk(starting_nodes, dim=1)  # (B,N)
        # topk_flat, logp_flat = topk_idx.reshape(-1), torch.log(topk_vals).reshape(-1)
        tours.append(final_idx)
        sumLogProbOfActions.append(final_logp) 
    
        out = repeat_interleave(
            starting_nodes,
            depots=depots,
            pickups=pickups,
            positions=positions,
            revenues=revenues,
            loads=loads,
            dist_matrix=dist_matrix,
            Q_max=Q_max,
            T_max=T_max,
            Q=Q,
            T=T,
            current=current,
            mask=mask,
            vis_node=vis_node,
            undel=undel,
        )    
        (
            depots, pickups, positions, revenues, loads,
            dist_matrix, Q_max, T_max, Q_flat, T_flat,
            current, mask, vis_node, undel
        ) = (
            out["depots"], out["pickups"], out["positions"], out["revenues"], out["loads"],
            out["dist_matrix"], out["Q_max"], out["T_max"], out["Q"], out["T"],
            out["current"], out["mask"], out["vis_node"], out["undel"]
        )
        
        b_e         = bsz * starting_nodes
        zero_to_b_e = torch.arange(b_e, device=device) # [0,1,...,bsz-1]

        out = repeat_interleave(
            starting_nodes,
            h_encoder=h_encoder,
            h_end=h_end,
            K_att_decoder=K_att_decoder,
            V_att_decoder=V_att_decoder,
            fixed_context=fixed_context,
        )    
        (
            h_encoder_flat, h_end_flat, K_att_decoder_flat, V_att_decoder_flat, fixed_context_flat
        ) = (
            out["h_encoder"], out["h_end"], out["K_att_decoder"], out["V_att_decoder"], out["fixed_context"]
        )
        
        while True:

            Q_flat += loads[zero_to_b_e, final_idx]
            prev_pos = positions[zero_to_b_e, current]   # (B,2)
            next_pos = positions[zero_to_b_e, final_idx]       # (B,2)
            cost     = (next_pos - prev_pos).norm(dim=1)  # (B,)
            T_flat[:, 0] += cost

            # print(T_flat.reshape(bsz, starting_nodes))
            # L       = compute_route_length(torch.stack(tours,dim=1), positions)
            # print(L.reshape(bsz, starting_nodes))

            # update masks and constraints
            with torch.no_grad():
                
                # 7a) pickups
                is_pick = (final_idx >= nb_depots) & (final_idx < nb_depots+nb_requests)       # (B,)
                if is_pick.any():
                    pic_idx = final_idx[is_pick] - nb_depots               # which request
                    # mark them undelivered
                    undel[is_pick, pic_idx] = True  

                # 7b) deliveries
                is_delv = (final_idx >= nb_depots+nb_requests)
                if is_delv.any():
                    del_idx = final_idx[is_delv] - nb_depots - nb_requests
                    undel[is_delv, del_idx] = False

                # 7c) record that we’ve visited idx
                vis_node[zero_to_b_e, final_idx] = True
            
            current = final_idx.clone()

            if (current == nb_depots-1).all():
                break

            h_t_flat = h_encoder_flat[zero_to_b_e, current, :] 

            with torch.no_grad():
                # 1) batch‐mask update
                mask, vis_node, _ = mask_cpdp(
                    mask, vis_node,
                    depots, pickups,
                    current.view(b_e,1),   # mask_pdp expects (B,1)
                    undel,
                    Q_flat.detach().clone().view(b_e,1),       # shapes must match your mask_pdp signature
                    T_flat.detach().clone().view(b_e,1),
                    positions, Q_max, T_max, dist_matrix, 
                    precise
                )
            
            assert (~mask).any(1).all(), "Some rows are fully masked out!"

            # Embedding of previous node + remaining capacity            
            query = fixed_context_flat + self.project_step_context(torch.cat(
                [h_t_flat.view(b_e,1,self.dim_emb), h_end_flat.view(b_e,1,self.dim_emb), 
                 Q_max.view(b_e,1,1)-Q_flat.view(b_e,1,1), 
                 T_max.view(b_e,1,1)-T_flat.view(b_e,1,1)
                 ], -1))
            
            # compute probability over the next node in the tour
            prob_next_node = self.decoder(query, K_att_decoder_flat, V_att_decoder_flat, mask) # size(prob_next_node)=(bsz, nb_nodes)
            
            # choose node with highest probability or sample with Bernouilli 
            if deterministic:
                final_idx = torch.argmax(prob_next_node, dim=1) # size(query)=(bsz,)
            else:
                final_idx = Categorical(prob_next_node).sample() # size(query)=(bsz,)

            # compute logprobs of the action items in the list sumLogProbOfActions   
            ProbOfChoices = prob_next_node[zero_to_b_e, final_idx] 
            sumLogProbOfActions.append( torch.log(ProbOfChoices) )  # size(query)=(bsz,)

            # update tour
            tours.append(final_idx)
        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        sumLogProbOfActions = torch.stack(sumLogProbOfActions,dim=1).sum(dim=1) # size(sumLogProbOfActions)=(bsz,)

        # convert the list of nodes into a tensor of shape (bsz,num_cities)
        tours = torch.stack(tours,dim=1) # size(col_index)=(bsz, nb_nodes)
        tours = F.pad(tours, (0, nb_nodes - tours.size(1)), value=nb_depots-1)
        
        return tours, sumLogProbOfActions


class CPDP_AM_net_SGBS(CPDP_AM_net):
    def forward(self,
                depots, pickups, deliveries,
                beta=5, gamma=5, 
                deterministic=False,
                beamsearch=False,
                precise=True):
        
        # initialize beam containers
        if not beamsearch:
            return super().forward(depots, pickups, deliveries, deterministic, precise)
        device      = pickups.device
        bsz         = depots.size(0)
        nb_requests = pickups.size(1)
        nb_depots   = depots.size(1)
        nb_nodes    = 2*nb_requests + nb_depots
        zero_to_bsz = torch.arange(bsz, device=device) # [0,1,...,bsz-1]

        assert gamma < nb_nodes

        # -- shared encoding and contexts (same for all beams) --
        h_depot    = self.input_emb_depot(depots)
        h_pickup   = self.input_emb_pickup(pickups)
        h_delivery = self.input_emb_delivery(deliveries)
        h = torch.cat([h_depot, h_pickup, h_delivery], dim=1)
        h_encoder, _ = self.encoder(h)
        fixed_context = self.project_fixed_context(h_encoder.mean(1))[:, None, :]
        K_att_decoder = self.WK_att_decoder(h_encoder)
        V_att_decoder = self.WV_att_decoder(h_encoder)

        # beam tensors shape: (bsz, B, ...)
        # sum_logp = torch.zeros(bsz, B, device=device)        
        mask        = torch.zeros(bsz, nb_nodes, dtype=torch.bool, device=device)
        vis_node    = torch.zeros_like(mask)
        undel       = torch.zeros(bsz, nb_requests, dtype=torch.bool, device=device)
        current     = torch.zeros(bsz, dtype=torch.long, device=device)      # shape (B,)
        Q           = torch.zeros(bsz, 1, device=device)                     # we won’t really use Q_max here
        T           = torch.zeros(bsz, 1, device=device)                     # accumulated distance

        positions, loads, revenues, Q_max, T_max = preprocess_data(depots, pickups)

        dist_matrix = torch.stack([torch.cdist(positions[b], positions[b]) 
                                for b in range(bsz)], dim=0)  # (bsz,N,N)
        
        # ---------- helpers ----------
        def flat(x):           # (bsz,B,...) -> (bsz*B,...)
            return x.reshape(-1, *x.shape[2:])
        def unflat(x, B):      # (bsz*B,...) -> (bsz,B,...)
            return x.reshape(bsz, B, *x.shape[1:])
        def repeat_KV(B):      # broadcast decoder KV to (bsz*B, ...)
            K = K_att_decoder.repeat_interleave(B, dim=0)
            V = V_att_decoder.repeat_interleave(B, dim=0)
            return K, V
        
        # time step t = 1: start at each depot in beam
        t = 1
        # use depot 0 as start for all beams
        h_t = h_encoder[zero_to_bsz, 0, :]
        h_end = h_encoder[zero_to_bsz, nb_depots-1, :]

        with torch.no_grad():
            # 1) batch‐mask update
            mask, vis_node, _ = mask_cpdp(
                mask, vis_node,
                depots, pickups,
                current.view(bsz,1),   # mask_pdp expects (B,1)
                undel,
                Q.detach().clone().view(bsz,1),       # shapes must match your mask_pdp signature
                T.detach().clone().view(bsz,1),
                positions, Q_max, T_max, dist_matrix, 
                precise
            )
        # Embedding of previous node + remaining capacity            
        query = fixed_context + self.project_step_context(torch.cat(
            [h_t.view(bsz,1,self.dim_emb), h_end.view(bsz,1,self.dim_emb), 
                Q_max.view(bsz,1,1)-Q.view(bsz,1,1), 
                T_max.view(bsz,1,1)-T.view(bsz,1,1)
                ], -1))

        start_b = min(gamma, beta)
        b_start_b = bsz*start_b
        zero_to_b_start_b = torch.arange(b_start_b, device=device)

        # compute probability over the next node in the tour
        prob_next_node = self.decoder(query, K_att_decoder, V_att_decoder, mask) # size(prob_next_node)=(bsz, nb_nodes)

        # choose nodes with top-B sumScores 
        _, top_idx = sample_actions(prob_next_node, mask, start_b)  # size(top_idx)=(bsz, B)

        # zero_to_B = torch.arange(B, device=device) # [0,1,...,B_t0-1]
        tours = torch.zeros(bsz, start_b, nb_nodes, device=device).long() # size(tours)=(bsz, B_t0, nb_nodes)
        tours[:,:,t] = top_idx # size(tours)=(bsz, B_t0, nb_nodes)

        # K, V = repeat_KV(B)
        # fixed_context = fixed_context.expand(bsz, start_b, -1).reshape(b_start_b, 1, self.dim_emb)    
        # h_end         = h_end.unsqueeze(1).expand(bsz, start_b, -1).reshape(b_start_b, 1, self.dim_emb)    
        
        out = repeat_interleave(
            start_b,
            depots=depots,
            pickups=pickups,
            positions=positions,
            revenues=revenues,
            loads=loads,
            dist_matrix=dist_matrix,
            Q_max=Q_max,
            T_max=T_max,
            Q=Q,
            T=T,
            mask=mask,
            vis_node=vis_node,
            undel=undel,
        )    
        (
            depots_c, pickups_c, positions_c, revenues_c, loads_c,
            dist_matrix_c, Q_max_c, T_max_c, Q_flat, T_flat,
            mask, vis_node, undel
        ) = (
            out["depots"], out["pickups"], out["positions"], out["revenues"], out["loads"],
            out["dist_matrix"], out["Q_max"], out["T_max"], out["Q"], out["T"],
            out["mask"], out["vis_node"], out["undel"]
        )    

        next_idx = top_idx.reshape(b_start_b)

        current  = tours[:,:,t-2].reshape(b_start_b)

        Q_flat  += loads_c[zero_to_b_start_b, next_idx]
        prev_pos = positions_c[zero_to_b_start_b, current]    # (B*E,2)
        next_pos = positions_c[zero_to_b_start_b, next_idx]  # (B*E,2)
        cost     = (next_pos - prev_pos).norm(dim=1)  # (B*E,)
        T_flat[:, 0] += cost

        # print(T_flat.reshape(bsz, B))
        # L       = compute_route_length(tours[:,:,:t], positions)
        # print(L.reshape(bsz, B))

        # update masks and constraints
        with torch.no_grad():
            
            # 7a) pickups
            is_pick = (next_idx >= nb_depots) & (next_idx < nb_depots+nb_requests)       # (B,)
            if is_pick.any():
                pic_idx = next_idx[is_pick] - nb_depots               # which request
                # mark them undelivered
                undel[is_pick, pic_idx] = True  

            # 7b) deliveries
            is_delv = (next_idx >= nb_depots+nb_requests)
            if is_delv.any():
                del_idx = next_idx[is_delv] - nb_depots - nb_requests
                undel[is_delv, del_idx] = False

            # 7c) record that we’ve visited idx
            vis_node[zero_to_b_start_b, next_idx] = True

        current = next_idx.clone() # (bsz*start_b)

        with torch.no_grad():
            # 1) batch‐mask update
            m, v, _ = mask_cpdp(
                mask, vis_node,
                depots_c, pickups_c,
                current.view(b_start_b,1),   # mask_pdp expects (B,1)
                undel,
                Q_flat.detach().clone().view(b_start_b,1),       # shapes must match your mask_pdp signature
                T_flat.detach().clone().view(b_start_b,1),
                positions_c, Q_max_c, T_max_c, dist_matrix_c, 
                precise
            )

        # ---------- helpers ----------
        # decode one step (returns log-prob and prob) for (bsz,B)
        def decode_step(current_d, Q_flat_d, T_flat_d, mask_d, B, h_encoder_flat, h_end_flat, fixed_context_flat):
            # gather h_t: (bsz,B,d)
            batch_idx = torch.arange(bsz*B, device=device)
            h_t = h_encoder_flat[batch_idx, current_d, :]                                  # (bsz,B,d)

            step_ctx = self.project_step_context(torch.cat([
                h_t.view(-1,1,self.dim_emb), h_end_flat.view(-1,1,self.dim_emb),
                Q_max_c.view(-1,1,1) - Q_flat_d.view(-1,1,1),      # remaining capacity
                T_max_c.view(-1,1,1) - T_flat_d.view(-1,1,1),      # remaining time
            ], dim=-1)).unsqueeze(2)                                         # (bsz,B,1,d)

            query = fixed_context_flat.unsqueeze(2) + step_ctx                               # (bsz,B,1,d)

            q  = flat(query)                                                  # (bsz*B,1,d)
            # m  = flat(mask)                                                   # (bsz*B,N)
            K, V = repeat_KV(B)

            # decoder expected to apply masking inside; it returns probabilities
            prob = self.decoder(q, K, V, mask_d)                                   # (bsz*B,N)
            logp = torch.log(prob.clamp_min(1e-12))                           # stable
            return logp, prob

        # write new_idx at position `step` and refresh all states; shapes are (bsz,B,·)
        def apply_step(B, *, parent_idx, new_idx, Q_flat_r, T_flat_r, mask_r, vis_node_r, undel_r, precise_):
            # write tour
            new_idx = flat(new_idx)
            # tours_r[:, :, step] = new_idx
            zero_to_b_B = torch.arange(bsz*B, device=device)
            # distance / load increment (prev -> new)
            prev_pos = positions_c[zero_to_b_B, flat(parent_idx)]
            new_pos  = positions_c[zero_to_b_B, new_idx]
            seg = (new_pos - prev_pos).norm(dim=1).view(bsz*B, 1)
            T_flat_r = T_flat_r + seg
            Q_flat_r = Q_flat_r + loads_c[zero_to_b_B, new_idx]

            # pickups / deliveries flags
            is_pick = (new_idx >= nb_depots) & (new_idx < nb_depots + nb_requests)
            is_delv = (new_idx >= nb_depots + nb_requests)
            if is_pick.any():
                pic_idx = new_idx[is_pick] - nb_depots
                undel_r[is_pick, pic_idx] = True
            if is_delv.any():
                del_idx = new_idx[is_delv] - nb_depots - nb_requests
                undel_r[is_delv, del_idx] = False

            # visited
            vis_node_r[zero_to_b_B, new_idx] = True

            # recompute mask for current = new_idx
            m, v, _ = mask_cpdp(
                mask_r, vis_node_r,
                depots_c, pickups_c,
                new_idx.unsqueeze(1),
                undel_r,
                Q_flat_r, T_flat_r,
                positions_c, Q_max_c, T_max_c, dist_matrix_c, 
                precise_,
            )
            return Q_flat_r, T_flat_r, m, v, undel_r

        B = start_b
        # ---------- SGBS main loop ----------
        for step in range(2, nb_nodes):
            # print(step)
            def is_terminal(current):
                return (current == (nb_depots - 1)).all()
            
            if is_terminal(current):
                tours[:,:,step] = nb_depots-1
                continue
            
            h_encoder_flat = h_encoder.repeat_interleave(B, dim=0) # (B*E, N, N)
            h_end_flat     = h_end.repeat_interleave(B, dim=0) # (B*E, N, N)
            fixed_context_flat = fixed_context.repeat_interleave(B, dim=0)

            # ===== Expansion: for each beam, pick top-γ children by πθ =====
            logp, prob = decode_step(current, Q_flat, T_flat, mask, B, h_encoder_flat, h_end_flat, fixed_context_flat)        # (bsz,B,N)
            # choose nodes with top-B sumScores 
            _, top_idx = sample_actions(prob, mask, gamma)  # size(top_idx)=(bsz, B)

            # build β×γ candidates (as a single (bsz,Bγ,·) state)
            cand_B = B * gamma
            parent_idx = current.view(bsz, -1, 1).expand(-1, -1, gamma)       # (bsz,B,γ)
            cand_parent = parent_idx.reshape(bsz, cand_B)
            cand_new    = top_idx.reshape(bsz, cand_B)

            tours_c = tours.unsqueeze(2).expand(-1, -1, gamma, -1).reshape(bsz, cand_B, nb_nodes)  
            out = repeat_interleave(
                gamma,
                depots=depots_c,
                pickups=pickups_c,
                positions=positions_c,
                revenues=revenues_c,
                loads=loads_c,
                dist_matrix=dist_matrix_c,
                Q_max=Q_max_c,
                T_max=T_max_c,
                Q=Q_flat,
                T=T_flat,
                mask=mask,
                vis_node=vis_node,
                undel=undel
            )
            (
                depots_c, pickups_c, positions_c, revenues_c, loads_c,
                dist_matrix_c, Q_max_c, T_max_c, Q_flat_c, T_flat_c,
                mask_c, vis_c, undel_c
            ) = (
                out["depots"], out["pickups"], out["positions"], out["revenues"], out["loads"],
                out["dist_matrix"], out["Q_max"], out["T_max"], out["Q"], out["T"],
                out["mask"], out["vis_node"], out["undel"]
            )

            # apply one step (write at `step`)
            tours_c[:, :, step] = cand_new
            Q_c, T_c, mask_c, vis_c, undel_c = apply_step(
                cand_B, 
                parent_idx=cand_parent, new_idx=cand_new, 
                Q_flat_r=Q_flat_c, T_flat_r=T_flat_c,
                mask_r=mask_c, vis_node_r=vis_c, undel_r=undel_c,
                precise_=precise
            )
            cur_c = cand_new

            # ===== Simulation: single greedy approximate rollout from each candidate =====
            def greedy_rollout(tours_r, cur_r, Q_r, T_r, mask_r, vis_r, undel_r, step_r, B_r):
                h_encoder_flat = h_encoder.repeat_interleave(B_r, dim=0) # (B*E, N, N)
                h_end_flat     = h_end.repeat_interleave(B_r, dim=0) # (B*E, N, N)
                fixed_context_flat = fixed_context.repeat_interleave(B_r, dim=0)
                # rollout up to remaining steps or until all terminal
                for t in range(step_r, nb_nodes):
                    logp_r, _ = decode_step(cur_r.view(-1), Q_r, T_r, mask_r, B_r, h_encoder_flat, h_end_flat, fixed_context_flat)
                    nxt = logp_r.argmax(dim=1) if deterministic else torch.distributions.Categorical(logp_r.exp()).sample()
                    nxt = nxt.reshape(bsz, B_r)
                    tours_r[:, :, t] = nxt
                    Q_r, T_r, mask_r, vis_r, undel_r = apply_step(
                        B_r,
                        parent_idx=cur_r, new_idx=nxt, Q_flat_r=Q_r, T_flat_r=T_r,
                        mask_r=mask_r, vis_node_r=vis_r, undel_r=undel_r,
                        precise_=False
                    )
                    cur_r = nxt
                # evaluate candidate by its simulated reward
                return tours_r

            tours_r = greedy_rollout(
                tours_c.clone(), cur_c.clone(), Q_c.clone(), T_c.clone(),
                mask_c.clone(), vis_c.clone(), undel_c.clone(),
                step_r=step+1, B_r=cand_B
            )  # (bsz, cand_B)
            R_c = compute_collected_revenue(tours_r, revenues_c)

            # ===== Pruning: keep β best candidates by R =====
            keep = min(beta, cand_B)

            def topk_unique_requestset_per_batch(
                    tours_r: torch.Tensor,   # (bsz, X, N) long
                    scores: torch.Tensor,    # (bsz, X)    float
                    keep: int,
                    D: int,                  # number of depots
                    R: int,                  # number of requests (pickups)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    """
                    For each batch b:
                    - Group routes by the SET of requests visited (ignoring order, depots, padding).
                    - From each group, keep only the best-scoring route.
                    - Then select up to `keep` highest-scoring distinct request-sets.

                    Returns:
                    best_idx_padded: (bsz, keep) long; index into tours_r[b], padded with pad_value
                    sel_mask:        (bsz, keep) bool; True where valid
                    """
                    device = tours_r.device
                    pad_value = D-1
                    bsz, X, N = tours_r.shape
                    best_idx_padded = torch.full((bsz, keep), pad_value, dtype=torch.long, device=device)
                    sel_mask        = torch.zeros((bsz, keep), dtype=torch.bool, device=device)

                    for b in range(bsz):
                        routes_b = tours_r[b]    # (X, N)
                        scores_b = scores[b]     # (X,)

                        if routes_b.numel() == 0:
                            continue

                        # --- build request presence signature (ignore depots/padding) ---
                        # pickups assumed in [D, D+R)
                        is_pick = (routes_b >= D) & (routes_b < D + R)
                        pick_idx = torch.where(is_pick, routes_b - D, torch.full_like(routes_b, -1))

                        pres = torch.zeros(X, R, dtype=torch.bool, device=device)
                        row_ids = torch.arange(X, device=device).unsqueeze(1).expand(X, N).reshape(-1)
                        pi = pick_idx.reshape(-1)
                        mask_p = pi >= 0
                        pres[row_ids[mask_p], pi[mask_p]] = True   # mark visited requests

                        # group by presence signature
                        uniq, inv = torch.unique(pres, dim=0, return_inverse=True)
                        n_groups = uniq.size(0)
                        if n_groups == 0:
                            continue

                        reps = []
                        for g in range(n_groups):
                            idxs = torch.nonzero(inv == g, as_tuple=False).squeeze(1)   # (k,)
                            local_best_pos = scores_b[idxs].argmax().item()
                            best_in_group = idxs[local_best_pos]
                            reps.append(best_in_group)

                        reps = torch.stack(reps, dim=0)         # (G,)
                        rep_scores = scores_b[reps]             # (G,)

                        k = min(keep, reps.numel())
                        top_vals, top_pos = torch.topk(rep_scores, k=k, dim=0)
                        chosen = reps[top_pos]                  # (k,)

                        best_idx_padded[b, :k] = chosen
                        sel_mask[b, :k] = True

                    return best_idx_padded, sel_mask

            # topR, best_idx = torch.topk(R_c.reshape(bsz,cand_B), k=keep, dim=1)  # (bsz, keep)
            best_idx, _ = topk_unique_requestset_per_batch(tours_r, unflat(R_c,cand_B), keep, nb_depots, nb_requests)

            def pick(X, is_nodes=False):
                if is_nodes:
                    idx = best_idx.unsqueeze(-1).expand(-1, -1, nb_nodes)
                else:
                    idx = best_idx.unsqueeze(-1).expand(-1, -1, X.size(-1))
                return torch.gather(X, 1, idx)

            tours    = pick(tours_c, is_nodes=True)     # (bsz, keep, N)
            current  = flat(torch.gather(cur_c, 1, best_idx)) # (bsz, keep)
            mask     = flat(pick(unflat(mask_c, cand_B), is_nodes=True))
            vis_node = flat(pick(unflat(vis_c, cand_B), is_nodes=True))
            undel    = flat(pick(unflat(undel_c, cand_B)))
            Q_flat   = flat(pick(unflat(Q_c, cand_B)))
            T_flat   = flat(pick(unflat(T_c, cand_B)))
            B        = keep

            out = repeat_interleave(
                keep,
                depots=depots,
                pickups=pickups,
                positions=positions,
                revenues=revenues,
                loads=loads,
                dist_matrix=dist_matrix,
                Q_max=Q_max,
                T_max=T_max,
            )
            (
                depots_c, pickups_c, positions_c, revenues_c, 
                loads_c, dist_matrix_c, Q_max_c, T_max_c, 
            ) = (
                out["depots"], out["pickups"], out["positions"], out["revenues"], 
                out["loads"], out["dist_matrix"], out["Q_max"], out["T_max"]
            )

        # ---------- pick final best tour ----------
        R_final = compute_collected_revenue(tours, revenues_c)             # (bsz,B)
        R_final = unflat(R_final, B)
        best = R_final.argmax(dim=1)
        # final_tours = tours[zero_to_bsz, best]      # (bsz, N)
        # final_R     = R_final[zero_to_bsz, best]
        return tours, R_final
