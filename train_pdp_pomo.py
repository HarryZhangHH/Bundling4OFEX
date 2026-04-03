import math
import argparse
import datetime
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.utils.rnn import pad_sequence
from scipy.stats import ttest_rel
from torch.cuda.amp import autocast

from nets.pdp_attention_model import PDP_AM_net
from nets.cpdp_attention_model import CPDP_POMO_net

from utils.pdp_functions import *
from utils.data_utils import load_dataset
from utils.functions import *
from algs.greedy_search import greedy_cpdp, two_opt_improve
from algs.large_neighborhood_search import hill_climb, large_negihborhood_search

def train_pdp_pomo(args, device, dataset, dataset_2, problem='cpdp', model='pomo'):
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
    
    seed_everything(seed=args.seed)

    assert len(args.dim_input_nodes) == 3

    r_max = max([req['revenue']
                 for instance in dataset
                 for req in instance['requests']])
    
    bsz_test = 100
    batch_test = sample_batch(dataset_2, bsz_test, seed=args.seed)

    depots_test, requests_test = collate_pdp(batch_test, device=device, seed=args.seed, c=args.c)
    positions_test, _, revenues_test, _, T_max_test = preprocess_data(depots_test, requests_test)
    depots_test_nor, pickups_test_nor, deliveries_test_nor = normalize_features(depots_test, requests_test, r_max)

    positions_test_dup = positions_test.repeat_interleave(args.starting_nodes, dim=0)    # (B*E, N, 2)
    revenues_test_dup  = revenues_test.repeat_interleave(args.starting_nodes, dim=0)    # (B*E, N, 2)
    T_max_test_dup     = T_max_test.repeat_interleave(args.starting_nodes, dim=0)        # (B*E, 1)


    model_train = CPDP_POMO_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.step_context_dim,
                args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads,
                batchnorm=args.batchnorm)
    
    optimizer = torch.optim.Adam( model_train.parameters() , lr = args.lr, weight_decay=args.weight_decay ) 

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[501,],   # at epoch=30 and epoch=80
        gamma=0.1              # multiply the lr by 0.1 at each milestone
    )

    # uncomment these lines if trained with multiple GPUs
    print(torch.cuda.device_count())
    if torch.cuda.device_count()>1:
        model_train = nn.DataParallel(model_train)
    # uncomment these lines if trained with multiple GPUs

    model_train = model_train.to(device)

    print(args); print('')

    # Logs
    #os.system("mkdir logs")
    time_stamp=datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
    file_name = 'logs_new'+'/'+ problem + '_' + model + args.prize_type + time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + ".txt"
    file = open(file_name,"w",1) 
    file.write(time_stamp+'\n\n') 
    for arg in vars(args):
        file.write(arg)
        hyper_param_val="={}".format(getattr(args, arg))
        file.write(hyper_param_val)
        file.write('\n')
    file.write('\n\n') 
    checkpoint_performance_train = []
    checkpoint_performance_baseline = []
    all_strings = []
    epoch_ckpt = 0
    tot_time_ckpt = 0

    # Uncomment these lines to re-start training with saved checkpoint
    # checkpoint_file = "checkpoint_new/checkpoint_cpdp_pomodistabs25-09-29--01-49-00-n60-gpu0.pkl"
    # checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    # epoch_ckpt = checkpoint['epoch'] + 1
    # tot_time_ckpt = checkpoint['tot_time']
    # checkpoint_performance_train = checkpoint['checkpoint_performance_train']
    # model_train.load_state_dict(checkpoint['model_train'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler.last_epoch = checkpoint['epoch']-1
    # print('Re-start training with saved checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))
    # del checkpoint
    # Uncomment these lines to re-start training with saved checkpoint

    ###################
    # Main training loop 
    ###################
    start_training_time = time.time()

    for epoch in range(0,args.nb_epochs):
        
        # re-start training with saved checkpoint
        epoch += epoch_ckpt
        
        if epoch > args.nb_epochs-1:
            break

        ###################
        # Train model for one epoch
        ###################
        start = time.time()
        model_train.train()

        # LR Decay
        scheduler.step() 
        loss_list = []
        for step in (range(1,args.nb_batch_per_epoch+1)):    

            batch = sample_batch(dataset, args.bsz)

            # trick
            # c = random.randint(-1, len(batch[0]['T']))

            depots, requests = collate_pdp(batch, device=device, c=args.c)

            depots_nor, pickups_nor, deliveries_nor = normalize_features(depots, requests, r_max)

            # compute tours for model
            with autocast(dtype=torch.bfloat16):
                tour_train, sumLogProbOfActions = model_train(depots_nor, pickups_nor, deliveries_nor, args.starting_nodes, deterministic=False) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)

            positions, _, revenues, _, T_max = preprocess_data(depots, requests)

            positions   = positions.repeat_interleave(args.starting_nodes, dim=0)    # (B*E, N, 2)
            revenues    = revenues.repeat_interleave(args.starting_nodes, dim=0)    # (B*E, N, 2)
            T_max       = T_max.repeat_interleave(args.starting_nodes, dim=0)        # (B*E, 1)
            
            L_train, R_train, obj_train = compute_loss(tour_train, positions, revenues, T_max, args.bsz*args.starting_nodes, args.penalty_factor, args.multi_obj)
            obj_baseline = obj_train.view(args.bsz, args.starting_nodes, -1).mean(dim=1)
            obj_baseline = obj_baseline.repeat_interleave(args.starting_nodes, dim=0).squeeze()

            assert obj_baseline.size() == sumLogProbOfActions.size()

            # backprop
            loss = torch.mean( (obj_train - obj_baseline) * sumLogProbOfActions )

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=1.0)
            optimizer.step()
            loss_list.append(loss)
            
        time_one_epoch = time.time()-start
        time_tot = time.time()-start_training_time + tot_time_ckpt

        # inference'
        #model_train.eval()
        #with torch.no_grad():
        #    tour_test, _ = model_train(depots_test_nor, pickups_test_nor, deliveries_test_nor, args.starting_nodes, deterministic=True, precise=True)

        #L_train, R_train, obj_train = compute_loss(tour_test, positions_test_dup, revenues_test_dup, T_max_test_dup, bsz_test*args.starting_nodes, args.penalty_factor, True)
        #max_idxs = obj_train.view(bsz_test, args.starting_nodes).argmin(dim=1)
        #tour_test = tour_test.view(bsz_test, args.starting_nodes, -1)         # (B, E, L)
        #batch_idx = torch.arange(bsz_test, device=device)
        #tour_test = tour_test[batch_idx, max_idxs]
        #L_baseline_test, R_baseline_test, obj_baseline_test = compute_loss(tour_test, positions_test, revenues_test, T_max_test, bsz_test, args.penalty_factor, True)
        #mean_tour_length_test, mean_tour_revenue_test, mean_tour_obj_test = L_baseline_test.mean().item(), R_baseline_test.mean().item(), obj_baseline_test.mean().item()
        
        #mean_obj_test_all = obj_train.mean().item()
        #checkpoint_performance_train.append([ (epoch+1), mean_obj_test_all])
        
        model_train.eval()
        with torch.no_grad():
            tour_test, _ = model_train(depots_test_nor, pickups_test_nor, deliveries_test_nor, 1, deterministic=True, precise=True)
        
        L_baseline_test, R_baseline_test, obj_baseline_test = compute_loss(tour_test, positions_test, revenues_test, T_max_test, bsz_test, args.penalty_factor, True)
        mean_tour_length_test_s, mean_tour_revenue_test_s, mean_tour_obj_test_s = L_baseline_test.mean().item(), R_baseline_test.mean().item(), obj_baseline_test.mean().item()
        
        checkpoint_performance_train.append([ (epoch+1), mean_tour_obj_test_s])
        
        # # Compute optimality gap
        # if args.nb_nodes==50: gap_train = mean_tour_length_train/5.692- 1.0
        # elif args.nb_nodes==100: gap_train = mean_tour_length_train/7.765- 1.0
        # else: gap_train = -1.0
        # gap_train = -1.0
        # Print and save in txt file
        # mystring_min = 'Epoch: {:d}, epoch time: {:.3f}min, tot time: {:.3f}day, L_test: {:.3f}, R_test: {:.3f}, obj_test: {:.3f}, obj_test_all: {:.3f}, L_test_s: {:.3f}, R_test_s: {:.3f}, obj_test_s: {:.3f}, avg_loss: {:.3f}'.format(epoch, time_one_epoch/60, time_tot/86400, mean_tour_length_test, mean_tour_revenue_test, mean_tour_obj_test, mean_obj_test_all, mean_tour_length_test_s, mean_tour_revenue_test_s, mean_tour_obj_test_s, torch.stack(loss_list, dim=0).mean(dim=0)) 
        mystring_min = 'Epoch: {:d}, epoch time: {:.3f}min, tot time: {:.3f}day, L_test_s: {:.3f}, R_test_s: {:.3f}, obj_test_s: {:.3f}, avg_loss: {:.3f}'.format(epoch, time_one_epoch/60, time_tot/86400, mean_tour_length_test_s, mean_tour_revenue_test_s, mean_tour_obj_test_s, torch.stack(loss_list, dim=0).mean(dim=0)) 
        print(mystring_min) # Comment if plot display

        file.write(mystring_min+'\n')
    
        # Saving checkpoint
        num = ''
        if epoch == 99:
          num = '-100'
        if epoch == 149:
          num = '-150'
        if epoch == 179:
          num = '180'
        if epoch == 199:
          num = '-200'

        checkpoint_dir = os.path.join("checkpoint_new")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save({
            'epoch': epoch,
            'time': time_one_epoch,
            'tot_time': time_tot,
            'loss': loss.item(),
            #'objectives': [torch.mean(obj_train).item(), torch.mean(obj_baseline).item(), mean_tour_obj_test],
            'checkpoint_performance_train': checkpoint_performance_train,
            'test_results': [mean_tour_length_test_s, mean_tour_revenue_test_s, mean_tour_obj_test_s],
            'model_train': model_train.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, '{}.pkl'.format(checkpoint_dir + "/checkpoint_" + problem + '_' + model + args.prize_type + time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + "{}".format(num)))

if __name__ == "__main__":

    device = torch.device("cpu"); gpu_id = -1 # select CPU

    gpu_id = '0' # select a single GPU  
    #gpu_id = '2,3' # select multiple GPUs  
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))   
        
    print(device)
        # args.max_len_PE = 1000

    ###################
    # Hyper-parameters
    ###################
            
    args = DotDict()
    args.nb_nodes = 20 # TSP20
    # args.nb_nodes = 50 # TSP50
    #args.nb_nodes = 100 # TSP100
    args.bsz = 64 # TSP20 TSP50

    args.dim_emb = 128
    args.dim_ff = 512
    
    # args.dim_input_nodes = [3,5,4]
    # args.step_context_dim = 2 * args.dim_emb + 1

    args.dim_input_nodes = [3,6,4]
    args.step_context_dim = 2 * args.dim_emb + 2

    args.nb_layers_encoder = 3
    args.nb_layers_decoder = 2
    args.nb_heads = 8
    args.nb_epochs = 200
    args.nb_batch_per_epoch = 2000
    args.nb_batch_eval = 20
    args.gpu_id = gpu_id
    args.lr = 1e-4
    args.weight_decay = 1e-6
    args.tol = 1e-3
    args.bl_alpha = 0.05
    args.penalty_factor = 1
    args.batchnorm = True  # if batchnorm=True  than batch norm is used
    args.seed = 1234
    args.c = None
    args.multi_obj = True
    args.prize_type = 'const'
    args.starting_nodes = int(args.nb_nodes / 2)
    #args.batchnorm = False # if batchnorm=False than layer norm is 
    
    filename = f"data/m1-pdstsp/m1-pdstsp_{args.prize_type}{args.nb_nodes}_capacitied_None_seed1234_1000000.pkl"
    # filename = f'data/m1-pdstsp/m1-pdstsp_{args.prize_type}{args.nb_nodes}_capacitied_None_seed1234.pkl'
    # filename = 'data/m1-pdstsp/m1-pdstsp20_uncapacitied_None_seed1234.pkl'

    dataset = load_dataset(filename)

    filename_2 = f"data/m1-pdstsp/m1-pdstsp_{args.prize_type}{args.nb_nodes}_capacitied_None_seed0_100000.pkl"

    dataset_2 = load_dataset(filename_2)

    train_pdp_pomo(args, device, dataset, dataset_2, problem='cpdp')