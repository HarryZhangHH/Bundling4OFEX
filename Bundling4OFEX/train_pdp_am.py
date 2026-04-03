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

from nets.pdp_attention_model import PDP_AM_net
from nets.cpdp_attention_model import CPDP_AM_net

from utils.pdp_functions import *
from utils.data_utils import load_dataset
from utils.functions import *
from algs.greedy_search import greedy_cpdp, two_opt_improve
from algs.large_neighborhood_search import hill_climb, large_negihborhood_search


def train_pdp_am(args, device, dataset, dataset_2, problem='cpdp', model='am', checkpoint=None):
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
    
    print(problem)
    
    seed_everything(seed=args.seed)

    assert len(args.dim_input_nodes) == 3

    r_max = max([req['revenue']
                 for instance in dataset
                 for req in instance['requests']])
    
    bsz_test = 100
    batch_test = sample_batch(dataset_2, bsz_test, seed=args.seed)

    depots_test, requests_test = collate_pdp(batch_test, device=device, seed=args.seed, c=args.c)

    # tour_list, dist = greedy_cpdp(depots_test[:bsz_test], requests_test[:bsz_test], True)
    # tour_list, dist = two_opt_improve(tour_list, depots_test, requests_test)

    # seqs = [torch.tensor(t, dtype=torch.long) for t in tour_list]
    # tour_greedy = pad_sequence(seqs, batch_first=True, padding_value=1)

    # tour_list = hill_climb(tour_list, depots_test[:bsz_test], requests_test[:bsz_test])
    # seqs = [torch.tensor(t, dtype=torch.long) for t in tour_list]
    # tour_hc = pad_sequence(seqs, batch_first=True, padding_value=1)

    # tour_list = large_negihborhood_search(tour_list, depots_test[:bsz_test], requests_test[:bsz_test], running_time=1)
    # seqs = [torch.tensor(t, dtype=torch.long) for t in tour_list]
    # tour_lns = pad_sequence(seqs, batch_first=True, padding_value=1)

    if problem == 'pdp':

        positions_test, revenues_test, T_max_test = preprocess_data_pdp(depots_test, requests_test)

        model_train = PDP_AM_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.step_context_dim,
                    args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads,
                    batchnorm=args.batchnorm)

        model_baseline = PDP_AM_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.step_context_dim,
                    args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads,
                    batchnorm=args.batchnorm)
    
    else:
        positions_test, _, revenues_test, _, T_max_test = preprocess_data(depots_test, requests_test)

        model_train = CPDP_AM_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.step_context_dim,
                    args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads,
                    batchnorm=args.batchnorm)

        model_baseline = CPDP_AM_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.step_context_dim,
                    args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads,
                    batchnorm=args.batchnorm)

    depots_test_nor, pickups_test_nor, deliveries_test_nor = normalize_features(depots_test, requests_test, r_max)

    # uncomment these lines if trained with multiple GPUs
    print(torch.cuda.device_count())
    if torch.cuda.device_count()>1:
        model_train = nn.DataParallel(model_train)
        model_baseline = nn.DataParallel(model_baseline)
    # uncomment these lines if trained with multiple GPUs

    optimizer = torch.optim.Adam( model_train.parameters() , lr = args.lr ) 

    model_train = model_train.to(device)
    model_baseline = model_baseline.to(device)
    model_baseline.eval()

    print(args); print('')

    # Logs
    #os.system("mkdir logs")
    time_stamp=datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
    file_name = 'logs_new'+'/'+ problem + '_' + model + args.prize_type + time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + "-bn={}".format(args.batchnorm) + ".txt"
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

    # tour_greedy_length, tour_greedy_revenue, tour_greedy_obj = compute_loss(tour_greedy, positions_test[:bsz_test], revenues_test[:bsz_test], T_max_test[:bsz_test], bsz_test, args.penalty_factor, True)
    # mystring_min = f'model (greedy + opt) - avg revenue: {tour_greedy_revenue.mean().item()}; avg tour length {tour_greedy_length.mean().item()}; avg obj {tour_greedy_obj.mean().item()}'
    # print(mystring_min)
    # file.write(mystring_min+'\n')

    # tour_hc_length, tour_hc_revenue, tour_hc_obj = compute_loss(tour_hc, positions_test[:bsz_test], revenues_test[:bsz_test], T_max_test[:bsz_test], bsz_test, args.penalty_factor, True)
    # mystring_min = f'model (hill climbing) - avg revenue: {tour_hc_revenue.mean().item()}; avg tour length {tour_hc_length.mean().item()}; avg obj {tour_hc_obj.mean().item()}'
    # print(mystring_min)
    # file.write(mystring_min+'\n')

    # tour_lns_length, tour_lns_revenue, tour_lns_obj = compute_loss(tour_lns, positions_test[:bsz_test], revenues_test[:bsz_test], T_max_test[:bsz_test], bsz_test, args.penalty_factor, True)
    # mystring_min = f'model (large negiborhood search - 1s) - avg revenue: {tour_lns_revenue.mean().item()}; avg tour length {tour_lns_length.mean().item()}; avg obj {tour_lns_obj.mean().item()}'
    # print(mystring_min)
    file.write(mystring_min+'\n')

    # Uncomment these lines to re-start training with saved checkpoint
    if checkpoint is not None:
        checkpoint_file = 'checkpoint_new'+'/'+ checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
        epoch_ckpt = checkpoint['epoch'] + 1
        tot_time_ckpt = checkpoint['tot_time']
        checkpoint_performance_train = checkpoint['checkpoint_performance_train']
        checkpoint_performance_baseline = checkpoint['checkpoint_performance_baseline']
        model_baseline.load_state_dict(checkpoint['model_baseline'])
        model_train.load_state_dict(checkpoint['model_train'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Re-start training with saved checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))
        del checkpoint
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
        
        loss_list = []

        for step in tqdm(range(1,args.nb_batch_per_epoch+1)):
            if step % 500 == 0:
                print(step, time.time()-start)

            # generate a batch of random TSP instances    
            # x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device) # size(x)=(bsz, nb_nodes, dim_input_nodes) 

            # `dataset` is now whatever object you saved — e.g. a list of instances

            batch = sample_batch(dataset, args.bsz)

            # trick
            # c = random.randint(-1, len(batch[0]['T']))

            depots, requests = collate_pdp(batch, device=device, c=args.c)

            depots_nor, pickups_nor, deliveries_nor = normalize_features(depots, requests, r_max)

            if problem == 'pdp':
                # compute tours for model
                tour_train, sumLogProbOfActions = model_train(depots_nor, pickups_nor, deterministic=False) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)
                
                # compute tours for baseline
                with torch.no_grad():
                    tour_baseline, _ = model_baseline(depots_nor, pickups_nor, deterministic=True)

                if args.multi_obj == True:
                    positions, revenues, T_max = preprocess_data_pdp(depots, requests)
                else:
                    positions, revenues, T_max = preprocess_data_pdp(depots_nor, pickups_nor)

            elif problem == 'cpdp':                
                # compute tours for model
                tour_train, sumLogProbOfActions = model_train(depots_nor, pickups_nor, deliveries_nor, deterministic=False) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)
  
                # compute tours for baseline
                with torch.no_grad():
                    tour_baseline, _ = model_baseline(depots_nor, pickups_nor, deliveries_nor, deterministic=True)

                if args.multi_obj == True:
                    positions, _, revenues, _, T_max = preprocess_data(depots, requests)
                else:
                    positions, _, revenues, _, T_max = preprocess_data(depots_nor, pickups_nor)
            
            L_train, R_train, obj_train = compute_loss(tour_train, positions, revenues, T_max, args.bsz, args.penalty_factor, args.multi_obj)
            L_baseline, R_baseline, obj_baseline = compute_loss(tour_baseline, positions, revenues, T_max, args.bsz, args.penalty_factor, args.multi_obj)
            # print(L_baseline, R_baseline, obj_baseline)

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

            
        ###################
        # Evaluate train model and baseline on 10k random TSP instances
        ###################
        model_train.eval()
        model_baseline.eval()
        mean_tour_objective_train = 0
        mean_tour_objective_baseline = 0
        candidate_vals = []
        baseline_vals = []

        for step in range(0,args.nb_batch_eval):
            
            batch = sample_batch(dataset_2, args.bsz)

            # trick
            # c = random.randint(-1, len(batch[0]['T']))
            
            depots, requests = collate_pdp(batch, device=device, c=args.c)

            depots_nor, pickups_nor, deliveries_nor = normalize_features(depots, requests, r_max)

            if problem == 'pdp':

                # compute tour for model and baseline
                with torch.no_grad():
                    tour_train, _ = model_train(depots_nor, pickups_nor, deterministic=True)
                    tour_baseline, _ = model_baseline(depots_nor, pickups_nor, deterministic=True)

                if args.multi_obj == True:
                    positions, revenues, T_max = preprocess_data_pdp(depots, requests)
                else:
                    positions, revenues, T_max = preprocess_data_pdp(depots_nor, pickups_nor)
            
            elif problem == 'cpdp':
                  
                # compute tours for model and baseline
                with torch.no_grad():
                    tour_train, _ = model_train(depots_nor, pickups_nor, deliveries_nor, deterministic=True) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)
                    tour_baseline, _ = model_baseline(depots_nor, pickups_nor, deliveries_nor, deterministic=True)

                if args.multi_obj == True:
                    positions, _, revenues, _, T_max = preprocess_data(depots, requests)
                else:
                    positions, _, revenues, _, T_max = preprocess_data(depots_nor, pickups_nor)
            
            # get the lengths of the tours
            L_train, R_train, obj_train = compute_loss(tour_train, positions, revenues, T_max, args.bsz, args.penalty_factor, args.multi_obj)
            L_baseline, R_baseline, obj_baseline = compute_loss(tour_baseline, positions, revenues, T_max, args.bsz, args.penalty_factor, args.multi_obj)

            # L_tr and L_bl are tensors of shape (bsz,). Compute the mean tour length
            mean_tour_objective_train += obj_train.mean().item()
            mean_tour_objective_baseline += obj_baseline.mean().item()
            candidate_vals.append(obj_train.detach().cpu())
            baseline_vals.append( obj_baseline.detach().cpu())

        # flatten to 1-D numpy arrays
        candidate_vals = torch.cat(candidate_vals).numpy()
        baseline_vals  = torch.cat(baseline_vals).numpy()

        mean_tour_objective_train =  mean_tour_objective_train/ args.nb_batch_eval
        mean_tour_objective_baseline =  mean_tour_objective_baseline/ args.nb_batch_eval

        # evaluate train model and baseline and update if train model is better
        update_baseline = mean_tour_objective_train+args.tol < mean_tour_objective_baseline
        update_baseline_ = False
        if update_baseline:
            # model_baseline.load_state_dict( model_train.state_dict() )

            # one-sided paired t-test baseline update
            # paired t-test: H0 = cand_mean ≥ base_mean, H1 = cand_mean < base_mean
            t_stat, p_two_sided = ttest_rel(candidate_vals, baseline_vals)
            p_one_sided = p_two_sided / 2
            assert t_stat < 0, f"T-statistic should be negative, not {t_stat}"

            # only update if actor is significantly better
            update_baseline_ = bool((t_stat < 0) and (p_one_sided < args.bl_alpha))
            if update_baseline_:
                model_baseline.load_state_dict(model_train.state_dict())

        model_baseline.eval()
        # Compute PDPs for small test set
        # Note : this can be removed
        if problem == 'pdp':
            with torch.no_grad():
                tour_baseline, _ = model_baseline(depots_test_nor, pickups_test_nor, deterministic=True, precise=True)
        elif problem == 'cpdp':
            with torch.no_grad():
                tour_baseline, _ = model_baseline(depots_test_nor, pickups_test_nor, deliveries_test_nor, deterministic=True, precise=True)

        L_baseline_test, R_baseline_test, obj_baseline_test = compute_loss(tour_baseline, positions_test, revenues_test, T_max_test, bsz_test, args.penalty_factor, True)
        mean_tour_length_test, mean_tour_revenue_test, mean_tour_obj_test = L_baseline_test.mean().item(), R_baseline_test.mean().item(), obj_baseline_test.mean().item()
        # For checkpoint
        checkpoint_performance_train.append([ (epoch+1), mean_tour_objective_train])
        checkpoint_performance_baseline.append([ (epoch+1), mean_tour_objective_baseline])
            
        # # Compute optimality gap
        # if args.nb_nodes==50: gap_train = mean_tour_length_train/5.692- 1.0
        # elif args.nb_nodes==100: gap_train = mean_tour_length_train/7.765- 1.0
        # else: gap_train = -1.0
        gap_train = -1.0
        # Print and save in txt file
        mystring_min = 'Epoch: {:d}, epoch time: {:.3f}min, tot time: {:.3f}day, obj_train: {:.3f}, obj_base: {:.3f}, L_test: {:.3f}, R_test: {:.3f}, obj_test: {:.3f}, avg loss: {:.3f}, update: {}'.format(
            epoch, time_one_epoch/60, time_tot/86400, mean_tour_objective_train, mean_tour_objective_baseline, mean_tour_length_test, mean_tour_revenue_test, mean_tour_obj_test, torch.stack(loss_list, dim=0).mean(dim=0), [update_baseline,update_baseline_]) 
        print(mystring_min) # Comment if plot display

        file.write(mystring_min+'\n')
        
        # Saving checkpoint
        num = ''
        if epoch == 99:
          num = '-100'
        #if epoch == 124:
        #  num = '-125'
        if epoch == 149:
          num = '-150'
        if epoch == 199:
          num = '-200'
    
        # Saving checkpoint
        checkpoint_dir = os.path.join("checkpoint_new")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save({
            'epoch': epoch,
            'time': time_one_epoch,
            'tot_time': time_tot,
            'loss': loss.item(),
            'objectives': [torch.mean(obj_train).item(), torch.mean(obj_baseline).item(), mean_tour_obj_test],
            'checkpoint_performance_train': checkpoint_performance_train,
            'checkpoint_performance_baseline': checkpoint_performance_baseline,
            'test_results': [mean_tour_length_test, mean_tour_revenue_test, mean_tour_obj_test],
            'model_baseline': model_baseline.state_dict(),
            'model_train': model_train.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, '{}.pkl'.format(checkpoint_dir + "/checkpoint_" + problem + '_' + model + time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + "-bn={}".format(args.batchnorm) + "{}".format(num)))
    
        # with torch.no_grad():
        #     tour_basline_plot_20, sumLogProbOfActions = model_baseline(x_test_20, deterministic=True)
        # with torch.no_grad():
        #     tour_basline_plot_50, sumLogProbOfActions = model_baseline(x_test_50, deterministic=True)
        # tour_basline_20_list.append(tour_basline_plot_20)
        # print(f'Node 20: {compute_tour_length(x_test_20, tour_basline_plot_20)}')
        # plot_tsp(x_test_20, tour_basline_plot_20, plot_concorde=False)
        # plt.show()
        # tour_basline_50_list.append(tour_basline_plot_50)
        # print(f'Node 50: {compute_tour_length(x_test_50, tour_basline_plot_50)}')
        # plot_tsp(x_test_50, tour_basline_plot_50, plot_concorde=False)
        # plt.show()

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
    args.nb_nodes = 10 # TSP20
    # args.nb_nodes = 50 # TSP50
    #args.nb_nodes = 100 # TSP100
    args.bsz = 512 # TSP20 TSP50

    args.dim_emb = 128
    args.dim_ff = 512
    
    # args.dim_input_nodes = [3,5,4]
    # args.step_context_dim = 2 * args.dim_emb + 1

    args.dim_input_nodes = [3,6,4]
    args.step_context_dim = 2 * args.dim_emb + 2

    args.nb_layers_encoder = 3
    args.nb_layers_decoder = 2
    args.nb_heads = 8
    args.nb_epochs = 125
    args.nb_batch_per_epoch = 2500
    args.nb_batch_eval = 20
    args.gpu_id = gpu_id
    args.lr = 1e-4
    args.tol = 1e-3
    args.bl_alpha = 0.05
    args.penalty_factor = 1
    args.batchnorm = True  # if batchnorm=True  than batch norm is used
    args.seed = 1234
    args.c = None
    args.multi_obj = True
    args.prize_type = 'distabs'
    #args.batchnorm = False # if batchnorm=False than layer norm is 
    
    filename = f'data/m1-pdstsp/m1-pdstsp_{args.prize_type}{args.nb_nodes}_capacitied_None_seed1234_5000000.pkl'
    # filename = 'data/m1-pdstsp/m1-pdstsp20_uncapacitied_None_seed1234.pkl'

    dataset = load_dataset(filename)

    filename_2 = f'data/m1-pdstsp/m1-pdstsp_{args.prize_type}{args.nb_nodes}_capacitied_None_seed0_100000.pkl'

    dataset_2 = load_dataset(filename_2)
    
    checkpoint = None

    train_pdp_am(args, device, dataset, dataset_2, problem='cpdp', checkpoint=checkpoint)