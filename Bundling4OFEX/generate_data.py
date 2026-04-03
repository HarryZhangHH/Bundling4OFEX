import os
import math
import numpy as np
from utils.data_utils import check_extension, save_dataset
import scipy.stats as stats
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_pdp_data(dataset_size, pdp_size, depot_size =2,
                      is_gaussian=True, coord_sigma=1.0,
                      load_min=1, load_max=5,
                      rev_sigma=0.5, prize_type=None):
    """
    Returns a list of `dataset_size` instances.  Each instance is a dict:
      {
        'depots':      [x,y],
        'requests': [
           {'pickup': [x,y],
            'delivery': [x,y],
            'load': int,
            'revenue': float},
           ...
        ]
      }
    """
    
    # industry fulltruck not linear with size only with distance
    # capacity of 1, fixed price per kilometer

    def sample_coords(n, is_gaussian, sigma):
        """Sample n 2D points in [0,1]^2 via truncated Normal or Uniform."""
        if is_gaussian:
            mu, low, high = 0.5, 0.0, 1.0
            X = stats.truncnorm((low-mu)/sigma, (high-mu)/sigma,
                                loc=mu, scale=sigma)
            pts = np.stack([X.rvs(n), X.rvs(n)], axis=1)
        else:
            pts = np.random.uniform(size=(n, 2))
        return pts

    dataset = []
    for _ in range(dataset_size):
        # 1) sample depot
        depot = sample_coords(depot_size, is_gaussian, coord_sigma)

        # 2) sample pickups & deliveries
        pickups   = sample_coords(pdp_size, is_gaussian, coord_sigma)
        deliveries= sample_coords(pdp_size, is_gaussian, coord_sigma)

        # 3) loadacities
        loadacities = np.random.randint(load_min, load_max + 1, size=pdp_size)

        # 4) compute Euclidean distances
        dists = np.linalg.norm(pickups - deliveries, axis=1)

        # 5) revenues ~ N(mean = dist * load,  std = rev_sigma)
        # Methods taken from Fischetti et al. 1998
        if prize_type == 'const':
            revenues = np.ones(pdp_size)
        elif prize_type == 'unif':
            revenues = (np.random.randint(10, 101, size=(pdp_size))) / 100.
        elif prize_type == 'distabs':
            revenues = dists
        elif prize_type == 'dist':
            means = dists
            revenues = np.random.normal(loc=means, scale=rev_sigma, size=pdp_size)
            revenues = np.clip(revenues, 0.5, 1.42 * 1)
        else:
            assert prize_type == 'distcap'
            means = dists * loadacities
            
            discount = (np.random.randint(50, 151, size=(pdp_size))) / 100.
            revenues = means*discount


        # 6) sample constraints
        if depot_size == 2:
            x1, y1 = depot[0]
            x2, y2 = depot[1]
            dist   = np.hypot(x1 - x2, y1 - y2)
            start  = max(math.ceil(3 * dist * (pdp_size/20)), 2)         # ceil(1.34)=2, and we require ≥2
            end    = max(math.ceil(10 * dist * (pdp_size/20)), 2)            # ceil(5.44)=6
            T_ints = list(range(start, end+1))    # range is inclusive of start, exclusive of end+1

        if load_min == load_max:
            Q_ints = [1]
        else:
            start  = max(math.ceil(2*load_max-2),2)         # ceil(1.34)=2, and we require ≥2
            end    = math.floor(4*load_max)               # floor(5.44)=5
            Q_ints = list(range(start, end+1))    # range is inclusive of start, exclusive of end+1

        # 7) pack requests
        reqs = []
        for j in range(pdp_size):
            reqs.append({
                'pickup':    pickups[j].tolist(),
                'delivery':  deliveries[j].tolist(),
                'load':  int(loadacities[j]),
                'revenue':   float(revenues[j])
            })

        dataset.append({
            'depots': depot.tolist(),
            'requests': reqs,
            'Q': Q_ints,
            'T': T_ints
        })

    return dataset

def plot_pdp_instance(instance, coord_range=1.0):
    """
    instance: dict with keys
        'depots': [x,y]
        'requests': list of {
            'pickup': [x,y],
            'delivery': [x,y],
            'load': int,
            'revenue': float
        }
    coord_range: maximum coordinate (assumes min=0)
    """
    depot = instance['depots']
    reqs  = instance['requests']
    n     = len(reqs)
    
    # pick n distinct colors
    cmap = mpl.colormaps['tab10']
    colors = [cmap(i % 10) for i in range(n)]
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    # Plot depot
    ax.scatter(*depot[0], marker='*', c='black', s=200, label='Origin')
    ax.scatter(*depot[1], marker='^', c='black', s=200, label='Destination')
    
    for i, req in enumerate(reqs):
        p    = req['pickup']
        d    = req['delivery']
        load = req['load']
        rev  = req['revenue']
        col  = colors[i]
        
        # Plot pickup & delivery
        ax.scatter(*p, marker='s', c=[col], s=100, edgecolors='black')
        ax.scatter(*d, marker='o', c=[col], s=100, edgecolors='black')
        # Arrow from pickup to delivery
        ax.annotate(
            "",
            xy=d, xytext=p,
            arrowprops=dict(arrowstyle='->', color=col, lw=1.5)
        )
        # Labels: load below pickup, revenue above delivery
        ax.text(p[0], p[1] - 0.02*coord_range,
                f"C={load}", ha='center', va='top', fontsize= 9, c=col)
        ax.text(d[0], d[1] + 0.02*coord_range,
                f"R={rev:.1f}", ha='center', va='bottom', fontsize= 9, c=col)
    
    # Set axis limits & aspect
    ax.set_xlim(0, coord_range)
    ax.set_ylim(0, coord_range)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='m1-pdstsp')
    parser.add_argument("--is_gaussian", type=int, default=0)
    parser.add_argument('--data_distribution', type=str, default=None,
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=100000, help="Size of the dataset")
    parser.add_argument('--request_size', type=int, nargs='+', default=[10, 20, 50, 100],
                        help="Sizes of problem instances (default 22, 42, 102, 202)")
    parser.add_argument("--depot_size", type=int, default=2, help="Size of the depots")
    parser.add_argument('--coord_sigma', type=float, nargs='+', default={0.6, 0.8, 1.0})
    parser.add_argument('--load_min',    type=int, default=2)
    parser.add_argument('--load_max',    type=int, default=5)
    parser.add_argument('--rev_sigma',  type=float, default=1.0)
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problem) == 1 and len(opts.request_size) == 1), \
        "Can only specify filename when generating a single dataset"

    # if opts.filename is None:
    #     opts.filname = f'{opts.problem}_{opts.dataset_size}_{opts.request_size}_sigma{opts.coord_sigma}_seed{opts.seed}'

    distributions_per_problem = {
        'm1-pdstsp': ['const', 'unif', 'dist', 'distcap','distabs']
    }
    problems = {
        opts.problem: [opts.data_distribution]
    }

    if opts.load_min == opts.load_max:
        capacitied = 'uncapacitied'
    else:
        capacitied = 'capacitied'

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for request_size in opts.request_size:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    if opts.is_gaussian:
                        filename = os.path.join(datadir, "{}{}{}_{}_{}_seed{}_{}_{}_{}.pkl".format(
                            problem,
                            "_{}".format(distribution) if distribution is not None else "",
                            request_size, capacitied, opts.name, opts.seed, 'gaussian', opts.coord_sigma[0], opts.dataset_size))

                    else:
                        filename = os.path.join(datadir, "{}{}{}_{}_{}_seed{}_{}.pkl".format(
                            problem,
                            "_{}".format(distribution) if distribution is not None else "",
                            request_size, capacitied, opts.name, opts.seed, opts.dataset_size))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'm1-pdstsp':
                    dataset = generate_pdp_data(opts.dataset_size,
                                    pdp_size=request_size,
                                    depot_size=opts.depot_size,
                                    is_gaussian=bool(opts.is_gaussian),
                                    coord_sigma=opts.coord_sigma[0] if isinstance(opts.coord_sigma, (list,tuple)) else opts.coord_sigma,
                                    load_min=opts.load_min,
                                    load_max=opts.load_max,
                                    rev_sigma=opts.rev_sigma,
                                    prize_type=opts.data_distribution)
                else:
                    assert False, "Unknown problem: {}".format(problem)
                print(dataset[0])
                
                revenues = [req['revenue']
                            for instance in dataset
                            for req in instance['requests']]

                # 2) Convert to a NumPy array
                revenues = np.array(revenues)

                # 3) Compute and print min, max, mean
                print(f"Min revenue:  {revenues.min():.4f}")
                print(f"Max revenue:  {revenues.max():.4f}")
                print(f"Mean revenue: {revenues.mean():.4f}")
                save_dataset(dataset, filename)

    plot_pdp_instance(dataset[0])
