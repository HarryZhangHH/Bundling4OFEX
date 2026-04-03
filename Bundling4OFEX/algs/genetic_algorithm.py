import math
import sys
import random
import time
import torch
import numpy as np

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# returns the subset of req that is on/in the ellipse defined by foci f1, f2 and T_max
def ell_sub(T_max, f1, f2, req):
    valid_req = []
    for r in req:
        if (distance(r[0], r[1], f1[0], f1[1]) + distance(r[0], r[1], r[2], r[3]) + distance(r[2], r[3], f2[0], f2[1]) <= T_max):
            valid_req.append(r)
    return valid_req

# returns a path through req with high value
def ellinit_replacement(requests, start, end, T_max):
    requests = list(requests)
    path = [start, end]
    length = distance(start[0], start[1], end[0], end[1])

    found = True
    while (found == True and len(requests) > 0):
        min_added_length = -1
        max_added_reward = 0
        for j in range(len(requests)):
            for k in range(len(path) - 1):
                added_length = (distance(path[k][2], path[k][3], requests[j][0], requests[j][1]) + 
                                distance(requests[j][0], requests[j][1], requests[j][2], requests[j][3]) + 
                                distance(requests[j][2], requests[j][3], path[k+1][0], path[k+1][1]) -
                                distance(path[k][2], path[k][3], path[k+1][0], path[k+1][1]))
                if (length + added_length < T_max and requests[j][4] > max_added_reward):
                    min_added_length = added_length
                    max_added_reward = requests[j][4]
                    min_point = j
                    path_point = k + 1
        if min_added_length > 0 :
            # add to path
            path.insert(path_point, requests.pop(min_point))
            length = length + min_added_length
        else:
            found = False
    return path

# returns a list of L paths with the best path in the first position by weight rather than length
def init_replacement(requests, start, end, T_max):
    """
    requests : 7-tuple (x1, y1, x2, y2, reward, index, weight)
    """
    requests = list(requests)
    L = len(requests) if len(requests) <=10 else 10
    if L == 0:
        return [[start, end]]
    
    # decorate and sort of weight
    d_sub = sorted([(x[-1], x) for x in requests])[::-1]
    ls = d_sub[:L]
    rest = d_sub[L:]
    paths = []
    for i in range(L):
        path = [start, ls[i][1], end]
        length = (distance(path[0][2], path[0][3], path[1][0], path[1][1]) + 
                  distance(path[1][0], path[1][1], path[1][2], path[1][3]) + 
                  distance(path[1][2], path[1][3], path[2][0], path[2][1]) )
        assert length < T_max
        a_rest = ls[:i] + ls[i+1:] + rest
        a_rest = [x[1] for x in a_rest] #undecorate
        assert len(a_rest) + len(path) == len(requests) + 2
        found = True
        while (found == True and len(a_rest) > 0):
            min_added_length = -1
            max_weight = 0
            for j in range(len(a_rest)):
                for k in range(len(path)-1):
                    added_length = (distance(path[k][2], path[k][3], a_rest[j][0], a_rest[j][1]) + 
                                    distance(a_rest[j][0], a_rest[j][1], a_rest[j][2], a_rest[j][3]) + 
                                    distance(a_rest[j][2], a_rest[j][3], path[k+1][0], path[k+1][1]) -
                                    distance(path[k][2], path[k][3], path[k+1][0], path[k+1][1]))
                if (length + added_length <= T_max and a_rest[j][-1] > max_weight):
                    min_added_length = added_length
                    max_weight = a_rest[j][-1]
                    min_point = j
                    path_point = k + 1
            if min_added_length > 0 :
                # add to path
                path.insert(path_point, a_rest.pop(min_point))
                length = length + min_added_length
            else:
                found = False
        if length < T_max:
            paths.append(path)

    assert len(paths) > 0
    # soting by descending reward and return that list of paths
    return [x[1] for x in sorted([(sum([y[4] for y in z]), z) for z in paths])[::-1]]

# returns a list of L paths with the best path in the first position
def initialize(requests, start, end, T_max):
    requests = list(requests)
    L = len(requests) if len(requests) <=10 else 10
    if L == 0:
        return [[start, end]]
    
    # decorate and sort of distance
    d_sub = sorted([((distance(path[0][2], path[0][3], path[1][0], path[1][1]) + 
                      distance(path[1][0], path[1][1], path[1][2], path[1][3]) + 
                      distance(path[1][2], path[1][3], path[2][0], path[2][1])), x) for x in requests])[::-1]
    ls = d_sub[:L]
    rest = d_sub[L:]
    paths = []
    for i in range(L):
        path = [start, ls[i][1], end]
        length = ls[i][0]
        a_rest = ls[:i] + ls[i+1:] + rest
        a_rest = [x[1] for x in a_rest] #undecorate
        assert len(a_rest) + len(path) == len(requests) + 2
        found = True
        while (found == True and len(a_rest) > 0):
            min_added = -1
            for j in range(len(a_rest)):
                for k in range(len(path)-1):
                    added_length = (distance(path[k][2], path[k][3], a_rest[j][0], a_rest[j][1]) + 
                                    distance(a_rest[j][0], a_rest[j][1], a_rest[j][2], a_rest[j][3]) + 
                                    distance(a_rest[j][2], a_rest[j][3], path[k+1][0], path[k+1][1]) -
                                    distance(path[k][2], path[k][3], path[k+1][0], path[k+1][1]))
                if (length + added_length < T_max and (added_length < min_added or min_added < 0)):
                    min_added = added_length
                    min_point = j
                    path_point = k + 1
            if min_added > 0 :
                # add to path
                path.insert(path_point, a_rest.pop(min_point))
                length = length + min_added
            else:
                found = False
        paths.append(path)
    assert (len([x[1] for x in sorted([(sum([y[4] for y in z]), z) for z in paths])[::-1]]) > 0)
    return [x[1] for x in sorted([(sum([y[4] for y in z]), z) for z in paths])[::-1]]  # 4 -> revenue

# fitness will take a set requests and a set of weights and return a tuple containing the fitness and the best path
def fitness(chrom, requests, start, end, T_max):
    requests = list(requests)
    augs = [ 
        (x1, y1, x2, y2, r, idx, weight+bonus)
        for (x1, y1, x2, y2, r, idx, weight), bonus in zip(requests, chrom)
    ]
    # best = ellinit_replacement(augs, start, end, T_max)

    # Build the “elliptical subset” of points that are even worth considering
    ell_set = ell_sub(T_max, start, end, augs)
    # best = initialize(ell_set, start, end, T_max)[0]

    # Run an “initialize‐and‐replace” procedure to pick a path inside that subset
    best = init_replacement(ell_set, start, end, T_max)[0]

    return (sum([ requests[int(x[5])-2][4] for x in best[1:len(best)-1] ]), best)

def crossover(c1, c2):
    assert len(c1) == len(c2)
    point = random.randrange(len(c1))
    first = random.randrange(2)
    if first:
        return c1[:point] + c2[point:]
    else:
        return c2[:point] + c1[point:]

def mutate(chrom, m_chance, m_sigma):
    return [x+random.gauss(0,m_sigma) if random.randrange(m_chance)==0 else x for x in chrom]

def genetic_search(requests, depots, T_max, pop_size=10, gen_limit=10, kt=5, i_sigma=10, m_sigma=7, m_chance=2, elitismn=2):
    B, R, D = requests.size(0), requests.size(1), depots.size(0)
    random.seed()

    routes_sol = []
    chrom_list = []
    time_list  = []

    for b in range(B):

        idx_col  = torch.arange(D+R, dtype=requests.dtype, device=requests.device).unsqueeze(1)
        zeros    = torch.zeros(R, 1, dtype=requests.dtype, device=requests.device)
        zero     = torch.tensor([0], dtype=requests.dtype, device=requests.device)
        c_points = torch.cat([requests, idx_col[D:], zeros], dim=1).detach().cpu().numpy()
        start    = torch.cat([depots[0, :2], depots[0, :], idx_col[0], zero], dim=0).detach().cpu().numpy()
        end      = torch.cat([depots[D-1, :2], depots[D-1, :], idx_col[D-1], zero], dim=0).detach().cpu().numpy()
        assert distance(start[0], start[1] ,end[0], end[1]) < T_max

        
        
        # kt        = 5  # tournament size
        # i_sigma   = 10 # initial chromosome stddev
        # m_sigma   = 7  # mutation stddev
        # m_chance  = 2  # 1/m_chance mutation prob per gene
        # elitismn  = 2  # how many top individuals to carry over unchanged

        start_time = time.time()

        # generate initial random population
        pop = []
        for i in range(pop_size + elitismn):
            # generate a random chromosome of length len(c_points)
            chrom = [ random.gauss(0, i_sigma) for _ in range(len(c_points)) ]
            chrom = (fitness(chrom, c_points, start, end, T_max)[0], chrom)

            # the last elitismn slots remain the best ones so far
            j_elite = 0
            while i - j_elite > 0 and j_elite < elitismn and chrom > pop[i-1-j_elite]:
                j_elite += 1
            pop.insert(i-j_elite, chrom)

        best_fit = 0
        for i in range(gen_limit):
            next_gen = []
            for j in range(pop_size):
                # select parents in k tournaments, sample kt members, sort them, take the top 2 → parents
                parents = sorted(random.sample(pop, kt))[kt-2:]

                # crossover and mutate
                offspring = mutate(crossover(parents[0][1], parents[1][1]), m_chance, m_sigma)
                offspring = (fitness(offspring, c_points, start, end, T_max)[0], offspring)
                if offspring[0] > best_fit:
                    best_fit = offspring[0]
                
                # compare to the worst of the current “elite” slice, do elitist insertion if better
                if elitismn > 0 and offspring > pop[pop_size]:
                    l = 0
                    while l < elitismn and offspring > pop[pop_size + l]:
                        l += 1
                    # insert it into the elite block at position (pop_size + l),
                    # then kick out the worst of the elite block (pop_size + 0)
                    pop.insert(pop_size + l, offspring)
                    next_gen.append(pop.pop(pop_size))
                else:
                    # no elitism needed: just keep the child for the next generation
                    next_gen.append(offspring)

            # f) Now `pop` = sorted list of length popsize+elitismn. The first popsize slots
            #    are replaced by nextgen (the newly created children). The last `elitismn`
            #    slots remain the best “elite” individuals from the old population.
            pop = next_gen + pop[pop_size:]

        best_chrom = sorted(pop)[pop_size + elitismn - 1]
        end_time = time.time()

        best_path = fitness( best_chrom[1], c_points, start, end, T_max )[1]
        best_path = np.vstack([np.array(x, dtype=np.float32) for x in best_path])
        best_path = best_path[:,5].astype(int).tolist()

        chrom_list.append(best_chrom[0])
        routes_sol.append(best_path)
        time_list.append(end_time-start_time)

        # print 'their stuff:'
        # stuff = oph.initialize( oph.ell_sub( tmax, start_point, end_point, cpoints )
        # , start_point, end_point, tmax )[0]
        # print 'fitness:', sum( [ x[2] for x in stuff ] )
        # print 'my stuff:'
        # stuff2 = oph.ellinit_replacement( cpoints, start_point, end_point, tmax )
        # print 'fitness:', sum( [ x[2] for x in stuff2 ] )
        # print 'checking correctness...',
        # total_distance = ( oph.distance( start_point, cpoints[ best_path[ 1                    ][3] - 2 ] ) + 
        #                    oph.distance( end_point,   cpoints[ best_path[ len( best_path ) - 2 ][3] - 2 ] ) )
        # for i in xrange( 1, len( best_path ) - 3 ):
        #     total_distance += oph.distance( cpoints[ best_path[ i     ][3] - 2 ], 
        #                                     cpoints[ best_path[ i + 1 ][3] - 2 ] )
        # print 'OK' if total_distance <= tmax else 'not OK'
        # print 'tmax:          ', tmax
        # print 'total distance:', total_distance

        # print(best_chrom)
    return chrom_list, routes_sol, time_list
