import itertools
import numpy as np


def pareto_dominate(x,y, epsilon = 1e-2):
    if ((x - y) >= 0).all():
        return 1
    if ((y - x) >= 0).all():
        return -1
    return 0


def chain_iterators(*it):
    for i in it:
        for j in i:
            yield j

def additivity(theta):
    return max([len(i) for i in theta])

def better(theta1, theta2):
    if additivity(theta1) < additivity(theta2):
        return 2
    if additivity(theta2) < additivity(theta1):
        return -2
    if len(theta1) > len(theta2):
        return -1
    if len(theta2) > len(theta1):
        return 1
    return 0

def dominated(theta, theta_list):
    for t in theta_list:
        if better(t, theta) > 0:
            return True
    return False

def keep_non_dominated(theta_list):
    non_dominated = []
    for t in theta_list:
        if not dominated(t, theta_list):
            non_dominated.append(t)
    return non_dominated

def vectorize_subset( x, model):
    vector = np.zeros(len(model))
    for subset in model:
        if all(s in x for s in subset):
            vector[model.index(subset)] += 1
    return vector

def vectorize_preference(x, y, model):
    vector = vectorize_subset(x, model) - vectorize_subset(y, model)
    return vector

def is_pareto_efficient_simple(costs):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def get_all_k_sets(items, k):
    """
    get all the subsets of size less or equal than k and that are included in the set of items.
    """
    subsets = []
    for i in range(1, k+1):
        k_subset = itertools.combinations(items, i)
        subsets = subsets + list(k_subset)
    return subsets

def get_exact_k_sets(items, k):
    """
    get all the subsets of size less or equal than k and that are included in the set of items.
    """
    k_subset = list(itertools.combinations(items, k))
    return k_subset

#TODO: Use subset instead of candidate
def get_k_candidates(connivents, k):
    L = []
    for connivent in connivents:
        s_1 = get_all_k_sets(connivent[0], k)
        s_2 = get_all_k_sets(connivent[1], k)
        L = L + s_1 + s_2
    return L

def get_exact_k_candidates(connivents, k):
    L = []
    for connivent in connivents:
        s_1 = get_exact_k_sets(connivent[0], k)
        s_2 = get_exact_k_sets(connivent[1], k)
        L = L + s_1 + s_2
    return L


def get_all_candidates(connivents):
    I = get_k_candidates(connivents, 1)
    return get_k_candidates(connivents, len(I))


def union(set_of_subsets):
    L = []
    for s in set_of_subsets:
        for i in s:
            if not i in L:
                L.append(i)
    return L

