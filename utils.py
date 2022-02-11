import itertools

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

