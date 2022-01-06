"""
@author: Ouaguenouni Mohamed
"""
import math
import itertools
import random
import numpy as np
from preferences import Preferences
from samplers import sample_subsets


def get_all_k_sets(items, k):
    """
    get all the subsets of size less or equal than k and that are included in the set of items.
    """
    subsets = []
    for i in range(1, k+1):
        k_subset = itertools.combinations(items, i)
        subsets = subsets + list(k_subset)
    return subsets

def sample_preferences_from_complete_order(items, indifference_rate=0.1):
    """
    Sample a complete order on a number of subsets and an associated preferences set.
    Params:
        -items: Set of alternatives among which the subset are sampled.
        -indifference_rate: Ratio of indifferent relations between the subsets
        of the order.
    """
    subsets = []
    for k in range(1, len(items)-1):
        subsets += list(itertools.combinations(items, k))
    random.shuffle(subsets)
    prefs = Preferences(items)
    for i, _ in enumerate(subsets):
        if random.random() <= indifference_rate and i<len(subsets)-1:
            prefs.add_indifference(subsets[i], subsets[i+1])
            continue
        for j in range(i+1, len(subsets)):
            prefs.add_preference(subsets[i], subsets[j])
    return prefs



def sample_preferences_from_order(items, n_relations, indifference_rate=0.1):
    """
    Sample an order on a number of subsets and an associated preferences set.
    Params:
        -items: Set of alternatives among which the subset are sampled.
        -n_relations: Number of preferences relations to sample.
        -indifference_rate: Ratio of indifferent relations between the subsets
        of the order.
    """
    n_subsets = int((1/2)*(math.sqrt(8*n_relations+1)+1))
    subsets = sample_subsets(items, n_subsets=n_subsets)
    random.shuffle(subsets)
    prefs = Preferences(items)
    for i, _ in enumerate(subsets):
        if random.random() <= indifference_rate and i<len(subsets)-1:
            prefs.add_indifference(subsets[i], subsets[i+1])
            continue
        for j in range(i+1, len(subsets)):
            prefs.add_preference(subsets[i], subsets[j])
    return prefs


if __name__ == "__main__":
    it = np.arange(10)
    L = get_all_k_sets(it, 2)
    print(L)

