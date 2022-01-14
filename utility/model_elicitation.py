import cvxpy as cp
import numpy as np
from preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order, get_all_k_sets
from samplers import sample_subsets, sample_subset
from utility_fitter import Utility_Fitter
from connivence_solver import Connivence_Solver
import time
import random


EMPTY_SET = tuple([])

def all_possible(connivents, k):
    L = []
    for connivent in connivents:
        s_1 = get_all_k_sets(connivent[0], k)
        s_2 = get_all_k_sets(connivent[1], k)
        L = L + s_1 + s_2
    return L



if __name__ == "__main__":
    print("Extension Solver Test: ")
    k = EMPTY_SET
    L = [(EMPTY_SET, 2)]
    print((EMPTY_SET, 2) in L)
    for i in range(1):
        items = np.arange(5)
        pref = sample_preferences_from_complete_order(items, empty = True)
        model = get_all_k_sets(items,1)
        print("Preferences:")
        print(pref)
        while True:
            CS = Connivence_Solver(pref, model)
            connivent = CS.check_connivences()
            if not connivent:
                break
            print("Model:" , model)
            print("Connivent:", connivent)
            for k in range(1, len(items)):
                L = [i for i in all_possible(connivent, k) if i not in model]
                print(f"Possible candidates for k={k}:", L)
                for i in L:
                    print(f"Comparison of {i} and nul gives {pref.is_preferred(i, EMPTY_SET)}")
                model += L
                if len(L) > 0:
                    break
            print("Model now:", model)
            zzz = input("Type enter to continue")






