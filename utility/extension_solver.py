import cvxpy as cp
import numpy as np
from preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order, get_all_k_sets
from samplers import sample_subsets, sample_subset
from utility_fitter import Utility_Fitter
from connivence_solver import Connivence_Solver
import time
import random


class Extension_Solver:

    def __init__(self, preferences, model):
        self.preferences = preferences
        self.model = model

    def involved_subsets(self, relation_set):
        x_c = [x[0] for x in connivent]
        y_c = [x[1] for x in connivent]
        c = x_c + y_c
        return c


    def get_random_candidate(self, connivent):
        c = self.involved_subsets(connivent)
        found = False
        while not found:
            s = random.choice(c)
            if s in self.model:
                continue
            mat = self.preferences.get_matrix([s], connivent)
            print(mat.sum(axis=0))
            if mat.sum(axis=0) != 0:
                found = True
        return s

    def build_reduction_constraints(self):
        pass

    def reduce(self):
        pass

if __name__ == "__main__":
    print("Extension Solver Test: ")
    for i in range(10):
        items = np.arange(4)
        pref = sample_preferences_from_complete_order(items)
        model = get_all_k_sets(items, 1)
        while(True):
            print("model:", model)
            es = Extension_Solver(pref, model)
            CS = Connivence_Solver(pref, model)
            connivent = CS.check_connivences()
            if connivent is None:
                break
            c = es.get_random_candidate(connivent)
            model += [c]
        print("Model:" , model)
        break
