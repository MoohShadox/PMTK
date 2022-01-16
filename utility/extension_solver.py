import time
import random
import cvxpy as cp
import numpy as np
from PMTK.random.preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order
from PMTK.utils import get_all_k_sets
from PMTK.random.subset_samplers import sample_subsets, sample_subset
from PMTK.utility.utility_fitter import Utility_Fitter
from PMTK.utility.connivence_solver import Connivence_Solver

class Extension_Solver:

    def __init__(self, preferences, model):
        self.preferences = preferences
        self.model = model

    def involved_subsets(self, relation_set):
        x_c = [x[0] for x in connivent]
        y_c = [x[1] for x in connivent]
        c = x_c + y_c
        return c

    def check_candidate(self, candidate, connivent):
        if candidate in self.model:
            return False
        mat = self.preferences.get_matrix([candidate], connivent)
        return mat.sum(axis=0) != 0

    def build_reduction_constraints(self):
        pass

    def get_random_candidate(self, connivent):
        c = self.involved_subsets(connivent)
        found = False
        while not found:
            s = random.choice(c)
            if self.check_candidate(s, connivent):
                break
        return s

    def reduce(self):
        pass

if __name__ == "__main__":
    print("Extension Solver Test: ")
    for i in range(10):
        items = np.arange(5)
        pref = sample_preferences_from_complete_order(items)
        model = get_all_k_sets(items, 1)
        while(True):
            print("model:", model)
            es = Extension_Solver(pref, model)
            CS = Connivence_Solver(pref, model)
            connivent = CS.check_connivences()
            print("connivent set:", connivent)
            if connivent is None:
                break
            c = es.get_random_candidate(connivent)
            model += [c]
        print("Model:" , model)
        break
