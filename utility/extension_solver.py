import time
import random
import cvxpy as cp
import numpy as np
from PMTK.random.preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order
from PMTK.utils import *
from PMTK.random.subset_samplers import sample_subsets, sample_subset
from PMTK.utility.utility_fitter import Utility_Fitter
from PMTK.utility.connivence_solver import Connivence_Solver
from PMTK.utils import get_all_candidates
from PMTK.utils import get_k_candidates

class Extension_Solver:

    def __init__(self, preferences, model):
        self.preferences = preferences
        self.model = model

    def check_candidate(self, candidate, connivent):
        if candidate in self.model:
            return False
        mat = self.preferences.get_matrix([candidate], connivent)
        return mat.sum(axis=0) != 0

    def build_reduction_constraints(self):
        pass

    def select_candidate(self, connivent):
        raise NotImplementedError("You must implement the method Select Candidate")

    def extend(self, verbose = True):
        while(True):
            CS = Connivence_Solver(self.preferences, self.model)
            connivent = CS.check_connivences()
            if connivent is None:
                break
            if verbose:
                print("model:", self.model)
                print("connivent set:", connivent)
            c = self.select_candidates(connivent)
            self.model += c

    def reduce(self):
        pass

class Random_Extension_Solver(Extension_Solver):

    def select_candidates(self, connivent):
        c = get_all_candidates(connivent)
        while True:
            s = random.choice(c)
            if self.check_candidate(s, connivent):
                break
        return [s]



class Smallest_Extension_Solver(Extension_Solver):
    def select_candidates(self, connivent):
        I = get_k_candidates(connivent, 1)
        for i in range(1, len(I) + 1):
            L = [S for S in get_k_candidates(connivent, i) if not S in self.model]
            if len(L):
                random.shuffle(L)
                return [L[0]]
        return None

class Cover_Extension_Solver(Extension_Solver):
    def select_candidates(self, connivent):
        I = set(get_exact_k_candidates(connivent, 1))
        c = []
        #print(f"I={I}")
        for k in range(1, len(I) + 1):
            L = [S for S in get_exact_k_candidates(connivent, k) if not S in self.model]
            #print(f"Testing for k={k}, L={L}")
            for i in L:
                appartenance = [all([k in i for k in j]) for j in c]
                if not any(appartenance):
                    #print("c=", c)
                    #print("Adding:", i)
                    c.append(i)
                    #input("Enter to continue")
                    #print("c=", c)
                #else:
                #    print("c=", c)
                #    print("Discarding:",i)
                #    input("Enter to continue")
        return c




