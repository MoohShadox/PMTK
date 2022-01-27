"""
This class provides an effective tool to represent the polyhedron of utilities.
that are n-additive and compatible with a set of preferences.
It could be used to find a specific utility function that is compatible 
with the preferences.
or to test the validity of an hypothesis on the full polyhedron.
"""
import cvxpy as cp
import numpy as np
import time
from PMTK.random.preferences_sampler import sample_preferences_from_order
from PMTK.utils import get_all_k_sets
from PMTK.random.subset_samplers import sample_subsets
from PMTK.utility.additive_utility import AdditiveUtility
from PMTK.preferences import Preferences



class Utility_Fitter:

    def __init__(self, items, model, preferences = None):
        self.items = items
        self.model = None
        self.preferences = None
        self.vars_ub = 1
        self.vars_lb = -1
        self.epsilon = 1e-6
        self.__gap_vars = None
        self.__cst = None
        self.__vars = None
        self.__prob = None
        self.__last_objectif = None
        self.__last_objectif_value = None
        self.__last_constraints_set = None

    def set_preferences(self, preferences):
        self.preferences=preferences
        return self

    def build_vars(self):
        variables = cp.Variable(len(self.model))
        self.__vars = variables
        return self

    def set_model(self, model):
        self.model = model
        return self

    def build_preferences_cst(self, bound_subsets_utilities = True, bound_model = False):
        if(self.__vars is None):
            print("Error: Did'nt build the variables")
        cst_l = []
        self.__gap_vars = cp.Variable(len(self.preferences.preferred))
        for i, (x, y) in enumerate(self.preferences.preferred):
            cst_l += [self.preferences.vectorize_preference(x, y, self.model) @ self.__vars >= self.__gap_vars[i]]
        for i, (x, y) in enumerate(self.preferences.preferred_or_indifferent):
            cst_l += [self.preferences.vectorize_preference(x, y, self.model) @ self.__vars >= 0]
        for i, (x, y) in enumerate(self.preferences.indifferent):
            cst_l += [self.preferences.vectorize_preference(x, y, self.model) @ self.__vars == 0]
        if len(cst_l) == 0:
            print("Warning: no preferences found!")
        cst_l += [self.__gap_vars >= self.epsilon]

        if bound_model:
            for m in self.model:
                v = self.preferences.vectorize_subset(m, self.model)
                cst_l.append(v @ self.__vars <= 1)
                cst_l.append(v @ self.__vars >= -1)

        if bound_subsets_utilities:
            for m,n in self.preferences:
                v = self.preferences.vectorize_subset(m, self.model)
                cst_l.append(v @ self.__vars <= 1)
                cst_l.append(v @ self.__vars >= -1)
                v = self.preferences.vectorize_subset(n, self.model)
                cst_l.append(v @ self.__vars <= 1)
                cst_l.append(v @ self.__vars >= -1)

        self.__cst = cst_l
        return self

    def __solve(self, cst_l, obj):
        self.prob = cp.Problem(obj, cst_l)
        self.prob.solve(solver = cp.GLPK)
        # We save the lines and columns of the last solved PL.
        self.__last_constraints_set = cst_l
        self.__last_objectif = obj
        self.__last_objectif_value = self.prob.value
        return self.prob

    def get_min_params_utility(self):
        cst_l = self.__cst
        if cst_l is None or len(cst_l) == 0:
            print("Warning: no preferences constraints! ")
            return None
        abs_val = cp.Variable(len(self.model))
        cst_abs_val_l = -self.__vars <= abs_val
        cst_abs_val_h = self.__vars <= abs_val
        cst_l += [cst_abs_val_h, cst_abs_val_l]
        obj = cp.Minimize(sum(abs_val))
        self.__solve(cst_l, obj)
        return self

    def regrets_matrix(self, subsets, verbose = False):
        MPRS = np.zeros((len(subsets), len(subsets)))
        for A in subsets:
            for B in subsets:
                MPRS[subsets.index(A),subsets.index(B)] = self.compute_MPR(A,B, verbose)
                MPRS[subsets.index(B) ,subsets.index(A)] = self.compute_MPR(B,A, verbose)
        return MPRS

    def get_robust_preferences(self, subsets, verbose = False):
        r_mat = self.regrets_matrix(subsets, verbose)
        pref_matrix = np.where(r_mat < 0)
        pref = Preferences(self.items)
        for x,y in zip(pref_matrix[0], pref_matrix[1]):
            pref.add_preference(subsets[x],subsets[y])

        indif_matrix = np.where(np.abs(r_mat) <= self.epsilon)
        indif_matrix = [(x,y) for x,y in zip(indif_matrix[0], indif_matrix[1])]
        for x,y in indif_matrix:
            if x==y:
                continue
            if (y,x) in indif_matrix:
                pref.add_indifference(subsets[x],subsets[y])
        return pref

    def solve_for_linear(self, vector):
        obj = cp.Maximize(self.__vars @ vector)
        self.__solve(self.cst, obj)
        return self

    def problem(self):
        return self.__prob

    def get_utility(self):
        if self.prob.status == cp.INFEASIBLE:
            #print("Infeasible! ")
            return None
        uf = AdditiveUtility(self.items)
        theta_values = {key: val for key, val in zip(self.model, self.__vars.value)}
        kv = [(x,theta_values[x]) for x in theta_values]
        uf.set_theta_values(*kv)
        return uf

    def get_most_discriminant_utility(self):
        if self.__cst is None or len(self.__cst) == 0:
            print("Warning: no preferences constraints! ")
            return None
        obj = cp.Maximize(sum(self.__gap_vars))
        self.__solve(self.__cst, obj)
        return self

    def get_min_additivity_utility(self):
        pass

    def compute_MPR(self, x, y, verbose = False):
        if x == y:
            return 0
        exp = self.preferences.vectorize_subset(y, self.model) @ self.__vars - self.preferences.vectorize_subset(x, self.model) @ self.__vars
        obj = cp.Maximize(exp)
        self.__solve(self.__cst, obj)
        if verbose:
            print(f"MPR({x}, {y}) = {self.__last_objectif_value}") 
        if self.prob.status == cp.INFEASIBLE:
            raise Exception("The model could not represent the preferences")
        return self.__last_objectif_value

    def compute_MMR(self, x):
        pass

    def run(self, f, *args, **kwargs):
        result = f(*args, **kwargs)
        return result

    def then(self, f, *args, **kwargs):
        self.__cst = self.__last_constraints_set
        self.__cst += [self.__last_objectif == self.__last_objectif_value]
        return self.run(f, *args, **kwargs)
