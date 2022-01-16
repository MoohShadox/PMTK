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

    def build_preferences_cst(self):
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
        for m in self.model:
            v = self.preferences.vectorize_subset(m, self.model)
            cst_l.append(v @ self.__vars <= 1)
            cst_l.append(v @ self.__vars >= -1)
        self.__cst = cst_l
        return self

    def __solve(self, cst_l, obj):
        self.prob = cp.Problem(obj, cst_l)
        self.prob.solve(solver="GLPK")
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

    def solve_for_linear(self, vector):
        obj = cp.Maximize(self.__vars @ vector)
        self.__solve(self.cst, obj)
        return self

    def problem(self):
        return self.__prob

    def get_utility(self):
        if self.prob.status == cp.INFEASIBLE:
            print("Infeasible! ")
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
        obj = cp.Maximize(self.__vars @ vector)
        self.solve(self.cst, obj)
        return self

    def run(self, f, *args, **kwargs):
        result = f(*args, **kwargs)
        return result

    def then(self, f, *args, **kwargs):
        self.__cst = self.__last_constraints_set
        self.__cst += [self.__last_objectif == self.__last_objectif_value]
        return f(*args, **kwargs)

if __name__ == "__main__":
    for i in range(20):
        items = np.arange(6)
        pref = sample_preferences_from_order(items, 100)
        model = get_all_k_sets(items, 4)
        UF = Utility_Fitter(items, model)
        UF.set_model(model).set_preferences(pref).build_vars().build_preferences_cst().run(UF.get_most_discriminant_utility)
        f = UF.get_utility()
        R = f.compute_relation(pref.subsets)
        if not R >= pref:
            n = input("...")
            for x,y in (pref - R).indifferent:
                print(x," = ",y) 
                print("f(x)=", f(x))
                print("f(y)=", f(y))
