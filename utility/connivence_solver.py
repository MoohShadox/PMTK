import cvxpy as cp
import numpy as np
from PMTK.random.preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order
from PMTK.utils import *
from PMTK.random.subset_samplers import sample_subsets
from PMTK.utility.utility_fitter import Utility_Fitter
import time
import random

class Connivence_Solver:

    def __init__(self, preferences = None, model = None, solver = None):
        self.preferences = preferences
        self.model = model
        self.problem = None
        if solver == None:
            solver = cp.GLPK_MI
        self.solver = solver
        
        
    def set_preferences(self, preferences):
        self.preferences = preferences
        return self

    def check_connivences(self, min_size = 1, log_output = False):
        #print("Model size:", len(self.model))
        #print("préférences in connivence:", len(self.preferences))

        variables = cp.Variable(len(self.preferences), integer = True)

        mat = self.preferences.get_matrix(self.model, self.preferences.preferred)
        sizes_pref = [len(x[0]) + len(x[1]) for x in self.preferences.preferred]

        mat2 = self.preferences.get_matrix(self.model, self.preferences.indifferent)
        sizes_pref += [len(x[0]) + len(x[1]) for x in self.preferences.indifferent]

        if len(mat2.shape) > 1:
            mat = np.vstack([mat, mat2]).astype(float)
        #print("Constraints matrix", mat.shape)
        mat = mat.T
        #print("Mat shape:", mat.shape)
        #print("Objective:", sum(variables))
        
        
        obj = cp.Minimize( sum( [ variables[i]*(sizes_pref[i]) for i in range(len(self.preferences))] ) )
        #obj = cp.Minimize( sum( [ variables[i]*(random.random()) for i in range(len(self.preferences))] ) )
        #obj = cp.Minimize(0)

        
        problem = cp.Problem(obj, [variables >= 0, sum(variables) >= min_size, mat.astype(float) @ variables == 0.0] )
        p = problem.solve(self.solver, reoptimize=True, verbose=False)
        #print(problem.status)
        #print(problem.status)
        #print(problem.value)
        
        if problem.status == cp.OPTIMAL:
            connivent = []
            for i in np.where(variables.value != 0)[0]:
                for k in range(int(variables.value[i])):
                    connivent.append(self.preferences[i])
            return connivent
        
        assert problem.status == cp.INFEASIBLE
        
        if problem.status == cp.INFEASIBLE:
            return []


if __name__ == "__main__":
    for i in range(10):
        items = np.arange(6)
        pref = sample_preferences_from_complete_order(items)
        model = get_all_k_sets(items, 3)
        CS = Connivence_Solver(pref, model)
        status = CS.check_connivences()
        UF = Utility_Fitter(items, model)
        UF.set_model(model).set_preferences(pref).build_vars().build_preferences_cst().run(UF.get_most_discriminant_utility)
        f = UF.get_utility()
        print("Status: ", status, " and function: ", f)
        #R = f.compute_relation(pref.subsets)
