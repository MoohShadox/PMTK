import cvxpy as cp
import numpy as np
from preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order, get_all_k_sets
from samplers import sample_subsets
from utility_fitter import Utility_Fitter
import time

class Connivence_Solver:

    def __init__(self, preferences = None, model = None):
        self.preferences = preferences
        self.model = model
        self.problem = None

    def set_preferences(self, preferences):
        self.preferences = preferences
        return self

    def check_connivences(self, min_size = 1):
        print("Model size:", len(self.model))
        print("Number of préférences:", len(self.preferences))
        variables = cp.Variable(len(self.preferences), integer = True)
        mat = self.preferences.get_matrix(self.model, self.preferences.preferred)
        mat2 = self.preferences.get_matrix(self.model, self.preferences.indifferent)
        if len(mat2.shape) > 1:
            mat = np.vstack([mat, mat2]).astype(float)
        #print("Constraints matrix", mat.shape)
        mat = mat.T
        #print("Mat shape:", mat.shape)
        #print("Objective:", sum(variables))
        obj = cp.Minimize(sum(variables))
        problem = cp.Problem(obj, [variables >= 0, sum(variables) >= min_size, mat.astype(float) @ variables == 0.0] )
        p = problem.solve()
        print(problem.status)
        print(problem.value)
        if problem.status == cp.INFEASIBLE:
            return None
        connivent = []
        for i in np.where(variables.value == 1 )[0]:
            connivent.append(self.preferences[i])
        return connivent

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
