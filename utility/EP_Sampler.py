import numpy as np
import matplotlib.pyplot as plt
import itertools
from PMTK.random.preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order
from PMTK.utils import *
from PMTK.random.subset_samplers import sample_subsets
from PMTK.utility.additive_utility import AdditiveUtility
from PMTK.preferences import *
from PMTK.utility.utility_fitter import Utility_Fitter
from PMTK.utility.connivence_solver import Connivence_Solver
#from PMTK.utility.utility_sampler import Utility_Sampler
from PMTK.utility.extension_solver import *
from tqdm.notebook import tqdm
from itertools import chain


class EP_Sampler:
    
    def __init__(self,items, preferences):
        self.items = items
        self.preferences = preferences
        self.epsilon = 1e-6
        self.theta = None
        self.problem = None
    
    def set_theta(self, theta):
        self.theta = theta
        return self
    
    def empty_polyhedron(self):
        UF = Utility_Fitter(self.items, self.theta)
        UF.epsilon = self.epsilon
        if len(self.preferences) == 0:
            return True
        UF.set_model(self.theta).set_preferences(self.preferences).build_vars().build_preferences_cst(bound_model = True)
        u = UF.solve_for_linear(np.random.rand(len(self.theta))).get_utility()
        return not u
    
    def sample_extreme_pts(self, n):
        if (self.empty_polyhedron()):
            return []
        UF = Utility_Fitter(self.items, self.theta)
        UF.epsilon = self.epsilon
        UF.set_model(self.theta).set_preferences(self.preferences).build_vars().build_preferences_cst(bound_model = True)
        ext_pts = []
        for i in range(n//2):
            d = np.random.rand(len(self.theta))
            u = UF.solve_for_linear(d).get_utility()
            v = np.array(list(u.theta_values.values()))
            if len(ext_pts) >= 1:
                ds = min([np.abs(v - i).sum() for i in ext_pts])
            if len(ext_pts) == 0 or ds > UF.epsilon:
                ext_pts.append(v)
            u = UF.solve_for_linear(-d).get_utility()
            v = np.array(list(u.theta_values.values()))
            if len(ext_pts) >= 1:
                ds = min([np.abs(v - i).sum() for i in ext_pts])
            if len(ext_pts) == 0 or ds > UF.epsilon:
                ext_pts.append(v)
        return ext_pts
    
    def get_best_subsets(self, k):
        ep = self.sample_extreme_pts(k)
        L = {}
        for e in ep:
            s = self.find_best_subset(e)
            L[s] = L.get(s,0) + 1
        ordered_subsets = [i[0] for i in sorted(L.items(), key = lambda x:x[1], reverse=True)] 
        return ordered_subsets

    
    def sample_points_in_polyhedron(self, ext_pts):
        for i in range(n):
            w = np.random.rand(len(ext_pts))
            w = w / w.sum()
            v = w @ np.array(ext_pts)
        return v
        
    
    def find_best_subset(self, u):
        self.selection_vars = cp.Variable(len(self.items),integer = True)
        cst_all = [self.selection_vars >= 0, self.selection_vars <= 1]
        ut_vars = []
        for x in self.theta:
            v = [self.selection_vars[list(self.items).index(i)] for i in x]
            if len(v) == 1:
                ut_vars.append(v[0])
                continue
            name = "&".join(["x_"+str(i) for i in x])
            s = cp.Variable(name = name)
            cst_l = [s <= i for i in v]
            cst_l += [s <= 1]
            cst_l += [s >= 0]
            cst_l += [s >= sum(v) - len(v) + 1 ]
            cst_all += cst_l
            ut_vars.append(s)
        obj = cp.Maximize(sum([u_i*x for u_i,x in zip(u, ut_vars)]))
        prob = cp.Problem(obj, cst_all)
        s = prob.solve()
        #print("value:", prob.value)
        #print("selected:", self.selection_vars.value)
        return tuple(np.where(self.selection_vars.value == 1)[0])
