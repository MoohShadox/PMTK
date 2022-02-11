import numpy as np
import matplotlib.pyplot as plt
import itertools
from PMTK.random.preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order
from PMTK.utils import *
from PMTK.random.subset_samplers import sample_subsets
from PMTK.utility.additive_utility import AdditiveUtility
from PMTK.preferences import *
from PMTK.utility.utility_fitter import Utility_Fitter
from PMTK.utility.extension_solver import *
from tqdm.notebook import tqdm
from numpy import array
from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices




class Utility_Sampler:
    def __init__(self,items, preferences):
        self.items = items
        self.preferences = preferences
        self.epsilon = 1e-2
        self.theta = None
        self.vertices = None
        self.problem = None
        self.vertices = None
        self.vert_computed = False
    
    def set_theta(self, theta):
        self.theta = theta
        self.vertices = None
        self.__vars = None
        return self
    
    def build_polyhedron(self):
        if not self.__vars:
            self.__vars = cp.Variable(len(self.theta))
        cst_l = []
        for i, (x, y) in enumerate(self.preferences.preferred):
            cst_l += [self.preferences.vectorize_preference(x, y, self.theta) @ self.__vars >= self.epsilon]
        for i, (x, y) in enumerate(self.preferences.preferred_or_indifferent):
            cst_l += [self.preferences.vectorize_preference(x, y, self.theta) @ self.__vars >= 0]
        for i, (x, y) in enumerate(self.preferences.indifferent):
            cst_l += [self.preferences.vectorize_preference(x, y, self.theta) @ self.__vars == 0]
        cst_l += [self.__vars >= -1, self.__vars <= 1]
        self.problem = cp.Problem(cp.Minimize(0), cst_l)
        self.vertices = None
        return self
        
    
    def getProblemMatrix(self):
        data, chain, inverse_data = self.problem.get_problem_data(cp.SCS)
        self.A = data["A"].toarray()
        self.b = data["b"]
        return self.A,self.b
    
    def sample_points(self, n):
        if not self.vert_computed:
            self.vertices = np.array(compute_polytope_vertices(self.A, self.b))
            if len(self.vertices) >= 1:
                self.vert_computed = True
        w = np.random.rand(len(self.vertices), n)
        w = (w / w.sum()).reshape((n,-1))
        u = w @ self.vertices
        return u
    
    def get_relative_volume(self, n):
        u_s = self.sample_points(n)
        if not self.vert_computed:
            return False
        try:
            v = ConvexHull(u_s).volume
        except Exception:
            return {}
        subsets_polyhedra = {}
        for u in u_s:
            s = self.found_best_subset(u)
            subsets_polyhedra[s] = subsets_polyhedra.get(s, []) + [u]
        vols = {}
        for s in subsets_polyhedra:
            try:
                vols[s] = ConvexHull(np.array(subsets_polyhedra[s])).volume / v 
            except Exception:
                vols[s] = 0
        return vols
    
    def found_best_subset(self, u):
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