import numpy as np
import matplotlib.pyplot as plt
import itertools
from PMTK.random.preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order
from PMTK.utils import *
from PMTK.random.subset_samplers import sample_subsets
from PMTK.utility.additive_utility import AdditiveUtility
from PMTK.preferences import *
from PMTK.utility.utility_fitter import Utility_Fitter
#from PMTK.utility.connivence_solver import Connivence_Solver
from PMTK.utility.extension_solver import *
from tqdm.notebook import tqdm
from itertools import chain



class Kernel_Finder:
    
    def __init__(self,items, preferences, model, epsilon = 1e-6, solver = None):
        self.items = items
        self.preferences = preferences
        self.model = model
        self.epsilon = epsilon
        #Solver variables
        self.vars = None
        self.is_not_zero = None
        self.cst = None
        self.additivity = None
        if not solver:
            solver = cp.GLPK_MI
        self.solver = solver
    
    def build_program(self):
        self.vars = cp.Variable(len(self.model))
        self.is_not_zero = cp.Variable(len(self.model), integer = True)
        self.additivity = cp.Variable()
        
        self.cst= [self.is_not_zero >= 0, self.is_not_zero <= 1]
        self.cst.append(self.additivity >= 0)
        
        
        for p in self.preferences.preferred:
            p_v = vectorize_subset(p[0], self.model) - vectorize_subset(p[1], self.model) 
            cst = self.vars @ p_v >= self.epsilon
            self.cst.append(cst)
            
        for p in self.preferences.indifferent:
            p_v = vectorize_subset(p[0], self.model) - vectorize_subset(p[1], self.model) 
            cst = self.vars @ p_v == 0
            self.cst.append(cst)
            
        #self.cst.append(self.vars <= 1)
        #self.cst.append(self.vars >= -1)
        self.cst.append(self.vars <= self.is_not_zero)
        self.cst.append(self.vars >= -self.is_not_zero)
        length_vector = np.array([len(s) for s in self.model])

        for nz, size in zip(self.is_not_zero, length_vector):
            self.cst.append(self.additivity >= nz*size)
            pass
        
    def compute_kernel(self):
        self.cardinality_obj = cp.Minimize(sum(self.is_not_zero))
        self.additivity_obj =  cp.Minimize(self.additivity)
        try:
            prob = cp.Problem(self.additivity_obj, self.cst)
            prob.solve(self.solver, reoptimize=True)
            #print(f"First solution: ({prob.status}, {self.additivity.value},  {sum(self.is_not_zero).value})")
            self.cst.append(self.additivity == self.additivity.value)
            prob = cp.Problem(self.cardinality_obj, self.cst)
            prob.solve(self.solver, reoptimize=True)
            #print(f"Second solution:  ({prob.status}, {self.additivity.value},  {sum(self.is_not_zero).value})") 
            self.kernel = []
            for i,v in enumerate(self.vars):
                if v.value != 0:
                    self.kernel.append(self.model[i])
            return self.kernel
            #return self.model
        except (Exception):
            print('[!!]', end = "")
            return self.model