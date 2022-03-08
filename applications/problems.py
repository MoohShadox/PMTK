import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from PMTK.random.preferences_sampler import sample_preferences_from_order, sample_preferences_from_complete_order
from PMTK.utils import *
from PMTK.random.subset_samplers import sample_subsets
from PMTK.utility.additive_utility import AdditiveUtility
from PMTK.preferences import *
from PMTK.utility.utility_fitter import Utility_Fitter
from PMTK.utility.connivence_solver import Connivence_Solver
from PMTK.utility.extension_solver import *
from PMTK.utility.kernel_finder import *
from PMTK.utility.model_solver import *
from tqdm.notebook import tqdm
from itertools import chain
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


class Random_Configuration_Problem: 
    
    def __init__(self, components, additivity = 1):
        self.components = components
        self.costs = np.random.randint(1, 100, size = (len(self.components), ))
        self.utilities = {i:np.abs(np.random.normal(0,10)*10) for i in get_all_k_sets(self.components,additivity)}
    
    def __str__(self):
        ch = "Model: \n"
        ch += f"Costs: {self.costs} \n"
        for u in self.utilities:
            ch += f"{u} : {self.utilities[u]} \n"
        return ch
    
    def __repr__(self):
        return self.__str__()
    
    def __call__(self, x):
        if len(x) == 0:
            return np.array([-np.inf, -np.inf])
        u_s = 0
        for u in self.utilities:
            if all([i in x for i in u]):
                u_s += self.utilities[u]
        cost = 0
        for i in x:
            cost += self.costs[list(self.components).index(i)]
            
        return np.array([u_s, -cost])
    
class MO_Objective_Function:
    def __init__(self, items, f):
        self.f = f
        self.items = items
        self.budget = 0
        self.saved = {}
        self.epsilon = 1e-6
        
    def relation(self):
        preferences = Preferences(self.items)
        for i in self.saved:
            for j in self.saved:
                if i == j:
                    continue
                if pareto_dominate(self.saved[i], self.saved[j], self.epsilon) > 0:
                    preferences.add_preference(i, j)
                elif pareto_dominate(self.saved[i], self.saved[j], self.epsilon) < 0:
                    preferences.add_preference(j,i) 
        return preferences
    
    def pareto_front(self):
        costs = - np.array(list(self.saved.values()))
        elements = np.array(list(self.saved.keys()))
        return elements[is_pareto_efficient_simple(costs)]
    
    def evaluated_elements(self): 
        return len(self.saved.keys())
    
    def __call__(self, x):
        if not x in self.saved:
            self.saved[x] = self.f(x)
            self.budget += 1
        return self.saved[x]
    
