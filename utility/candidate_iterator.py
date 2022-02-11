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

from PMTK.utility.extension_solver import *
from tqdm.notebook import tqdm
from itertools import chain

class Candidate_Iterator:
    
    def __init__(self, items, subsets, banned = None, max_size = None, min_size = 1, pred = None):
        if not max_size:
            max_size = len(items)
        if not banned:
            banned=[]
        if not pred:
            pred = lambda x:True
        self.items = items
        self.subsets = subsets
        self.banned = banned
        self.min_size = min_size
        self.max_size = max_size
        self.pred = pred
        self.comb_it = []
        self.generated = []
        for k in range(1, len(self.items)):
            its = []
            for s in subsets:
                if len(s) >= k:
                    its.append(itertools.combinations(s, k))
            self.comb_it.append(chain(*its))
        self.k = 1
        self.id = 0
    
    def __iter__(self):
        return self
    
    def ban(self, *els):
        for el in els:
            self.banned += [tuple(sorted(el))]
    
    def __next__(self):
        self.k = max(self.k, self.min_size)
        while self.k <= min(self.max_size, len(self.items)-1):
            for c in self.comb_it[self.k-1]:
                if not c in self.banned and self.pred(c) and not c in self.generated:
                    self.generated.append(c)
                    return c
            self.k += 1
        raise StopIteration