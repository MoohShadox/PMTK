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
from PMTK.utility.candidate_iterator import Candidate_Iterator
from PMTK.utility.extension_solver import *
from tqdm.notebook import tqdm
from itertools import chain

class TNode:
    def __init__(self, tree, theta, banned = None, level = 0):
        if not banned:
            banned = []
        if theta == []:
            theta.append(EMPTY_SET)
        self.banned = banned
        self.theta = theta
        self.connivent = None
        self.child_iterator = None
        self.tree = tree
        self.childs = []
        self.level = level
        self.solved = False
        
    def __iter__(self):
        return self.__it
    
    def __str__(self):
        c = f"-----Node({self.theta}) [Banned = {self.banned}]\n"
        for i in self.childs:
            c += self.level*"\t"+ f"{i}"
        return c
        
    
    def __repr__(self):
        return self.__str__()
    
    def get_next_child(self, min_size = None, max_size = None, log_output = False):
        if not self.connivent:
            #print("Solving node:",self.theta," on level:", self.level)
            self.compute_connivent(self.tree.preferences, log_output=True)
            #print("Solved!")
        try:
            if min_size:
                self.child_iterator.min_size = min_size
            if max_size:
                self.child_iterator.max_size = max_size
            e = next(self.__it)
            return e
        except StopIteration:
            return None
    
    def build(self, log_output = False):
        
        if len(self.theta) > self.tree.best_n_elements:
            #print("Theta: ", self.theta, ", descendant of a size >= ", len(self.theta) + 1 , " CUT ! ") 
            return 
        e = self.get_next_child(min_size = self.tree.min_size, max_size = self.tree.max_size)
        all_bans = []
        cpt = 1
        while e:
            if e in all_bans:
                continue
            #print(f"Node:{self.theta} with id: ", self.idx," -> son's id :", str(self.idx + f".{cpt}"))
            child = TNode(self.tree, self.theta + [e], banned =list(set( self.banned + list(all_bans))) , level = self.level + 1)
            child.build()
            all_bans.append(e)
            e = self.get_next_child(min_size = self.tree.min_size, max_size = self.tree.max_size, log_output=log_output)
            self.childs.append(child)
            cpt = cpt + 1
    
    def get_all_thetas(self, L = None):
        if not L:
            L = []
        if self.solved:
            if not self.theta in L:
                L.append(self.theta)
        for c in self.childs:
            c_L = c.get_all_thetas(L)
            for c in c_L:
                if not c in L:
                    L.append(c)
        return L
                

            
    def compute_connivent(self, preferences, log_output = False):
        CS = Connivence_Solver(preferences, self.theta)
        self.connivent = CS.check_connivences(log_output = log_output)
        if not self.connivent:
            self.connivent = []
            m = max([len(i) for i in self.theta])
            self.tree.max_size = min(self.tree.max_size, m)
            self.tree.best_n_elements = min(len(self.theta), self.tree.best_n_elements)
            self.solved = True
            print("Solved for : ", self.theta," k=",{self.tree.max_size}, " s = ", self.tree.best_n_elements)
            
            
        subsets = []
        #print(self.connivent)
        for x,y in self.connivent:
            if not x in subsets:
                subsets.append(x)
            if not y in subsets:
                subsets.append(y)
        self.child_iterator = Candidate_Iterator(self.tree.items,subsets,banned = self.theta + self.banned)
        self.__it = iter(self.child_iterator)
    

class Ttree:
    def __init__(self, items, preferences, initial_theta = None):
        if not initial_theta:
            initial_theta = []
        self.head = TNode(self, initial_theta)
        self.max_size = len(items)
        self.min_size = 0
        
        self.best_n_elements = np.inf
        
        self.preferences = preferences
        self.items = items
    
    def __str__(self):
        c = f"==========Theta Tree===== \n"
        c += str(self.head)
        return c
    
    def get_all_thetas(self):
        return self.head.get_all_thetas()
    
    def get_min_thetas(self):
        thetas = self.head.get_all_thetas()
        min_size = min([len(i) for i in thetas])
        min_t = []
        for t in thetas:
            if len(t) == min_size:
                min_t.append(t)
        return min_t
        
    
    def __repr__(self):
        return self.__str__()