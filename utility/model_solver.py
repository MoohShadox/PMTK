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
from PMTK.utility.kernel_finder import *
from tqdm.notebook import tqdm
from itertools import chain

def chain_iterators(*it):
    for i in it:
        for j in i:
            yield j

def additivity(theta):
    return max([len(i) for i in theta])

def get_unifying_model(prf, init_mdl):
    print("Init model", init_mdl)
    T = Tree(prf.items, prf ,init_mdl)
    T.head.open_node()
    return union(T.found_theta)

def better(theta1, theta2):
    if additivity(theta1) < additivity(theta2):
        return 2
    if additivity(theta2) < additivity(theta1):
        return -2
    if len(theta1) > len(theta2):
        return -1
    if len(theta2) > len(theta1):
        return 1
    return 0

def dominated(theta, theta_list):
    for t in theta_list:
        if better(t, theta) > 0:
            return True
    return False

def keep_non_dominated(theta_list):
    non_dominated = []
    for t in theta_list:
        if not dominated(t, theta_list):
            non_dominated.append(t)
    return non_dominated

def vectorize_subset( x, model):
    vector = np.zeros(len(model))
    for subset in model:
        if all(s in x for s in subset):
            vector[model.index(subset)] += 1
    return vector

def vectorize_preference(x, y, model):
    vector = vectorize_subset(x, model) - vectorize_subset(y, model)
    return vector

class Node:
    
    def __init__(self, tree, theta, banned, level = 0):
        self.theta = theta
        self.k = additivity(theta)
        self.s = len(theta)
        self.tree = tree
        self.banned = banned
        self.children = []
        self.connivent = None
        self.level = level
        self.found_connivent = False
        self.solved = False
    
    def get_next_child(self, min_size = None, max_size = None):
        if not self.connivent:
            self.compute_connivent(self.tree.preferences)
        try:
            if min_size:
                self.child_iterator.min_size = min_size
            if max_size:
                self.child_iterator.max_size = max_size
            e = next(self.__it)
            return e
        except StopIteration:
            return None
        
    def __str__(self):
        c = f"-----Node({self.theta}) [Banned = {self.banned}]\n"
        for i in self.children:
            c += self.level*"\t"+ f"{i}"
        return c
        
    
    def __repr__(self):
        return self.__str__()
    
    
    def open_node(self):
        if dominated(self.theta, self.tree.found_theta):
            print("#", end="")
            return
        
        print(f".", end="")
        
        if not self.found_connivent:
            self.connivent = self.tree.found_connivence(self.theta)
            self.found_connivent = True
        if not self.connivent:
            self.solved = True
            return 
            
        subsets = []
        for x,y in self.connivent:
            if not x in subsets:
                subsets.append(x)
            if not y in subsets:
                subsets.append(y)
            
        self.child_iterator = Candidate_Iterator(self.tree.items,subsets,banned = self.theta + self.banned)
        self.__it = iter(self.child_iterator)
        e = self.get_next_child(min_size = 0, max_size = self.tree.get_additivity())
        all_bans = []
        cpt  = 0
        while e:
            cpt = cpt + 1
            if e in all_bans:
                continue
            #print(f"Node:{self.theta} with id: ", self.idx," -> son's id :", str(self.idx + f".{cpt}"))
            child = Node(self.tree, self.theta + [e], banned =list(set( self.banned + list(all_bans))) , level = self.level + 1)
            #print(f"..{cpt}..", end="")
            child.open_node()
            
            all_bans.append(e)
            e = self.get_next_child(min_size = 0, max_size = self.tree.get_additivity())
            
            self.children.append(child)
            
    
class Tree:
    def __init__(self, items, preferences, init_theta, epsilon=1e-4):
        self.items = items
        self.preferences = preferences
        self.found_theta = []
        self.connivent_calculated = []  
        self.head = Node(self, init_theta, [])
        self.epsilon = epsilon
        
    def __str__(self):
        c = f"==========Theta Tree===== \n"
        c += str(self.head)
        return c
    
    def get_additivity(self):
        if len(self.found_theta)>0:
            return additivity(self.found_theta[0])
        return np.inf

    def found_connivence(self, theta):
        for connivent in self.connivent_calculated:
            if self.is_connivent(connivent, theta):
                return connivent
        CS = Connivence_Solver(self.preferences, theta)        
        #print("Solving...", end="")
        c = CS.check_connivences()
        #print(f"  Solved in {(time.time() - t):.2f} s    ",end="")
        if not c:
            self.found_theta.append(theta)
            print(f"Found theta: {additivity(theta)} , {len(theta)}") 
            KF = Kernel_Finder(self.items, self.preferences, theta, epsilon=self.epsilon)
            KF.build_program()
            kernel = KF.compute_kernel()
            self.found_theta.append(kernel)
            self.found_theta = keep_non_dominated(self.found_theta)
            print("Found:",theta)
            print("New size: ", len(self.found_theta))
            print(f"! {len(self.found_theta)} \n \n ")
        #if not c in self.connivent_calculated:
        #    self.connivent_calculated.append(c)
        return c
    
    def is_connivent(self, preference_set, theta):
        L = []
        for x,y in preference_set:
            L.append(vectorize_preference(x,y,theta))
        L = np.array(L)
        return ((L.sum(axis=0) == 0).all())