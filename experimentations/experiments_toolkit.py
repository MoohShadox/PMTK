import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from itertools import chain, combinations
import matplotlib.pyplot
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
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
import numpy as np
import pandas as pd
from PMTK.preferences import *
from PMTK.utility.EP_Sampler import *
from PMTK.utils import *
from PMTK.utility.model_solver import *
from sklearn.svm import SVC
from tqdm.notebook import tqdm
from PMTK.preferences import Preferences
import numpy as np
import pandas as pd
from datetime import datetime
import cvxpy as cp
import os

def get_exp_id(root_path, EXP_TITLE):
  s = datetime.today().strftime('%d%m-%H%M%S')
  p = os.path.join(root_path, EXP_TITLE+s)
  return p
  

def train_clf(clf_class, preferences, theta, **kwargs):
  clf = clf_class(**kwargs)
  ds = preferences.to_dataset(theta)
  X,y = ds[:, :-1], ds[:, -1]
  clf.fit(X,y)
  return clf

def predict_clf(items, clf, theta, subsets):
  p = Preferences(items)
  for s1 in subsets:
    for s2 in subsets:
      x = list(vectorize_subset(s1, theta)) + list(vectorize_subset(s2, theta))
      x = np.array(x)
      x = x.reshape((1, -1))
      if (clf.predict(x) == 1):
          p.add_preference(s1, s2)
      elif clf.predict(x) == 2:
          p.add_preference(s2, s1)
      else:
          print("ERROR !!!! ")
  return p

def train_ordinal(items, function, max_budget, start_budget = 10, n_ext_pts = 20):
  objective = TL_Objective_Function(items, function)
  thetas = None
  for s in sample_subsets(items, n_subsets = start_budget):
    objective(s)
  preferences = objective.relation()
  thetas = get_all_thetas(preferences, [EMPTY_SET])
  while objective.budget < max_budget:
    timeout = 0
    preferences = objective.relation()
    print("Preferences: ", len(preferences))
    best_subsets = []
    precedent_thetas = list(thetas)
    thetas = [theta for theta in thetas if not EP_Sampler(items, preferences).set_theta(intersection(thetas)).empty_polyhedron()]
    print(f"thetas = {thetas} intersection: {intersection(precedent_thetas)}, precedent thetas: {precedent_thetas}")
    if len(thetas) == 0:
      thetas = get_all_thetas(preferences, intersection(precedent_thetas) if len(intersection(precedent_thetas)) > 0 else [EMPTY_SET])
    for theta in thetas:
      EPS = EP_Sampler(items, preferences).set_theta(theta)
      bs = EPS.get_best_subsets(n_ext_pts)
      best_subsets = best_subsets + bs
      found = False
    for s in best_subsets:
      if not s in objective.saved:
        print("evaluating ", s)
        objective(s)
        found = True
        break
      while not found:
        timeout += 1
        s = sample_subset(items)
        if not s in objective.saved:
          objective(s)
          print("evaluating random", s)
          found = True
          break
        if timeout > 1000:
          break
    if not found:
      print("no new subset to evaluate :)")
      break
  thetas = get_all_thetas(preferences, intersection(thetas) + [EMPTY_SET])
  return preferences, thetas
      

def predict_ordinal_multiple_thetas(items, preferences, thetas, subsets):
  pref = Preferences(items)
  UFS = [Utility_Fitter(items, theta).set_preferences(preferences).set_model(theta).build_vars().build_preferences_cst() for theta in thetas]
  for s1 in subsets:
    for s2 in subsets:
      if s1 == s2:
        continue
      dominance = True
      for UF in UFS:
        mpr = UF.compute_MPR(s1, s2)
        if mpr > 0:
          dominance = False
          break
      if dominance:
        pref.add_preference(s1, s2)
        print("|", end = "")
      else:
        print(".", end= "")
  print("")
  return pref


def get_all_thetas(prf, init_mdl, use_kernel = True):
    T = Tree(prf.items, prf ,init_mdl, use_kernel = use_kernel)
    T.head.open_node()
    return T.found_theta



def predict_ordinal(items, preferences, theta, subsets):
  EPS = EP_Sampler(items, preferences).set_theta(theta)
  pref = Preferences(items)
  if not EPS.empty_polyhedron():
    UF = Utility_Fitter(items, theta)
    UF.set_preferences(preferences).set_model(theta).build_vars().build_preferences_cst()
    for s1 in subsets:
      for s2 in subsets:
        if s1 == s2:
          continue
        mpr = UF.compute_MPR(s1, s2)
        if mpr < 0:
          pref.add_preference(s1, s2)
          print("|", end = "")
        else:
          print(".", end= "")
    print("")
  else:
    print("Error polyhedron is empty !")
    return False
  return pref



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def complete_theta(theta):
  n_theta = []
  for s in theta:
    for k in powerset(s):
      if not k in n_theta:
        n_theta.append(k)
  return n_theta


def vectorize_subsets(films, theta):
    vec = []
    for i in films:
        v = vectorize_subset(i,theta)
        vec.append(v)
    return np.array(vec)


def get_all_thetas(prf, init_mdl, use_kernel = True):
    T = Tree(prf.items, prf ,init_mdl, use_kernel = use_kernel)
    T.head.open_node()
    return T.found_theta



def get_random_model(prf, init_mdl, use_kernel = True):
    print("Init model", init_mdl)
    T = Tree(prf.items, prf ,init_mdl, use_kernel = use_kernel)
    T.head.open_node()
    return random.choice(T.found_theta)

def intersection(thetas):
  s = union(thetas)
  for t in thetas:
    s = list(set(t).intersection(set(s)))
  return s


def clf_to_pref(clf, subsets, items, sampled_ext_pts = 10, theta = None):
    if theta == None:
        theta = get_all_k_sets(items, 1)
    preferences = Preferences(items)
    for s1 in subsets:
        for s2 in subsets:
            if s1 == s2:
                continue
            x = list(vectorize_subset(s1, theta)) + list(vectorize_subset(s2, theta))
            x = np.array(x)
            x = x.reshape((1, -1))
            if (clf.predict(x) == 1):
                preferences.add_preference(s1, s2)
            elif clf.predict(x) == 2:
                preferences.add_preference(s2, s1)
            else:
                print("ERROR !!!! ")
    return preferences


def preference_evaluation(preferences, function):
  correct = 0
  wrong = 0
  total = 0
  for x,y in preferences.preferred:
    v_x = function(x)
    v_y = function(y)
    if v_x > v_y:
      correct += 1
    elif v_y > v_x:
      wrong += 1
    total += 1
  return correct, wrong, total


class Random_Tierlist_Decider: 
    
    def __init__(self, items, p = 0.2, sigma = 100, alpha = 0.2, n_theta = None, n_tiers = 5):
      self.items = items
      self.utilities = {}
      self.p = p
      self.n_theta = n_theta
      self.n_tiers = n_tiers
      if not self.n_theta:
        self.n_theta = int(alpha*(len(items) + len(items)**2))+1
      for k in get_all_k_sets(items, 1):
        self.utilities[k] = np.random.normal(0, sigma)
      for i in range(self.n_theta):
        theta = self.sample_geometric_subset()
        while theta in self.utilities:
          theta = self.sample_geometric_subset()
        self.utilities[theta] = np.random.normal(0, sigma)
        self.tiers = self.compute_tiers()


    def sample_geometric_subset(self):
      it = list(self.items)
      s = random.choice(it)
      it.remove(s)
      theta = [s]
      while len(it) != 0:
        s = random.choice(it)
        theta.append(s)
        it.remove(s)
        r = random.random()
        if r > self.p:
          break
      return tuple(sorted(theta))

    def find_max_min(self):
      vars = cp.Variable(len(self.items), integer = True)
      ut_vars = {}
      cst_all = [vars >= 0 , vars <= 1]
      exp1 = []
      for u in self.utilities:
        v = [vars[list(self.items).index(i)] for i in u]
        if len(u) == 1:
          ut_vars[u] = v[0]
          exp1.append(self.utilities[u] * v[0])
          continue
        s = cp.Variable(integer = True)
        cst_l = [s <= i for i in v]
        cst_l += [s <= 1]
        cst_l += [s >= 0]
        cst_l += [s >= sum(v) - len(v) + 1]
        cst_all += cst_l
        ut_vars[u] = s
        exp1.append(self.utilities[u] * s)
      exp1 = sum(exp1)
      obj1 = cp.Maximize(exp1)
      prob = cp.Problem(obj1, cst_all)
      s = prob.solve()
      s1 = tuple(np.where(vars.value == 1)[0])
      max_v = obj1.value
      obj2 = cp.Minimize(exp1)
      prob = cp.Problem(obj2, cst_all)
      s = prob.solve()
      s2 = tuple(np.where(vars.value == 1)[0])
      min_v = obj2.value
      return max_v, min_v
    
    def compute_tiers(self):
      max_v, min_v = self.find_max_min()
      return np.linspace(min_v, max_v, self.n_tiers + 1)
    
    def __str__(self):
        ch = "Model: \n"
        for u in self.utilities:
            ch += f"{u} : {self.utilities[u]} \n"
        return ch
    
    def __repr__(self):
        return self.__str__()
    
    def __call__(self, x):
      if len(x) == 0:
        return 0
      u_s = 0
      for u in self.utilities:
          if all([i in x for i in u]):
              u_s += self.utilities[u]
      return np.argmax((self.tiers > u_s).astype(int))


class TL_Objective_Function:
    def __init__(self, items, f):
        self.f = f
        self.items = items
        self.budget = 0
        self.saved = {}
        self.epsilon = 2
        
    def relation(self):
        preferences = Preferences(self.items)
        for i in self.saved:
            for j in self.saved:
              if i == j:
                continue
              if self.saved[i] > self.saved[j]:
                preferences.add_preference(i,j)
              elif self.saved[j] > self.saved[i]:
                preferences.add_preference(j, i)
        return preferences
    
    def pareto_front(self):
        costs = np.array(list(self.saved.values()))
        elements = np.array(list(self.saved.keys()))
        ids = np.where(costs == np.max(costs))[0]
        print(ids)
        print(costs)
        print(elements)
        return elements[ids]
    
    def evaluated_elements(self): 
        return len(self.saved.keys())
    
    def __call__(self, x):
        if not x in self.saved:
            self.saved[x] = self.f(x)
            self.budget += 1
        return self.saved[x]
