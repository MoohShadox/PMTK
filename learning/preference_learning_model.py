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

class Preference_Model: 
    
    def __init__(self, item, **kwargs):
        self.item = item
        self.preference = None
        self.data = []
        
    def update_model(self, x, y, order, **kwargs):
        D = {"x":x,"y":y, "order":order}
        D.update(kwargs)
        self.data.append(D)
    
    def __str__(self):
        return str(pd.DataFrame(self.data))
    
    def fit(self, preferences):
        self.preferences = preferences
        for x in self.preferences.subsets:
            for y in self.preferences.subsets:
                if x==y:
                    continue
                if self.preferences.is_preferred(x,y) != 0:
                    self.update_model(x,y,">")
                    self.update_model(y,x,"<")
                elif self.preferences.is_indifferent(x,y):
                    self.update_model(x,y,"=")
                    self.update_model(y,x,"=")
                else:
                    self.update_model(x,y,"?")
                    self.update_model(y,x,"?")
        pass
    
    def predict(self, x, y):
        pass


    
    
    
class  Clf_cp_mdl(Preference_Model):
    
    def __init__(self, item, **kwargs):
        super().__init__(item, **kwargs)
        assert "clf" in kwargs, "A Classification Preference model needs a classifier to be specified with the keyword clf"
        self.clf = kwargs["clf"]
        self.class_dict = {
            ">":0,
            "<":1,
            "=":2,
        }
        self.df = None
        self.X = None
        self.y = None
        self.clf_mdl = None
        self.model = None
        self.cost_vector = None
    
    def train(self,preferences, model, cost_vector, **kwargs):
        self.fit(preferences)
        self.df = pd.DataFrame(self.data)
        self.model = model
        self.cost_vector = cost_vector
        self.items = preferences.items
        arr = []
        classes = []
        singletons = get_all_k_sets(preferences.items, 1)
        for x,y,o in zip(self.df["x"], self.df['y'], self.df["order"]):
            if o == "?":
                continue
            v_x = vectorize_subset(x, self.model)
            v_y = vectorize_subset(y, self.model)
            cost_dif = cost_vector @ (vectorize_subset(x, singletons) - vectorize_subset(y,singletons))
            line = list(v_x) + list(v_y) + [cost_dif]
            line = np.array(line)
            arr.append(line)
            classes.append(self.class_dict[o])
        arr = np.array(arr)
        self.X = arr
        self.y = classes
        self.clf_mdl = self.clf(**kwargs)
        self.clf_mdl.fit(self.X,self.y)
    
    def predict(self, x, y):
        v_x = vectorize_subset(x, self.model)
        v_y = vectorize_subset(y, self.model)
        singletons = get_all_k_sets(self.items, 1)
        cost_dif = self.cost_vector @ (vectorize_subset(x, singletons) - vectorize_subset(y,singletons))
        line = list(v_x) + list(v_y) + [cost_dif]
        line = np.array(line).reshape((1,-1))
        return self.clf_mdl.predict(line)[0]
    
    def score(self, subsets, preferences):
        union_size = 0
        intersection_size = 0
        predicted_mdl = 0
        true_pref = 0
        for x in subsets:
            for y in subsets:
                if x == y:
                    continue
                if self.predict(x,y) != 3:
                    predicted_mdl += 1
                if  not preferences.is_incomparable(x,y):
                    true_pref += 1
                if self.predict(x,y) != 3 or not preferences.is_incomparable(x,y):
                    union_size += 1
                if preferences.is_preferred(x,y) == 1 and self.predict(x,y) == 0:
                    intersection_size += 1
                elif preferences.is_preferred(y,x) == 1 and self.predict(x,y) == 1:
                    intersection_size += 1
                elif preferences.is_indifferent(y,x) and self.predict(x,y) == 2:
                    intersection_size += 1
                else:
                    pass
                    #print(preferences.is_preferred(y,x), " vs ", self.predict(x,y))
        print("Intersection size:", intersection_size)
        print("Union size", union_size)
        print("Predicted mdl: ", predicted_mdl)
        print("true pref:", true_pref)
        return intersection_size/union_size