""" @Author: Ouaguenouni Mohamed """ 
import numpy as np

EMPTY_SET = tuple([])

class Preferences:
    """
    This class encapsulates different representations (sparse, matricial) of
    a set of preferences.
    These preferences could be represented with 2 predicates:
        - Strict preference.
        - Indifference.
    These two predicates could be extended by defining:
        - Prefered or indifferend as a disjonction of the two.
        - Incomparability if we don't have A preferred to B nor B
        preferred to A.
    """

    def __init__(self, items):
        self.items = items
        self.preferred = []
        self.preferred_or_indifferent = []
        self.indifferent = []
        self.subsets = []
        self.relation_matrix = None

    def vectorize_subset(self, x, model):
        vector = np.zeros(len(model))
        for subset in model:
            if all(s in x for s in subset):
                vector[model.index(subset)] += 1
        return vector

    def vectorize_preference(self, x, y, model):
        vector = self.vectorize_subset(x, model) - self.vectorize_subset(y, model)
        return vector

    def get_matrix(self, model, relation_set):
        vectors = []
        for x,y in relation_set:
            v = self.vectorize_preference(x,y,model)
            vectors.append(v.astype(float))
        vectors = np.array(vectors).astype(float)
        return vectors

    def __getitem__(self, index):
        all_prefs = self.preferred + self.indifferent
        return all_prefs[index]

    def __set_item__(self, index, item):
        all_prefs = self.preferred + self.indifferent
        all_prefs[index] = item

    def contradictions(self, other):
        contr = []
        for x,y in self.preferred:
            if [y,x] in other.preferred:
                contr.append((x,y))
        #for x,y in self.indifferent:
        #    if not [x,y] in other.indifferent:
        #        contr.append((x,y))
        return contr


    def __add_subsets(self, subset):
        """
        Used to maintain a list of the subsets concerned by the preferences.
        """
        if subset == EMPTY_SET:
            return subset
        if type(subset) != tuple:
            print(f"Warning: adding {subset} which is not a tuple")
            subset = [subset]
        try:
            subset = [i for i in subset if i in self.items]
            subset = tuple(sorted(set(subset)))
            if subset not in self.subsets:
                self.subsets.append(subset)
            return subset
        except TypeException:
            print(f"Exception because of the type of {subset}")
            raise TypeException
        
    def add_preference(self, s_1, s_2):
        """
        Create a strict preference between x and y.
        """
        s_1 = self.__add_subsets(s_1)
        s_2 = self.__add_subsets(s_2)
        t_1 = [s_1, s_2]
        if t_1 not in self.preferred:
            self.preferred.append(t_1)

    def add_indifference(self, s_1, s_2):
        """
        Create an indifference relation between x and y.
        """
        s_1 = self.__add_subsets(s_1)
        s_2 = self.__add_subsets(s_2)
        t_1 = [s_1, s_2]
        if t_1 not in self.indifferent:
            self.indifferent.append(t_1)

    def is_preferred(self, s_1, s_2):
        """
        Test the preference between two subsets
        return
                1 if the first subset is preferred to the second,
                -1 if the opposite is true
                0 if neither case is true.
        """
        if [s_1, s_2] in self.preferred:
            return 1
        if [s_2, s_1] in self.preferred:
            return -1
        return 0

    def is_preferred_or_indifferent(self, s_1, s_2):
        """
        Test the preference or indifference between two subsets
        return
                1 if the first subset is preferred to the second,
                -1 if the opposite is true
                0 if neither case is true.
        """
        if [s_1, s_2] in self.preferred or [s_1, s_2] in self.indifferent:
            return 1
        if [s_2, s_1] in self.preferred or [s_2, s_1] in self.indifferent:
            return -1
        return 0

    def is_indifferent(self, s_1, s_2):
        """
        Test the indifference between two subsets
        return
                1:  If the decision-maker is indifferent to the choice
                    of one alternative over the other
                0:  If not.
        """
        if [s_1, s_2] in self.indifferent or [s_2, s_1] in self.indifferent:
            return 1
        return 0

    def is_incomparable(self, s_1, s_2):
        """
        Checks if we cannot compare two subsets.
        return
                1:  If we cannot compare s_1 to s_2 neither with the
                    indifference nor with the preference
                0:  If we can.
        """
        if [s_1, s_2] not in self.preferred \
           and [s_1, s_2] not in self.indifferent \
           and [s_2, s_1] not in self.indifferent \
           and [s_2, s_1] not in self.preferred:
            return 1
        return 0

    def __len__(self):
        return len(self.indifferent) + len(self.preferred)
        pass

    def __str__(self):
        r_ch = "Preference relation : \n"
        for s_1, s_2 in self.preferred:
            r_ch += f"{s_1} > {s_2} \n"
        for s_1, s_2 in self.indifferent:
            r_ch += f"{s_1} = {s_2} \n"
        return r_ch

    def __ge__(self, other):
        for s_1 in other.preferred:
            if s_1 not in self.preferred:
                return False
        for s_1 in other.indifferent:
            if s_1 not in self.indifferent:
                return False
        return True

    def __eq__(self, other):
        return other >= self and self >= other

    def __gt__(self, other):
        return self >= other and not self == other

    def __le__(self, other):
        return other >= self

    def __lt__(self, other):
        return other >= self and not other == self

    def __sub__(self, other):
        prf = Preferences(self.items)
        for s in self.preferred:
            if s not in other.preferred:
                prf.preferred.append(s)
        for s in self.indifferent:
            if s not in other.indifferent:
                prf.indifferent.append(s)
        return prf



if __name__ == "__main__":
    pass
