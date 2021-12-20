"""
@Author: Ouaguenouni Mohamed
"""


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
        self.alternatives = items
        self.preferred = []
        self.preferred_or_indifferent = []
        self.indifferent = []
        self.subsets = []
        self.relation_matrix = None

    def __add_subsets(self, subset):
        """
        Used to maintain a list of the subsets concerned by the preferences.
        """
        subset = [i for i in subset if i in self.alternatives]
        subset = tuple(sorted(set(subset)))
        if subset not in self.subsets:
            self.subsets.append(subset)
        return subset

    def add_preference(self, s_1, s_2):
        """
        Create a strict preference between x and y.
        """
        s_1 = self.__add_subsets(s_1)
        s_2 = self.__add_subsets(s_2)
        t_1 = sorted([s_1, s_2])
        if t_1 not in self.preferred:
            self.preferred.append(t_1)

    def add_indifference(self, s_1, s_2):
        """
        Create an indifference relation between x and y.
        """
        s_1 = self.__add_subsets(s_1)
        s_2 = self.__add_subsets(s_2)
        t_1 = sorted([s_1, s_2])
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
        if (s_1, s_2) in self.preferred:
            return 1
        if (s_2, s_1) in self.preferred:
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
        if (s_1, s_2) in self.preferred or (s_1, s_2) in self.indifferent:
            return 1
        if (s_2, s_1) in self.preferred or (s_2, s_1) in self.indifferent:
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
        if (s_1, s_2) in self.indifferent or (s_2, s_1) in self.indifferent:
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
        if (s_1, s_2) not in self.preferred \
           and (s_1, s_2) not in self.indifferent \
           and (s_2, s_1) not in self.indifferent \
           and (s_2, s_1) not in self.preferred:
            return 1
        return 0

    def __str__(self):
        r_ch = "Preference relation : \n"
        for s_1, s_2 in self.preferred:
            r_ch += f"{s_1} > {s_2} \n"
        for s_1, s_2 in self.indifferent:
            r_ch += f"{s_1} = {s_2} \n"
        return r_ch


if __name__ == "__main__":
    pass
