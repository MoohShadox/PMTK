from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from feature_selection.evaluation.Evaluator import Subset_Evaluator


class Classification_Evaluator(Subset_Evaluator):

    def __init__(self,X,y, *args, **kwargs):
        super().__init__(X, y, *args, **kwargs)
        self.cls = kwargs["cls"] if "cls" in kwargs else RandomForestClassifier()
        self.k = kwargs["k"] if "k" in kwargs else 2 #k in the k-fold validation

    def evaluate(self, subset):
        p = cross_val_score(self.cls, self.X[:, subset], self.y, cv= self.k )
        return p.mean(), -len(subset)


if __name__ == "__main__":
    pass

