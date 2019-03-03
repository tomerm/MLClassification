from models.sklearn_models.sklearn_models import SklearnModels
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.multiclass import OneVsRestClassifier


class PassiveAggresive(SklearnModels):

    def __int__(self,**kw_args):

        super(PassiveAggresive, self).__init__()
        self.max_iter = kw_args.get("max_iter",20)
        self.model = OneVsRestClassifier(PassiveAggressiveClassifier(n_jobs=-1, max_iter = self.max_iter))