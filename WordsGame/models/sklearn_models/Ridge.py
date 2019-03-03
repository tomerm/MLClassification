from models.sklearn_models.sklearn_models import SklearnModels
from sklearn.linear_model import RidgeClassifierCV
from sklearn.multiclass import OneVsRestClassifier


class Ridge(SklearnModels):

    def __int__(self,**kw_args):


        super(Ridge, self).__init__()
        self.alpha = kw_args.get("alpha",1)
        self.model = OneVsRestClassifier(RidgeClassifierCV(self.alpha))