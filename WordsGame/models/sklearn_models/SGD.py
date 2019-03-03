from models.sklearn_models.sklearn_models import SklearnModels
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier


class SGD(SklearnModels):

    def __int__(self,**kw_args):

        super(SGD, self).__init__()
        self.alpha = kw_args.get("alpha",1e-4)
        self.loss = kw_args.get("loss","modified_huber")
        self.penalty = kw_args.get("penalty","elasticnet")
        self.max_iter = kw_args.get("max_iter",10)
        self.model = OneVsRestClassifier(SGDClassifier(loss=self.loss, penalty=self.penalty, alpha=self.alpha, max_iter=self.max_iter, n_jobs=-1))