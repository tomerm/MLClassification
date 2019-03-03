from models.sklearn_models.sklearn_models import SklearnModels
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


class SVM(SklearnModels):

    def __int__(self,**kw_args):

        super(SVM, self).__init__()
        self.multi_class = kw_args.get("multi_class","ovr")
        self.model = OneVsRestClassifier(LinearSVC(multi_class=self.multi_class))