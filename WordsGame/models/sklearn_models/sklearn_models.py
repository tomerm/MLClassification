from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from Utils.config import categories



class SklearnModels(BaseEstimator, RegressorMixin):


    def __init__(self,**kw_args):

        super(SklearnModels,self).__init__()


    def prepare_data(self,trainDocs,testDocs):

        mlb = MultiLabelBinarizer(classes=categories)
        wev = TfidfVectorizer(ngram_range=(1, 3), max_df=0.50).fit([x.lines for x in trainDocs],
                                                                   [x.nlabs for x in trainDocs])
        X_train = wev.transform([x.lines for x in trainDocs])
        y_train = mlb.fit_transform([x.nlabs for x in trainDocs])
        X_test = wev.transform([x.lines for x in testDocs])
        y_test = mlb.fit_transform([x.nlabs for x in testDocs])

        return X_train,y_train,X_test,y_test

    def model_save(self):
        '''
        Save the model with the model name

        :return:
        '''
        pass


    def reload_model(self):
        '''

        Load saved model.
        :return:
        '''

