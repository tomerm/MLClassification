import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Utils.utils import fullPath

class SGDModel(BaseModel):
    def __init__(self, Config):
        super().__init__(Config)
        if len(Config["binarizerpath"]) == 0 or not os.path.isfile(fullPath(Config, "binarizerpath")):
            if Config["runfor"] == "test" or (len(Config["binarizerpath"]) != 0 and not os.path.isdir(
                    os.path.dirname(fullPath(Config, "binarizerpath")))):
                print ("Wrong path to binarizer. Stop.")
                Config["error"] = True
                return
        if len(Config["vectorizerpath"]) == 0 or not os.path.isfile(fullPath(Config, "vectorizerpath")):
            if Config["runfor"] == "test" or (len(Config["vectorizerpath"]) != 0 and not os.path.isdir(
                    os.path.dirname(fullPath(Config, "vectorizerpath")))):
                print ("Wrong path to vectorizer. Stop.")
                Config["error"] = True
                return
        self.useProbabilities = True
        self.handleType = "vectorize"
        if Config["runfor"] != "crossvalidation":
            self.prepareData()
        self.launchProcess()

    def prepareData(self):
        print("Start data preparation...")
        dp = DataPreparation(self, False)
        dp.getDataForSklearnClassifiers()

    def createModel(self):
        return OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='elasticnet',
                                                 alpha=1e-4, max_iter=10, tol=1e-3, n_jobs=-1))

    def loadModel(self):
        self.model = self.loadSKLModel()

    def trainModel(self):
        self.trainSKLModel()

    def testModel(self):
        self.testSKLModel()

    def saveAdditions(self):
        if not "vectorizer" in self.Config["resources"]:
            self.Config["resources"]["vectorizer"] = fullPath(self.Config, "vectorizerpath")
        self.resources["vectorizer"] = "yes"

