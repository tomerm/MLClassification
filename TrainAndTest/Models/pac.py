import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Utils.utils import fullPath

class PacModel(BaseModel):
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
        self.useProbabilities = False
        self.prepareData()
        self.launchProcess()

    def prepareData(self):
        print("Start data preparation...")
        dp = DataPreparation(self, False)
        dp.getDataForSklearnClassifiers()

    def createModel(self):
        return OneVsRestClassifier(PassiveAggressiveClassifier(n_jobs=-1, max_iter=20))

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



