import os
import gensim
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Models.metrics import ModelMetrics
from Utils.utils import fullPath, showTime

class SnnModel(BaseModel):
    def __init__(self, Config):
        super().__init__(Config)
        if self.Config["w2vmodel"] == None:
            if len(Config["w2vmodelpath"]) == 0 or not os.path.isfile(fullPath(Config, "w2vmodelpath")):
                print ("Wrong path to W2V model. Stop.")
                Config["error"] = True
                return
        try:
            self.valSize = float(Config["valsize"])
        except ValueError:
            self.valSize = 0
        if self.valSize <= 0 or self.valSize >= 1:
            print ("Wrong size of validation data set. Stop.")
            Config["error"] = True
            return
        try:
            self.ndim = int(self.Config["w2vdim"])
        except ValueError:
            print ("Wrong size of vectors' dimentions. Stop.")
            Config["error"] = True
            return
        self.tempSave = Config["tempsave"] == "yes"
        self.useProbabilities = True
        self.w2vModel = None
        self.loadW2VModel()
        self.prepareData()
        self.launchProcess()

    def prepareData(self):
        print ("Start data preparation...")
        dp = DataPreparation(self, True)
        dp.getWordVectorsSum()

    def createModel(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=self.ndim))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.Config["cats"]), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def loadModel(self):
        self.model = self.loadNNModel()

    def trainModel(self):
        self.trainNNModel()

    def testModel(self):
        self.testNNModel()

    def saveAdditions(self):
        self.resources["w2v"] = "yes"