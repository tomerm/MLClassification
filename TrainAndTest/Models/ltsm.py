import os
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Models.metrics import ModelMetrics
from Utils.utils import fullPath, showTime

class LTSMModel(BaseModel):
    def __init__(self, Config):
        super().__init__(Config)
        if self.Config["w2vmodel"] == None:
            if len(Config["w2vmodelpath"]) == 0 or not os.path.isfile(fullPath(Config, "w2vmodelpath")):
                print ("Wrong path to W2V model. Stop.")
                Config["error"] = True
                return
        if len(Config["indexerpath"]) == 0 or not os.path.isfile(fullPath(Config, "indexerpath")):
            if Config["runfor"] == "test" or (len(Config["indexerpath"]) != 0 and not os.path.isdir(
                    os.path.dirname(fullPath(Config, "indexerpath")))):
                print ("Wrong path to indexer. Stop.")
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
        self.embMatrix, self.maxWords = dp.getWordVectorsMatrix()

    def createModel(self):
        model = Sequential()
        model.add(Embedding(self.maxWords, self.ndim, input_length=self.Config["maxseqlen"]))
        model.layers[0].set_weights([self.embMatrix])
        model.layers[0].trainable = False
        model.add(LSTM(self.Config["maxseqlen"]))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
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
        if not "indexer" in self.Config["resources"]:
            self.Config["resources"]["indexer"] = fullPath(self.Config, "indexerpath")
        self.resources["indexer"] = "yes"
        self.resources["handleType"] = "wordVectorsMatrix"