import gensim
import datetime
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from Utils.utils import fullPath
from Models.metrics import metricsNames, printMetrics
from Models.metrics import ModelMetrics
from Utils.utils import leftAlign, showTime

class BaseModel:
    def __init__(self, Config):
        self.Config = Config
        self.trainArrays = []
        self.trainLabels = []
        self.testArrays = []
        self.testLabels = []
        self.valArrays = []
        self.valLabels = []
        self.predictions = []
        self.metrics = {}
        self.resources = {}

        self.epochs = int(Config["epochs"])
        self.verbose = int(Config["verbose"])
        if self.verbose != 0:
            self.verbose = 1
        self.trainBatch = int(Config["trainbatch"])

    def launchProcess(self):
        if self.Config["runfor"] != "test":
            self.model = self.createModel()
            self.trainModel()
            if self.Config["runfor"] != "train":
                self.testModel()
        else:
            self.loadModel()
            self.testModel()

    def createModel(self):
        pass

    def loadModel(self):
        pass

    def trainModel(self):
        pass

    def testModel(self):
        pass

    def loadW2VModel(self):
        if self.Config["w2vmodel"] != None:
            print ("W2V model is already loaded...")
            self.w2vModel = self.Config["w2vmodel"]
            return
        print ("Load W2V model... ")
        ds = datetime.datetime.now()
        self.w2vModel = gensim.models.KeyedVectors.load_word2vec_format(fullPath(self.Config, "w2vmodelpath"))
        de = datetime.datetime.now()
        print("Load W2V model (%s) in %s" % (fullPath(self.Config, "w2vmodelpath"), showTime(ds, de)))
        self.Config["resources"]["w2v"]["modelPath"] = fullPath(self.Config, "w2vmodelpath")
        self.Config["resources"]["w2v"]["ndim"] = self.ndim

    def loadNNModel(self):
        return load_model(fullPath(self.Config, "modelpath", opt="name"))

    def loadSKLModel(self):
        return joblib.load(fullPath(self.Config, "modelpath", opt="name"))

    def trainNNModel(self):
        checkpoints = []
        if self.tempSave:
            checkpoint = ModelCheckpoint(fullPath(self.Config, "temppath") + "/tempModel.hdf5", monitor='val_acc',
                                     verbose=self.verbose, save_best_only=True, mode='auto')
            checkpoints.append(checkpoint)
        print("Start training..")
        ds = datetime.datetime.now()
        self.model.fit(self.trainArrays, self.trainLabels, epochs=self.epochs,
                validation_data=(self.valArrays, self.valLabels),
                batch_size=self.trainBatch, verbose=self.verbose, callbacks=checkpoints, shuffle=False)
        de = datetime.datetime.now()
        self.model.save(fullPath(self.Config, "modelpath", opt="name"))
        print("Model is trained in %s" %  (showTime(ds, de)))
        print ("Model evaluation...")
        scores1 = self.model.evaluate(self.testArrays, self.testLabels, verbose=self.verbose)
        print("Final model accuracy: %.2f%%" % (scores1[1] * 100))
        if self.tempSave:
            model1 = load_model(fullPath(self.Config, "temppath") + "/tempModel.hdf5")
            scores2 = model1.evaluate(self.testArrays, self.testLabels, verbose=self.verbose)
            print("Last saved model accuracy: %.2f%%" % (scores2[1] * 100))
            if scores1[1] < scores2[1]:
                model = model1
            pref = "The best model "
        else:
            pref = "Model "
        self.model.save(fullPath(self.Config, "modelpath", opt="name"))
        print (pref + "is saved in %s"%(fullPath(self.Config, "modelpath", opt="name")))

    def trainSKLModel(self):
        de = datetime.datetime.now()
        print("Start training...")
        self.model.fit(self.trainArrays, self.trainLabels)
        ds = datetime.datetime.now()
        print("Model is trained in %s" % (showTime(de, ds)))
        joblib.dump(self.model, fullPath(self.Config, "modelpath", opt="name"))
        print ("Model is saved in %s"%(fullPath(self.Config, "modelpath", opt="name")))
        print("Model evaluation...")
        prediction = self.model.predict(self.testArrays)
        print('Final accuracy is %.2f'%(accuracy_score(self.testLabels, prediction)))
        de = datetime.datetime.now()
        print("Prediction in %s" % (showTime(ds, de)))

    def testNNModel(self):
        print ("Start testing...")
        ds = datetime.datetime.now()
        self.predictions = self.model.predict(self.testArrays)
        de = datetime.datetime.now()
        print("Test dataset containing %d documents predicted in %s\n" % (len(self.testArrays), showTime(ds, de)))
        self.saveResources("keras")
        self.getMetrics()
        self.saveResults()

    def testSKLModel(self):
        print ("Start testing...")
        ds = datetime.datetime.now()
        self.predictions = self.model.predict(self.testArrays)
        de = datetime.datetime.now()
        print("Test dataset containing %d documents predicted in %s\n" % (self.testArrays.shape[0], showTime(ds, de)))
        self.saveResources("skl")
        self.getMetrics()
        self.saveResults()

    def getMetrics(self):
        print ("Calculate metrics...")
        ModelMetrics(self)
        if self.Config["showmetrics"] == "yes":
            printMetrics(self)


    def saveResults(self):
        self.Config["results"][self.Config["name"]] = self.predictions

    def saveResources(self, type):
        self.resources["modelPath"] = fullPath(self.Config, "modelpath", opt="name")
        self.resources["modelType"] = type
        self.saveAdditions()
        self.Config["resources"]["models"]["Model" + str(self.Config["modelid"])] = self.resources

    def saveAdditions(self):
        pass
