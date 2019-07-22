import gensim
import os
import shutil
import datetime
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from Utils.utils import fullPath
from Models.metrics import metricsNames, printMetrics, printAveragedMetrics
from Models.metrics import ModelMetrics
from Models.dataPreparation import DataPreparation
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
        self.cvDocs = []
        self.predictions = []
        self.metrics = {}
        self.resources = {}
        self.addValSet = False
        self.valSize = 0
        self.isCV = False
        self.handleType = ""
        self.useProbabilities = False

        self.epochs = int(Config["epochs"])
        self.verbose = int(Config["verbose"])
        self.kfold = int(Config["kfold"])
        if self.verbose != 0:
            self.verbose = 1
        if Config["customrank"] == "yes":
            self.rankThreshold = float(Config["rankthreshold"])
        else:
            self.rankThreshold = 0.5
        if self.rankThreshold == 0:
            self.rankThreshold = 0.5
        self.trainBatch = int(Config["trainbatch"])

    def launchProcess(self):
        if self.Config["runfor"] == "crossvalidation":
            self.isCV = True
            self.launchCrossValidation()
        elif self.Config["runfor"] != "test":
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
        if self.tempSave and not self.isCV:
            checkpoint = ModelCheckpoint(fullPath(self.Config, "temppath") + "/tempModel.hdf5", monitor='val_acc',
                                     verbose=self.verbose, save_best_only=True, mode='auto')
            checkpoints.append(checkpoint)
        print("Start training...              ")
        ds = datetime.datetime.now()
        self.model.fit(self.trainArrays, self.trainLabels, epochs=self.epochs,
                validation_data=(self.valArrays, self.valLabels),
                batch_size=self.trainBatch, verbose=self.verbose, callbacks=checkpoints, shuffle=False)
        de = datetime.datetime.now()
        print("Model is trained in %s" %  (showTime(ds, de)))
        if self.isCV:
            return
        self.model.save(fullPath(self.Config, "modelpath", opt="name"))
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
        if self.isCV:
            return
        joblib.dump(self.model, fullPath(self.Config, "modelpath", opt="name"))
        print ("Model is saved in %s"%(fullPath(self.Config, "modelpath", opt="name")))
        print("Model evaluation...")
        prediction = self.model.predict(self.testArrays)
        print('Final accuracy is %.2f'%(accuracy_score(self.testLabels, prediction)))
        de = datetime.datetime.now()
        print("Evaluated in %s" % (showTime(ds, de)))

    def testNNModel(self):
        print ("Start testing...")
        print("Rank threshold: %.2f" % (self.rankThreshold))
        ds = datetime.datetime.now()
        self.predictions = self.model.predict(self.testArrays)
        de = datetime.datetime.now()
        print("Test dataset containing %d documents predicted in %s\n" % (len(self.testArrays), showTime(ds, de)))
        if self.isCV:
            return
        self.saveResources("keras")
        self.getMetrics()
        self.saveResults()

    def testSKLModel(self):
        print ("Start testing...")
        if self.useProbabilities:
            print ("Rank threshold: %.2f" % (self.rankThreshold))
        else:
            print ("Model doesn't evaluate probabilities.")
        ds = datetime.datetime.now()
        if not self.useProbabilities:
            self.predictions = self.model.predict(self.testArrays)
        else:
            self.predictions = self.model.predict_proba(self.testArrays)
        de = datetime.datetime.now()
        print("Test dataset containing %d documents predicted in %s" % (self.testArrays.shape[0], showTime(ds, de)))
        if self.isCV:
            return
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
        self.Config["metrics"][self.Config["name"]] = self.metrics
        if self.useProbabilities:
            self.Config["ranks"][self.Config["name"]] = self.rankThreshold
        else:
            self.Config["ranks"][self.Config["name"]] = 1.0

    def saveResources(self, type):
        self.resources["modelPath"] = fullPath(self.Config, "modelpath", opt="name")
        self.resources["modelType"] = type
        self.resources["rankThreshold"] = self.rankThreshold
        self.saveAdditions()
        if type == "skl":
            self.resources["handleType"] = "vectorize"
        self.Config["resources"]["models"]["Model" + str(self.Config["modelid"])] = self.resources

    def saveAdditions(self):
        pass

    def launchCrossValidation(self):
        print ("Start cross-validation...")
        ds = datetime.datetime.now()
        dp = DataPreparation(self, self.addValSet)
        pSize = len(self.cvDocs) // self.kfold
        ind = 0
        f1 = 0
        arrMetrics =[]
        for i in range(self.kfold):
            print ("Cross-validation, cycle %d from %d..."%((i+1), self.kfold))
            if i == 0:
                self.Config["cvtraindocs"] = self.cvDocs[pSize:]
                self.Config["cvtestdocs"] = self.cvDocs[:pSize]
            elif i == self.kfold - 1:
                self.Config["cvtraindocs"] = self.cvDocs[:ind]
                self.Config["cvtestdocs"] = self.cvDocs[ind:]
            else:
                self.Config["cvtraindocs"] = self.cvDocs[:ind] + self.cvDocs[ind+pSize:]
                self.Config["cvtestdocs"] = self.cvDocs[ind:ind+pSize]
            ind += pSize
            dp.getVectors(self.handleType)
            self.model = self.createModel()
            self.trainModel()
            self.testModel()
            ModelMetrics(self)
            arrMetrics.append(self.metrics)
            cycleF1 = self.metrics["all"]["f1"]
            print ("Resulting F1-Measure: %f\n"%(cycleF1))
            if cycleF1 > f1:
                if self.Config["cvsave"]:
                    self.saveDataSets()
                f1 = cycleF1
        de = datetime.datetime.now()
        print ("Cross-validation is done in %s"%(showTime(ds, de)))
        printAveragedMetrics(arrMetrics, self.Config)
        print ("The best result is %f"%(f1))
        print ("Corresponding data sets are saved in the folder %s"%(fullPath(self.Config, "cvpath")))


    def saveDataSets(self):
        root = fullPath(self.Config, "cvpath")
        shutil.rmtree(root)
        os.mkdir(root)
        trainPath = root + "/train"
        testPath = root + "/test"
        folds = {}
        os.mkdir(trainPath)
        for i in range(len(self.Config["cvtraindocs"])):
            doc = self.Config["cvtraindocs"][i]
            for j in range(len(doc.nlabs)):
                foldPath = trainPath + "/" + doc.nlabs[j]
                if doc.nlabs[j] not in folds:
                    os.mkdir(foldPath)
                    folds[doc.nlabs[j]] = True
                with open(foldPath + '/' + doc.name, 'w', encoding="utf-8") as file:
                    file.write(doc.lines)
                file.close()
        folds = {}
        os.mkdir(testPath)
        for i in range(len(self.Config["cvtestdocs"])):
            doc = self.Config["cvtestdocs"][i]
            for j in range(len(doc.nlabs)):
                foldPath = testPath + "/" + doc.nlabs[j]
                if doc.nlabs[j] not in folds:
                    os.mkdir(foldPath)
                    folds[doc.nlabs[j]] = True
                with open(foldPath + '/' + doc.name, 'w', encoding="utf-8") as file:
                    file.write(doc.lines)
                file.close()
