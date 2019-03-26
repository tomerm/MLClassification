import os
from Utils.utils import fullPath, updateParams
from Models.snn import SnnModel
from Models.ltsm import LTSMModel
from Models.cnn import CNNModel
from Models.pac import PacModel
from Models.ridge import RidgeModel
from Models.svc import SVCModel
from Models.perceptron import PerceptronModel
from Models.sgd import SGDModel
from Models.bert import BertModel

modelTypes = ["snn", "ltsm", "cnn", "pac", "perceptron", "ridge", "sgd", "svc", "bert"]
modelGoals = ["trainandtest", "train", "test", "crossvalidation", "none"]
userInfo = {
    "trainandtest": "training and testing",
    "train": "training only",
    "test": "testing only",
    "crossvalidation": "cross-validation",
    "none": "set parameters only"}

class ModelController:
    def __init__(self, Config, DefConfig, kwargs):
        Config["modelid"] += 1
        print ("=== Model " + str(Config["modelid"]) + " ===")
        updateParams(Config, DefConfig, kwargs)
        Config["type"] = Config["type"].lower()
        Config["runfor"] = Config["runfor"].lower()
        if Config["runfor"] != "none" and Config["type"] not in modelTypes:
            print("Request contains definition of model with wrong type. Stop.")
            Config["error"] = True
            return
        if Config["runfor"] not in modelGoals:
            print("Request doesn't define the goal of the model process")
            print("It should be one of 'trainAndTest', 'train', 'test', 'crossValidation' or 'none'. Stop.")
            Config["error"] = True
            return
        if Config["runfor"] != "none":
            print ("Model type: " + Config["type"].upper() + ", " + userInfo[Config["runfor"]])
        else:
            print("Model : " +  userInfo[Config["runfor"]])
        if Config["runfor"] == "none":
            return
        self.Config = Config
        self.DefConfig = DefConfig;
        if "cats" not in Config or "traindocs" not in Config or "testdocs" not in Config:
            print ("Input data isn't loaded. Stop.")
            Config["error"] = True
            return
        stop = False
        try:
            self.testSize = float(Config["testsize"])
        except ValueError:
            self.testSize = -1
        if len(Config["trainpath"]) == 0 or not os.path.isdir(fullPath(Config, "trainpath")):
            if Config["runfor"] != "test" or len(Config["testpath"]) == 0:
                print ("Wrong path to the training set: folder %s doesn't exist."%(fullPath(Config, "trainpath")))
                stop = True
        if len(Config["testpath"]) == 0 or not os.path.isdir(fullPath(Config, "testpath")):
            if not (len(Config["testpath"]) == 0 and self.testSize > 0 and self.testSize < 1):
                print ("Wrong path to the testing set: folder %d doesn't exist."%(fullPath(Config, "testpath")))
                stop = True
        if len(Config["modelpath"]) == 0 or not os.path.isdir(fullPath(Config, "modelpath")):
            print ("Wrong path to the models' folder.")
            stop = True
        if len(Config["name"]) == 0:
            Config["name"] = Config["type"] + str(Config["modelid"])
        mPath = fullPath(Config, "modelpath", opt="name")
        if Config["runfor"] == "test" and not os.path.isfile(mPath):
            print ("Wrong path to the tested model.")
            stop = True
        if Config["runfor"] != "test":
            try:
                epochs = int(Config["epochs"])
            except ValueError:
                print ("Wrong quantity of epochs for training.")
                stop = True
            try:
                self.trainBatch = int(Config["trainbatch"])
            except ValueError:
                print ("Wrong batch size for training.")
                stop = True
            try:
                self.verbose = int(Config["verbose"])
            except ValueError:
                print ("Wrong value of 'verbose' flag for training.")
                stop = True
            if Config["tempsave"] == "yes":
                if len(Config["temppath"]) == 0 or not os.path.isdir(fullPath(Config, "temppath")):
                    print ("Wrong path to folder with intermediate results.")
                    stop = True
        if Config["runfor"].lower() != "train":
            if Config["modelinfo"] == "yes":
                if len(Config["infopath"]) == 0 or not os.path.isdir(fullPath(Config, "infopath")):
                    print ("Wrong path to folder containing model info.")
                    stop = True
        if Config["runfor"] == "crossvalidation":
            try:
                kfold = int(Config["kfold"])
            except ValueError:
                print ("Wrong k-fold value.")
                stop = True
            try:
                pSize = float(Config["psize"])
            except ValueError:
                print ("Wrong pSize value.")
                stop = True
        if stop:
            print ("Stop.")
            Config["error"] = True
            return
        if Config["type"].lower() == "snn":
            SnnModel(Config)
        elif Config["type"].lower() == "ltsm":
            LTSMModel(Config)
        elif Config["type"].lower() == "cnn":
            CNNModel(Config)
        elif Config["type"].lower() == "pac":
            PacModel(Config)
        elif Config["type"].lower() == "ridge":
            RidgeModel(Config)
        elif Config["type"].lower() == "svc":
            SVCModel(Config)
        elif Config["type"] == "perceptron":
            PerceptronModel(Config)
        elif Config["type"] == "sgd":
            SGDModel(Config)
        elif Config["type"] == "bert":
            BertModel(Config)