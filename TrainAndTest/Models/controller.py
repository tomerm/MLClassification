import os
import logging
from Utils.utils import get_abs_path, correct_path, test_path
from Models.base import BaseModel
from Models.snn import SnnModel
from Models.ltsm import LTSMModel
from Models.cnn import CNNModel
from Models.pac import PacModel
from Models.ridge import RidgeModel
from Models.svc import SVCModel
from Models.perceptron import PerceptronModel
from Models.sgd import SGDModel
from Models.bert import BertModel
import General.settings as settings

logger = logging.getLogger(__name__)

model_types = ["snn", "ltsm", "cnn", "pac", "perceptron", "ridge", "sgd", "svc", "bert"]
model_goals = ["trainandtest", "train", "test", "crossvalidation", "none"]
user_info = {
    "trainandtest": "training and testing",
    "train": "training only",
    "test": "testing only",
    "crossvalidation": "cross-validation",
    "none": "set parameters only"}


def job_model_controller():
    worker = Controller()
    worker.run()


class Controller:
    def __init__(self):
        settings.dynamic_store["modelid"] += 1
        logger.info("=== Model " + str(settings.dynamic_store["modelid"]) + " ===")
        if settings.Config["type_of_execution"] != "none" and settings.Config["type"] not in model_types:
            raise ValueError("Request contains definition of model with wrong type. Stop.")
        if settings.Config["type_of_execution"] not in model_goals:
            raise ValueError("Request doesn't define the goal of the model process. "
                             "It should be one of 'trainAndTest', 'train', 'test', 'crossvalidation' or 'none'. Stop.")
        if settings.Config["type_of_execution"] != "none":
            logger.info("Model type: " + settings.Config["type"].upper() + ", " + user_info[settings.Config["type_of_execution"]])
        else:
            logger.info("Model : " +  user_info[settings.Config["type_of_execution"]])
        if settings.Config["type_of_execution"] == "none":
            return
        if "predefined_categories" not in settings.dynamic_store or "train_docs" not in settings.dynamic_store or "test_docs" not in settings.dynamic_store:
            raise ValueError("Input data isn't loaded. Stop.")
        
    def run(self):
        try:
            self.test_data_size = float(settings.Config["test_data_size"])
        except ValueError:
            self.test_data_size = -1
        if not correct_path(settings.Config, "train_data_path"):
            if settings.Config["type_of_execution"] != "test" or not settings.Config["test_data_path"]:
                raise ValueError("Wrong path to the training set: folder %s doesn't exist."
                                 % get_abs_path(settings.Config, "train_data_path"))
        if not correct_path(settings.Config, "test_data_path"):
            if not (len(settings.Config["test_data_path"]) == 0 and self.test_data_size > 0 and self.test_data_size < 1):
                raise ValueError("Wrong path to the testing set: folder %d doesn't exist."
                                 % get_abs_path(settings.Config, "test_data_path"))
        test_path(settings.Config, "created_model_path", "Wrong path to the models' folder.")
        if not settings.Config["name"]:
            settings.Config["name"] = settings.Config["type"] + str(settings.dynamic_store["modelid"])
        m_path = get_abs_path(settings.Config, "created_model_path", opt="name")
        if settings.Config["type_of_execution"] == "test" and not os.path.isfile(m_path):
            raise ValueError("Wrong path to the tested model.")
        if settings.Config["type_of_execution"] != "test":
            try:
                self.epochs = int(settings.Config["epochs"])
            except ValueError:
                raise ValueError("Wrong quantity of epochs for training.")
            try:
                self.train_batch = int(settings.Config["train_batch"])
            except ValueError:
                raise ValueError("Wrong batch size for training.")
            try:
                self.verbose = int(settings.Config["verbose"])
            except ValueError:
                raise ValueError("Wrong value of 'verbose' flag for training.")
            if settings.Config["save_intermediate_results"] == "True":
                if not settings.Config["intermediate_results_path"] or \
                        not os.path.isdir(get_abs_path(settings.Config, "intermediate_results_path")):
                    raise ValueError("Wrong path to folder with intermediate results.")
        if settings.Config["type_of_execution"] != "train" and settings.Config["customrank"] == "True":
            try:
                self.rank_threshold = float(settings.Config["rank_threshold"])
            except ValueError:
                raise ValueError("Wrong custom rank threshold.")
        if settings.Config["type_of_execution"] == "crossvalidation":
            if settings.Config["save_cross_validations_datasets"] == "True":
                test_path(settings.Config, "cross_validations_datasets_path",
                          "Wrong path to the cross-validation's resulting folder.")
            try:
                cross_validations_total = int(settings.Config["cross_validations_total"])
            except ValueError:
                raise ValueError("Wrong k-fold value.")
        if settings.Config["type"].lower() == "snn":
            model = SnnModel()
        elif settings.Config["type"].lower() == "ltsm":
            model = LTSMModel()
        elif settings.Config["type"].lower() == "cnn":
            model = CNNModel()
        elif settings.Config["type"].lower() == "pac":
            model = PacModel()
        elif settings.Config["type"].lower() == "ridge":
            model = RidgeModel()
        elif settings.Config["type"].lower() == "svc":
            model = SVCModel()
        elif settings.Config["type"] == "perceptron":
            model = PerceptronModel()
        elif settings.Config["type"] == "sgd":
            model = SGDModel()
        elif settings.Config["type"] == "bert":
            model = BertModel()

        model.launch_process() 