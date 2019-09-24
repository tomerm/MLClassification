import gensim
import os
import shutil
import datetime
import logging
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from Models.metrics import ModelMetrics, print_metrics, print_averaged_metrics
from Models.dataPreparation import DataPreparation
from Utils.utils import get_formatted_date, get_abs_path, correct_path
from abc import ABC, abstractmethod
import General.settings as settings
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self):

        self.train_arrays = []
        self.train_labels = []
        self.test_arrays = []
        self.test_labels = []
        self.valArrays = []
        self.valLabels = []
        self.cvDocs = []
        self.predictions = []
        self.metrics = {}
        self.resources = {}
        self.add_val_set = False
        self.validation_data_size = 0
        self.isCV = False
        self.handle_type = ""
        self.use_probabilities = False

        self.epochs = int(settings.Config["epochs"])
        self.verbose = int(settings.Config["verbose"])
        self.cross_validations_total = int(settings.Config["cross_validations_total"])
        if self.verbose != 0:
            self.verbose = 1
        if settings.Config["customrank"] == "True":
            self.rank_threshold = float(settings.Config["rank_threshold"])
        else:
            self.rank_threshold = 0.5
        if self.rank_threshold == 0:
            self.rank_threshold = 0.5
        self.train_batch = int(settings.Config["train_batch"])

    def is_correct_path(self):
        if not correct_path(settings.Config, "binarizer_path"):
            if settings.Config["type_of_execution"] == "test" or settings.Config["binarizer_path"]:
                logger.error("Wrong path to binarizer. Stop.")
                return False
        if not correct_path(settings.Config, "vectorizer_path"):
            if settings.Config["type_of_execution"] == "test" or settings.Config["vectorizer_path"]:
                logger.error("Wrong path to vectorizer. Stop.")
                return False
        return True

    def launch_process(self):
        if settings.Config["type_of_execution"] == "crossvalidation":
            self.isCV = True
            self.launch_crossvalidation()
        elif settings.Config["type_of_execution"] != "test":
            self.model = self.create_model()
            self.train_model()
            if settings.Config["type_of_execution"] != "train":
                self.test_model()
        else:
            self.load_model()
            self.test_model()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass

    def load_w2v_model(self):
        if settings.Config["w2vmodel"] != None:
            logger.warning("W2V model is already loaded.")
            self.w2vModel = settings.Config["w2vmodel"]
            return
        logger.info("Load W2V model... ")
        ds = datetime.datetime.now()
        self.w2vModel = gensim.models.KeyedVectors.load_word2vec_format(get_abs_path(settings.Config, "model_path"))
        de = datetime.datetime.now()
        logger.info("Load W2V model (%s) in %s" % (get_abs_path(settings.Config, "model_path"), get_formatted_date(ds, de)))
        settings.dynamic_store["resources"]["w2v"]["created_model_path"] = get_abs_path(settings.Config, "model_path")
        settings.dynamic_store["resources"]["w2v"]["ndim"] = self.ndim

    def load_nn_model(self):
        return load_model(get_abs_path(settings.Config, "created_model_path", opt="name"))

    def load_skl_model(self):
        return joblib.load(get_abs_path(settings.Config, "created_model_path", opt="name"))

    def train_nn_model(self):
        checkpoints = []
        if self.save_intermediate_results and not self.isCV:
            checkpoint = ModelCheckpoint(get_abs_path(settings.Config, "intermediate_results_path") + "/tempModel.hdf5",
                                         monitor='val_acc', verbose=self.verbose, save_best_only=True, mode='auto')
            checkpoints.append(checkpoint)
        logger.info("Start training...              ")
        ds = datetime.datetime.now()
        self.model.fit(self.train_arrays, self.train_labels, epochs=self.epochs,
                validation_data=(self.valArrays, self.valLabels),
                batch_size=self.train_batch, verbose=self.verbose, callbacks=checkpoints, shuffle=False)
        de = datetime.datetime.now()
        logger.info("Model is trained in %s" %  (get_formatted_date(ds, de)))
        if self.isCV:
            return
        self.model.save(get_abs_path(settings.Config, "created_model_path", opt="name"))
        logger.info("Model evaluation...")
        scores1 = self.model.evaluate(self.test_arrays, self.test_labels, verbose=self.verbose)
        logger.info("Final model accuracy: %.2f%%" % (scores1[1] * 100))
        if self.save_intermediate_results:
            model1 = load_model(get_abs_path(settings.Config, "intermediate_results_path") + "/tempModel.hdf5")
            scores2 = model1.evaluate(self.test_arrays, self.test_labels, verbose=self.verbose)
            logger.info("Last saved model accuracy: %.2f%%" % (scores2[1] * 100))
            if scores1[1] < scores2[1]:
                model = model1
            pref = "The best model "
        else:
            pref = "Model "
        self.model.save(get_abs_path(settings.Config, "created_model_path", opt="name"))
        logger.info(pref + "is saved in %s" % get_abs_path(settings.Config, "created_model_path", opt="name"))

    def train_skl_model(self):
        de = datetime.datetime.now()
        logger.info("Start training...")
        self.model.fit(self.train_arrays, self.train_labels)
        ds = datetime.datetime.now()
        logger.info("Model is trained in %s" % (get_formatted_date(de, ds)))
        if self.isCV:
            return
        joblib.dump(self.model, get_abs_path(settings.Config, "created_model_path", opt="name"))
        logger.info("Model is saved in %s" % get_abs_path(settings.Config, "created_model_path", opt="name"))
        logger.info("Model evaluation...")
        prediction = self.model.predict(self.test_arrays)
        logger.info('Final accuracy is %.2f' % accuracy_score(self.test_labels, prediction))
        de = datetime.datetime.now()
        logger.info("Evaluated in %s" % get_formatted_date(ds, de))

    def test_nn_model(self):
        logger.info("Start testing...")
        logger.info("Rank threshold: %.2f" % self.rank_threshold)
        ds = datetime.datetime.now()
        self.predictions = self.model.predict(self.test_arrays)
        de = datetime.datetime.now()
        logger.info("Test dataset containing %d documents predicted in %s\n" % (len(self.test_arrays), get_formatted_date(ds, de)))
        if self.isCV:
            return
        self.prepare_resources_for_runtime("keras")
        self.get_metrics()
        self.save_results()

    def test_skl_model(self):
        logger.info("Start testing...")
        if self.use_probabilities:
            logger.info("Rank threshold: %.2f" % self.rank_threshold)
        else:
            logger.warning("Model doesn't calculate probabilities.")
        ds = datetime.datetime.now()
        if not self.use_probabilities:
            self.predictions = self.model.predict(self.test_arrays)
        else:
            self.predictions = self.model.predict(self.test_arrays)
        de = datetime.datetime.now()
        logger.info("Test dataset containing %d documents predicted in %s"
              % (self.test_arrays.shape[0], get_formatted_date(ds, de)))
        if self.isCV:
            return
        self.prepare_resources_for_runtime("skl")
        self.get_metrics()
        self.save_results()

    def get_metrics(self):
        logger.info("Calculate metrics...")
        ModelMetrics(self)
        if settings.Config["show_test_results"] == "True":
            print_metrics(self)

    def save_results(self):
        settings.dynamic_store["results"][settings.Config["name"]] = self.predictions
        settings.dynamic_store["metrics"][settings.Config["name"]] = self.metrics
        if self.use_probabilities:
            settings.dynamic_store["ranks"][settings.Config["name"]] = self.rank_threshold
        else:
            settings.dynamic_store["ranks"][settings.Config["name"]] = 1.0

    def prepare_resources_for_runtime(self, type):
        self.resources["created_model_path"] = get_abs_path(settings.Config, "created_model_path", opt="name")
        self.resources["modelType"] = type
        if self.use_probabilities:
            self.resources["rank_threshold"] = self.rank_threshold
        else:
            self.resources["rank_threshold"] = 1.0
        self.save_additions()
        if type == "skl":
            self.resources["handleType"] = "vectorize"
        settings.dynamic_store["resources"]["models"]["Model" + str(settings.dynamic_store["modelid"])] = self.resources

    def save_additions(self):
        pass

    def launch_crossvalidation(self):
        logger.info("Start cross-validation...")
        ds = datetime.datetime.now()
        dp = DataPreparation(self, self.add_val_set)
        psize = len(self.cvDocs) // self.cross_validations_total
        ind = 0
        f1 = 0
        attr_metrics =[]
        for i in range(self.cross_validations_total):
            logger.info("Cross-validation, cycle %d from %d..." % ((i+1), self.cross_validations_total))
            if i == 0:
                settings.dynamic_store["cross_validations_train_docs"] = self.cvDocs[psize:]
                settings.dynamic_store["cross_validations_test_docs"] = self.cvDocs[:psize]
            elif i == self.cross_validations_total - 1:
                settings.dynamic_store["cross_validations_train_docs"] = self.cvDocs[:ind]
                settings.dynamic_store["cross_validations_test_docs"] = self.cvDocs[ind:]
            else:
                settings.dynamic_store["cross_validations_train_docs"] = self.cvDocs[:ind] + self.cvDocs[ind+psize:]
                settings.dynamic_store["cross_validations_test_docs"] = self.cvDocs[ind:ind+psize]
            ind += psize
            dp.get_vectors(self.handle_type)
            self.model = self.create_model()
            self.train_model()
            self.test_model()
            ModelMetrics(self)
            attr_metrics.append(self.metrics)
            cycle_f1 = self.metrics["all"]["f1"]
            logger.info("Resulting F1-Measure: %f\n" % cycle_f1)
            if cycle_f1 > f1:
                if settings.Config["save_cross_validations_datasets"]:
                    self.save_data_sets()
                f1 = cycle_f1
        de = datetime.datetime.now()
        logger.info("Cross-validation is done in %s" % get_formatted_date(ds, de))
        print_averaged_metrics(attr_metrics)
        logger.info("The best result is %f"%(f1))
        logger.info("Corresponding data sets are saved in the folder %s"
               % get_abs_path(settings.Config, "cross_validations_datasets_path"))


    def save_data_sets(self):
        root = get_abs_path(settings.Config, "cross_validations_datasets_path")
        shutil.rmtree(root)
        os.mkdir(root)
        train_data_path = root + "/train"
        test_data_path = root + "/test"
        folds = {}
        os.mkdir(train_data_path)
        for doc in settings.dynamic_store["cross_validations_train_docs"]:
            for nlab in doc.nlabs:
                fold_path = train_data_path + "/" + nlab
                if nlab not in folds:
                    os.mkdir(fold_path)
                    folds[nlab] = True
                with open(fold_path + '/' + doc.name, 'w', encoding="utf-8") as file:
                    file.write(doc.lines)
                file.close()
        folds = {}
        os.mkdir(test_data_path)
        for doc in settings.dynamic_store["cross_validations_test_docs"]:
            for nlab in doc.nlabs:
                fold_path = test_data_path + "/" + nlab
                if nlab not in folds:
                    os.mkdir(fold_path)
                    folds[nlab] = True
                with open(fold_path + '/' + doc.name, 'w', encoding="utf-8") as file:
                    file.write(doc.lines)
                file.close()
