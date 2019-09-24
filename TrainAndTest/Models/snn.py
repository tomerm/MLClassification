from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Utils.utils import test_path
import General.settings as settings
import logging

logger = logging.getLogger(__name__)

class SnnModel(BaseModel):
    def __init__(self):
        super().__init__()
        if settings.Config["w2vmodel"] == None:
            test_path(settings.Config, "model_path", "Wrong path to W2V model. Stop.")
        try:
            self.validation_data_size = float(settings.Config["validation_data_size"])
        except ValueError:
            self.validation_data_size = 0
        if self.validation_data_size <= 0 or self.validation_data_size >= 1:
            raise ValueError("Wrong size of validation data set. Stop.")
        try:
            self.ndim = int(settings.Config["vectors_dimension"])
        except ValueError:
            raise ValueError("Wrong size of vectors' dimentions. Stop.")
        self.add_val_set = True
        self.handle_type = "wordVectorsSum"
        self.save_intermediate_results = settings.Config["save_intermediate_results"] == "True"
        self.use_probabilities = True
        self.w2vModel = None
        self.load_w2v_model()
        if settings.Config["type_of_execution"] != "crossvalidation":
            self.prepare_data()

    def prepare_data(self):
        logger.info("Start data preparation...")
        dp = DataPreparation(self, self.add_val_set)
        dp.get_word_vectors_sum()

    def create_model(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=self.ndim))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(settings.dynamic_store["predefined_categories"]), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def load_model(self):
        self.model = self.load_nn_model()

    def train_model(self):
        self.train_nn_model()

    def test_model(self):
        self.test_nn_model()

    def save_additions(self):
        self.resources["w2v"] = "True"
        self.resources["handleType"] = "wordVectorsSum"
