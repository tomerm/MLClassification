from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import AlphaDropout
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Utils.utils import arabic_charset

from keras import backend as K
import tensorflow as tf
import General.settings as settings
import logging

logger = logging.getLogger(__name__)

class CNNModel(BaseModel):
    def __init__(self):
        super().__init__()
        try:
            self.validation_data_size = float(settings.Config["validation_data_size"])
        except ValueError:
            self.validation_data_size = 0
        if self.validation_data_size <= 0 or self.validation_data_size >= 1:
            raise ValueError("Wrong size of validation data set. Stop.")
        self.add_val_set = True
        self.handle_type = "charVectors"
        self.save_intermediate_results = settings.Config["save_intermediate_results"] == "True"
        self.use_probabilitiesh = True
        if settings.Config["type_of_execution"] != "crossvalidation":
            self.prepare_data()

    def prepare_data(self):
        logger.info("Start data preparation...")
        dp = DataPreparation(self, self.add_val_set)
        dp.get_char_vectors()

    def create_model(self):
        embedding_size = 128
        max_seq_length = settings.Config["max_chars_seq_len"]
        conv_layers_data = [[256, 10], [256, 7], [256, 5], [256, 3]]
        dropout_p = 0.1
        optimizer = 'adam'
        inputs = Input(shape=(max_seq_length,), dtype='int64')
        x = Embedding(len(arabic_charset()) + 1, embedding_size, input_length=max_seq_length)(inputs)
        convolution_output = []
        for num_filters, filter_width in conv_layers_data:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_width,
                                 activation='tanh')(x)
            pool = GlobalMaxPooling1D()(conv)
            convolution_output.append(pool)
        x = Concatenate()(convolution_output)
        x = Dense(1024, activation='selu', kernel_initializer='lecun_normal')(x)
        x = AlphaDropout(dropout_p)(x)
        x = Dense(1024, activation='selu', kernel_initializer='lecun_normal')(x)
        x = AlphaDropout(dropout_p)(x)
        predictions = Dense(len(settings.dynamic_store["predefined_categories"]), activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_model(self):
        self.model = self.load_nn_model()

    def train_model(self):
        """
        cf = tf.ConfigProto(inter_op_parallelism_threads=5)
        session = tf.Session(config=cf)
        K.set_session(session)
        """
        self.train_nn_model()

    def test_model(self):
        self.test_nn_model()

    def save_additions(self):
        self.resources["handleType"] = "charVectors"
