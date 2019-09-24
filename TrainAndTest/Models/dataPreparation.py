import numpy
import pickle
import datetime
import random
import logging
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from Utils.utils import get_formatted_date, get_abs_path, arabic_charset
import General.settings as settings

logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, model, add_val_set):
        self.model = model
        self.add_val_set = add_val_set
        self.maxWords = 300000
        self.ndim = int(settings.Config["vectors_dimension"])
        if add_val_set:
            self.validation_data_size = float(settings.Config["validation_data_size"])
            if settings.Config["type_of_execution"] != "test":
                logger.info("Validation: %.2f" % (self.validation_data_size))
        else:
            self.validation_data_size = 0
        if settings.Config["type_of_execution"] == "crossvalidation":
            self.model.cvDocs = settings.dynamic_store["train_docs"] + settings.dynamic_store["test_docs"]
            random.shuffle(self.model.cvDocs)
            self.key_train = "cross_validations_train_docs"
            self.key_test = "cross_validations_test_docs"
        else:
            self.key_train = "train_docs"
            self.key_test = "test_docs"

    def get_vectors(self, type):
        if type == "wordVectorsSum":
            self.get_word_vectors_sum()
        elif type == "wordVectorsMatrix":
            self.get_word_vectors_matrix()
        elif type == "charVectors":
            self.get_char_vectors()
        elif type == "vectorize":
            self.get_data_for_sklearn_classifiers()
        else:
            pass

    def get_word_vectors_sum(self):
        self.nfWords = 0
        self.sdict = dict()
        self.tmpCount = 0
        if settings.Config["type_of_execution"] != "test":
            ds = datetime.datetime.now()
            self.model.train_arrays = numpy.concatenate([self.get_docs_array(x.words, 'Train')
                                                        for x in settings.dynamic_store[self.key_train]])
            self.model.train_labels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(settings.dynamic_store["predefined_categories"]))
                                                        for x in settings.dynamic_store[self.key_train]])

            if self.add_val_set:
                ind = int(len(self.model.train_arrays) * (1 - self.validation_data_size))
                self.model.valArrays = self.model.train_arrays[ind:]
                self.model.valLabels = self.model.train_labels[ind:]
                self.model.train_arrays = self.model.train_arrays[:ind]
                self.model.train_labels = self.model.train_labels[:ind]

                de = datetime.datetime.now()
                logger.info("Prepare train and validation data in %s" % (get_formatted_date(ds, de)))
            else:
                de = datetime.datetime.now()
                logger.info("Prepare train data in %s" % (get_formatted_date(ds, de)))

        self.tmpCount = 0
        ds = datetime.datetime.now()
        self.model.test_arrays = numpy.concatenate([self.get_docs_array(x.words, "Test")
                                                   for x in settings.dynamic_store[self.key_test]])
        self.model.test_labels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(settings.dynamic_store["predefined_categories"]))
                                                   for x in settings.dynamic_store[self.key_test]])
        if self.model.isCV:
            return
        de = datetime.datetime.now()
        logger.info("Prepare test data in %s" % (get_formatted_date(ds, de)))
        logger.info("Unique words in all documents: %d" % (len(self.sdict)))
        logger.info("Words not found in the w2v vocabulary: %d" % (self.nfWords))

    def get_docs_array(self, tokens, data_type):
        self.tmpCount += 1
        if self.tmpCount != 0 and self.tmpCount % 1000 == 0:
            logger.info(data_type + ": prepare " + str(self.tmpCount))
        vec = numpy.zeros(self.ndim).reshape((1, self.ndim))
        count = 0.
        for word in tokens:
            if word not in self.sdict:
                self.sdict[word] = 1
            else:
                self.sdict[word] = self.sdict[word] + 1
        for word in tokens:
            try:
                vec += self.model.w2vModel[word].reshape((1, self.ndim))
                count += 1.
            except KeyError:
                if self.sdict[word] == 1:
                    self.nfWords += 1
                continue
        if count != 0:
            vec /= count
        return vec

    def get_word_vectors_matrix(self):
        tokenizer = None
        ds = datetime.datetime.now()
        if settings.Config["type_of_execution"] != "test":
            tokenizer = Tokenizer(num_words=self.maxWords)
            train_texts = []
            for t in settings.dynamic_store[self.key_train]:
                train_texts.append(t.lines)
            tokenizer.fit_on_texts(train_texts)
            if not self.model.isCV:
                with open(get_abs_path(settings.Config, "indexer_path"), 'wb') as handle:
                    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                if settings.Config["max_doc_len"] > settings.Config["max_seq_len"]:
                    logger.warning("Most of documents from training set have less then %d tokens. Longer documents will be truncated."%(
                        settings.Config["max_seq_len"]))
            self.model.train_arrays = pad_sequences(tokenizer.texts_to_sequences(train_texts),
                                                    maxlen=settings.Config["max_seq_len"])
            self.model.train_labels = numpy.concatenate([numpy.array(x.labels).
                            reshape(1, len(settings.dynamic_store["predefined_categories"]))
                                                        for x in settings.dynamic_store[self.key_train]])
            if self.add_val_set:
                ind = int(len(self.model.train_arrays) * (1 - self.validation_data_size))
                self.model.valArrays = self.model.train_arrays[ind:]
                self.model.valLabels = self.model.train_labels[ind:]
                self.model.train_arrays = self.model.train_arrays[:ind]
                self.model.train_labels = self.model.train_labels[:ind]
        if tokenizer == None:
            with open(get_abs_path(settings.Config, "indexer_path"), 'rb') as handle:
                tokenizer = pickle.load(handle)
            handle.close()
        test_texts = []
        for t in settings.dynamic_store[self.key_test]:
            test_texts.append(t.lines)
        self.model.test_arrays = pad_sequences(tokenizer.texts_to_sequences(test_texts),
                                              maxlen=settings.Config["max_seq_len"])
        self.model.test_labels = numpy.concatenate([numpy.array(x.labels).
                            reshape(1, len(settings.dynamic_store["predefined_categories"]))
                                                   for x in settings.dynamic_store[self.key_test]])
        embedding_matrix = numpy.zeros((self.maxWords, self.ndim))
        word_index = tokenizer.word_index
        nf = 0
        for word, i in word_index.items():
            if i < self.maxWords:
                try:
                    embedding_vector = self.model.w2vModel[word]
                except KeyError:
                    nf += 1
                    continue
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        self.model.embMatrix =  embedding_matrix
        self.model.maxWords = self.maxWords
        if self.model.isCV:
            return
        de = datetime.datetime.now()
        logger.info('Found %s unique tokens.' % len(tokenizer.word_index))
        logger.info('Tokens not found in W2V vocabulary: %d'%nf)
        logger.info("All data prepared and embedding matrix built in %s"%(get_formatted_date(ds, de)))
        return embedding_matrix, self.maxWords

    def get_char_vectors(self):
        ds = datetime.datetime.now()
        if settings.Config["type_of_execution"] != "test":
            self.model.train_arrays = numpy.concatenate([self.string_to_indexes(" ".join(x.words))
                                            for x in settings.dynamic_store[self.key_train]])
            self.model.train_labels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(settings.dynamic_store["predefined_categories"]))
                                                        for x in settings.dynamic_store[self.key_train]])
            if self.add_val_set:
                ind = int(len(self.model.train_arrays) * (1 - self.validation_data_size))
                self.model.valArrays = self.model.train_arrays[ind:]
                self.model.valLabels = self.model.train_labels[ind:]
                self.model.train_arrays = self.model.train_arrays[:ind]
                self.model.train_labels = self.model.train_labels[:ind]
        self.model.test_arrays = numpy.concatenate([self.string_to_indexes(" ".join(x.words))
                                            for x in settings.dynamic_store[self.key_test]])
        self.model.test_labels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(settings.dynamic_store["predefined_categories"]))
                                                   for x in settings.dynamic_store[self.key_test]])
        if self.model.isCV:
            return
        de = datetime.datetime.now()
        logger.info("Prepare all data in %s" % (get_formatted_date(ds, de)))

    def string_to_indexes(self, str):
        ch_dict = arabic_charset()
        str2ind = numpy.zeros(settings.Config["max_chars_seq_len"], dtype='int64')
        str_len = min(len(str), settings.Config["max_chars_seq_len"])
        for i in range(1, str_len + 1):
            c = str[-i]
            if c in ch_dict:
                str2ind[i - 1] = ch_dict[c]
        return str2ind.reshape(1, settings.Config["max_chars_seq_len"])

    def get_data_for_sklearn_classifiers(self):
        mlb = None
        ds = datetime.datetime.now()
        if settings.Config["type_of_execution"] != "test":
            nm_cats = [""] * len(settings.dynamic_store["predefined_categories"])
            for k in list(settings.dynamic_store["predefined_categories"].keys()):
                nm_cats[settings.dynamic_store["predefined_categories"][k]] = k
            mlb = MultiLabelBinarizer(classes=nm_cats)
            wev = (TfidfVectorizer(ngram_range=(1, 3), max_df=0.50)
                   .fit([x.lines for x in settings.dynamic_store[self.key_train]],
                                                                [x.nlabs for x in settings.dynamic_store[self.key_train]]))
            self.model.train_arrays = wev.transform([x.lines for x in settings.dynamic_store[self.key_train]])
            self.model.train_labels = mlb.fit_transform([x.nlabs for x in settings.dynamic_store[self.key_train]])
            if not self.model.isCV:
                with open(get_abs_path(settings.Config, "binarizer_path"), 'wb') as handle:
                    pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                with open(get_abs_path(settings.Config, "vectorizer_path"), 'wb') as handle:
                    pickle.dump(wev, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
        if not mlb:
            with open(get_abs_path(settings.Config, "binarizer_path"), 'rb') as handle:
                mlb = pickle.load(handle)
            handle.close()
            with open(get_abs_path(settings.Config, "vectorizer_path"), 'rb') as handle:
                wev = pickle.load(handle)
            handle.close()
        self.model.test_arrays = wev.transform([x.lines for x in settings.dynamic_store[self.key_test]])
        self.model.test_labels = mlb.fit_transform([x.nlabs for x in settings.dynamic_store[self.key_test]])
        de = datetime.datetime.now()
        logger.info("Prepare all data in %s" % (get_formatted_date(ds, de)))
