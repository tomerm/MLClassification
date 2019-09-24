import os
import numpy
import datetime
import gensim
from gensim.models.word2vec import Word2Vec
from Utils.utils import get_abs_path, get_formatted_date, test_path
from WordEmbedding.epochlogger import EpochLogger
import General.settings as settings
import logging

logger = logging.getLogger(__name__)

def job_word_embedding():
    worker = Embedding()
    worker.run()


class Embedding:
    def __init__(self):
        logger.info("=== Word Embedding ===")
        test_path(settings.Config, "model_path", "Wrong path to W2V model. Word Embedding can't be done.")
        if settings.Config["need_create_model"] != "True":
            return
        test_path(settings.Config, "data_corpus_path", "Wrong corpus path. W2V model can't be created.")
        try:
            self.epochs = int(settings.Config["epochs_total"])
        except ValueError:
            raise ValueError("Wrong quantity of epochs for training. W2V model can't be created.")
        try:
            self.ndim = int(settings.Config["vectors_dimension"])
        except ValueError:
            raise ValueError("Wrong size of resulting vectors. W2V model can't be created.")
        #self.createW2VModel()

    def run(self): #create W2V Model
        sentences = []
        count = 0
        logger.info("Start to create W2V model...")
        logger.info("Get input data...")
        ds = datetime.datetime.now()
        with open(get_abs_path(settings.Config, "data_corpus_path"), 'r', encoding='UTF-8') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                count += 1
                words = [w for w in line.strip().split()]
                sentences.append(words)
        f.close()
        de = datetime.datetime.now()
        logger.info("Got %d lines from file %s in %s"
              % (count, get_abs_path(settings.Config, "data_corpus_path"), get_formatted_date(ds, de)))
        numpy.random.shuffle(sentences)

        epoch_logger = EpochLogger(self.epochs)
        w2v = Word2Vec(size=self.ndim, window=10, min_count=3, workers=10)
        ds = datetime.datetime.now()
        logger.info("Build vocabulary...")
        w2v.build_vocab(sentences)
        de = datetime.datetime.now()
        logger.info("Vocabulary is built in %s" % (get_formatted_date(ds, de)))
        logger.info("Train model...")
        ds = datetime.datetime.now()
        w2v.train(sentences, epochs=int(settings.Config["epochs_total"]), total_examples=len(sentences), callbacks=[epoch_logger])
        de = datetime.datetime.now()
        logger.info("W2V model is completed in %s" % (get_formatted_date(ds, de)))

        created_model_path = get_abs_path(settings.Config, "model_path")
        if settings.Config["include_current_time_in_model_name"]:
            modelName = os.path.basename(created_model_path)
            dt = "-" + datetime.datetime.now().strftime("%Y-%b-%d-%H%M%S")
            pInd = modelName.rfind(".")
            if pInd > 0:
                modelName = modelName[:pInd] + dt + modelName[pInd:]
            else:
                modelName += dt
        finalPath = os.path.dirname(created_model_path) + "/" + modelName
        ds = datetime.datetime.now()
        w2v.wv.save_word2vec_format(finalPath, binary=False)
        de = datetime.datetime.now()
        logger.info("W2V model %s is saved in the text format in %s\n" % (finalPath, get_formatted_date(ds, de)))

