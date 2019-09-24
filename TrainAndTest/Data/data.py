import os
import glob
import random
import gensim
import math
import statistics
import datetime
import nltk
import subprocess
import logging
from subprocess import run, PIPE
from nltk.corpus import stopwords
from collections import namedtuple
from Data.plots import *
from Preprocess.utils import ArabicNormalizer
from Utils.utils import get_abs_path, get_formatted_date, test_path
import stanfordnlp
import General.settings as settings

LabeledDocument = namedtuple('LabeledDocument', 'lines words labels nlabs q_labs name')
stop_words = set(stopwords.words('arabic'))
logger = logging.getLogger(__name__)

def job_data_loader():
    worker = DataLoader()
    worker.run()


class DataLoader:
    def __init__(self):
        logger.info("=== Loading data ===")
        self.exclude_categories = settings.Config["exclude_categories"].split(",")
        self.sz = 0
        self.splitTrain = False
        self.topBound = 0.9
        self.charsTopBound = 0.6
        self.run()

    def run(self):
        test_path(settings.Config, "train_data_path", "Wrong path to training set. Data can't be loaded.")
        if settings.Config["test_data_path"]:
            test_path(settings.Config, "test_data_path", "Wrong path to testing set. Data can't be loaded.")
        else:
            self.splitTrain = True
            try:
                self.sz = float(settings.Config["test_data_size"])
            except ValueError:
                self.sz = 0
            if not settings.Config["test_data_path"] and (self.sz <= 0 or self.sz >= 1):
                raise ValueError("Wrong size of testing set. Data can't be loaded.")
        if settings.Config["enable_tokenization"] == "True":
            if settings.Config["language_tokenization"] == "True":
                logger.info("use single_doc_lang_tokenization")
                if settings.Config["use_java"] == "True":
                    test_path(settings.Config, 'single_doc_lang_tokenization_lib_path',
                              "Wrong path to the tagger's jar. Preprocessing can't be done.")
                    lib_path = get_abs_path(settings.Config, 'single_doc_lang_tokenization_lib_path')
                    command_line = 'java -Xmx2g -jar ' + lib_path + ' "' + settings.Config["exclude_positions"] + '"'
                    self.jar = subprocess.Popen(command_line, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, shell=True, encoding="utf-8")
                else:
                    self.nlp_tokenizer = stanfordnlp.Pipeline(lang="ar", processors='tokenize,mwt', use_gpu=True)
            if settings.Config["stop_words"] == "True":
                self.stop_words = set(nltk.corpus.stopwords.words('arabic'))
            else:
                self.stop_words = set()
            if settings.Config["normalization"] == "True":
                self.normalizer = ArabicNormalizer()
        if settings.Config["load_w2v_model"] == "True":
            if not settings.Config["model_path"] or not os.path.isfile(get_abs_path(settings.Config, "model_path")):
                raise ValueError("Wrong path to W2V model. Stop.")
            try:
                self.ndim = int(settings.Config["vectors_dimension"])
            except ValueError:
                raise ValueError("Wrong size of vectors' dimentions. Stop.")
            settings.dynamic_store["resources"]["w2v"]["created_model_path"] = get_abs_path(settings.Config, "model_path")
            settings.dynamic_store["resources"]["w2v"]["ndim"] = self.ndim
            self.load_w2v_model()
        else:
            settings.Config["w2vmodel"] = None

        self.load_data()
        if settings.Config["analysis"] == "True":
            self.analysis()

    def load_data(self):
        if settings.Config["enable_tokenization"] == "True":
            logger.info("Start loading and preprocessing of data...")
        else:
            logger.info("Start loading data...")
        ds = datetime.datetime.now()
        settings.dynamic_store["predefined_categories"] = self.get_categories(get_abs_path(settings.Config, "train_data_path"))
        train_docs = self.get_data_docs(get_abs_path(settings.Config, "train_data_path"))
        if not self.splitTrain:
            test_docs = self.get_data_docs(get_abs_path(settings.Config, "test_data_path"))
        else:
            ind = int(len(train_docs) * (1 - self.sz))
            random.shuffle(train_docs)
            test_docs = train_docs[ind:]
            train_docs = train_docs[:ind]
        de = datetime.datetime.now()
        settings.dynamic_store["train_docs"] = random.sample(train_docs, len(train_docs))
        settings.dynamic_store["test_docs"] = random.sample(test_docs, len(test_docs))
        self.get_max_seq_len()
        self.get_max_chars_length()
        if settings.Config["enable_tokenization"] == "True" \
                and settings.Config["language_tokenization"] == "True" \
                and settings.Config["use_java"] == "True":
            self.jar.stdin.write('!!! STOP !!!\n')
            self.jar.stdin.flush()
        logger.info("Input data loaded in %s"%(get_formatted_date(ds, de)))
        logger.info("Training set contains %d documents."%(len(settings.dynamic_store["train_docs"])))
        logger.info("Testing set contains %d documents."%(len(settings.dynamic_store["test_docs"])))
        logger.info("Documents belong to %d categories."%(len(settings.dynamic_store["predefined_categories"])))

    def get_categories(self, path):
        cats = dict()
        n_cats = 0
        os.chdir(path)
        for f in glob.glob("*"):
            if os.path.isdir(f) and not f in self.exclude_categories:
                cats[f] = n_cats
                n_cats += 1
        return cats

    def get_data_docs(self, path):
        files = dict()
        f_in_cats = [0] * len(settings.dynamic_store["predefined_categories"])
        n_files = 0
        act_files = 0
        cur_category = 0
        docs = []
        os.chdir(path)
        for f in glob.glob("*"):
            if f in self.exclude_categories:
                continue
            cur_category = settings.dynamic_store["predefined_categories"][f]
            cat_path = path + "/" + f
            os.chdir(cat_path)
            for fc in glob.glob("*"):
                act_files += 1
                if fc not in files:
                    n_files += 1
                    doc_cont = ''
                    with open(fc, 'r', encoding='UTF-8') as tc:
                        for line in tc:
                            doc_cont += line.strip() + " "
                    tc.close()
                    if settings.Config["enable_tokenization"] == "True":
                        doc_cont = self.preprocess(doc_cont)
                    words = doc_cont.strip().split()
                    labels = [0] * len(settings.dynamic_store["predefined_categories"])
                    labels[cur_category] = 1
                    nlabs = [f]
                    files[fc] = LabeledDocument(doc_cont.strip(), words, labels, nlabs, [1], fc)
                else:
                    files[fc].labels[cur_category] = 1
                    files[fc].nlabs.append(f)
                    files[fc].q_labs[0] += 1
                f_in_cats[cur_category] += 1
        for k, val in files.items():
            docs.append(val)
        return docs

    def get_max_seq_len(self):
        max_doc_len = max(len(x.words) for x in settings.dynamic_store["train_docs"])
        max_len = math.ceil(max_doc_len / 100) * 100 + 100
        input_length_list = []
        for i in range(100, max_len, 100):
            input_length_list.append(i)
        input_length_dict = {x: 0 for x in input_length_list}
        for t in settings.dynamic_store["train_docs"]:
            cur_len = len(t.words)
            dic_len = max_len
            for ln in input_length_dict:
                if cur_len < ln:
                    dic_len = ln
                    break
            input_length_dict[dic_len] = input_length_dict[dic_len] + 1
        input_length_dict_percentage = {}
        for k, v in input_length_dict.items():
            v = v / len(settings.dynamic_store["train_docs"])
            input_length_dict_percentage[k] = v
        max_seq_length = 0
        accumulate_percentage = 0
        for length, percentage in input_length_dict_percentage.items():
            accumulate_percentage += percentage
            if accumulate_percentage > self.topBound:
                max_seq_length = length
                break
        settings.Config["max_doc_len"] = max_doc_len
        settings.Config["max_seq_len"] = max_seq_length

    def get_max_chars_length(self):
        max_doc_len = max(len(x.lines) for x in settings.dynamic_store["train_docs"])
        max_len = math.ceil(max_doc_len / 100) * 100 + 100
        input_length_list = []
        for i in range(100, max_len, 100):
            input_length_list.append(i)
        input_length_dict = {x: 0 for x in input_length_list}
        for t in settings.dynamic_store["train_docs"]:
            cur_len = len(t.lines)
            dic_len = max_len
            for ln in input_length_dict:
                if cur_len < ln:
                    dic_len = ln
                    break
            input_length_dict[dic_len] = input_length_dict[dic_len] + 1
        input_length_dict_percentage = {}
        for k, v in input_length_dict.items():
            v = v / len(settings.dynamic_store["train_docs"])
            input_length_dict_percentage[k] = v
        max_seq_length = 0
        accumulate_percentage = 0
        for length, percentage in input_length_dict_percentage.items():
            accumulate_percentage += percentage
            if accumulate_percentage > self.charsTopBound:
                max_seq_length = length
                break
        settings.Config["max_chars_doc_len"] = max_doc_len
        settings.Config["max_chars_seq_len"] = min(max_seq_length, 512)

    def analysis(self):
        max_doc_len = max(len(x.words) for x in settings.dynamic_store["train_docs"])
        min_doc_len = min(len(x.words) for x in settings.dynamic_store["train_docs"])
        avrg_doc_len = round(statistics.mean(len(x.words) for x in settings.dynamic_store["train_docs"]), 2)
        max_chars_doc_len = max(len(x.lines) for x in settings.dynamic_store["train_docs"])
        min_chars_doc_len = min(len(x.lines) for x in settings.dynamic_store["train_docs"])
        avrg_chars_doc_len = round(statistics.mean(len(x.lines) for x in settings.dynamic_store["train_docs"]), 2)
        dls, q_labs = self.get_label_sets()
        f_in_cats1 = self.files_by_category(settings.dynamic_store["train_docs"], settings.dynamic_store["predefined_categories"])
        f_in_cats2 = self.files_by_category(settings.dynamic_store["test_docs"], settings.dynamic_store["predefined_categories"])
        logger.info("Length of train documents: maximum: %d, minimum: %d, average: %d" % (
                        max_chars_doc_len, min_chars_doc_len, avrg_chars_doc_len))
        logger.info("Tokens in train documents: maximum: %d, minimum: %d, average: %d" % (max_doc_len, min_doc_len, avrg_doc_len))
        logger.info("Length of %.1f%% documents from training set is less then %d tokens." % (
            self.topBound * 100, settings.Config["max_seq_len"]))
        if settings.Config["show_plots"] == "True":
            showDocsByLength(settings.dynamic_store);
        logger.info("Documents for training in category : maximum: %d, minimum: %d, avegare: %d" % (
        max(f_in_cats1), min(f_in_cats1), round(statistics.mean(f_in_cats1), 2)))
        logger.info("Documents for testing  in category : maximum: %d, minimum: %d, avegare: %d" % (
        max(f_in_cats2), min(f_in_cats2), round(statistics.mean(f_in_cats2), 2)))
        if settings.Config["show_plots"] == "True":
            showDocsByLabs(settings.dynamic_store)
        logger.info("Training dataset properties:")
        logger.info("  Distinct Label Set: %d" % (dls))
        logger.info("  Proportion of Distinct Label Set: %.4f" % (dls / len(settings.dynamic_store["train_docs"])))
        logger.info("  Label Cardinality: %.4f" % (q_labs / len(settings.dynamic_store["train_docs"])))
        logger.info("  Label Density: %.4f" % (
                q_labs / len(settings.dynamic_store["train_docs"]) / len(settings.dynamic_store["predefined_categories"])))

    def get_label_sets(self):
        labels = [x[2] for x in settings.dynamic_store["train_docs"]]
        results = [labels[0]]
        q_labs = 0
        for label in labels:
            q_labs += sum(label)
            count = 0
            for res in results:
                for k in range(len(settings.dynamic_store["predefined_categories"])):
                    if label[k] != res[k]:
                        count += 1
                        break
            if count == len(results):
                results.append(label)
        return len(results), q_labs

    def files_by_category(docs, cats):
        f_in_cats = [0] * len(cats)
        for doc in docs:
            for j in range(len(cats)):
                if doc.labels[j] == 1:
                    f_in_cats[j] += 1
        return f_in_cats

    def load_w2v_model(self):
        logger.info("Load W2V model...")
        ds = datetime.datetime.now()
        settings.Config["w2vmodel"] = \
            gensim.models.KeyedVectors.load_word2vec_format(get_abs_path(settings.Config, "model_path"))
        de = datetime.datetime.now()
        logger.info("Load W2V model (%s) in %s" % (get_abs_path(settings.Config, "model_path"), get_formatted_date(ds, de)))

    def preprocess(self, text):
        if settings.Config["language_tokenization"] == "True":
            if settings.Config["use_java"] == "True":
                self.jar.stdin.write(text + '\n')
                self.jar.stdin.flush()
                text = self.jar.stdout.readline().strip()
                words = [w for w in text.strip().split() if w not in self.stop_words]
                words = [w for w in words if w not in settings.Config["extra_words"]]
            else:
                doc = self.nlp_tokenizer(text)
                words = []
                for sentence in doc.sentences:
                    for token in sentence.tokens:
                        for word in token.words:
                            new_word = word.text
                            if new_word not in self.stop_words and new_word not in settings.Config["extra_words"]:
                                words.append(new_word)

            if settings.Config["normalization"] == "True":
                words = [self.normalizer.normalize(w) for w in words]
            text = " ".join(words)
        return text


def compose_tsv(model, type):
    c_names = [''] * len(settings.dynamic_store["predefined_categories"])
    for k, v in settings.dynamic_store["predefined_categories"].items():
        c_names[v] = k
#    if type == "train":
#        pretrained_bert_model_path = get_abs_path(settings.Config, "resulting_bert_files_path", opt="/train.tsv")
#        data = settings.dynamic_store[model.key_train]
#    else:
#        pretrained_bert_model_path = get_abs_path(settings.Config, "resulting_bert_files_path", opt="/dev.tsv")
#        data = settings.dynamic_store[model.key_test]

    pre_trained_bert_model_path = get_abs_path(settings.Config, "resulting_bert_files_path",
                                                    opt=("/train.tsv" if type == "train" else "/dev.tsv"))
    data = settings.dynamic_store[model.key_test]
    target = open(pre_trained_bert_model_path, "w", encoding="utf-8")
    for i in range(len(data)):
        conts = data[i].lines.replace('\r','').replace('\n','.')
        nl = '\n'
        if i == 0:
            nl = ''
        string = nl + ",".join(data[i].nlabs) + "\t" + conts
        target.write(string)
    target.close()
