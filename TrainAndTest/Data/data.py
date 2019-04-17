import os
import glob
import random
import gensim
import math
import statistics
import datetime
import nltk
import subprocess
from subprocess import run, PIPE
from nltk.corpus import stopwords
from collections import namedtuple
from Data.plots import *
from Preprocess.utils import ArabicNormalizer
from Utils.utils import fullPath, updateParams, showTime

LabeledDocument = namedtuple('LabeledDocument', 'lines words labels nlabs qLabs name')
stop_words = set(stopwords.words('arabic'))

class DataLoader:
    def __init__(self, Config, DefConfig, kwargs):
        print ("=== Loading data ===")
        updateParams(Config, DefConfig, kwargs)
        self.Config = Config
        self.DefConfig = DefConfig;
        self.exCats = Config["excats"].split(",")
        self.sz = 0
        self.splitTrain = False
        self.topBound = 0.9
        self.charsTopBound = 0.6

        if len(Config["trainpath"]) == 0 or not os.path.isdir(fullPath(Config, "trainpath")):
            print ("Wrong path to training set. Data can't be loaded.")
            Config["error"] = True
            return
        if len(Config["testpath"]) > 0 and not os.path.isdir(fullPath(Config, "testpath")):
            print ("Wrong path to testing set. Data can't be loaded.")
            Config["error"] = True
            return
        elif len(Config["testpath"]) == 0:
            self.splitTrain = True
            try:
                self.sz = float(Config["testsize"])
            except ValueError:
                self.sz = 0
            if len(Config["testpath"]) == 0 and (self.sz <= 0 or self.sz >= 1):
                print ("Wrong size of testing set. Data can't be loaded.")
                Config["error"] = True
                return
        if Config["datatoks"] == "yes":
            if Config["actualtoks"] == "yes":
                taggerPath = fullPath(Config, 'rttaggerpath')
                if (self.Config["rttaggerpath"] == 0 or not os.path.exists(taggerPath)):
                    print("Wrong path to the tagger's jar. Preprocessing can't be done")
                    Config["error"] = True
                    return
                self.jar = subprocess.Popen(
                        'java -Xmx2g -jar ' + taggerPath + ' "' + self.Config["expos"] + '"',
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                        encoding="utf-8")
            if self.Config["stopwords"] == "yes":
                self.stopWords = set(nltk.corpus.stopwords.words('arabic'))
            else:
                self.stopWords = set()
            if self.Config["normalization"] == "yes":
                self.normalizer = ArabicNormalizer()
        if Config["w2vload"] == "yes":
            if len(Config["w2vmodelpath"]) == 0 or not os.path.isfile(fullPath(Config, "w2vmodelpath")):
                print("Wrong path to W2V model. Stop.")
                Config["error"] = True
                return
            try:
                self.ndim = int(self.Config["w2vdim"])
            except ValueError:
                print("Wrong size of vectors' dimentions. Stop.")
                Config["error"] = True
                return
            self.Config["resources"]["w2v"]["modelPath"] = fullPath(Config, "w2vmodelpath")
            self.Config["resources"]["w2v"]["ndim"] = self.ndim
            self.loadW2VModel()
        else:
            self.Config["w2vmodel"] = None

        self.loadData()
        if Config["analysis"] == "yes":
            self.analysis()

    def loadData(self):
        if self.Config["datatoks"] == "yes":
            print("Start loading and preprocessing of data...")
        else:
            print ("Start loading data...")
        ds = datetime.datetime.now()
        self.Config["cats"] = self.getCategories(fullPath(self.Config, "trainpath"))
        traindocs = self.getDataDocs(fullPath(self.Config, "trainpath"))
        if not self.splitTrain:
            testdocs = self.getDataDocs(fullPath(self.Config, "testpath"))
        else:
            ind = int(len(traindocs) * (1 - self.sz))
            testdocs = traindocs[ind:]
            traindocs = traindocs[:ind]
        de = datetime.datetime.now()
        self.Config["traindocs"] = random.sample(traindocs, len(traindocs))
        self.Config["testdocs"] = random.sample(testdocs, len(testdocs))
        self.getMaxSeqLen()
        self.getMaxCharsLength()
        if self.Config["datatoks"] == "yes" and self.Config["actualtoks"] == "yes":
            self.jar.stdin.write('!!! STOP !!!\n')
            self.jar.stdin.flush()
        print ("Input data loaded in %s"%(showTime(ds, de)))
        print ("Training set contains %d documents."%(len(self.Config["traindocs"])))
        print ("Testing set contains %d documents."%(len(self.Config["testdocs"])))
        print ("Documents belong to %d categories."%(len(self.Config["cats"])))

    def getCategories(self, path):
        cats = dict()
        nCats = 0
        os.chdir(path)
        for f in glob.glob("*"):
            if os.path.isdir(f) and not f in self.exCats:
                cats[f] = nCats
                nCats += 1
        return cats

    def getDataDocs(self, path):
        files = dict()
        fInCats = [0] * len(self.Config["cats"])
        nFiles = 0
        actFiles = 0
        curCategory = 0
        docs = []
        os.chdir(path)
        for f in glob.glob("*"):
            if f in self.exCats:
                continue
            curCategory = self.Config["cats"][f]
            catPath = path + "/" + f
            os.chdir(catPath)
            for fc in glob.glob("*"):
                actFiles += 1
                if fc not in files:
                    nFiles += 1
                    docCont = ''
                    with open(fc, 'r', encoding='UTF-8') as tc:
                        for line in tc:
                            docCont += line.strip() + " "
                    tc.close()
                    if self.Config["datatoks"] == "yes":
                        docCont = self.preprocess(docCont)
                    words = docCont.strip().split()
                    labels = [0] * len(self.Config["cats"])
                    labels[curCategory] = 1
                    nlabs = [f]
                    files[fc] = LabeledDocument(docCont.strip(), words, labels, nlabs, [1], fc)
                else:
                    files[fc].labels[curCategory] = 1
                    files[fc].nlabs.append(f)
                    files[fc].qLabs[0] += 1
                fInCats[curCategory] += 1
        for k, val in files.items():
            docs.append(val)
        return docs

    def getMaxSeqLen(self):
        maxDocLen = max(len(x.words) for x in self.Config["traindocs"])
        maxLen = math.ceil(maxDocLen / 100) * 100 + 100
        input_length_list = []
        for i in range(100, maxLen, 100):
            input_length_list.append(i)
        input_length_dict = {x: 0 for x in input_length_list}
        for i in range(len(self.Config["traindocs"])):
            curLen = len(self.Config["traindocs"][i].words)
            dicLen = maxLen
            for ln in input_length_dict:
                if curLen < ln:
                    dicLen = ln
                    break
            input_length_dict[dicLen] = input_length_dict[dicLen] + 1
        input_length_dict_percentage = {}
        for k, v in input_length_dict.items():
            v = v / len(self.Config["traindocs"])
            input_length_dict_percentage[k] = v
        maxSeqLength = 0
        accumulate_percentage = 0
        for length, percentage in input_length_dict_percentage.items():
            accumulate_percentage += percentage
            if accumulate_percentage > self.topBound:
                maxSeqLength = length
                break

        self.Config["maxdoclen"] = maxDocLen
        self.Config["maxseqlen"] = maxSeqLength

    def getMaxCharsLength(self):
        maxDocLen = max(len(x.lines) for x in self.Config["traindocs"])
        maxLen = math.ceil(maxDocLen / 100) * 100 + 100
        input_length_list = []
        for i in range(100, maxLen, 100):
            input_length_list.append(i)
        input_length_dict = {x: 0 for x in input_length_list}
        for i in range(len(self.Config["traindocs"])):
            curLen = len(self.Config["traindocs"][i].lines)
            dicLen = maxLen
            for ln in input_length_dict:
                if curLen < ln:
                    dicLen = ln
                    break
            input_length_dict[dicLen] = input_length_dict[dicLen] + 1
        input_length_dict_percentage = {}
        for k, v in input_length_dict.items():
            v = v / len(self.Config["traindocs"])
            input_length_dict_percentage[k] = v
        maxSeqLength = 0
        accumulate_percentage = 0
        for length, percentage in input_length_dict_percentage.items():
            accumulate_percentage += percentage
            if accumulate_percentage > self.charsTopBound:
                maxSeqLength = length
                break

        self.Config["maxcharsdoclen"] = maxDocLen
        self.Config["maxcharsseqlen"] = min(maxSeqLength, 512)

    def analysis(self):
        maxDocLen = max(len(x.words) for x in self.Config["traindocs"])
        minDocLen = min(len(x.words) for x in self.Config["traindocs"])
        avrgDocLen = round(statistics.mean(len(x.words) for x in self.Config["traindocs"]), 2)
        maxCharsDocLen = max(len(x.lines) for x in self.Config["traindocs"])
        minCharsDocLen = min(len(x.lines) for x in self.Config["traindocs"])
        avrgCharsDocLen = round(statistics.mean(len(x.lines) for x in self.Config["traindocs"]), 2)
        dls, qLabs = self.getLabelSets()
        fInCats1 = self.filesByCategory(self.Config["traindocs"], self.Config["cats"])
        fInCats2 = self.filesByCategory(self.Config["testdocs"], self.Config["cats"])
        print("Length of train documents: maximum: %d, minimum: %d, average: %d" % (
                        maxCharsDocLen, minCharsDocLen, avrgCharsDocLen))
        """
        print("Length of %.1f%% documents from training set is less then %d characters." % (
                    self.charsTopBound * 100, self.Config["maxcharsseqlen"]))
        """
        print("Tokens in train documents: maximum: %d, minimum: %d, average: %d" % (maxDocLen, minDocLen, avrgDocLen))
        print("Length of %.1f%% documents from training set is less then %d tokens." % (
            self.topBound * 100, self.Config["maxseqlen"]))
        if self.Config["showplots"] == "yes":
            showDocsByLength(self.Config);
        print("Documents for training in category : maximum: %d, minimum: %d, avegare: %d" % (
        max(fInCats1), min(fInCats1), round(statistics.mean(fInCats1), 2)))
        print("Documents for testing  in category : maximum: %d, minimum: %d, avegare: %d" % (
        max(fInCats2), min(fInCats2), round(statistics.mean(fInCats2), 2)))
        if self.Config["showplots"] == "yes":
            showDocsByLabs(self.Config)
        print("Training dataset properties:")
        print("  Distinct Label Set: %d" % (dls))
        print("  Proportion of Distinct Label Set: %.4f" % (dls / len(self.Config["traindocs"])))
        print("  Label Cardinality: %.4f" % (qLabs / len(self.Config["traindocs"])))
        print("  Label Density: %.4f" % (qLabs / len(self.Config["traindocs"]) / len(self.Config["cats"])))

    def getLabelSets(self):
        labels = [x[2] for x in self.Config["traindocs"]]
        results = [labels[0]]
        qLabs = 0
        for i in range(len(labels)):
            qLabs += sum(labels[i])
            count = 0
            for j in range(len(results)):
                for k in range(len(self.Config["cats"])):
                    if labels[i][k] != results[j][k]:
                        count += 1
                        break
            if count == len(results):
                results.append(labels[i])
        return len(results), qLabs

    def filesByCategory(self, docs, cats):
        fInCats = [0] * len(cats)
        for i in range(len(docs)):
            for j in range(len(cats)):
                if docs[i].labels[j] == 1:
                    fInCats[j] += 1
        return fInCats

    def loadW2VModel(self):
        print ("Load W2V model...")
        ds = datetime.datetime.now()
        self.Config["w2vmodel"] = gensim.models.KeyedVectors.load_word2vec_format(fullPath(self.Config, "w2vmodelpath"))
        de = datetime.datetime.now()
        print("Load W2V model (%s) in %s" % (fullPath(self.Config, "w2vmodelpath"), showTime(ds, de)))

    def preprocess(self, text):
        if self.Config["actualtoks"] == "yes":
            self.jar.stdin.write(text + '\n')
            self.jar.stdin.flush()
            text = self.jar.stdout.readline().strip()
        words = [w for w in text.strip().split() if w not in self.stopWords]
        words = [w for w in words if w not in self.Config["extrawords"]]
        if self.Config["normalization"] == "yes":
            words = [self.normalizer.normalize(w) for w in words]
        text = " ".join(words)
        return text


def composeTsv(model, type):
    cNames = [''] * len(model.Config["cats"])
    for k, v in model.Config["cats"].items():
        cNames[v] = k
    if type == "train":
        bertPath = fullPath(model.Config, "bertoutpath", opt="/train.tsv")
        data = model.Config[model.keyTrain]
    else:
        bertPath = fullPath(model.Config, "bertoutpath", opt="/dev.tsv")
        data = model.Config[model.keyTest]
    target = open(bertPath, "w", encoding="utf-8")
    for i in range(len(data)):
        conts = data[i].lines.replace('\r','').replace('\n','.')
        nl = '\n'
        if i == 0:
            nl = ''
        string = nl + ",".join(data[i].nlabs) + "\t" + conts
        target.write(string)
    target.close()