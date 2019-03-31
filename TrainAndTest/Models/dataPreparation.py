import numpy
import pickle
import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from Utils.utils import showTime, fullPath, getDictionary

class DataPreparation:
    def __init__(self, model, addValSet):
        self.model = model
        self.addValSet = addValSet
        self.maxWords = 300000
        self.ndim = int(model.Config["w2vdim"])
        if addValSet:
            self.valSize = float(model.Config["valsize"])
            if model.Config["runfor"] != "test":
                print("Validation: %f" % (self.valSize))
        else:
            self.valSize = 0

    def getWordVectorsSum(self):
        self.nfWords = 0
        self.sdict = dict()
        self.tmpCount = 0

        if self.model.Config["runfor"] != "test":
            ds = datetime.datetime.now()
            self.model.trainArrays = numpy.concatenate([self.getDocsArray(x.words, 'Train') for x in self.model.Config["traindocs"]])
            self.model.trainLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(self.model.Config["cats"])) for x in self.model.Config["traindocs"]])

            if self.addValSet:
                ind = int(len(self.model.trainArrays) * (1 - self.valSize))
                self.model.valArrays = self.model.trainArrays[ind:]
                self.model.valLabels = self.model.trainLabels[ind:]
                self.model.trainArrays = self.model.trainArrays[:ind]
                self.model.trainLabels = self.model.trainLabels[:ind]

                de = datetime.datetime.now()
                print("Prepare train and validation data in %s" % (showTime(ds, de)))
            else:
                de = datetime.datetime.now()
                print("Prepare train data in %s" % (showTime(ds, de)))


        self.tmpCount = 0
        ds = datetime.datetime.now()
        self.model.testArrays = numpy.concatenate([self.getDocsArray(x.words, "Test") for x in self.model.Config["testdocs"]])
        self.model.testLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(self.model.Config["cats"])) for x in self.model.Config["testdocs"]])
        de = datetime.datetime.now()
        print("Prepare test data in %s" % (showTime(ds, de)))

        print("Unique words in all documents: %d" % (len(self.sdict)))
        print("Words not found in the w2v vocabulary: %d" % (self.nfWords))

    def getDocsArray(self, tokens, dataType):
        self.tmpCount += 1
        if self.tmpCount != 0 and self.tmpCount % 1000 == 0:
            print(dataType + ": prepare ", self.tmpCount, end="\r")
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

    def getWordVectorsMatrix(self):
        tokenizer = None
        ds = datetime.datetime.now()
        if self.model.Config["runfor"] != "test":
            tokenizer = Tokenizer(num_words=self.maxWords)
            trainTexts = []
            for i in range(len(self.model.Config["traindocs"])):
                trainTexts.append(self.model.Config["traindocs"][i].lines)
            tokenizer.fit_on_texts(trainTexts)
            with open(fullPath(self.model.Config, "indexerpath"), 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if self.model.Config["maxdoclen"] > self.model.Config["maxseqlen"]:
                print("Most of documents from training set have less then %d tokens. Longer documents will be truncated."%(
                    self.model.Config["maxseqlen"]))
            self.model.trainArrays = pad_sequences(tokenizer.texts_to_sequences(trainTexts),
                                                    maxlen=self.model.Config["maxseqlen"])
            self.model.trainLabels = numpy.concatenate([numpy.array(x.labels).
                            reshape(1, len(self.model.Config["cats"])) for x in self.model.Config["traindocs"]])
            if self.addValSet:
                ind = int(len(self.model.trainArrays) * (1 - self.valSize))
                self.model.valArrays = self.model.trainArrays[ind:]
                self.model.valLabels = self.model.trainLabels[ind:]
                self.model.trainArrays = self.model.trainArrays[:ind]
                self.model.trainLabels = self.model.trainLabels[:ind]
        if tokenizer == None:
            with open(fullPath(self.model.Config, "indexerpath"), 'rb') as handle:
                tokenizer = pickle.load(handle)
        testTexts = []
        for i in range(len(self.model.Config["testdocs"])):
            testTexts.append(self.model.Config["testdocs"][i].lines)
        self.model.testArrays = pad_sequences(tokenizer.texts_to_sequences(testTexts),
                                              maxlen=self.model.Config["maxseqlen"])
        self.model.testLabels = numpy.concatenate([numpy.array(x.labels).
                            reshape(1, len(self.model.Config["cats"])) for x in self.model.Config["testdocs"]])
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
        de = datetime.datetime.now()
        print('Found %s unique tokens.' % len(tokenizer.word_index))
        print ('Tokens not found in W2V vocabulary: %d'%nf)
        print("All data prepared and embedding matrix built in %s"%(showTime(ds, de)))
        return embedding_matrix, self.maxWords

    def getCharVectors(self):
        ds = datetime.datetime.now()
        if self.model.Config["runfor"] != "test":
            self.model.trainArrays = numpy.concatenate([self.stringToIndexes(x.lines)
                                            for x in self.model.Config["traindocs"]])
            self.model.trainLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(self.model.Config["cats"])) for x in self.model.Config["traindocs"]])
            if self.addValSet:
                ind = int(len(self.model.trainArrays) * (1 - self.valSize))
                self.model.valArrays = self.model.trainArrays[ind:]
                self.model.valLabels = self.model.trainLabels[ind:]
                self.model.trainArrays = self.model.trainArrays[:ind]
                self.model.trainLabels = self.model.trainLabels[:ind]
        self.model.testArrays = numpy.concatenate([self.stringToIndexes(x.lines)
                                            for x in self.model.Config["testdocs"]])
        self.model.testLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(self.model.Config["cats"])) for x in self.model.Config["testdocs"]])
        de = datetime.datetime.now()
        print("Prepare all data in %s" % (showTime(ds, de)))

    def stringToIndexes(self, str):
        chDict = getDictionary()
        str2ind = numpy.zeros(self.model.Config["maxseqlen"], dtype='int64')
        strLen = min(len(str), self.model.Config["maxseqlen"])
        for i in range(1, strLen + 1):
            c = str[-i]
            if c in chDict:
                str2ind[i - 1] = chDict[c]
        return str2ind.reshape(1, self.model.Config["maxseqlen"])

    def getDataForSklearnClassifiers(self):
        mlb = None
        ds = datetime.datetime.now()
        if self.model.Config["runfor"] != "test":
            nmCats = [""] * len(self.model.Config["cats"])
            cKeys = list(self.model.Config["cats"].keys())
            for i in range(len(cKeys)):
                nmCats[self.model.Config["cats"][cKeys[i]]] = cKeys[i]
            mlb = MultiLabelBinarizer(classes=nmCats)
            wev = TfidfVectorizer(ngram_range=(1, 3), max_df=0.50).fit([x.lines for x in self.model.Config["traindocs"]],
                                                                   [x.nlabs for x in self.model.Config["traindocs"]])
            self.model.trainArrays = wev.transform([x.lines for x in self.model.Config["traindocs"]])
            self.model.trainLabels = mlb.fit_transform([x.nlabs for x in self.model.Config["traindocs"]])
            with open(fullPath(self.model.Config, "binarizerpath"), 'wb') as handle:
                pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(fullPath(self.model.Config, "vectorizerpath"), 'wb') as handle:
                pickle.dump(wev, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if mlb == None:
            with open(fullPath(self.model.Config, "binarizerpath"), 'rb') as handle:
                mlb = pickle.load(handle)
            with open(fullPath(self.model.Config, "vectorizerpath"), 'rb') as handle:
                wev = pickle.load(handle)
        self.model.testArrays = wev.transform([x.lines for x in self.model.Config["testdocs"]])
        self.model.testLabels = mlb.fit_transform([x.nlabs for x in self.model.Config["testdocs"]])
        de = datetime.datetime.now()
        print("Prepare all data in %s" % (showTime(ds, de)))