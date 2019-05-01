import os
import numpy
import shutil
import pickle
import json
import datetime
from Models.metrics import ModelMetrics, printMetrics
from Utils.utils import fullPath, showTime
from Models.reports import Report

class Collector:
    def __init__(self, Config):
        self.Config = Config
        if "testdocs" not in Config or len(Config["results"]) == 0:
            print ("Documents have not been classified in this process chain.")
            print ("Consolidation can't be performed.")
            return
        self.testLabels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1, len(self.Config["cats"])) for x in self.Config["testdocs"]])
        self.qLabs = len(self.Config["cats"])
        self.predictions = numpy.zeros([len(self.testLabels), self.qLabs])
        self.rankThreshold = 0.5
        self.diffThreshold = 10
        self.metrics = {}
        self.useProbabilities = False
        self.reports = False
        self.runtime = False
        print ("\nCalculate consolidated metrics...")
        if len(self.Config["results"]) == 0:
            print ("No results to consolidate them.")
            print ("Consolidation can't be performed.")
            return
        if Config["reports"] == "yes":
            if len(Config["reportspath"]) == 0 or not os.path.isdir(fullPath(Config, "reportspath")):
                print ("Wrong path to the folder, containing reports.")
                print ("Reports can't be created.")
            else:
                self.reports = True
        if Config["saveresources"] == "yes":
            if len(Config["resourcespath"]) == 0 or not os.path.isdir(fullPath(Config, "resourcespath")):
                print ("Wrong path to the folder, containing resources for runtime.")
                print ("Resources can't be saved.")
            else:
                self.runtime = True
        if self.reports or self.Config["showresults"] == "yes":
            self.getConsolidatedResults()
            self.getMetrics()
            if self.reports:
                self.saveReports()
        if self.runtime:
            if len(os.listdir(fullPath(self.Config, "resourcespath"))) > 0:
                print ("Warning: folder %s is not empty. All its content will be deleted."%(
                                fullPath(self.Config, "resourcespath")))
                shutil.rmtree(fullPath(self.Config, "resourcespath"))
                os.makedirs(fullPath(self.Config, "resourcespath"), exist_ok=True)
            print("\nCollect arfifacts for runtime...")
            self.saveResources()


    def getConsolidatedResults(self):
        for key, res in self.Config["results"].items():
            for i in range(len(res)):
                for j in range(self.qLabs):
                    if res[i][j] == 1:
                        self.predictions[i][j] += 1
                    elif res[i][j] >= self.rankThreshold:
                        self.predictions[i][j] += 1
        qModels = len(self.Config["results"])
        for i in range(len(self.predictions)):
            for j in range(len(self.predictions[i])):
                if self.predictions[i][j] >= qModels / 2.0:
                    self.predictions[i][j] = 1
                else:
                    self.predictions[i][j] = 0

    def getMetrics(self):
        ModelMetrics(self)
        if self.Config["showresults"] == "yes":
            printMetrics(self)

    def saveReports(self):
        print ("Save report...")
        report = Report()
        report.requestId = self.Config["reqid"]
        report.sourcesPath = self.Config["actualpath"]
        report.datasetPath = self.Config["testpath"]

        tokOpts = ["actualtoks", "normalization", "stopwords", "expos", "extrawords", "excats"]
        for i in range(len(tokOpts)):
            report.preprocess[tokOpts[i]] = self.Config[tokOpts[i]]
        for i in range(len(self.Config["testdocs"])):
            report.docs[self.Config["testdocs"][i].name] = {}
            report.docs[self.Config["testdocs"][i].name]["actual"] = ",".join(self.Config["testdocs"][i].nlabs)
        if len(self.Config["excats"]) == 0:
            exCats = []
        else:
            exCats = self.Config["excats"].split(",")
        cNames = [''] * (len(self.Config["cats"]) - len(exCats))
        for k, v in self.Config["cats"].items():
            if k not in exCats:
                cNames[v] = k
        report.categories = cNames
        for key, val in self.Config["results"].items():
            for i in range(len(val)):
                labs = []
                for j in range(self.qLabs):
                    if val[i][j] >= self.rankThreshold:
                        labs.append(cNames[j])
                report.docs[self.Config["testdocs"][i].name][key] = ",".join(labs)
        for key, val in self.Config["metrics"].items():
            report.models[key] = val
        if len(self.Config["results"]) > 1:
            for i in range(len(self.predictions)):
                labs = []
                for j in range(self.qLabs):
                    if self.predictions[i][j] == 1:
                        labs.append(cNames[j])
                report.docs[self.Config["testdocs"][i].name]["consolidated"] = ",".join(labs)
            report.models["consolidated"] = self.metrics
        rPath = fullPath(self.Config, "reportspath") + "/" + self.Config["reqid"] + ".json"
        with open(rPath, 'w', encoding="utf-8") as file:
            json.dump(report.toJSON(), file, indent=4)
        file.close()

    def saveResources(self):
        tokOpts = ["actualtoks", "normalization", "stopwords", "expos", "extrawords",
                   "maxseqlen", "maxcharsseqlen", "rttaggerpath"]
        self.Config["resources"]["tokenization"] = {}
        ds = datetime.datetime.now()
        self.outDir = fullPath(self.Config, "resourcespath") + "/"
        for i in range(len(tokOpts)):
            self.Config["resources"]["tokenization"][tokOpts[i]] = self.Config[tokOpts[i]]
            if self.Config["actualtoks"] == "yes" and tokOpts[i] == "rttaggerpath":
                self.Config["resources"]["tokenization"]["rttaggerpath"] = \
                    self.copyFile(fullPath(self.Config, "rttaggerpath"))
        isW2VNeeded = False
        for key, val in self.Config["resources"]["models"].items():
            val["modelPath"] = self.copyFile(val["modelPath"])
            if "w2v" in val and val["w2v"] == "yes":
                isW2VNeeded = True
        if not isW2VNeeded and "w2v" in self.Config["resources"]:
            self.Config["resources"].pop("w2v", None)
        if "w2v" in self.Config["resources"]:
            w2vDict = {}
            isFirstLine = True
            fEmbeddings = open(self.Config["resources"]["w2v"]["modelPath"], encoding="utf-8")
            for line in fEmbeddings:
                if isFirstLine == True:
                    isFirstLine = False
                    continue
                split = line.strip().split(" ")
                word = split[0]
                vector = numpy.array([float(num) for num in split[1:]])
                w2vDict[word] = vector
            fEmbeddings.close()
            with open(self.Config["resources"]["w2v"]["modelPath"] + '.pkl', 'wb') as file:
                pickle.dump(w2vDict, file, pickle.HIGHEST_PROTOCOL)
            file.close()
            self.Config["resources"]["w2v"]["modelPath"] = self.copyFile(self.Config["resources"]["w2v"]["modelPath"] + '.pkl')
        if "indexer" in self.Config["resources"]:
            self.Config["resources"]["indexer"] = self.copyFile(self.Config["resources"]["indexer"])
        if "vectorizer" in self.Config["resources"]:
            self.Config["resources"]["vectorizer"] = self.copyFile(self.Config["resources"]["vectorizer"])
        if "ptBertModel" in self.Config["resources"]:
            self.Config["resources"]["ptBertModel"] = self.copyFile(self.Config["resources"]["ptBertModel"])
            self.Config["resources"]["vocabPath"] = self.copyFile(self.Config["resources"]["vocabPath"])
        cNames = [''] * len(self.Config["cats"])
        for k, v in self.Config["cats"].items():
            cNames[v] = k
        with open(self.outDir + 'labels.txt', 'w', encoding="utf-8") as file:
            file.write(",".join(cNames))
        file.close()
        self.Config["resources"]["labels"] = "labels.txt"
        with open(self.outDir + 'config.json', 'w', encoding="utf-8") as file:
            json.dump(self.Config["resources"], file, indent=4)
        file.close()
        de = datetime.datetime.now()
        print("\nArtifacts are copied into the folder %s in %s"%(fullPath(self.Config, "resourcespath"), showTime(ds, de)))

    def copyFile(self, inPath):
        dir, name = os.path.split(inPath)
        outPath = self.outDir + name
        shutil.copy(inPath, outPath)
        return name

