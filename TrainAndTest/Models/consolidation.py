import os
import numpy
import shutil
import json
import datetime
from Models.metrics import ModelMetrics, printMetrics
from Utils.utils import fullPath, showTime

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
        if self.runtime:
            if len(os.listdir(fullPath(self.Config, "resourcespath"))) > 0:
                print ("Warning: folder %s is not empty. All its content will be deleted."%(
                                fullPath(self.Config, "resourcespath")))
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
                    else:
                        notActuals = 0
                        for k in range(self.qLabs):
                            if k == j:
                                continue
                            if self.testLabels[i][k] == 0:
                                notActuals += 1
                                if res[i][k] == 0 or ((res[i][j] / res[i][k]) < self.diffThreshold):
                                    notActuals = 0
                                    break
                        if notActuals > 0:
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

    def saveResources(self):
        tokOpts = ["actualtoks", "normalization", "stopwords", "expos", "extrawords"]
        self.Config["resources"]["tokenization"] = {}
        ds = datetime.datetime.now()
        for i in range(len(tokOpts)):
            self.Config["resources"]["tokenization"][tokOpts[i]] = self.Config[tokOpts[i]]
        #print (json.dumps(self.Config["resources"]))
        self.outDir = fullPath(self.Config, "resourcespath") + "/"
        for key, val in self.Config["resources"]["models"].items():
            val["modelPath"] = self.copyFile(val["modelPath"])
        if "w2v" in self.Config["resources"]:
            self.Config["resources"]["w2v"]["modelPath"] = self.copyFile(self.Config["resources"]["w2v"]["modelPath"])
        if "indexer" in self.Config["resources"]:
            self.Config["resources"]["indexer"] = self.copyFile(self.Config["resources"]["indexer"])
        if "vectorizer" in self.Config["resources"]:
            self.Config["resources"]["vectorizer"] = self.copyFile(self.Config["resources"]["vectorizer"])
        if "ptBertModel" in self.Config["resources"]:
            self.Config["resources"]["ptBertModel"] = self.copyFile(self.Config["resources"]["ptBertModel"])
        with open(self.outDir + 'config.json', 'w', encoding="utf-8") as file:
            json.dump(self.Config["resources"], file, indent=4)
        de = datetime.datetime.now()
        print("\nArtifacts are copied into the folder %s in %s"%(fullPath(self.Config, "resourcespath"), showTime(ds, de)))

    def copyFile(self, inPath):
        dir, name = os.path.split(inPath)
        outPath = self.outDir + name
        shutil.copy(inPath, outPath)
        return name

