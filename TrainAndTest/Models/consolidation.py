import numpy
from Models.metrics import ModelMetrics, printMetrics

class ConsolidatedResults:
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
        print ("Calculate consolidated metrics...")
        self.getConsolidatedResults()
        self.getMetrics()

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
        printMetrics(self)

