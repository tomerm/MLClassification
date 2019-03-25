from Utils.utils import leftAlign

metricsNames = {
    "d_docs": "Documents",
    "dd_ex": "Exactly classified",
    "dd_cf": "Classified competely with errors",
    "dd_p": "Partially classified",
    "dd_pf": "Classified partially with errors",
    "dd_f": "Falsely classified",
    "dd_n": "Not classified",
    "d_actual": "Actual labels",
    "d_predicted": "Predicted labels",
    "d_correctly": "Correctly predicted labels",
    "d_falsely": "Falsely predicted labels",
    "d_notPredicted": "Not predicted labels",
    "emr": "Exact Match Ratio",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Measure",
    "hl": "Hamming Loss",
    "macroPrecision": "Macro-Averaged Precision",
    "macroRecall": "Macro-Averaged Recall",
    "macroF1": "Macro-Averaged F1-Measure",
    "microPrecision": "Micro-Averaged Precision",
    "microRecall": "Micro-Averaged Recall",
    "microF1": "Micro-Averaged F1-Measure"
}

class ModelMetrics:
    def __init__(self, model):
        self.model = model
        self.labels = model.testLabels
        self.predictions = model.predictions
        self.cNames = [''] * len(model.Config["cats"])
        for k, v in model.Config["cats"].items():
            self.cNames[v] = k
        self.cats = model.Config["cats"]
        self.useProbabilities = model.useProbabilities
        self.rankThreshold = 0.5
        self.diffThreshold = 10
        self.model.metrics = self.initDicts()
        self.getMetrics()

    def initDicts(self):
        metrics = dict()
        metrics["all"] = dict()
        for key in self.cats.keys():
            metrics[key] = dict()
        for key, val in metrics.items():
            for k in metricsNames.keys():
                val[k] = 0
        return metrics

    def rankIndicator(self, labels, predictions, index):
        actual = labels[index] == 1
        if self.useProbabilities:
            predicted = predictions[index] >= self.rankThreshold
            if actual and not predicted and self.diffThreshold > 0:
                notActual = 0
                for i in range(len(predictions)):
                    if i == index:
                        continue
                    if labels[i] == 0:
                        notActual += 1
                        if predictions[i] > 0 and (predictions[index] / predictions[i]) < self.diffThreshold:
                            return True, False
                if notActual > 0:
                    return True, True
        else:
            predicted = predictions[index] == 1
        return actual, predicted

    def getMetrics(self):
        lenCats = len(self.cats)
        # General results
        self.model.metrics["all"]["d_docs"] = len(self.labels)
        emr = 0
        accuracy = 0
        precision = 0
        recall = 0
        hl = 0
        start = False
        for i in range(len(self.labels)):
            exact = True
            labels = sum(self.labels[i])
            tp = 0
            tn = 0
            trueLabs = 0
            falseLabs = 0
            for j in range(lenCats):
                actual, predicted = self.rankIndicator(self.labels[i], self.predictions[i], j)
                if actual:
                    self.model.metrics["all"]["d_actual"] += 1
                    self.model.metrics[self.cNames[j]]["d_actual"] += 1
                    self.model.metrics[self.cNames[j]]["d_docs"] += 1
                if predicted:
                    self.model.metrics["all"]["d_predicted"] += 1
                    self.model.metrics[self.cNames[j]]["d_predicted"] += 1
                    if actual:
                        self.model.metrics["all"]["d_correctly"] += 1
                        self.model.metrics[self.cNames[j]]["d_correctly"] += 1
                        tp += 1
                        trueLabs += 1
                    else:
                        self.model.metrics["all"]["d_falsely"] += 1
                        self.model.metrics[self.cNames[j]]["d_falsely"] += 1
                        exact = False
                        tn += 1
                        hl += 1
                        falseLabs += 1
                elif actual:
                    self.model.metrics["all"]["d_notPredicted"] += 1
                    self.model.metrics[self.cNames[j]]["d_notPredicted"] += 1
                    exact = False
                    hl += 1
            if not exact:
                emr += 1
            if labels + tn > 0:
                accuracy += tp / (labels + tn)
            if labels > 0:
                precision += tp / labels
            if tp + tn > 0:
                recall += tp / (tp + tn)
            if trueLabs == labels:
                if falseLabs == 0:
                    self.model.metrics["all"]["dd_ex"] += 1
                else:
                    self.model.metrics["all"]["dd_cf"] += 1
            elif trueLabs > 0:
                if falseLabs == 0:
                    self.model.metrics["all"]["dd_p"] += 1
                else:
                    self.model.metrics["all"]["dd_pf"] += 1
            elif falseLabs > 0:
                self.model.metrics["all"]["dd_f"] += 1
            else:
                self.model.metrics["all"]["dd_n"] += 1

        # Exact Match Ratio
        self.model.metrics["all"]["emr"] = (len(self.labels) - emr) / len(self.labels)

        # Accuracy
        self.model.metrics["all"]["accuracy"] = accuracy / len(self.labels)

        # Precision
        for key, val in self.model.metrics.items():
            if key == "all":
                self.model.metrics["all"]["precision"] = precision / len(self.labels)
            else:
                if self.model.metrics[key]["d_actual"] > 0:
                    self.model.metrics[key]["precision"] = self.model.metrics[key]["d_correctly"] / (
                        self.model.metrics[key]["d_actual"])

        # Recall
        for key, val in self.model.metrics.items():
            if key == "all":
                self.model.metrics["all"]["recall"] = recall / len(self.labels)
            else:
                if self.model.metrics[key]["d_correctly"] + self.model.metrics[key]["d_falsely"] > 0:
                    self.model.metrics[key]["recall"] = self.model.metrics[key]["d_correctly"] / (
                            self.model.metrics[key]["d_correctly"] + self.model.metrics[key]["d_falsely"])

        # F1-Measure
        for key, val in self.model.metrics.items():
            p = self.model.metrics[key]["precision"]
            r = self.model.metrics[key]["recall"]
            if p + r > 0:
                self.model.metrics[key]["f1"] = 2 * ((p * r) / (p + r))

        # Hamming Loss
        self.model.metrics["all"]["hl"] = hl / (len(self.labels) * lenCats)

        precision = 0
        recall = 0
        f1 = 0
        mtp = 0
        mtn = 0
        mlabs = 0
        for i in range(lenCats):
            tp = 0
            tn = 0
            labels = 0
            for j in range(len(self.labels)):
                actual, predicted = self.rankIndicator(self.labels[j], self.predictions[j], i)
                if actual:
                    labels += 1
                    if predicted:
                        tp += 1
                elif predicted:
                    tn += 1
            mlabs += labels
            mtp += tp
            mtn += tn
            if labels > 0:
                precision += tp / labels
            if tp + tn > 0:
                recall += tp / (tp + tn)
            if tp + tn + labels > 0:
                f1 += 2 * tp / (tp + tn + labels)

        # Macro-Averaged Precision
        self.model.metrics["all"]["macroPrecision"] = precision / lenCats

        # Macro-Averaged Recall
        self.model.metrics["all"]["macroRecall"] = recall / lenCats

        # Macro-Averaged F1-Measure
        self.model.metrics["all"]["macroF1"] = f1 / lenCats

        # Micro-Averaged Precision
        if mlabs > 0:
            self.model.metrics["all"]["microPrecision"] = mtp / mlabs

        # Micro-Averaged Recall
        if mtp + mtn > 0:
            self.model.metrics["all"]["microRecall"] = mtp / (mtp + mtn)

        # Micro-Averaged F1-Measure
        if mtp + mtn + mlabs > 0:
            self.model.metrics["all"]["microF1"] = 2 * mtp / (mtp + mtn + mlabs)

def printMetrics(model):
    if len(model.metrics) == 0:
        print("Metrics isn't calculated yet...")
        return
    print("Model's metrics:")
    dt = model.metrics["all"]
    print("  General:")
    for key, val in dt.items():
        if key.startswith("d_") or key.startswith("dd_"):
            print(f"    {leftAlign(metricsNames[key], 35)}   {'%5d' % val}")
        else:
            print(f"    {leftAlign(metricsNames[key], 35)}   {'%3.2f%%' % (val * 100)}")
    sortedDict = sorted(model.metrics.items(), key=lambda x: x[1]["f1"], reverse=True)
    print("\n  F1-Measure by category in descent order:")
    for i in range(len(sortedDict)):
        if sortedDict[i][0] != "all":
            print(f"    {leftAlign(sortedDict[i][0], 35)}\u200e   {'%.2f%%' % (sortedDict[i][1]['f1'] * 100)}")


