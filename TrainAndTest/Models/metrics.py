import logging
from Utils.utils import align_to_left
import General.settings as settings

logger = logging.getLogger(__name__)

metrics_names = {
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
        self.labels = model.test_labels
        self.predictions = model.predictions
        self.cNames = [''] * len(settings.dynamic_store["predefined_categories"])
        for k, v in settings.dynamic_store["predefined_categories"].items():
            self.cNames[v] = k
        self.cats = settings.dynamic_store["predefined_categories"]
        self.use_probabilities = model.use_probabilities
        self.rank_threshold = model.rank_threshold
        self.diffThreshold = 10
        #self.model.metrics = self.initDicts()

        self.model.metrics = dict()
        self.model.metrics["all"] = dict()
        for key in self.cats.keys():
            self.model.metrics[key] = dict()
        for key, val in self.model.metrics.items():
            for k in metrics_names.keys():
                val[k] = 0

        self.get_metrics()


    def rank_indicator(self, labels, predictions, index):
        actual = labels[index] == 1
        if self.use_probabilities:
            predicted = predictions[index] >= self.rank_threshold
            """
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
            """
        else:
            predicted = predictions[index] == 1
        return actual, predicted

    def get_metrics(self):
        len_cats = len(self.cats)
        # General results
        self.model.metrics["all"]["d_docs"] = len(self.labels)
        emr = 0
        accuracy = 0
        precision = 0
        recall = 0
        hl = 0
        for i in range(len(self.labels)):
            exact = True
            labels = sum(self.labels[i])
            tp = 0
            tn = 0
            true_labs = 0
            false_labs = 0
            for j in range(len_cats):
                actual, predicted = self.rank_indicator(self.labels[i], self.predictions[i], j)
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
                        true_labs += 1
                    else:
                        self.model.metrics["all"]["d_falsely"] += 1
                        self.model.metrics[self.cNames[j]]["d_falsely"] += 1
                        exact = False
                        tn += 1
                        hl += 1
                        false_labs += 1
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
            if true_labs == labels:
                if false_labs == 0:
                    self.model.metrics["all"]["dd_ex"] += 1
                else:
                    self.model.metrics["all"]["dd_cf"] += 1
            elif true_labs > 0:
                if false_labs == 0:
                    self.model.metrics["all"]["dd_p"] += 1
                else:
                    self.model.metrics["all"]["dd_pf"] += 1
            elif false_labs > 0:
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
        self.model.metrics["all"]["hl"] = hl / (len(self.labels) * len_cats)

        precision = 0
        recall = 0
        f1 = 0
        mtp = 0
        mtn = 0
        mlabs = 0
        for i in range(len_cats):
            tp = 0
            tn = 0
            labels = 0
            for j in range(len(self.labels)):
                actual, predicted = self.rank_indicator(self.labels[j], self.predictions[j], i)
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
        self.model.metrics["all"]["macroPrecision"] = precision / len_cats

        # Macro-Averaged Recall
        self.model.metrics["all"]["macroRecall"] = recall / len_cats

        # Macro-Averaged F1-Measure
        self.model.metrics["all"]["macroF1"] = f1 / len_cats

        # Micro-Averaged Precision
        if mlabs > 0:
            self.model.metrics["all"]["microPrecision"] = mtp / mlabs

        # Micro-Averaged Recall
        if mtp + mtn > 0:
            self.model.metrics["all"]["microRecall"] = mtp / (mtp + mtn)

        # Micro-Averaged F1-Measure
        if mtp + mtn + mlabs > 0:
            self.model.metrics["all"]["microF1"] = 2 * mtp / (mtp + mtn + mlabs)


def print_metrics(model):
    if not model.metrics:
        logger.error("Metrics were not calculated yet...")
        return
    logger.info("Model's metrics:")
    dt = model.metrics["all"]
    logger.info("  General:")
    for key, val in dt.items():
        if key.startswith("d_") or key.startswith("dd_"):
            logger.info(f"    {align_to_left(metrics_names[key], 35)}   {'%5d' % val}")
        else:
            logger.info(f"    {align_to_left(metrics_names[key], 35)}   {'%3.2f%%' % (val * 100)}")
    sorted_dict = sorted(model.metrics.items(), key=lambda x: x[1]["f1"], reverse=True)
    logger.info("\n  F1-Measure by category in descend order:")
    for d in sorted_dict:
        if d[0] != "all":
            logger.info(f"    {align_to_left(d[0], 35)}\u200e   {'%.2f%%' % (d[1]['f1'] * 100)}")


def print_averaged_metrics(attr_metrics):
    logger.info("Averaged metrics:")
    model = SimpleModel()
    for it_metrics in attr_metrics:
        for key1, val1 in it_metrics.items():
            for key2, val2 in val1.items():
                if not key2.startswith("d"):
                    model.metrics[key1][key2] += val2
    for key1, val1 in model.metrics.items():
        for key2, val2 in val1.items():
            if not key2.startswith("d"):
                model.metrics[key1][key2] /= len(attr_metrics)
    logger.info("  General:")
    dt = model.metrics["all"]
    for key, val in dt.items():
        if not (key.startswith("d_") or key.startswith("dd_")):
            logger.info(f"    {align_to_left(metrics_names[key], 35)}   {'%3.2f%%' % (val * 100)}")
    sorted_dict = sorted(model.metrics.items(), key=lambda x: x[1]["f1"], reverse=True)
    logger.info("\n  Averaged F1-Measure by category in descend order:")
    for d in sorted_dict:
        if d[0] != "all":
            logger.info(f"    {align_to_left(d[0], 35)}\u200e   {'%.2f%%' % (d[1]['f1'] * 100)}")


class SimpleModel:
    def __init__(self):
        self.metrics = dict()
        self.metrics["all"] = dict()
        for key in settings.dynamic_store["predefined_categories"].keys():
            self.metrics[key] = dict()
        for key, val in self.metrics.items():
            for k in metrics_names.keys():
                val[k] = 0
