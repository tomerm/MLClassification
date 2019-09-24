import os
import numpy
import shutil
import pickle
import json
import datetime
from Models.metrics import ModelMetrics, print_metrics
from Utils.utils import get_abs_path, get_formatted_date
from Models.reports import Report
import General.settings as settings
import logging

logger = logging.getLogger(__name__)

def job_collector():
    worker = Collector()
    worker.run()


class Collector:
    def __init__(self):
        if "test_docs" not in settings.dynamic_store or not settings.dynamic_store["results"]:
            logger.warning("Documents have not been classified in this process chain.")
            logger.warning("Consolidation can't be performed.")
            return

        self.rank_threshold = 0.5
        if settings.Config['consolidatedrank'] == "True":
                try:
                    self.rank_threshold = float(settings.Config["consolidated_rank_threshold"])
                except ValueError:
                    self.rank_threshold = 0.5
        self.test_labels = numpy.concatenate([numpy.array(x.labels).
                                reshape(1,
                                        len(settings.dynamic_store["predefined_categories"])) for x in settings.dynamic_store["test_docs"]])
        self.qLabs = len(settings.dynamic_store["predefined_categories"])
        self.predictions = numpy.zeros([len(self.test_labels), self.qLabs])
        self.metrics = {}
        self.use_probabilities = False
        self.do_save_reports = False
        self.runtime = False

    def run(self):
        logger.info("\nCalculate consolidated metrics...")
        if not settings.dynamic_store["results"]:
            logger.error("No results to consolidate them. Consolidation can not be performed.")
            return
        if settings.Config["save_reports"] == "True":
            if not settings.Config["reports_path"] or not os.path.isdir(get_abs_path(settings.Config, "reports_path")):
                logger.error("Wrong path to the folder, containing reports. Reports can not be created.")
            else:
                self.do_save_reports = True
        if settings.Config["prepare_resources_for_runtime"] == "True":
            if (not settings.Config["saved_resources_path"] or
                    not os.path.isdir(get_abs_path(settings.Config, "saved_resources_path"))):
                logger.error("Wrong path to the folder, containing resources for runtime. Resources can not be saved.")
            else:
                self.runtime = True
        logger.info("Rank threshold for consolidated results: %.2f" % (self.rank_threshold))
        if self.do_save_reports or settings.Config["show_consolidated_results"] == "True":
            self.get_consolidated_results()
            self.get_metrics()
            if self.do_save_reports:
                self.save_reports()
        if self.runtime:
            saved_rc_path = get_abs_path(settings.Config, "saved_resources_path")
            if len(os.listdir(saved_rc_path)) > 0:
                logger.warning("Warning: folder %s is not empty. All its content will be deleted." % saved_rc_path)
                shutil.rmtree(saved_rc_path)
                os.makedirs(saved_rc_path, exist_ok=True)
            logger.info("\nCollect arfifacts for runtime...")
            self.prepare_resources_for_runtime()


    def get_consolidated_results(self):
        for key, res in settings.dynamic_store["results"].items():
            for i in range(len(res)):
                for j in range(self.qLabs):
                    if res[i][j] == 1:
                        self.predictions[i][j] += 1
                    #elif res[i][j] >= self.rank_threshold:
                    elif res[i][j] >= settings.dynamic_store["ranks"][key]:
                        self.predictions[i][j] += 1
        q_models = len(settings.dynamic_store["results"])
        for p1 in self.predictions:
            for p in p1:
                if p >= q_models * self.rank_threshold:
                    p = 1
                else:
                    p = 0

    def get_metrics(self):
        ModelMetrics(self)
        if settings.Config["show_consolidated_results"] == "True":
            print_metrics(self)

    def save_reports(self):
        logger.info("Save report...")
        report = Report()
        report.requestId = settings.Config["reqid"]
        report.sourcesPath = settings.Config["actual_path"]
        report.datasetPath = settings.Config["test_data_path"]

        tokenization_options = ["language_tokenization", "normalization", "stop_words", "exclude_positions",
                                "extra_words", "exclude_categories"]
        for t in tokenization_options:
            report.preprocess[t] = settings.Config[t]
        for t in settings.dynamic_store["test_docs"]:
            report.docs[t.name] = {}
            report.docs[t.name]["actual"] = ",".join(t.nlabs)
        if not settings.Config["exclude_categories"]:
            exclude_categories = []
        else:
            exclude_categories = settings.Config["exclude_categories"].split(",")
        c_names = [''] * (len(settings.dynamic_store["predefined_categories"]) - len(exclude_categories))
        for k, v in settings.dynamic_store["predefined_categories"].items():
            if k not in exclude_categories:
                c_names[v] = k
        report.categories = c_names
        for key, val in settings.dynamic_store["results"].items():
            for i in range(len(val)):
                labs = []
                for j in range(self.qLabs):
                    #if val[i][j] >= self.rank_threshold:
                    if val[i][j] >= settings.dynamic_store["ranks"][key]:
                        labs.append("%s[%.2f]" % (c_names[j], val[i][j]))
                report.docs[settings.dynamic_store["test_docs"][i].name][key] = ",".join(labs)
        for key, val in settings.dynamic_store["metrics"].items():
            report.models[key] = val
        for key, val in settings.dynamic_store["ranks"].items():
            report.ranks[key] = val
        if len(settings.dynamic_store["results"]) > 1:
            for i in range(len(self.predictions)):
                labs = []
                for j in range(self.qLabs):
                    if self.predictions[i][j] == 1:
                        labs.append(c_names[j])
                report.docs[settings.dynamic_store["test_docs"][i].name]["consolidated"] = ",".join(labs)
            report.models["consolidated"] = self.rank_threshold
        rPath = get_abs_path(settings.Config, "reports_path") + "/" + settings.Config["reqid"] + ".json"
        with open(rPath, 'w', encoding="utf-8") as file:
            json.dump(report.to_json(), file, indent=4)
        file.close()

    def prepare_resources_for_runtime(self):
        tokenization_options = ["language_tokenization", "normalization", "stop_words", "exclude_positions",
                        "extra_words", "max_seq_len", "max_chars_seq_len", "single_doc_lang_tokenization_lib_path"]
        settings.dynamic_store["resources"]["tokenization"] = {}
        ds = datetime.datetime.now()
        self.outDir = get_abs_path(settings.Config, "saved_resources_path") + "/"
        for t in tokenization_options:
            if t != "single_doc_lang_tokenization_lib_path":
                settings.dynamic_store["resources"]["tokenization"][t] = settings.Config[t]
            elif settings.Config["language_tokenization"] == "True":
                settings.dynamic_store["resources"]["tokenization"]["single_doc_lang_tokenization_lib_path"] = \
                    self.copy_file(get_abs_path(settings.Config, "single_doc_lang_tokenization_lib_path"))
        is_w2v_needed = False
        for key, val in settings.dynamic_store["resources"]["models"].items():
            val["created_model_path"] = self.copy_file(val["created_model_path"])
            if "w2v" in val and val["w2v"] == "True":
                is_w2v_needed = True
        if not is_w2v_needed and "w2v" in settings.dynamic_store["resources"]:
            settings.dynamic_store["resources"].pop("w2v", None)
        if "w2v" in settings.dynamic_store["resources"]:
            w2v_dict = {}
            is_first_line = True
            f_embeddings = open(settings.dynamic_store["resources"]["w2v"]["created_model_path"], encoding="utf-8")
            for line in f_embeddings:
                if is_first_line == True:
                    is_first_line = False
                    continue
                split = line.strip().split(" ")
                word = split[0]
                vector = numpy.array([float(num) for num in split[1:]])
                w2v_dict[word] = vector
            f_embeddings.close()
            with open(settings.dynamic_store["resources"]["w2v"]["created_model_path"] + '.pkl', 'wb') as file:
                pickle.dump(w2v_dict, file, pickle.HIGHEST_PROTOCOL)
            file.close()
            settings.dynamic_store["resources"]["w2v"]["created_model_path"] = \
                self.copy_file(settings.dynamic_store["resources"]["w2v"]["created_model_path"] + '.pkl')
        if "indexer" in settings.dynamic_store["resources"]:
            settings.dynamic_store["resources"]["indexer"] = self.copy_file(settings.dynamic_store["resources"]["indexer"])
        if "vectorizer" in settings.dynamic_store["resources"]:
            settings.dynamic_store["resources"]["vectorizer"] = self.copy_file(settings.dynamic_store["resources"]["vectorizer"])
        if "ptBertModel" in settings.dynamic_store["resources"]:
            settings.dynamic_store["resources"]["ptBertModel"] = self.copy_file(settings.dynamic_store["resources"]["ptBertModel"])
            settings.dynamic_store["resources"]["vocabPath"] = self.copy_file(settings.dynamic_store["resources"]["vocabPath"])
        c_names = [''] * len(settings.dynamic_store["predefined_categories"])
        for k, v in settings.dynamic_store["predefined_categories"].items():
            c_names[v] = k
        with open(self.outDir + 'labels.txt', 'w', encoding="utf-8") as file:
            file.write(",".join(c_names))
        file.close()
        settings.dynamic_store["resources"]["labels"] = "labels.txt"
        settings.dynamic_store["resources"]["consolidatedRank"] = self.rank_threshold
        with open(self.outDir + 'config.json', 'w', encoding="utf-8") as file:
            json.dump(settings.dynamic_store["resources"], file, indent=4)
        file.close()
        de = datetime.datetime.now()
        logger.info("\nArtifacts are copied into the folder %s in %s"%(
            get_abs_path(settings.Config, "saved_resources_path"), get_formatted_date(ds, de)))

    def copy_file(self, in_path):
        dir, name = os.path.split(in_path)
        out_path = self.outDir + name
        shutil.copy(in_path, out_path)
        return name
