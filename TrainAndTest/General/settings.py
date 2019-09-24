import datetime
from pathlib import Path
from configparser import ConfigParser

from Utils.utils import test_path
import json
import logging

logger = logging.getLogger(__name__)

dynamic_store = {}
Config = {'show_plots': 'False', 'use_java': 'True', 'language_tokenization': 'True',
           'exclude_positions': '', 'normalization': 'True', 'stop_words': 'True',
           'extra_words': '', 'need_create_model': 'True', 'vectors_dimension': '100',
           'epochs_total': '100', 'include_current_time_in_model_name': 'False',
           'test_data_size': '0', 'validation_data_size': '0.15', 'exclude_categories': '',
           'analysis': 'False', 'load_w2v_model': 'True', 'enable_tokenization': 'False',
           'type': 'bert', 'name': '', 'epochs': '20', 'train_batch': '128', 'test_batch': '8',
           'verbose': '1', 'save_intermediate_results': 'True', 'type_of_execution': 'trainandtest',
           'cross_validations_total': '2', 'save_cross_validations_datasets': 'True',
           'show_test_results': 'True', 'customrank': 'True', 'rank_threshold': '0.5',
           'show_consolidated_results': 'False', 'save_reports': 'True',
           'prepare_resources_for_runtime': 'True', 'consolidatedrank': 'True',
           'consolidated_rank_threshold': '0.5'}
def_config = {}
jobs_list = []

jobs_def = {"P": "", "W": "", "D": "", "M": "", "C" : ""}

def parse(path, init_def_config):
    if path.endswith('.cfg'):
        parser = ConfigParser()
        parser.read_file(open(path))
        for s in parser.sections():
            if init_def_config:
                def_config[s] = parser.items(s)
            for opt in parser.items(s):
                Config[opt[0]] = opt[1]
    elif path.endswith('.json'):
        with open(path) as json_file:
            config_data = json.load(json_file)
            for k, v in config_data.items():
                if init_def_config:
                    def_config[k] = [(_k, _v) for _k, _v in v.items() ]
                for key, value in v.items():
                    Config[key] = value
    else:
        logger.error('Wrong configuration file format, only cfg or json are allowed')
        return False

    if "home" not in Config:
        Config["home"] = str(Path.home())
		
    Config["type_of_execution"] = Config["type_of_execution"].lower()
    return True

def parse_config(path):
    if False == parse(path, True):
        return False
    Config["reqid"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dynamic_store["predefined_categories"] = {}
    dynamic_store["metrics"] = {}
    dynamic_store["ranks"] = {}
    dynamic_store["modelid"] = 0	
    dynamic_store["results"] = {}
    dynamic_store["resources"] = {}	
    dynamic_store["resources"]["w2v"] = {}	
    dynamic_store["resources"]["models"] = {}
    return True

def parse_request(req):
    logger.info("=== Request " + Config["reqid"] + " ===")
    logger.info(req)
    if not req:
        req = (Config["request"]).strip().replace(" ", "")
    if not req:
        raise ValueError("Request is not defined, nothing to do")
    tasks = req.split("|")
    for task in tasks:
        task_name = task[0]
        if task_name not in jobs_def.keys():
            raise ValueError("Wrong task name, should be one of P,W,M,D,C: " + task_name)
        if not (task[1] == "(" and task[-1] == ")"):
            raise ValueError("Wrong definition of task name ('%s'). Exit." % task)
        definition = task[2:-1]
        kwargs = {}
        if definition != "":
            for option in definition.split(";"):
                kvs = option.split("=")
                if kvs[0].lower() not in Config:
                    raise ValueError("Wrong parameter ('%s') of task name '%s'. Stop." % (kvs[0], task_name))
                for k in range(len(kvs)):
                    kwargs[kvs[0].lower()] = kvs[1]
        jobs_list.append((task_name, kwargs))

def parse_config_info(path):
    if False == parse(path, False):
        return False
    if 'info_from' not in Config:
        Config['info_from'] = "today"
    if Config["info_from"] != "today":
        chk = Config["info_from"].split()
        if len(chk) != 2 and not chk[1].startswith("day"):
            logger.error("Wrong value of 'info_from' option. Exit.")
            return False
        try:
            days = int(chk[0])
        except ValueError:
            logger.error("Wrong value of 'info_from' option. Exit.")
            return False
    test_path(Config, "reports_path", "Wrong path to the folder, containing reports. Exit.")
    test_path(Config, "actual_path",
              "Warning: wrong path to the folder containing original documents. It will not be possible to view them.")

    return True


