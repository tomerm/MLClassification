import datetime
from pathlib import Path
from configparser import ConfigParser, Error
from Tokenization.tokenization import Tokenizer
from WordEmbedding.vectors import Embedding
from Data.data import DataLoader
from Models.controller import ModelController
from Models.consolidation import ConsolidatedResults
from Utils.utils import defaultOptions

Config = {}

def parseConfig(path):
    parser = ConfigParser()
    parser.read_file(open(path))
    try:
        sections = parser.sections()
        for i in range(len(sections)):
            options = parser.items(sections[i])
            if sections[i] == "requests":
                if len(options) == 0 or not parser.has_option("requests", "request"):
                    print ("Config file doesn't contain request for any process. Exit.")
                    return
            for j in range(len(options)):
                Config[options[j][0]] = options[j][1]
        if not Config["home"]:
            Config["home"] = str(Path.home())
        Config["reqid"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        Config["modelid"] = 0
        Config["results"] = {}
        Config["error"] = False
        parseRequestAndLaunchPipe(parser, Config["request"])
    except Error:
        print ("Config file's parsing error. Exit.")
        return


def parseRequestAndLaunchPipe(parser, req):
    print ("=== Request " + Config["reqid"] + " ===")
    req = req.strip()
    tasks = req.split("|")
    for i in range(len(tasks)):
        task = tasks[i].replace(" ", "")
        process = task[0]
        if not (process == "T" or process == "W" or process == "D" or process == "M" or process == "C"):
            print ("Request contains wrong name of process ('%s')."%(process))
            print ("It should be one of 'T' (tokenization), 'W' (word embedding), " +
                   "'D' (data definition), 'M' (model) or 'C' (consolidate results). Exit.")
            return
        if  not (task[1] == "(" and task[-1] == ")"):
            print ("Request contains wrong definition of process ('%s'). Exit."%(task))
            return
        definition = task[2:-1]
        kwargs = {}
        if definition != "":
            options = definition.split(";")
            for j in range(len(options)):
                kvs = options[j].split("=");
                if kvs[0] not in Config:
                    print ("Request contains wrong parameter ('%s') of process '%s'. Stop."%(kvs[0], process))
                    return
                for k in range(len(kvs)):
                    kwargs[kvs[0].lower()] = kvs[1]
        if process == "T":   #Tokenization
            DefConfig = defaultOptions(parser, "tokenization")
            Tokenizer(Config, DefConfig, kwargs);
        elif process == "W":  #Word Embedding
            DefConfig = defaultOptions(parser, "word_embedding")
            Embedding(Config, DefConfig, kwargs);
        elif process == "D":  #Load data
            DefConfig = defaultOptions(parser, "data")
            lastD = (i == len(tasks)-1)
            DataLoader(Config, DefConfig, lastD, kwargs)
        elif process == "C": #Consolidated results
            ConsolidatedResults(Config)
        else:    #Model
            DefConfig = defaultOptions(parser, "model")
            ModelController(Config, DefConfig, kwargs)
        if Config["error"]:
            return
