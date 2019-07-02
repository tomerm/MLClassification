import os
import datetime
from pathlib import Path
from configparser import ConfigParser, Error
from Preprocess.preprocess import Preprocessor
from WordEmbedding.vectors import Embedding
from Data.data import DataLoader
from Models.controller import ModelController
from Models.consolidation import Collector
from Utils.utils import defaultOptions, fullPath
from Info.creator import InfoCreator

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
        Config["metrics"] = {}
        Config["ranks"] = {}
        Config["resources"] = {}
        Config["resources"]["reqid"] = Config["reqid"]
        Config["resources"]["models"] = {}
        Config["resources"]["w2v"] = {}
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
        if not (process == "P" or process == "W" or process == "D" or process == "M" or process == "C"):
            print ("Request contains wrong name of process ('%s')."%(process))
            print ("It should be one of 'P' (preprocess), 'W' (word embedding), " +
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
                kvs = options[j].split("=")
                if kvs[0].lower() not in Config:
                    print ("Request contains wrong parameter ('%s') of process '%s'. Stop."%(kvs[0], process))
                    return
                for k in range(len(kvs)):
                    kwargs[kvs[0].lower()] = kvs[1]
        if process == "P":   #Preprocess
            DefConfig = defaultOptions(parser, "preprocess")
            Preprocessor(Config, DefConfig, kwargs)
        elif process == "W":  #Word Embedding
            DefConfig = defaultOptions(parser, "word_embedding")
            Embedding(Config, DefConfig, kwargs)
        elif process == "D":  #Load data
            DefConfig = defaultOptions(parser, "data")
            DataLoader(Config, DefConfig, kwargs)
        elif process == "C": #Collector
            Collector(Config)
        else:    #Model
            DefConfig = defaultOptions(parser, "model")
            ModelController(Config, DefConfig, kwargs)
        if Config["error"]:
            return

def parseConfigInfo(path):
    parser = ConfigParser()
    parser.read_file(open(path))
    try:
        sections = parser.sections()
        for i in range(len(sections)):
            options = parser.items(sections[i])
            for j in range(len(options)):
                Config[options[j][0]] = options[j][1]
        if not Config["home"]:
            Config["home"] = str(Path.home())
        if not Config["infofrom"]:
            Config["infofrom"] = "today"
        if Config["infofrom"] != "today":
            chk = Config["infofrom"].split()
            if len(chk) != 2 and not chk[1].startswith("day"):
                print ("Wrong value of 'infofrom' option. Exit.")
                return
            try:
                days = int(chk[0])
            except ValueError:
                print ("Wrong value of 'infofrom' option. Exit.")
                return
        if len(Config["reportspath"]) == 0 or not os.path.isdir(fullPath(Config, "reportspath")):
            print("Wrong path to the folder, containing reports. Exit.")
            return
        if len(Config["actualpath"]) == 0 or not os.path.isdir(fullPath(Config, "actualpath")):
            print("Warning: wrong path to the folder, containing original documents.")
            print("It will not be possible to view this documents.")
    except Error:
        print ("Config file's parsing error. Exit.")
        return
    InfoCreator(Config)


