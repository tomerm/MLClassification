import numpy
import random
import glob
import shutil
import datetime
import os
import subprocess
import threading
import time
import signal
from subprocess import Popen, PIPE
from pathlib import Path
from nltk.parse.corenlp import CoreNLPParser
from Utils.utils import get_formatted_date, get_abs_path
from Tokenization.utils import join_tokens
import logging

logger = logging.getLogger(__name__)
process_started = False
initial_dir = ""


def tokens_from_server(Config):
    global initial_dir, process_started
    logger.info("run tokens_from_server()")
    initial_dir = os.getcwd()
    if Config["servstop"] == "True":
        stop_server()
    process_started = True
    start_server(Config)
    tokenize(Config)
    stop_server()


def start_server(Config):
    stanford_path = get_abs_path(Config, "servsource") + "/"
    os.chdir(stanford_path)
    os.environ["CLASSPATH"] = "*"

    def run_server(restore_initial_dir, popenArgs):
        def runInThread(restore_initial_dir, popenArgs):
            srv = Popen('java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties ' +
                        stanford_path + 'StanfordCoreNLP-arabic.properties -preload tokenize,ssplit,pos ' +
                        '-status_port ' + Config["servport"] +
                        ' -port ' + Config["servport"] + ' -timeout 20000',
                        shell=True)
            srv.wait()
            restore_initial_dir()
            return

        thread = threading.Thread(target=runInThread, args=(restore_initial_dir, ''))
        thread.start()
        return thread

    def restore_initial_dir():
        os.chdir(initial_dir)
        logger.warning("Server is down")

    run_server(restore_initial_dir, '')
    time.sleep(10)
    logger.info("Server is running")


def tokenize(Config):
    parser = CoreNLPParser(url='http://localhost:' + Config["servport"], tagtype='pos')
    inPath = Config["home"] + "/" + Config["source_path"]
    outPath = Config["home"] + "/" + Config["target_path"]

    fds = datetime.datetime.now()
    tokenize_data(Config, parser, inPath, outPath)
    fde = datetime.datetime.now()
    logger.info("Tokenization complited in %s" % (get_formatted_date(fds, fde)))


def tokenize_data(Config, parser, inPath, outPath):
    global initial_dir, process_started
    if not os.path.exists(inPath):
        logger.error("Source file or folder %s doesn't exist. Tokenization can't be done." % inPath)
        #Config["error"] = True
        return
    if not os.path.isdir(inPath):
        tokenize_file(Config, parser, inPath, outPath)
        if process_started:
            return
    if process_started:
        if os.path.exists(outPath):
            shutil.rmtree(outPath)
        os.mkdir(outPath)
    process_started = False
    os.chdir(inPath)
    for ff in glob.glob("*"):
        if os.path.isdir(inPath + "/" + ff):
            dPath = inPath + "/" + ff
            tPath = outPath + "/" + ff
            if os.path.exists(tPath):
                shutil.rmtree(tPath)
            os.mkdir(tPath)
            tokenize_data(Config, parser, dPath, tPath)
        else:
            iPath = inPath + "/" + ff
            oPath = outPath + "/" + ff
            tokenize_file(Config, parser, iPath, oPath)


def tokenize_file(Config, parser, inPath, outPath):
    outFile = open(outPath, 'w', encoding='UTF-8')
    ds = datetime.datetime.now()
    q = 0
    qt = 0
    with open(inPath, 'r', encoding='UTF-8') as f:
        for line in f:
            q += 1
            if q > 1:
                result = '\n'
            else:
                result = ''
            line = line.replace('\r', '').replace('\n', '')
            if not line:
                continue
            toks = line.split()
            if len(toks) < 3:
                continue
            qt += len(toks)
            tArr = parser.tag(line.split())
            result += join_tokens(tArr, Config).strip()
            outFile.write(result)
    de = datetime.datetime.now()
    logger.info("File %s (%d lines, %d tokens): in %s" % (outPath, q, qt, get_formatted_date(ds, de)))
    f.close()
    outFile.close()


def stop_server():
    cmd = """ps -ef | grep StanfordCoreNLPServer | awk '$8=="java" {print $2}'"""
    ps = Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    reply = ps.communicate()
    try:
        txt = reply[0].decode()
        os.kill(int(txt), signal.SIGTERM)
    except ValueError:
        return
