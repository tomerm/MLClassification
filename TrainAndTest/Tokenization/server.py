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
from Utils.utils import showTime
from Utils.utils import fullPath
from Tokenization.utils import joinTokens

class TokensFromServer:
    def __init__(self, Config):
        self.Config = Config
        self.curdir = os.getcwd()
        if Config["servstop"] == "yes":
            self.stopServer()
        self.startProcess = True
        self.startServer()
        self.tokenize()
        self.stopServer()

    def startServer(self):
        stanford_path = fullPath(self.Config, "servsource") + "/"
        os.chdir(stanford_path)
        os.environ["CLASSPATH"] = "*"

        def runServer(onExit, popenArgs):
            def runInThread(onExit, popenArgs):
                srv = Popen('java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties ' +
                            stanford_path + 'StanfordCoreNLP-arabic.properties -preload tokenize,ssplit,pos ' +
                            '-status_port ' + self.Config["servport"] +
                            ' -port ' + self.Config["servport"] + ' -timeout 20000',
                            shell=True)
                srv.wait()
                onExit()
                return

            thread = threading.Thread(target=runInThread, args=(onExit, ''))
            thread.start()
            return thread

        def onExit():
            os.chdir(self.curdir)
            print("Server is down")

        runServer(onExit, '')
        time.sleep(10)
        print ("Server is running")

    def tokenize(self):
        parser = CoreNLPParser(url='http://localhost:' + self.Config["servport"], tagtype='pos')
        inPath = self.Config["home"] + "/" + self.Config["sourcepath"]
        outPath = self.Config["home"] + "/" + self.Config["targetpath"]

        fds = datetime.datetime.now()
        self.tokenizeData(parser, inPath, outPath)
        fde = datetime.datetime.now()
        print("Tokenization complited in %s" % (showTime(fds, fde)))


    def tokenizeData(self, parser, inPath, outPath):
        if not os.path.exists(inPath):
            print ("Source file or folder %s doesn't exist. Tokenization can't be done."%(inPath))
            self.Config["error"] = True
            return
        if not os.path.isdir(inPath):
            self.tokenizeFile(parser, inPath, outPath)
            if self.startProcess:
                return
        if self.startProcess:
            if os.path.exists(outPath):
                shutil.rmtree(outPath)
            os.mkdir(outPath)
        self.startProcess = False
        os.chdir(inPath);
        for ff in glob.glob("*"):
            if os.path.isdir(inPath + "/" + ff):
                dPath = inPath + "/" + ff
                tPath = outPath + "/" + ff
                if os.path.exists(tPath):
                    shutil.rmtree(tPath)
                os.mkdir(tPath)
                self.tokenizeData(parser, dPath, tPath)
            else:
                iPath = inPath + "/" + ff
                oPath = outPath + "/" + ff
                self.tokenizeFile(parser, iPath, oPath)

    def tokenizeFile(self, parser, inPath, outPath):
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
                if len(line) == 0:
                    continue
                toks = line.split()
                if len(toks) < 3:
                    continue
                qt += len(toks)
                tArr = parser.tag(line.split())
                result += joinTokens(tArr, self.Config).strip()
                outFile.write(result)
        de = datetime.datetime.now()
        print("File %s (%d lines, %d tokens): in %s" % (outPath, q, qt, showTime(ds, de)))
        f.close()
        outFile.close()

    def stopServer(self):
        cmd = """ps -ef | grep StanfordCoreNLPServer | awk '$8=="java" {print $2}'"""
        ps = Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        reply = ps.communicate()
        try:
            txt = reply[0].decode()
            os.kill(int(txt), signal.SIGTERM)
        except ValueError:
            return
