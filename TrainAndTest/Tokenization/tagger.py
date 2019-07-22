import os
import subprocess
import datetime
from subprocess import Popen, PIPE
from nltk.corpus import stopwords
from Utils.utils import showTime
from Utils.utils import fullPath

class TokensFromTagger:
    def __init__(self, Config):
        self.Config = Config
        self.tokenize(Config)

    def tokenize(self, Config):
        taggerPath = fullPath(Config, "taggerpath")
        if (len(taggerPath) == 0 or not os.path.exists(taggerPath)):
            print ("Wrong path to the tagger's jar. Tokenization can't be done")
            Config["error"] = True
            return
        inPath = Config["home"] + "/" + Config["sourcepath"]
        outPath = Config["home"] + "/" + Config["targetpath"]
        stopWords = ""
        if Config["stopwords"] == "yes":
            sWords = list(stopwords.words('arabic'))
            for i in range(len(sWords)):
                if i > 0:
                    stopWords += ","
                stopWords += sWords[i]
        ds = datetime.datetime.now()
        srv = subprocess.Popen('java -Xmx2g -jar ' + taggerPath + ' "' + inPath +  '" "'  +
                               outPath + '" "' + Config["expos"] + '" "'+ stopWords + '" "' +
                               Config["extrawords"] + '" "' + Config["normalization"] + '" "' +
                               Config["actualtoks"] + '"',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        srv.wait()
        reply = srv.communicate()
        de = datetime.datetime.now()
        print(reply[0].decode())
        print("All process is done in %s" % (showTime(ds, de)))