import os
import subprocess
import datetime
from subprocess import Popen, PIPE
from nltk.corpus import stopwords
from Utils.utils import get_formatted_date, get_abs_path
import General.settings as settings
import logging

logger = logging.getLogger(__name__)

def job_preprocessor():
    worker = Preprocessor()
    worker.run()

class Preprocessor:
    def __init__(self):
        logger.info("=== Preprocessing ===")
        #self.process(settings.Config)

    def run(self):
        lib_path = get_abs_path(settings.Config, "set_of_docs_lang_tokenization_lib_path")
        logger.info("use set_of_docs_lang_tokenization")
        if not lib_path or not os.path.exists(lib_path):
            raise ValueError("Wrong path to the tagger's jar. Tokenization can't be done")
        in_path = settings.Config["home"] + "/" + settings.Config["source_path"]
        if not settings.Config["source_path"] or settings.Config["source_path"] == settings.Config["target_path"]:
            raise ValueError("Wrong source/target path(s). Tokenization can't be done.")
        out_path = settings.Config["home"] + "/" + settings.Config["target_path"]
        stop_words = ""
        stop_words = ",".join(list(stopwords.words('arabic'))) if settings.Config["stop_words"] == "True" else ""
        ds = datetime.datetime.now()
        srv = subprocess.Popen('java -Xmx2g -jar ' + lib_path + ' "' + in_path + '" "' +
                               out_path + '" "' + settings.Config["exclude_positions"] + '" "'+ stop_words + '" "' +
                               settings.Config["extra_words"] + '" "' + settings.Config["normalization"] + '" "' +
                               settings.Config["language_tokenization"] + '"',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        srv.wait()
        reply = srv.communicate()
        de = datetime.datetime.now()
        logger.info(reply[0].decode())
        logger.info("All process is done in %s" % (get_formatted_date(ds, de)))
