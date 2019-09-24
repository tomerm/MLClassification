import os
import subprocess
import datetime
from subprocess import Popen, PIPE
from nltk.corpus import stopwords
from Utils.utils import get_formatted_date, get_abs_path, test_path
import logging

logger = logging.getLogger(__name__)

def tokens_from_tagger(Config):
    logger.info("run tokens_from_tagger()")
    test_path(Config, "set_of_docs_lang_tokenization_lib_path",
              "Wrong path to the tagger's jar. Tokenization can't be done")
    tagger_path = get_abs_path(Config, "set_of_docs_lang_tokenization_lib_path")
    source_path = Config["home"] + "/" + Config["source_path"]
    target_path = Config["home"] + "/" + Config["target_path"]
    stop_words = ",".join(list(stopwords.words('arabic'))) if Config["stop_words"] == "True" else ""
    ds = datetime.datetime.now()
    srv = subprocess.Popen('java -Xmx2g -jar ' + tagger_path + ' "' + source_path + '" "' + target_path + '" "' +
                                Config["exclude_positions"] + '" "'+ stop_words + '" "' +
                                Config["extra_words"] + '" "' + Config["normalization"] + '" "' +
                                Config["language_tokenization"] + '"',
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    srv.wait()
    reply = srv.communicate()
    de = datetime.datetime.now()
    logger.info(reply[0].decode())
    logger.info("All process is done in %s" % (get_formatted_date(ds, de)))
