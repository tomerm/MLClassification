from Tokenization.server import tokens_from_server
from Tokenization.tagger import tokens_from_tagger
import General.settings as settings
import logging

logger = logging.getLogger(__name__)

class Tokenizer:
    def __init__(self):
        logger.info("=== Tokenization ===")
        #if settings.Config["language_tokenization"] != "True":
        #    return
        if not settings.Config["source_path"] or settings.Config["source_path"] == settings.Config["target_path"]:
            logger.error("Wrong source/target path(s). Tokenization can't be done.")
            #settings.Config["error"] = True
            return
        if settings.Config["typetoks"] == "server":
            tokens_from_server(settings.Config)
        elif settings.Config["typetoks"] == "tagger":
            tokens_from_tagger(settings.Config)
        else:
            logger.error("Wrong tokenization type. Tokenization can't be done.")
