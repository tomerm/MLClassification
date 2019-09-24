import os
import logging

logger = logging.getLogger(__name__)

def get_formatted_date(ds,de):
    result = ''
    seconds = (de-ds).total_seconds()
    if seconds < 1:
        return "less than 1 sec"
    hh = seconds//3600
    if hh > 0:
        result = "%d h:"%(hh)
    seconds = seconds%(3600)
    mm = seconds//60
    if mm > 0:
        result += "%d min:"%(mm)
    ss = seconds%60
    result += "%d sec"%(ss)
    return result


def test_path(Config, path, error_msg):
    if not correct_path(Config, path):
        raise ValueError(error_msg)


def correct_path(Config, path):
    if not path:
        return False
    if not Config[path]:
        return False
    if not os.path.exists(get_abs_path(Config, path)):
        return False
    return True

''' get absolute path from Config property '''
def get_abs_path(Config, relPath, opt=""):
    result = ""
    if relPath in Config:
        result =  Config["home"] + "/" + Config[relPath]
    else:
        result =  Config["home"] + "/" + relPath
    if len(opt) > 0:
        if opt in Config:
            result += "/" + Config[opt]
        else:
            result += "/" + opt
    return result

def updateParams(Config, def_config, kwargs):
    if not kwargs:  # Reset default values
        return
    if "reset" in  kwargs.keys():
        if kwargs["reset"] == "True":
            for option, value in def_config.items():
                Config[option] = value
            logger.info("Reset parameters")
        del kwargs["reset"]
    if len(kwargs) > 0:
        for option, value in kwargs.items():
            Config[option] = value
        logger.info("Update parameters")


def align_to_left(str, size):
    if len(str) >= size:
        return str[:size]
    return str + "".join([" "] * (size - len(str)))


def arabic_charset():
    start = ord('\u0600')
    end = ord('\u06ff')
    alphabet = ''
    for i in range(start, end + 1):
        ch = chr(i)
        alphabet = alphabet + ch
    start = ord('\u0750')
    end = ord('\u077f')
    for i in range(start, end + 1):
        ch = chr(i)
        alphabet = alphabet + ch
    charDict = {}
    for idx, char in enumerate(alphabet):
        charDict[char] = idx + 1
    return charDict
