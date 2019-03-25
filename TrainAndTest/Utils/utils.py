
def showTime(ds,de):
    result = ''
    seconds = (de-ds).total_seconds()
    if seconds < 1:
        return "less than 1 sec"
    hh = seconds//(60*24);
    if hh > 0:
        result = "%d h:"%(hh);
    seconds = seconds%(60*24)
    mm = seconds//60;
    if mm > 0:
        result += "%d min:"%(mm)
    ss = seconds%60;
    result += "%d sec"%(ss)
    return result

def fullPath(Config, relPath, opt = ""):
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

def defaultOptions(parser, section):
    DefConfig = {}
    options = parser.items(section)
    for i in range(len(options)):
        DefConfig[options[i][0]] = options[i][1]
    return DefConfig

def updateParams(Config, DefConfig, kwargs):
    if len(kwargs) == 0:  # Reset default values
        return
    if "reset" in  kwargs.keys():
        if kwargs["reset"] == "yes":
            for option, value in DefConfig.items():
                Config[option] = value
            print("Reset parameters")
        del kwargs["reset"]
    if len(kwargs) > 0:
        for option, value in kwargs.items():
            Config[option] = value
        print("Update parameters")

def leftAlign(str, size):
    if len(str) >= size:
        return str[:size]
    return str + "".join([" "] * (size - len(str)))

def getDictionary():
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
