import re
from nltk.corpus import stopwords
from nltk.stem.util import suffix_replace


def join_tokens(tArr, Config):
    toks = [x[0] for x in tArr]
    tags = [x[1] for x in tArr]
    result = ''
    normalizer = ArabicNormalizer()
    if Config["stop_words"]:
        stop_words = set(stopwords.words('arabic'))
    else:
        stop_words = set()
    exclude_positions = Config["exclude_positions"].split(",")
    exWords = Config["extra_words"].split(",")
    for i in range(len(tArr)):
        ftok = ''
        if i > 0:
            result += ' '
        if tags[i] in exclude_positions or tags[i] in stop_words or tags[i] in exWords:
            continue
        else:
            ftok = toks[i]
            if Config["normalization"]:
                ftok = normalizer.normalize(ftok)
            result += ftok
    return result

class ArabicNormalizer(object):
    __vocalization = re.compile(r'[\u064b-\u064c-\u064d-\u064e-\u064f-\u0650-\u0651-\u0652]')
    __kasheeda = re.compile(r'[\u0640]') # tatweel/kasheeda
    __arabic_punctuation_marks = re.compile(r'[\u060C-\u061B-\u061F]')
    __last_hamzat = ('\u0623', '\u0625', '\u0622', '\u0624', '\u0626')
    __initial_hamzat = re.compile(r'^[\u0622\u0623\u0625]')
    __waw_hamza = re.compile(r'[\u0624]')
    __yeh_hamza = re.compile(r'[\u0626]')
    __alefat = re.compile(r'[\u0623\u0622\u0625]')

    def normalize(self, token):
        """
        :param token: string
        :return: normalized token type string
        """
        # strip diacritics
        token = self.__vocalization.sub('', token)
        #strip kasheeda
        token = self.__kasheeda.sub('', token)
        # strip punctuation marks
        token = self.__arabic_punctuation_marks.sub('', token)
        # normalize last hamza
        for hamza in self.__last_hamzat:
            if token.endswith(hamza):
                token = suffix_replace(token, hamza, '\u0621')
                break
        # normalize other hamzat
        token = self.__initial_hamzat.sub('\u0627', token)
        token = self.__waw_hamza.sub('\u0648', token)
        token = self.__yeh_hamza.sub('\u064a', token)
        token = self.__alefat.sub('\u0627', token)
        return token
