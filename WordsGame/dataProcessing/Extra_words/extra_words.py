from Utils.config import extra_words_to_remove
from dataProcessing.data_processing import DataProcessing

class ExtraWords(DataProcessing):

    def __init__(self,docs):

        self.docs = docs


    def fit_transform(self):

        '''

        remove all extra words from extra_words_to_remove

        :return: docs after removing
        '''

        return self.get_docs()





