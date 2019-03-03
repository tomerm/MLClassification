from Utils.config import part_of_speach_to_remove
from dataProcessing.Tokenization.parser_opperator import run_Server,stop_Server
from dataProcessing.data_processing import DataProcessing

class Tokenize(DataProcessing):

    def __init__(self,docs,part_of_speach_to_remove):

        self.docs = docs

    def fit_transform(self):

        '''

        parse the documents and remove unwanted words

        :return: docs after removing unwanted part os speech.
        '''

        run_Server()
        #tokenization
        stop_Server()
        return self.get_docs()

