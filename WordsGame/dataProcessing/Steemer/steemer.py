from dataProcessing.data_processing import DataProcessing

class Steemer(DataProcessing):

    def __init__(self,docs):

        self.docs = docs


    def fit_transform(self):

        '''

        stemming the words to more simple base.

        :return: docs after stemming
        '''

        return self.get_docs()



