from dataProcessing.data_processing import DataProcessing

class StopWords(DataProcessing):


    def __init__(self,docs,**kw_args):

        '''

        :param docs: the documents in namedtuple structure
        '''

        self.docs = docs


    def fit_transform(self):

        '''

        remove all unwanted words (stop words)

        :return: docs after removing words.
        '''
        return self.get_docs()