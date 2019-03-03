from Utils.config import w2v_path
from WordEmbedding.word2vec import W2V


class NNModels:

    def __init__(self):
        pass


    def load_w2v(self):
        '''
        load word vectors from w2v_path path

        :return: word vectors
        '''

        return W2V.load_w2v(w2v_path)


    def getModel(self):
        raise NotImplementedError


    def prepare_data(self):

        '''

        prepare the data for the model

        :return:
        '''

        raise NotImplementedError

    def fit(self):

        '''

        All the NN models have the same fit, it should be written here. It should worked in either case if I'm using
        NN or SKlearn

        :return:
        '''

        pass

    def predict(self):

        '''

        predict for set documents, suppose to be like SKlearn.

        :return:
        '''

        pass


