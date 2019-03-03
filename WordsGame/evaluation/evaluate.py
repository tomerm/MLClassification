
from dataProcessing.data_processing import DataProcessing
from Utils.config import categories,doc_path,nn_flag
from Utils.utils import Loader
if nn_flag:
    from Utils.config import test_path


class evaluate:

    def __init__(self,**kw_args):
        self.docpath = doc_path
        self.testpath = test_path

    def eval(self):
        '''
        main code. here the main process is conducted. the user chose the method to use. run function suppose to
        run fit and predict the test data

        :return:
        '''

        if self.method != 'train_test':
            docs = Loader(self.docpath,categories)
            docs = DataProcessing(docs).process()
            self.run(docs)

        else:
            train = Loader(self.docpath,categories)
            test = Loader(self.testpath,categories)
            train = DataProcessing(train).process()
            test = DataProcessing(test).process()
            self.run(train,test)

    def run(self):
        raise NotImplementedError

