from evaluation.evaluate import evaluate
from Utils.config import categories,p,models_ro_run
from evaluation.calculate_metrics import CalculateMetrics


class TrainTestRandomSplit(evaluate):

    def __int__(self,docs):

        self.docs = docs
        self.categories = categories
        self.p = p
        self.documents_in_train = []


    def run(self):

        '''

        split the docs into train set and test set in the size of p (if p = 0.2 than the test size will be 20% from the data,
        each category will "give" 80% from the category data for the training in order that all categories will be in similar size
        After it we will train on the train test and try it on the test set and will retrieve the results

        :return: dictionary with the metrics and their values.
        '''

        self.split_data(self.docs)
        for model in models_ro_run:
            model.fit()
            model.predict()
            CalculateMetrics()


    def get_documents_in_train(self):
        return self.documents_in_train

    def split_data(self):
        '''

        split data into p% test and (1-p)% train.

        :return:
        '''
