from evaluation.evaluate import evaluate
from Utils.config import kfold,categories,models_ro_run
from evaluation.calculate_metrics import CalculateMetrics


class CrossValidation(evaluate):

    def __init__(self,docs):

        self.docs = docs
        self.best_results = dict()
        self.worst_results = dict()
        self.documents_in_train_for_best_results = []
        self.documents_in_train_for_worst_results = []


    def run(self):

        '''

        run kfold iterations on the data where (kfold-1) parts from the data are set to be train and the rest for testing.
        In the end the results will be the average for all iterations. we will track the best results and worst and will
        save the documents we trained on in the best/worst results. this is done in order to analize the data after it.

        :return: dictionary of the average data for all the metrics
        '''

        split_data(self.docs)
        for model in models_ro_run:
            model.fit()
            model.predict()
            CalculateMetrics()



    def split_data(self):
        '''

        split data into k train and test sets.

        :return:
        '''



    def get_best_results(self):
        return self.best_results

    def get_wrost_results(self):
        return self.worst_results

    def get_documents_in_train_for_best_results(self):
        return self.documents_in_train_for_best_results

    def get_get_best_results(self):
        return self.documents_in_train_for_worst_results