from evaluation.evaluate import evaluate


class TrainTest(evaluate):

    def __init__(self, traindocs, testdocs,categories):

        self.categories = categories
        self.traindocs = traindocs
        self.testdocs = testdocs


    def run(self):

        '''

        In case we want train the data from specific path and test it on different data from other path (similar to the
        notebooks) than we will use this class. the results will be the same.
        :return:
        '''

        for model in models_ro_run:
            model.fit()
            model.predict()
            CalculateMetrics()




