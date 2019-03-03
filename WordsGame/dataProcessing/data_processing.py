
'''

In case the implementation will be after loading the data than here will be the "main" code for running all the
data processing. each element will run if the user wanted.

for example if the user wanted stemming than we will have a parameter stemming = True. It can be written in the config
file.

If the data process will be done in the loader than it can be done by calling the classes directly from it (I think it better)


'''

from dataProcessing.Steemer.steemer import Steemer
from dataProcessing.stopwords.stop_words import StopWords
from dataProcessing.Tokenization.tokenize import Tokenize
from Utils.config import data_preperation_pipeline
from sklearn.pipeline import Pipeline


class DataProcessing:


    def __init__(self, doc, **kw_args):

        self.doc = doc


    def process(self):

        pipeline = Pipeline(data_preperation_pipeline)
        return pipeline.fit_transform(self.doc)


    def get_docs(self):
        return self.docs