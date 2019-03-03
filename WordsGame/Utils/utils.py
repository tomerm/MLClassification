


class Loader:

    def __init__(self,path,**kw_args):

        '''

        Init for loader data.

        :param path: Path for a directory which contain folders. Each folder is a category which contains txt files,
        each file is a document
        :param kw_args: More parameters for the model. categories [list] - if the user want to load only specific
        categories
        '''

        self.path = path
        self.categories = kw_args.get("categories",None)

    def load(self):
        '''
        the function will load the data into namedtuple structure. this will be the entire data (not split into test and
        train)

        :return: namedtuple structure
        '''

    def getCategories(self):
        '''

        get mapping between categories and integers. If specific categories were selected than only those
        will be mapped

        :return: mapping between categories and integers
        '''

    def create_evel(self):
        '''
        split train to train and evel for NN models

        :return:
        '''

class documentAnalysis:

    def __init__(self,plt,docs,mode):
        '''

        :param plt: plt for ploting
        :param docs: the docs to run analysis on
        :param mode: train/test
        '''

        self.plt = plt
        self.docs = docs
        self.mode = mode

    def run_analysis(self):
        self.getLabelSets()
        self.showDocsByLength()
        self.showDocsByLabs()

    def getLabelSets(self):
        pass

    def showDocsByLength(self):
        pass

    def showDocsByLabs(self):
        pass


def showTime(ds,de):
    '''

    :param ds: start time
    :param de: end time
    :return: string of amount of time the function ran.
    '''


def prepare_categories(categories):
    nmCats = [""] * len(categories)
    cKeys = list(categories.keys())
    for i in range(len(cKeys)):
        nmCats[categories[cKeys[i]]] = cKeys[i]
    return nmCats