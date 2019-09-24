
class Report(object):
    def __init__(self):
        self.requestId = ""
        self.sourcesPath = ""
        self.datasetPath = ""
        self.preprocess = {}
        self.categories = []
        self.docs = {}
        self.models = {}
        self.ranks = {}

    def to_json(self):
        obj = {}
        obj["requestId"] = self.requestId
        obj["sourcesPath"] = self.sourcesPath
        obj["datasetPath"] = self.datasetPath
        obj["preprocess"] = self.preprocess
        obj["categories"] = self.categories
        obj["docs"] = self.docs
        obj["models"] = self.models
        obj["ranks"] = self.ranks
        return obj
