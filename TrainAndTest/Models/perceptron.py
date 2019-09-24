import logging
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import Perceptron
from Models.base import BaseModel
from Models.dataPreparation import DataPreparation
from Utils.utils import get_abs_path
import General.settings as settings

logger = logging.getLogger(__name__)

class PerceptronModel(BaseModel):
    def __init__(self):
        super().__init__()
        if not self.is_correct_path():
            raise Exception
        self.use_probabilities = False
        self.handle_type = "vectorize"
        if settings.Config["type_of_execution"] != "crossvalidation":
            self.prepare_data()

    def prepare_data(self):
        logger.info("Start data preparation...")
        dp = DataPreparation(self, False)
        dp.get_data_for_sklearn_classifiers()

    def create_model(self):
        return OneVsRestClassifier(Perceptron(n_jobs=-1, max_iter=20, tol=1e-3))

    def load_model(self):
        self.model = self.load_skl_model()

    def train_model(self):
        self.train_skl_model()

    def test_model(self):
        self.test_skl_model()

    def save_additions(self):
        if not "vectorizer" in settings.dynamic_store["resources"]:
            settings.dynamic_store["resources"]["vectorizer"] = get_abs_path(settings.Config, "vectorizer_path")
        self.resources["vectorizer"] = "True"
