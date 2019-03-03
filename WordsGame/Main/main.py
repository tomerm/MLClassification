from Utils.config import method
from evaluation.cross_validation import CrossValidation
from evaluation.train_test import TrainTest
from evaluation.train_test_p_split import TrainTestRandomSplit


def main():

    if method == "cross_validation":
        cv = CrossValidation()
        cv.eval()
    elif method == "train_test":
        tt = TrainTest()
        tt.eval()
    elif method == "train_test_random_split":
        ttrs = TrainTestRandomSplit()
        ttrs.eval()


