from dataProcessing.stopwords.stop_words import StopWords
from dataProcessing.Steemer.steemer import Steemer
from dataProcessing.Tokenization.tokenize import Tokenize
from models.sklearn_models.Ridge import Ridge
from models.sklearn_models.SGD import SGD
from models.nn_models.NN import NN
from dataProcessing.Extra_words.extra_words import ExtraWords

#preprocess

'''
Below are all the parameters that concern the preprocesse part, such as which methods to use and more
'''
########


'''In the list below each method is a part in the preprocesse, like remove stopwords. The process is build
on python pipeline. each process has a function name "fit_transform" which responsible for the preocess. In
order to add a new process just add ("name",Process class()) to the list.
'''
data_preperation_pipeline = [("stem",Steemer()),("sw",StopWords()),("ew",ExtraWords()),("tk",Tokenize())]

#tokenization parameters

'''
Below add the POS tags you wish to remove from the text. leave the list empty if there are no POS you wish
to remove
'''
part_of_speach_to_remove = ["NN","PUNC"]

#words to remove

'''
below add specific words that you wish to remove from the text. leave the list empty in case there are no
words you wish to remove
'''
extra_words_to_remove = ["dev","extract","sheet"]


#choose categories

'''

below add specific words that you wish to train and predict on. In case you want to run over all categories leave
it empty
'''

categories = ["sport","economic"]

#models

'''
Below add specfic models you wish to train with. you must have at least 1 model.
'''
models_ro_run = [Ridge(),SGD(),NN()]

#if there are nn models in the list above

nn_flag = True

#path configoration

doc_path = "path"
models_path = "path"


#methodology

'''
There are 3 methodologies. 
1)cross_validation
2)train_test_p_split
3)train_test

Please select one of the following and write it in the parameter "method" below
'''

method = "cross_validation"

'''
In case the method is train_test than you must add test folder path 
'''
if method == "train_test":
    test_path = "path"

'''
In case the method is cross_validation than you should specify the k parameter
'''
if method == "cross_validation":
    kfold = 10

'''
In case the method is train_test_p_split than you should specify the  parameter
'''
if method == "train_test_p_split":
    p = 0.2


## NN models

'''
In case there are NN algorithms in the set of models the following parameters are needed to be set.
'''

if nn_flag:
    w2v_path = "path"
    val = 0.1
    epoc = 15
    n_dim = 100 # should be like the vectors you intend to load
    batch_size = 128



