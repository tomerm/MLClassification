# Project "TrainAndTest"
Goal of this project - provide the user with a convenient way to start the processes necessary to create ML models, used
for multi-label document classification. Processes may be launched independently or in the chain, where previous process 
can prepare data or set parameters for the next one. Each user's request should be configured and composed in the standard 
configuration file with the list of predefined sections and options (as an example, see file _config.cfg_ in the current folder).

## Execution of user's requests
Each user's request can be executed by launching script _start.py_ with one required parameter, which defines the path to 
specific configuration file, for example:

`python start.py config.cfg`

Content of the request should be placed into configuration file as a value of option _request_ from the section _requests_.

## Content of user's request. Default and actual parameters.
Currently we support 5 types of processes. Each of them has its own symbol, by which it is designated in the request:
- P  -  preprocessing of the source text
- W  -  word embedding
- D  -  data loading
- M  -  create/train/test specific model
- C  -  collector (consolidate results of all models in the chain and/or collect artifacts for runtime)

One-char sign of the process should be followed by the list of its actual parameters, enclosed in parentheses. 
This list can be empty which means, that actual parameters are the same as default. 
Default parameters are defined by the options from configuration file. Each process has there its
own section, but sometimes options from other sections can be used as well. If some actual parameter appears in the list, 
its value changes the value of corresponding option. Then it is used as a default for all subsequent processes in the chain.

The rules of creating user's request are the following:
- Definitions of specific processes in the request should be separated by symbol "|". 
- Actual parameters in the list should be seperated by symbol ";". 
- Name and value of each actual parameter should be separated by symbol "=". 
- If value of some actual parameter is a list, items of this list should be separated by comma. 

Here is the example of the request:

`request = D(w2vload=no) | M(type=perceptron; name=perceptron; runfor=test) | M(type=svc; name=svc) | C(saveResources=no)`

This means: 
- load train and test data from default folders, but not load file of word vectors;
- load model of type and name 'perceptron' from the default folder and get its predictions for testing data, 
loaded by the previous process;
- do the same with the model of type and name 'svc'
- get consolidated results (by default), but don't collect artifacts

_Note: Configuration file additionaly contains section 'root', which doesn't belong to any process. Its option 'home'
defines the root folder, where all related data should be placed. All other options, which defines references to different 
places in the file system, should contain paths relative to the root._

## Preprocess
_Preprocess_ is completely independent process, but it can be included into the chain with other process, if they need 
to load the data, which was not preprocessed yet. In this case source file (or folder) for these processes should be set the 
same, as target file (or folder) for _Preprocess_.

#### Parameters:

Name | Possible values | Comments
--- | --- | ---
actualToks | *yes*/no | 'yes' defines langiage-specific tokenization, no - simple white space tokenization
sourcePath | <path> | Relative path to the folder or file, containing source data.
targetPath | <path> | Relative path to the folder or file, which will contain results of tokenization
taggerPath | <path> | Relative path to jar with tagger, which performs all preprocessing
rtTaggerPath | <path> | Relative path to jar with tagger, which performs tokenization only (used by DataLoader and in runtime)
exPOS | PUNC,DT,IN,CD,PRP,RP,RB,W,PDT | List of POS's, which should be excluded from results of tokenization.
normalization | yes/no | Defines the need in text normalization
stopWords | yes/no | Defines the need to exclude stop words from results of tokenization.
extraWords | <list> / empty | List of extra words, which should be excluded from results of tokenization.

_Note: 'list' can be empty, but if it contains few items, they should be separated by comma without spaces._

_Note: if source and target paths define folders, all content of the target path will be removed and recreated 
in accordance with the files tree of the source path._

## Word Embedding
_WordEmbedding_ is a process, used to create new W2V model and save it into the file of vectors.

#### Parameters:

Name | Possible Values | Comments
--- | --- | ---
w2vCreate | yes/*no* | If 'yes', new W2V model will be created. Use 'no', if you need change some parameters only.
w2vCorpusPath | <path> | Relative path to the file, containing text corpus.
w2vEpochs | 100 | Number of epochs in training W2V model
w2vDim | 100 | Dimension of created word vectors
w2vModelPath | <path> | Relative path to W2V model (file with word vectors)

_Note: 'w2vDim' and 'w2vModelPath' can be used by other processes, which need word embedding._

## Data Loading
_DataLoader_ load all data, needed for training and testing models (including W2V data, if required). All this is placed into 
the internal cache and can be reused by the other processes in the chain.     
In addition it can perform per-line preprocessing of the data.

_Note: per-line preprocessing significantly slows down the data loading._

#### Parameters:

Name | Possible Values | Comments
--- | --- | ---
trainPath | <path> | Relative path to the folder, containing train or all data.
testPath | <path> / empty | Relative path to the folder, containing test data
testSize | 0 | Size of test data set as a part of train data set (used if testPath is empty).
valSize | 0.15 | Size of validation data set as a part of train data set.
exCats | <list> / empty | List of categories, which should be excluded from training and testing.
analysis | *yes*/no | Defines the need to show data set analysis.
w2vLoad | *yes*/no | Defines if W2V model should be loaded.
dataToks | yes/no | Defines if data should be preprocessed.

_Note: If W2V model will not be loaded by _DataLoader_, it will be loaded and placed into the internal cache by first 
in the chain _Model_ process, which require word embedding._

## Model
_Model_ is a process, which can create, train and test ML model of one of supported types. It uses the data, loaded by 
_DataLoader_.

#### Parameters:

Name | Possible Values | Comments
--- | --- | ---
type |   | Type of the model. One of SNN, LTSM, CNN, PAC, Perceptron, Ridge, SGD, SVC, BERT
name |   | Name of the model (e.g., name of the file, containing model)
modelPath | <path> | Relative path to the folder, containing models
runFor |   | Type of execution. One of trainAndTest, train, test, crossValidation and none.
epochs | 15 | Number of epochs in model's training. Depends on specific model.
trainBatch | 128 | Batch size for training.
verbose | 0/1 | Logging level in model's training
tempSave | yes/no | Defines the need to save intermediate results in the process of model's training
tempPath | <path> | Relative path to the folder, containing intermediate results.
indexerPath | <path> | Relative path to indexer (saved with some specific models)
binarizerPath | <path> | Relative path to binarizer (saved with some specifc models)
vectorizerPath | <path> | Relative path to vectorizer (saved with some specific models)
bertPath | <.../pytorch_bert.gz> | Relative path to the gz file, containing pre-trained BERT model
bertOutPath | <path> | Path to the folder with resulting BERT files
showMetrics | yes/no | Need to show testing results (model's metrics) 
kfold | 10 | Number of cross-validation cycles
cvSave | yes/no | Need to save datasets, correpond to the cross-validation cycle with the best results
cvPath | <path> | Path to the folder containing datasets used in cross-validation cycle with the best results
customRank | yes/no | Use default (0.5) or custom rank threshold
rankThreshold | <=1.0 | Value of custom rank threshold

_Note: Models PAC, Perceptron, Ridge and SVC don't return probabilities of predictions, so their results can't be
controlled by `rankThreshold` option._

## Collector
_Collector_ is a process which, if required, can perform the following tasks:
- Show consolidated results of testing of all models, used in the current chain
- Create HTML reports
- Create configuration file and collect all artifacts, needed for launching current chain in runtime.

#### Parameters

Name | Possible Values | Comments
--- | --- | ---
showResults | yes/no | Show consolidated results
reports | yes/no | Calculate and save reports (as a files in json format)
reportsPath | <path> | Path to the folder containing reports
saveResources | yes/no | Collect artifacts for runtime
resourcesPath | <path> | Path to the folder containing collected artifacts. Content of this folder should be copied into folder /app/resources belonging to repository MLClassificationDocker.
consolidatedRank | yes/no | Use default (0.5) or custom rank for consolidated results
consolidatedRankThreshold | <=1.0 | Value of custom rank threshold for consolidated results

_Note: folder `resourcePath` will contain only those resources, which belong to the model(s), tested in the current pipe. Resources, used by models in cross-validation mode, aren't saved._

## Show and compare results of user's requests
Results of user's requests, previously created and saved by the process _Collector_, can be converted into HTML form and shown by  the script _info.py_, launched with one required parameter - path to specific configuration file, for example:

`python info.py config.cfg`

In general it is the same configuration file, which was used by the script `start`. Data needed for `info.py`, should be placed there as a value of the option _infofrom_ from the section _requests_. This option defines the period, in which requests were created, and may contain or word "today", or text in the form `X days`, here X is any number.    

Example of resulting HTML file (`curInfo.html`) can be found in the current folder.
