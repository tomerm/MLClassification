## Prerequisites
Before launching the notebook **Multi-label text classification with BERT(Pytorch)** for the first time, you need to install **pytorch-pretrained-BERT** in your Anaconda environment:
```
pip install pytorch-pretrained-BERT
```

## Dataset properties
Before creating some specific model for document classification, it is important to understand the specifics of dataset, 
used for its training. Given a multi-label dataset, following data properties can be defined to compare different sets of data:
1. **Distinct Label Set** is the total count of number of distinct label combinations observed in the given dataset.
2. **Proportion of Distinct Label Set** is the Distinct Label Set, normalized by total number of examples.
3. **Label Cardinality** is the average number of labels per example.
4. **Label Density** is Label Cardinality, normalized by the the number of labels.

## Evaluation Metrices
Evaluation of some learning algorithm is a measurement of how far the learning system predictions are from the actual 
class labels, tested on some unseen data. To determine the quality of the created model, we use the following evaluation
metrices:
1. **Exact Match Ratio** considers only correctly predicted labels.
2. **Accuracy** is defined for each specific instance as the proportion of the correctly predicted labels to the sum 
of actual and incorrectly predicted labels for that instance. Overall accuracy is the average across all instances.
3. **Precision** is the proportion of correctly predicted labels to the total number of actual labels, 
averaged over all instances.
4. **Recall** is the proportion of correctly predicted labels to the total number of predicted labels, averaged 
over all instances.
5. **F1-Measure** is a harmonic mean of Precision and Recal
6. **Macro Averaged Measures (Precision, Recall and F1)** are computed on individual class labels first and then 
averaged over all classes.
7. **Micro Averaged Measures (Precision, Recall and F1)** are computed globally over all instances and all class labels.
8. **Hamming  Loss** reports  how  many  times  on  average,  the  relevance  of  an example to a class 
label is incorrectly predicted.
9. **One error** measures how many times the top ranked predicted label is not in the set of true labels of the instance.
10. **Coverage** is the metric that evaluates how far on average a learning algorithm need to go down in the ordered 
list of prediction to cover all the true labels of an instance.
11. **Average precision** computes for each correctly predicted label the proportion of incorrectly labels 
that are ranked before it, and finally averages over all relevant labels.

_Note: unlike other metrices, smaller value of last four means the better performance of the learning algorithm._

## Comparison of models
After creating model, we save all related data into the file with specific structure. In addition to general information, including evaluation metrices, it contains data in the context of individual categories and documents. By default this file has the same name, as the model itself and it is placed into directory ``~/MLClassificationData/modelinfo``.    
Notebook **Comparator** can be used to create **html report** on the base of contents of all files, located in this directory. This allows to compare results of the testing, produced by different models. Example of such report (_modelCompareExample.html_) can be found in the current subfolder.
