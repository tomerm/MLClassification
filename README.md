# MLClassification
Classification using ML approach for English / Hebrew / Arabic data sets


# Setting up development environment
High level overview of development environment is presented on the image shown below:
![image](https://user-images.githubusercontent.com/5329257/46391999-40ff8500-c6e8-11e8-962b-3da09533e2fd.png)

We are using RedHat Linux. For more details on the version please see the image below: 
![image](https://user-images.githubusercontent.com/5329257/46408850-31019880-c71c-11e8-97c3-6fe222f61317.png)

1.	**Download Anaconda**

Anaconda is a free and easy-to-use environment for scientific Python. 

Download from  https://www.anaconda.com/download/#linux.  Use installer for Python version >= 3.6.

2.	**Install Anaconda & Python**

See http://docs.anaconda.com/anaconda/install/linux


3.	**Start Anaconda**

http://docs.anaconda.com/anaconda/user-guide/getting-started/

See also http://docs.anaconda.com/anaconda/

4.	**Create an Anaconda Environment**

https://conda.io/docs/user-guide/tasks/manage-environments.html

After creating the new environment, few mashin learning libraries should be installed.

5.	**Install Mashin Learning Libraries**

**TensorFlow** is a tool for machine learning. While it contains a wide range of functionality, TensorFlow is mainly designed for deep neural network models ("deep learning").

For installing TensorFlow, open Terminal to type the following commands.
To install the CPU-only version of TensorFlow:

`pip install --ignore-installed --upgrade tensorflow`

To install the GPU version of TensorFlow:

`pip install --ignore-installed --upgrade tensorflow-gpu`

See https://www.tensorflow.org/tutorials/

**Keras** is a high-level neural networks API, written in Python and capable of running on top of TensorFlow and few other similar tools.
For installing Keras, open Terminal to type the following command:

`pip install keras`

See https://keras.io/#keras-the-python-deep-learning-library.

**Sklearn** is a collection of efficient tools for data analysis and classification. It applies specific machine-learning technique ("shallow learning")  using different algorithms, like SVM, nearest neighbors, random forest, etc.
For installing sklearn, open Terminal to type the following command:

`pip install sklearn`

See http://scikit-learn.org/stable/  

**Gensim** is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. 
For installing gensim, open Terminal to type the following command:

`pip install gensim`

See https://pypi.org/project/gensim/

**Matplotlib** is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. To install, actuvate your environment and type the following command:

`conda install matplotlib`

See https://matplotlib.org/

6.  **Jupyter notebook**
Python scripts in this project are presented in Jupyter Notebooks. Jupyter is automatically installed for base anaconda environment only. For each new envionment new install (from Navigator or from Teminal, using pip) is required.

Launch notebook from example/notebooks to check, that development environment is set correctly.

# Project file tree
We assume that
1. Project is cloned into user's home directory, so its root folder is `~/MLClassification`.
2. Folder, containing inputs and outputs of the project (mostly large text files) isn't stored in the github. Its path is `~/MLClassificationData`.
3. Folder MLClassificationData has the following structure:
``````
MLClassificationData
 |
 |==== w2v
 |       |
 |       |====> source     Folder containing original file(s), used for creating Word2Vec models
 |       |
 |       |====> target     Folder containing tokenized file(s), used for creating Word2Vec models
 |       |
 |       |====> models     Folder containing Word2Vec model(s) in binary mode
 |       |
 |       |====> vectors    Folder containing file(s) with word vectors
 |
 |==== train
 |       |
 |       |====> source     Folder containing original documents, used for model's training
 |       |
 |       |====> target     Folder containing tokenized documents, used for model's training
 |       
 |==== test
 |       |
 |       |====> source     Folder containing original documents, used for model's testing
 |       |
 |       |====> target     Folder containing tokenized documents, used for model's testing
 |      
 |==== work                Folder containing documents, classification of which should be predicted
 |
 |==== models              Folder containing models, used for document classification
 
 ``````
 *Note: If you need to work with another project file tree you should update notebooks correspondingly.*
 
 # Initial data
 You can get some files (and use them as initial data for this project) from the following links:
 1. [Cleaned dump of Arabic Wikipedia (source text for w2v models)](https://ibm.ent.box.com/file/344972068403)
 2. [Tokenized content of Arabic Wikipedia (input for training w2v models)](https://ibm.ent.box.com/file/344966829805)
 3. [W2V model in binary format, created on Arabic Wikipedia](https://ibm.ent.box.com/file/344964480859)
 4. [File of vectors, created on Arabic Wikipedia](https://ibm.ent.box.com/file/344961720796)
 5. [Example of the dataset containing documents, tagged by their place](https://ibm.ent.box.com/file/344967756393)
 
