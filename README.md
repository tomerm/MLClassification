# MLClassification
Classification using ML approach for English / Hebrew / Arabic data sets


# Setting up development environment
High level overview of development environment is presented on the image shown below:
![image](https://user-images.githubusercontent.com/5329257/46391999-40ff8500-c6e8-11e8-962b-3da09533e2fd.png)


1.	**Download Anaconda**

Anaconda is a free and easy-to-use environment for scientific Python. 

Download from  https://www.anaconda.com/download/#linux.  Use installer for Python version >= 3.6.

Anaconda contains many useable packages, including scikit-learn (sklearn, as it is called from Pyton).
Sklearn is a collection of efficient tools for data analysis and classification. It applies specific machine-learning technique ("shallow learning")  using different algorithms, like SVM, nearest neighbors, random forest, etc.

See http://scikit-learn.org/stable/  


2.	**Install Anaconda & Python**

See http://docs.anaconda.com/anaconda/install/linux


3.	**Start Anaconda**

http://docs.anaconda.com/anaconda/user-guide/getting-started/

See also http://docs.anaconda.com/anaconda/

4.	**Create an Anaconda Environment**

https://conda.io/docs/user-guide/tasks/manage-environments.html

5.	**Install Deep Learning Libraries (TensorFlow & Keras)**

TensorFlow is a tool for machine learning. While it contains a wide range of functionality, TensorFlow is mainly designed for deep neural network models.

For installing TensorFlow, open Anaconda Prompt to type the following commands.
To install the CPU-only version of TensorFlow:

`pip install --ignore-installed --upgrade tensorflow`

To install the GPU version of TensorFlow:

`pip install --ignore-installed --upgrade tensorflow-gpu`

See https://www.tensorflow.org/tutorials/

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow and few other similar tools.
For installing Keras, open Anaconda Prompt to type the following command:

`pip install keras`

See https://keras.io/#keras-the-python-deep-learning-library.
