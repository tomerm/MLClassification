This directory includes a sample Jupyter Notebook with simple Arabic data set and pipe for single label classification. 
The sole purpose of this example is to assure that development environment is working properly. 

# Dataset info.
Dataset contains several thousand articles written on Arabic language.
Articles are downloaded from the following news sites:
- alhayat.com
- aawsat.com
- al-ayyam.com

All articles are divided into 4 categories (classes) and placed into directory example/data/docs.
This directory contains 4 subdirectories (one per category): Culture, Sport, Politics and Economy.
Each subdirectory contains 2 files: **docs.txt** and **tocs.txt**.
File **docs.txt** contains text of all articles belonging to current category.
File **tocs.txt** contains tokenized text of the same articles. 

# How to run.

1. Clone current repository into your home directory.
2. In Terminal, go to  ~/example/notebooks, type 'jupyter notebook'.
3.  From Home page, load classificationWithW2V.ipynb (by click on its name)
4. Clean output of all cells (Cell -> All output -> Clear)
5. Launch notebook (Cell -> Run All). Alternatively you can run each cell individually (from top to bottom, strictly following the sequence).
6. If you want rerun the notebook, restart the Kernel and clean all outputs (Kernel -> Restart & Clear Output).


At the bottom of Jupyter notebook (/example/notebooks/classificationWithW2V.ipynb) you should see the results of data processing:
![image](https://user-images.githubusercontent.com/5329257/46580291-c2c61a00-ca2a-11e8-8a28-8c31a23ae948.png)
