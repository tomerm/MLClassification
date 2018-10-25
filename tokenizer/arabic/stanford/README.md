### Stanford Part-Of-Speach tagger 
Stanford Part-Of-Speech Tagger (POS Tagger) is a piece of software that reads text in some language and assigns parts of speech (such as noun, verb, adjective, etc.) to each token.

#### Install:

1.  **Install nltk** (which is the python library containing modules for interfacing with the Stanford taggers):
    - Type in Terminal:     
        `conda install nltk`    

2.  **Get the software**
    - Type in Terminal to download the main part of software:  
        `wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip`
    - Unzip downloaded file
    - Go to directory stanford-corenlp-full-2016-10-31
    - Type in Terminal to download the files, containing Arabic models and properties:        
        `wget http://nlp.stanford.edu/software/stanford-arabic-corenlp-2016-10-31-models.jar`    
        `wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/src/edu/stanford/nlp/pipeline/StanfordCoreNLP-arabic.properties`

3. **Run the local server**
	- You can launch script from "Start Stanford Server from Python" notebook
    - Alternatively you can type in Terminal:    
        `java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-arabic.properties -preload tokenize,ssplit,pos,parse -status_port 9005  -port 9005 -timeout 15000`
        
Now you can use Stanford POS tagger.

#### Example of using
``````
from nltk.parse.corenlp import CoreNLPParser
parser = CoreNLPParser(url='http://localhost:9005', tagtype='pos')
text = "مشيتُ من بيتي إلى المدرسة."
parser.tag(text.split())
``````

Output of the tagger is an array of tokens with their POS tags:
``````
[('مشيت', 'VBD'),('من', 'IN'),('بيتي', 'NNP'),
 ('الى', 'IN'),('المدرسة', 'DTNN'),('.', 'PUNC')]
``````
This output should be additionally handled to remove punctuation, articles etc. and merge remaining tokens into text lines.
