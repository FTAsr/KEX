<b> Unsupervised Keyword Extraction </b>

The program executes an unsupervised keyword extraction model that works on individual documents.

Folder Contents: 
1) KEX.py - The entire code of the project
2) tf_idf_dict.txt - Pretrained Wikipedia TF-IDF scores 
3) stopwords_en.txt - Stop word list with more than 550 words.
4) Docs folder - Samples documents to run and test the code.
5) Simple Instructions - This document has instructions to have an on-the-go run at the program. 
6) Detailed Instructions - This document delves a bit into the subtle intricacies of the arguments, and gives a breif explanation of all the modes listed.

Preliminaries:
1) Make sure you have pre-trained Google News word2vec file. If not, download it from https://github.com/3Top/word2vec-api
2) Make sure you have the following packages installed: Numpy, Scipy, NLTK, Summa, and Gensim.

To have a simple run of our best method, use this command below.

    >>> python KEX.py clouds.txt 1

This returns the keywords from the document about clouds.
