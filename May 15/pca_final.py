# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:29:37 2017

@author: prudh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:08:03 2017

@author: prudh
"""


import numpy as np # For fast numeric computing
import gensim # Word2vec data processor
import re # regular expressions to remove fill stops and commas
import os #manipulate directories
from sklearn.decomposition import PCA # Principal component anaysis 
from scipy.spatial.distance import cosine # Distance metric

np.random.seed(2) #Random Seeder

'''
Since all the words in the given input may not be present
in the pre-trained word2vec model, we use a single random
vector to replace them 
'''
random_vect = np.random.rand(300)

'''
Loading Word2Vec Model 

Pre-trained vectors can be downloaded from
https://github.com/3Top/word2vec-api (Google News)
'''
diction = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary= True)

#Loading Files
files = open('testing.txt').read().split() #The original text file
stop = open('stopwords_en.txt').read().splitlines() # Stop word list
gt = open('ground_truth.txt').read().splitlines() # Manually Labeled Ground truth keywords seperated by new line

'''
This snippet removes stop words from the original text
'''
clean_files = []
for f in files:
    if f.lower() not in stop:
        clean_files.append(f)

'''
This snippet removes full stops (.) and commas (,) from the text.
'''
clean_words = []
for f in clean_files:
    current = re.sub("\.(?!\d)", " ", f)
    current = re.sub("\,(?!\d)", " ", current)
    clean_words.append(current)

#Converting the added words into a Numpy array
words = np.array(clean_words)

'''
Creates a n x d matrix where n is the number of
words and d is the dimensionality of each word. In 
this case, d is 300.
'''

w2v_matrix = []
for w in words:
    if w in diction:
        w2v_matrix.append(diction[w])
    else:
        w2v_matrix.append(random_vect)
        
#Converting this to a Numpy array and changing to 16-bit floating pointer
#to make things run quicker.
w2v_matrix = np.array(w2v_matrix).astype(np.float16)

#Apply Sklearn's PCA
pca = PCA()
fit_pca = pca.fit(w2v_matrix)


'''
The function components () takes two parameters:
    Parameter 1: The components generated from PCA
    Parameter 2: The word2vec matrix
    
    First we add small decimals to both our parameters 
    to avoid any division by zero errors.
    
    For each principal component:
        a. Find the closest word using cosine distance in the original n x d matrix 
        b. Add the word to the list of keywords

    If the predicted word is already in the keyword list, 
    do not add it to the list. Go to the next component. 

    Repeat until all the components are completed.       
'''

def components(reduced, w2v_matrix):  
    reduced += 1e-10
    w2v_matrix+= 1e-10
    
    add_words = []
    
    for i in range(0, reduced.shape[0]):
        dist_list = []
        first_word = reduced[i, :]
        
        for g in range(0, w2v_matrix.shape[0]):
            dist_list.append(cosine(first_word, w2v_matrix[g, :]))
        
        min_arg = np.argmin(dist_list)
        
        if words[min_arg] not in add_words:
            add_words.append(words[min_arg])
            
    return add_words

# Calling components function ()
pca_words = components(fit_pca.components_, w2v_matrix)

print "Predicted Keywords :"
print pca_words
print "Ground Truth Keywords: "
print gt
print "Matching Keywords: "
print np.intersect1d(pca_words, gt)
print 'Hit Rate: ' + str(1.0 * len(np.intersect1d(pca_words, gt))/ len(gt))
