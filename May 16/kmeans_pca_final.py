# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:16:36 2017

@author: prudh
"""

import numpy as np # Library for fast numerical computing
np.random.seed(2) # Random number seeder

import warnings # To ignore gensim warnings for windows systems
warnings.filterwarnings('ignore')

import gensim # Library to load and work with pretrained w2v models
import re # Library for regular expressions

from sklearn.cluster import KMeans # Scikit-learn's k-means clustering algorithm
from sklearn.decomposition import PCA # Scikit-learn Principal Component Analysis
from scipy.spatial.distance import cosine # Distance metric
 
random_vect = np.random.rand(300) # Random substitution vector in case a word is not present

# Loading word2vec pretrained model
diction = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary= True)

# Loading files
files = open('testing.txt').read().split() # The document file
stop = open('stopwords_en.txt').read().splitlines() # Stopword list
gt = open('ground_truth.txt').read().splitlines() # Ground truth file

# Removing stop words
clean_files = []
for f in files:
    if f.lower() not in stop:
        clean_files.append(f)

# Removing special characters, etc.
clean_words = []
for f in clean_files:
    current = re.sub("\.(?!\d)", "", f)
    current = re.sub("\,(?!\d)", "", current)
    current = re.sub('\(', "", current)
    current = re.sub('\)', "", current)
    #current = re.sub(")", "", current)
    clean_words.append(current)

#Convert the words to numpy array
words = np.array(clean_words)

#Building a word2vec matrix
w2v_matrix = []
for w in words:
    if w in diction:
        w2v_matrix.append(diction[w])
    else:
        w2v_matrix.append(random_vect)
        
# Converting the values to float16 to make things faster
w2v_matrix = np.array(w2v_matrix).astype(np.float16)

# Creating a PCA object
pca = PCA()

'''
The components() function takes three parameters:
    reduced - the principal components ot the current sub_matrix
    w2v_matrix - the word2vec sub_matrix created for each cluster
    current_words - words belonging to the current cluster
    
This module takes the principal components as input and for
each of them, compares it to the word vectors in the 
current cluster's sub_matrix using a simple distance
metric (cosine).

This returns a list of keywords
'''
def components(reduced, w2v_matrix, current_words):  
    reduced += 1e-10
    w2v_matrix+= 1e-10
    add_words = []
    for i in range(0, reduced.shape[0]):
        dist_list = []
        first_word = reduced[i, :]
        #dist_list = np.sqrt(np.sum(first_word - glove_matrix, axis = 1) ** 2)
        for g in range(0, w2v_matrix.shape[0]):
            dist_list.append(cosine(first_word, w2v_matrix[g, :]))
        min_arg = np.argmin(dist_list)
#        print current_words[min_arg], i
        if current_words[min_arg] not in add_words:
            add_words.append(current_words[min_arg])
    return add_words

'''
The sub_matrix() function takes two parameters:
    label - the label of the current cluster to be evaluated
    fitter - Sklearn's k-means clustering object
    
From the parameters above, this function first collects
all the words belonging to the current cluster, and creates
a new word2vec matrix.

Then we convert this matrix to float16 to make computations 
faster, and we apply principal component analysis (PCA) on
this sub_matrix.
'''
def sub_matrix(label, fitter):
    current_words = words[fitter.labels_ == label]
    sub_matrix = []
    for c in current_words:
        if c in diction:
            sub_matrix.append(diction[c])
        else:
            sub_matrix.append(random_vect)

    sub_matrix = np.array(sub_matrix).astype(np.float16)
    pca_fit= pca.fit(sub_matrix)
    return components(pca_fit.components_, sub_matrix, current_words)


'''
The clustering() function takes two parameters:
    n_clus - the number of clusters
    w2v_matrix - the full w2v_matrix 
    
After passing the number of clusters, we use
sklearn's k-means algorithm to cluster various words. The intuition
is that words belonging to the same class would form a single cluster.

Then for each cluster, we collect the keywords evaluated from that, and
we concatenate all of them to form our final list.
'''
def clustering(n_clus, w2v_matrix):
    kmeans = KMeans(n_clusters= n_clus, random_state= 2)
    fitter = kmeans.fit(w2v_matrix)

    concat = []
    for i in range(0, n_clus):
        concat += sub_matrix(i, fitter)

    return concat    

# Number of Clusters

n_clus = 3

concat = clustering(n_clus, w2v_matrix)

print "Predicted Keywords: "
print concat
print "\nGround Truth: "
print gt
print "\nMatching Keywords: "
print list(np.intersect1d(concat, gt))
#print '\n\n\n\n'
#print len(concat)
#print len(np.intersect1d(concat, gt))
print "\nHit Rate: " +str(1.0 * len(np.intersect1d(concat, gt))/len(gt))
