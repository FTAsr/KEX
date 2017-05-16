# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:16:36 2017

@author: prudh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:02:33 2017

@author: prudh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:54:52 2017

@author: prudh
"""


import numpy as np
np.random.seed(2)

import gensim
import warnings
warnings.filterwarnings('ignore')
import os
import re

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

random_vect = np.random.rand(300)
diction = gensim.models.KeyedVectors.load_word2vec_format('C:\Users\prudh\Quora\GoogleNews-vectors-negative300.bin', binary= True)
'''
diction = {}

for f in files:
    vals = f.split()
    diction[vals[0]] = np.array(vals[1:]).astype(np.float16)

del files
'''

files = open('testing.txt').read().split()
stop = open('stopwords_en.txt').read().splitlines()
gt = open('ground_truth.txt').read().splitlines()
clean_files = []
for f in files:
    if f.lower() not in stop:
        clean_files.append(f)
clean_words = []
for f in clean_files:
    current = re.sub("\.(?!\d)", "", f)
    current = re.sub("\,(?!\d)", "", current)
    current = re.sub('\(', "", current)
    current = re.sub('\)', "", current)
    #current = re.sub(")", "", current)
    clean_words.append(current)

words = np.array(clean_words)

w2v_matrix = []
for w in words:
    if w in diction:
        w2v_matrix.append(diction[w])
    else:
        w2v_matrix.append(random_vect)
        
w2v_matrix = np.array(w2v_matrix).astype(np.float16)

pca = PCA()

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
        if len(add_words) == reduced.shape[0]:
            break
    return add_words

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

def clustering(n_clus, w2v_matrix):
    kmeans = KMeans(n_clusters= n_clus, random_state= 2)
    fitter = kmeans.fit(w2v_matrix)

    concat = []
    for i in range(0, n_clus):
        concat += sub_matrix(i, fitter)

    return concat    

n_clus = 5
concat = clustering(n_clus, w2v_matrix)

print "Predicted Keywords: "
print concat
print "Ground Truth: "
print gt
print "Matching Keywords: "
print list(np.intersect1d(concat, gt))
#print '\n\n\n\n'
#print len(concat)
#print len(np.intersect1d(concat, gt))
print "Hit Rate: " +str(1.0 * len(np.intersect1d(concat, gt))/len(gt))
