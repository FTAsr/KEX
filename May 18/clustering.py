# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:18:15 2017

@author: prudh
"""


import numpy as np # Library for fast numerical computing
np.random.seed(2) # Random number seeder

import warnings # To ignore gensim warnings for windows systems
warnings.filterwarnings('ignore')

import gensim # Library to load and work with pretrained w2v models
import re # Library for regular expressions
import copy
from sklearn.cluster import KMeans # Scikit-learn's k-means clustering algorithm
from sklearn.decomposition import PCA # Scikit-learn Principal Component Analysis
from scipy.spatial.distance import cosine # Distance metric
 
random_vect = np.random.rand(300) # Random substitution vector in case a word is not present

# Loading word2vec pretrained model
diction = gensim.models.KeyedVectors.load_word2vec_format('C:\Users\prudh\Quora\GoogleNews-vectors-negative300.bin', binary= True)
import os
os.chdir('NLP_2')
files = open('sentences_45.txt').read().splitlines()

#files = open('testing_2.txt').read().split() #The original text file
stop = open('stopwords_en.txt').read().splitlines() # Stopword list

len_list = []
for f in files:
    len_list.append(len(f.split()))
len_list = np.array(len_list)
current_file = files[3].split()
clean_files = []
#current_file = copy.deepcopy(files)
for f in current_file:
    if f.lower() not in stop:
        clean_files.append(f)

clean_words = []
for f in clean_files:
    current = re.sub("\.(?!\d)", "", f)
    current = re.sub("\;(?!\d)", "", current)
    current = re.sub("\,(?!\d)", "", current)
    current = re.sub('\(', "", current)
    current = re.sub('\)', "", current)
    #current = re.sub(")", "", current)
    clean_words.append(current)

clean_files = []
for f in clean_words:
    if f.lower() not in stop:
        clean_files.append(f)

clean_words = copy.deepcopy(clean_files)
new_words = np.array(clean_words).astype(np.str)

def remove_adverbs(new_words, adv_sfx):
    words = []
    for n in new_words:
        if not n.endswith(adv_sfx):
            if n not in words:
                words.append(n)
    return words
import copy
words = copy.deepcopy(new_words)
adv_suffs = ['ed', 'ly', 'ily', 'ically', 'es']
for a in adv_suffs:
    words = remove_adverbs(words, a)
words = np.array(words).astype(np.str)

#Building a word2vec matrix
cover = 0
w2v_matrix = []
for w in words:
    if w in diction:
        cover += 1
        w2v_matrix.append(diction[w])
    else:
        w2v_matrix.append(random_vect)
        
print "Coverage:" + str(cover * 1.0/ len(words))
# Converting the values to float16 to make things faster
w2v_matrix = np.array(w2v_matrix).astype(np.float16)

random = np.random.randint(0, w2v_matrix.shape[0])

curr = w2v_matrix[random, :]
dist= []
for i in range(0, len(w2v_matrix)):
    if i != random:
        dist.append(cosine(curr, w2v_matrix[i, :]))
    amin = np.argmin(dist)


    


from sklearn.cluster import KMeans

def clustering(w2v_matrix, n_clus, threshold):
    kmeans = KMeans(n_clusters = n_clus, random_state = 12)
    fitter = kmeans.fit(w2v_matrix)
    keywords = []
    centers = fitter.cluster_centers_
    print words[fitter.labels_ == 1]
    for c in range(0, len(centers)):
        current = centers[c, :]
        dist_list = []
        for w in range(0, len(w2v_matrix)):
            dist_list.append(cosine(current, w2v_matrix[w, :]))
        min_arg = np.argmin(dist_list)
        if dist_list[min_arg] < threshold:
            keywords.append(words[min_arg])
            #print words[min_arg], dist_list[min_arg]
    return keywords

keywords = clustering(w2v_matrix, 2, 0.1)