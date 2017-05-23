# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:41:33 2017

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
files = open('biology_11.txt').read().splitlines()

#files = open('testing_2.txt').read().split() #The original text file
stop = open('stopwords_en.txt').read().splitlines() # Stopword list

len_list = []
for f in files:
    len_list.append(len(f.split()))
len_list = np.array(len_list)
current_file = files[8].split()
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
    current = re.sub("\:(?!\d)", "", current)
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
#adv_suffs = ['ed', 'ly', 'ily', 'ically', 'es']
adv_suffs = ['ly', 'ily', 'ically']
for a in adv_suffs:
    words = remove_adverbs(words, a)
words = np.array(words).astype(np.str)



#Tagger
from nltk import pos_tag


from nltk.stem import *
stemmer = SnowballStemmer("english")

new_words = []
book = pos_tag(words)
for b in book:
    if b[1] == 'NNP' or b[1] == 'NNPS':
        #print b[0]
        new_words.append(b[0])
    else:
        #print b[0]
        if stemmer.stem(b[0]) in diction:
            new_words.append(stemmer.stem(b[0]))
        else:
            new_words.append(b[0])
            
new_words = np.array(new_words).astype(np.str)
#Building a word2vec matrix
cover = 0
words = copy.deepcopy(new_words)
w2v_matrix = []
for w in words:
    if w in diction:
        cover += 1
        w2v_matrix.append(diction[w])
    else:
        w2v_matrix.append(random_vect)
        print w
        
print "Coverage:" + str(cover * 1.0/ len(words))

w2v_matrix = np.array(w2v_matrix)

words_original = copy.deepcopy(words)

number_of_iterations = 100
keyword_count = len(words_original)/2
iter_words = []
iter_sum = []
iter_indexes = []

for iterations in range(0, number_of_iterations):
    current_word_index = np.random.randint(0, len(words))
    current_word = words[current_word_index]

    add_words = []
    add_indexes = []


    add_indexes.append(current_word_index)
    add_words.append(current_word)

    min_values = []
    
    for i in range(0, keyword_count-1):
        
        distance_list = []
        for w in range(0, len(words_original)):
            
            if w not in add_indexes:
                cosine_distance = cosine(w2v_matrix[current_word_index, :], w2v_matrix[w, :])
                distance_list.append(cosine_distance)
            
            else:
                distance_list.append(100.0)
        
        min_index = np.argmin(distance_list)
        min_value = np.min(distance_list)

        add_indexes.append(min_index)
        min_values.append(min_value)
        
        current_word_index = min_index
        
        if words_original[current_word_index] not in add_words:
            add_words.append(words_original[current_word_index])
    
    iter_words.append(add_words)
    iter_sum.append(np.sum(min_values))
    iter_indexes.append(add_indexes)
keywords = iter_words[np.argmin(iter_sum)]
print keywords
#locations  = iter_indexes[np.argmin(iter_sum)]
#visualization
'''
all_words= copy.deepcopy(w2v_matrix)

key_words = []
for k in keywords:
    key_words.append(diction[k])

key_words= np.array(key_words)

pca = PCA(n_components = 2, random_state = 7)
total_reduction = pca.fit_transform(all_words)
keyword_reduction = pca.fit_transform(key_words)

import matplotlib.pyplot as plt


#plt.scatter(total_reduction[:, 0], total_reduction[:, 1], color = 'black')
plt.scatter(total_reduction[locations, 0], total_reduction[locations, 1], color = 'red')
'''