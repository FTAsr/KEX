# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:10:42 2017

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
#os.chdir('NLP_2')
files = open('sentences_45.txt').read().splitlines()

#files = open('testing_2.txt').read().split() #The original text file
stop = open('stopwords_en.txt').read().splitlines() # Stopword list

len_list = []
for f in files:
    len_list.append(len(f.split()))
len_list = np.array(len_list)
current_file = files[9].split()
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
#adv_suffs = ['ed', 'ly', 'ily', 'ically', 'es']
adv_suffs = ['ly', 'ily', 'ically']
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
        print w
        
print "Coverage:" + str(cover * 1.0/ len(words))

w2v_matrix = np.array(w2v_matrix)

'''
current_word_index = np.random.randint(0, len(words))
current_word = words[current_word_index]

distance_list = []
word_indexes = []
min_distances = []
for n in range(0, 3):
    for w in range(len(words)):
        if w != word_indexes:
            print w, words[w]
            distance_list.append(cosine(w2v_matrix[current_word_index, :], w2v_matrix[w, :]))
    word_indexes.append(np.argmin(distance_list))
    min_distances.append(np.min(distance_list))
    current_word_index = np.argmin(distance_list)
    
'''
    

words_original = copy.deepcopy(words)
#current_word_index = np.random.randint(0, len(words))
#current_word = words[current_word_index]

#add_words = []
#add_indexes = []

#add_indexes.append(current_word_index)
#add_words.append(current_word)
#print add_words
#print words
#words = np.delete(words, current_word_index)
#print words

#Calculate distance


number_of_iterations = 100
keyword_count = 17
iter_words = []
iter_sum = []

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
        add_words.append(words_original[current_word_index])
    iter_words.append(add_words)
    iter_sum.append(np.sum(min_values))

keywords = iter_words[np.argmin(iter_sum)]

#Vizualization

'''
mat = w2v_matrix[add_indexes, :]

pca = PCA(n_components = 2, random_state = 2)
fitter = pca.fit_transform(mat)

x = fitter[:, 0]
y = fitter[:, 1]

import matplotlib.pyplot as plt

pca2 = PCA(n_components = 2, random_state = 2)
fitter2 = pca.fit_transform(w2v_matrix)

x1 = fitter2[:, 0]
y1 = fitter2[:, 1]

x = fitter2[add_indexes, 0]
y = fitter2[add_indexes, 1]

plt.figure()
plt.scatter(x1, y1,color = 'black')
#plt.scatter(x, y, color = 'red')
'''


