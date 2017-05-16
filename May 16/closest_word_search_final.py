# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:24:04 2017

@author: prudh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:08:39 2017

@author: prudh
"""


import numpy as np
np.random.seed(2)

import gensim

import os
import re

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
#os.chdir('NLP_2')
files = open('testing_2.txt').read().split()
stop = open('stopwords_en.txt').read().splitlines()
gt = open('ground_truth_2.txt').read().splitlines()
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

    clean_words.append(current)

words = np.array(clean_words).astype(np.str)

w2v_matrix = []
for w in words:
    if w in diction:
        w2v_matrix.append(diction[w])
    else:
        w2v_matrix.append(random_vect)
        
w2v_matrix = np.array(w2v_matrix).astype(np.float16)

def closest_word_search(w2v_matrix, threshold):
    add_list = []  
    bad_list = []
    for i in range(1, len(w2v_matrix)):
        current = w2v_matrix[i-1, :]
        next_ = w2v_matrix[i, :]
        dist = cosine(current, next_)
        if dist < threshold:
            if words[i] not in add_list:
                add_list.append(words[i])
        else:
            bad_list.append(words[i])
    return add_list, bad_list
    
concat, bad_list = closest_word_search(w2v_matrix, 0.9)
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

print "Non-keyword List: "
print bad_list