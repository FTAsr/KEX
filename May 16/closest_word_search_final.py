# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:24:04 2017

@author: prudh
"""

import numpy as np # For fast numerical computing in Python
np.random.seed(2) # Random number seeder
              
import warnings 
warnings.filterwarnings("ignore") # To ignore gensim warnings on windows systems

import gensim # Library to load and work with pre-trained word2vec models
import re # Library for regular expressions
from scipy.spatial.distance import cosine # Distance metric

random_vect = np.random.rand(300) #Random vector if any word is not present in the model

# Loading pre-trained word2vec model
diction = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary= True)

#Loading Files
files = open('testing_2.txt').read().split() # The document to extract keywords
stop = open('stopwords_en.txt').read().splitlines() # Stopword list
gt = open('ground_truth_2.txt').read().splitlines() # ground truth file

#Remove all the stop words
clean_files = []
for f in files:
    if f.lower() not in stop:
        clean_files.append(f)

# Remove special characters, etc.
clean_words = []
for f in clean_files:
    current = re.sub("\.(?!\d)", "", f)
    current = re.sub("\,(?!\d)", "", current)
    current = re.sub('\(', "", current)
    current = re.sub('\)', "", current)

    clean_words.append(current)

# Convert the words to a Numpy array
words = np.array(clean_words).astype(np.str)

# Create a word2vec matrix
w2v_matrix = []
for w in words:
    if w in diction:
        w2v_matrix.append(diction[w])
    else:
        w2v_matrix.append(random_vect)
        
#Converting the values to float16 to make things run faster
w2v_matrix = np.array(w2v_matrix).astype(np.float16)

'''
The closest_word_search() takes two parameters:
    Parameter 1: The word2vec matrix
    Parameter 2: Maximum Distance Threshold
    
The method takes the current word and the proceeding 
word and calculates the cosine distance between them.
If this distance is larger than some set "high" threshold,
then it makes the word unrelated in the context of other words,
therefore, could be considered as a "filler" word.
'''
def closest_word_search(w2v_matrix, threshold):
    add_list = []  # Initialize list for keywords
    bad_list = [] # Initialize list for non-keywords 
    
    for i in range(1, len(w2v_matrix)):
    
        current = w2v_matrix[i-1, :] #current word
        next_ = w2v_matrix[i, :] # next word
        
        dist = cosine(current, next_) #calculate distance
        
        if dist < threshold: #Threshold checker
            if words[i] not in add_list:
                add_list.append(words[i]) # Add keywords
        else:
            bad_list.append(words[i]) # Add non-keywords
    return add_list, bad_list
    
threshold = 0.9
good_list, bad_list = closest_word_search(w2v_matrix, threshold)
print "\nPredicted Keywords: "
print good_list
print "\nGround Truth: "
print gt
print "\nMatching Keywords: "
print list(np.intersect1d(good_list, gt))
#print '\n\n\n\n'
#print len(concat)
#print len(np.intersect1d(concat, gt))
print "\nHit Rate: " +str(1.0 * len(np.intersect1d(good_list, gt))/len(gt))

print "\nNon-keyword List: "
print bad_list