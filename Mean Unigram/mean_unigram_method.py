# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:03:41 2017

@author: prudh
"""
"""
Unigram keyword model based on intermediate means
"""


import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from collections import defaultdict
import operator
import sys

filename = sys.argv[-1]
tokenizer = RegexpTokenizer(r'\w+')
file_name = open(filename, 'r').read()
words = tokenizer.tokenize(file_name)
stop_words = open('stopwords.txt', 'r').read().split()
#print stop_words

min_occurance = 2
min_word_length = 3
k = 10

def convert_to_lower(words):
    new_list = []
    for w in words:
        new_list.append(w.lower())
    return new_list
#words = convert_to_lower(words)

def add_word_indexes_to_dict(words, stop_words):
    d = defaultdict(list)
    for w in range(len(words)):
        if words[w] not in stop_words:
            d[words[w]].append(w)
    return d

#full_dict = add_word_indexes_to_dict(words, stop_words)    

def truncate_minimum_occurances(d, threshold):
    trunc_d = defaultdict(list)
    for de in d:
        if len(d[de]) > threshold:
            trunc_d[de] = d[de]
    return trunc_d

#trunc_d = truncate_minimum_occurances(full_dict, 4)

def calculate_fractions(trunc_d):

    final = dict()
    epsilon = 1e-5

    subtractor = defaultdict(list)
    
    for j in range(0, len(trunc_d)):
        current = trunc_d[trunc_d.keys()[j]]
    
        for i in range(0, len(current)-1):
            subtractor[trunc_d.keys()[j]].append((current[i+1] - current[i]))

        current_word = trunc_d.keys()[j]
        #print current_word
    #print subtractor[current_word]
        average = np.mean(subtractor[current_word])
    #print average
        less_than_mean = np.sum(subtractor[current_word] < average)
        greater_than_mean = np.sum(subtractor[current_word] > average)

    #print less_than_mean + greater_than_mean
        fraction = 1.0 * less_than_mean /(less_than_mean + greater_than_mean + epsilon)
    #print fraction
        final[current_word] = round(fraction, 4)
        
    return final

#final = calculate_fractions(trunc_d)
        
def sort_final_outputs(final): 
    sorted_final = sorted(final.items(), key = operator.itemgetter(1), reverse = True)
    return sorted_final

#sorted_final = sort_final_outputs(final)

def truncate_minimum_word_length(sorted_final, threshold):
    new_list = []
    for s in sorted_final:
        if len(s[0]) > threshold:
            new_list.append(s)
    return new_list

#output = truncate_minimum_word_length(sorted_final, 4)

def print_top_k_keywords(output, k):
    print output[0:k]

words = convert_to_lower(words)
full_dict = add_word_indexes_to_dict(words, stop_words)
trunc_d = truncate_minimum_occurances(full_dict, min_occurance)
final = calculate_fractions(trunc_d)
sorted_final = sort_final_outputs(final)
output = truncate_minimum_word_length(sorted_final, min_word_length)

print_top_k_keywords(output, k)
