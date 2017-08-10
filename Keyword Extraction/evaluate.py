# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:49:57 2017

@author: prudh
"""


import warnings
warnings.filterwarnings("ignore")
import os
from KEX3 import KEX3, Extractor
import numpy as np
from nltk.probability import FreqDist
from nltk import WordNetLemmatizer
import gensim
import pickle
import pandas as pd
import sys
from summa import keywords
import copy
def generate_ground_truth(gt, col_word):
    student_truth  = []
    for g in range(0, len(gt['stimuli'])):
        if gt['stimuli'][g] == col_word:
            student_truth.append(gt['selection'][g]) 
    fdist = dict(FreqDist(student_truth))
    val = np.median(fdist.values())
        
    words = np.array(fdist.keys())
    freqs = np.array(fdist.values())
    new_words = words[freqs > val]
    #print len(words)
    return new_words

def evaluate(gt, keywords):
    common = len(np.intersect1d(gt, keywords))
    #print len(gt), len(keywords), common
    #print list(np.intersect1d(gt, keywords))
    precision = 1.0 * common/len(keywords)
    recall = 1.0 * common/len(gt)
    return precision, recall

def score(gt, keywords):
    precision, recall = evaluate(gt, keywords)
    F_score = (2 * precision * recall)/ (precision + recall + 1e-5) 
    return F_score

def lemmatization(clean_phrases):
    lm = WordNetLemmatizer()
    cleanest_phrases = []
    for c in clean_phrases:
        cleanest_phrases.append(np.str(lm.lemmatize(c.lower())))
        #cleanest_phrases.append(np.str(lm.lemmatize(c.lower())))
    #print cleanest_phrases
    clean_phrases = copy.deepcopy(cleanest_phrases)
    clean_phrases = np.unique(clean_phrases)
    return clean_phrases


word_vectors = "GoogleNews-vectors-negative300.bin"
stopwords = "stopwords_en.txt"

direc = word_vectors
diction = gensim.models.KeyedVectors.load_word2vec_format(direc, binary = True)

stopwords = open(stopwords).read().splitlines()

open_name = open("tf_idf_dict.txt", "rb")
di = pickle.load(open_name)

#direc = "C:\Users\prudh\NLP_2\New folder\Keyword stimuli and data\\"

ground_truth = "Keyword-Selection.csv"
#gt = pd.read_csv(direc + ground_truth)
gt = pd.read_csv(ground_truth)


ext = Extractor(stopwords, diction, di)
docs = os.listdir("docs/")
col_names = []
for d in docs:
    col_names.append(d.split('.', 1)[0])

method = sys.argv[1]
metric = sys.argv[2]
#method = 'best'
#metric = 'avg'
for c in range(0, len(col_names)):
    text= open("docs\\" +docs[c]).read()
    new_words = generate_ground_truth(gt, col_names[c])
    if method == '1':
        keyw = ext.clustering_keywords(text, metric)
    if method == '2':
        keyw = ext.skip_agglomeration_keywords(text, metric)
    if method == '3':
        k = KEX3()
        lemmatized = k.relative_keywords(text, stopwords)
        count = len(lemmatized)/2
        new_key = keywords.keywords(text, words = count).split()
        summa_key = []
        for n in new_key:
            summa_key.append(str(n))
        keyw = copy.deepcopy(summa_key)
    if method == '4':
        k = KEX3()
        lemmatized = k.relative_keywords(text, stopwords)
        count = len(lemmatized)/2
        tf_scored_words = k.tfidf_score(di, lemmatized, count)
        #print distance_metric
        keyw = copy.deepcopy(tf_scored_words)
    if method == 'best':
        keyw = ext.clustering_keywords(text, 'chain')
    #keyw = ext.skip_agglomeration_with_related_word(text, 10, col_names[c])
    #print keyw
    print col_names[c], score(lemmatization(new_words), keyw)