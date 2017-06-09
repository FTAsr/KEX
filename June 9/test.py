# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 13:33:19 2017

@author: prudh
"""



import pandas as pd
import numpy as np
import gensim
import os
import re
import operator
from scipy.spatial.distance import cosine
from nltk.stem.wordnet import WordNetLemmatizer
import copy
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import nltk
from nltk.probability import FreqDist
import sys
np.random.seed(2)

def pre_stripper(seperated):
    sep = []
    for s in seperated:
        curr = s.replace('.', ' .')
        curr = curr.replace(',', ' ,')
        curr = curr.replace(';', ' ;')
        curr = curr.replace("'", ' ')
        curr = curr.replace('-', ' ')
        sep.append(curr)
    new_sep = []
    for se in sep:
        if len(se.split()) > 1:
           for ses in se.split():
               new_sep.append(ses)
        else:
            new_sep.append(se)
    new_sep.append('.')
    return new_sep

def candidate_phrase_extraction(stopwords, seperated):
    phrases = []
    for s in seperated:
        if s.lower() not in stopwords and s.isalnum() == True:
            phrases.append(s)
            #phrases.append(current_word)
        if s.lower() in stopwords or s.isalnum() == False:
            pass
            #current_word = ''
    return phrases

def lemmatization(clean_phrases):
    lm = WordNetLemmatizer()
    cleanest_phrases = []
    for c in clean_phrases:
        cleanest_phrases.append(np.str(lm.lemmatize(c.lower())))
    #print cleanest_phrases
    clean_phrases = copy.deepcopy(cleanest_phrases)
    clean_phrases = np.unique(clean_phrases)
    return clean_phrases

def distrib_represent_conversion(clean_phrases, diction):
    mapper = dict()
    for c in clean_phrases:
        if len(c.split()) == 1:
            if c in diction:
                mapper[c] = diction[c]
            else:
                mapper[c] = random_vect
        else:
            summer = 0
            for s in c.split():
                if s in diction:
                    summer += diction[s]
            summer = summer/np.linalg.norm(summer)
            mapper[c] = summer
    return mapper

def close_topic_clustering(mapper, k_count):
    words_original = mapper.keys()
    w2v_matrix = np.array(mapper.values())
    keyword_count = k_count
    iter_words = []
    iter_sum = []
    iter_indexes = []

    for iterations in range(0, len(words_original)):
        current_word_index = iterations
        current_word = words_original[current_word_index]

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
    #print keywords
    locations  = iter_indexes[np.argmin(iter_sum)]
    
    #return keywords, locations, iter_words
    return keywords

def keyword_cluster(mapper, k_count):
    k_count -= 1
    distance_dict = dict()
    keyword_dict = dict()
    for m in mapper:
        current = mapper[m]
        li = (cosine_broad(current, np.array(mapper.values())))    
        li[np.argmin(li)] = 100.0
    #print np.argsort(li)
        keyword_dict[m] = np.array(mapper.keys())[np.argsort(li)[0:k_count]]
        distance_dict[m] = li[np.argsort(li)][0:k_count]

    check_list = dict()
    for d in distance_dict:
        check_list[d] = np.sum(distance_dict[d])/k_count

    sorted_list = sorted(check_list.items(), key=operator.itemgetter(1))
    best_key = sorted_list[0][0]
    keywords = [best_key] + list(keyword_dict[best_key])
    
    return keywords, keyword_dict

def cluster_chain_evaluation(mapper, keyword_dict, k_count):
    k_count -= 1
    check_list = dict()
    for k in keyword_dict:
        current_list = [k] + list(keyword_dict[k])
        check_list[k] = 0
        for c in range(0, len(current_list)-1):
            check_list[k] += cosine(mapper[current_list[c]], mapper[current_list[c+1]])
        check_list[k] /= k_count
             
    sorted_list = sorted(check_list.items(), key=operator.itemgetter(1))
    best_key = sorted_list[0][0]
    keywords=  ([best_key] + list(keyword_dict[best_key]))
    return keywords

def cosine_broad(A, B):
    num = np.dot(A, B.T)
    denA = np.linalg.norm(A)
    denB = np.linalg.norm(B, axis = 1)
    den = np.dot(denA, denB)
    return 1- num/den


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

def representation_reduction(mapper):
    pca = PCA(n_components = 3, random_state = 2)
    fitter = pca.fit_transform(mapper.values() * 300)
    return fitter

'''
A simple function that is used to find
the locations of keywords in the
document
'''
def location_finder(mapper, keywords):
    locs = []
    for k in keywords:
        #print locs
        #print k
        if len(np.where(np.array(mapper.keys()) == k)[0]) > 0:
            locs.append(np.where(np.array(mapper.keys()) == k)[0][0])
        else:
            locs.append(-1)
    return locs

def vizualize_predictions(mapper, keywords, gt, fitter, filename):
    words_original = mapper.keys() 
    locations = location_finder(mapper, keywords)
    fig = plt.figure(figsize = (20, 20))
    #ax = fig.add_subplot(111, projection = '3d')
    ax = Axes3D(fig)
    
    #Scatter all words
    
    #Scatter all the keywords predicted
    ax.scatter(fitter[locations, 0], fitter[locations, 1], fitter[locations, 2])
    
    #Annotate all the keywords predicted
    for i, txt in enumerate(keywords):
        ax.text(fitter[locations, 0][i], fitter[locations, 1][i], fitter[locations, 2][i], 
                txt + '(' +str(i+1) +')', bbox=dict(facecolor='red', alpha=0.5), fontsize= 25)
        
    #Annotate the first word with a different color
    ax.text(fitter[locations[0], 0], fitter[locations[0], 1], fitter[locations[0], 2], 
            words_original[locations[0]] + '(' +str(1) +')', bbox=dict(facecolor='yellow', alpha=0.95), 
                          fontsize= 25)
    
    ax.tick_params(labelsize = 25)
    plt.savefig(filename)


def vizualize_hits_and_misses(mapper, keywords, gt, fitter, filename):
    words_original = mapper.keys() 
    locations = location_finder(mapper, keywords)
    fig = plt.figure(figsize = (20, 20))
    #ax = fig.add_subplot(111, projection = '3d')
    ax = Axes3D(fig)
    
    matched = np.intersect1d(keywords, gt)
    match_locs = location_finder(mapper, matched)
    #print matched
    ax.scatter(fitter[match_locs, 0], fitter[match_locs, 1], fitter[match_locs, 2])
    
    for i, txt in enumerate(matched):
        ax.text(fitter[match_locs, 0][i], fitter[match_locs, 1][i], fitter[match_locs, 2][i], 
                #txt + '(' +str(i+1) +')', bbox=dict(facecolor='green', alpha=0.5), fontsize= 25)
                txt, bbox=dict(facecolor='red', alpha=0.5), fontsize= 25)
    
    union = (set(keywords).symmetric_difference(gt))
    misses = np.intersect1d(gt,list(union))
    #print misses
    miss_locs = location_finder(mapper, misses)
    for i, txt in enumerate(misses):
        ax.text(fitter[miss_locs, 0][i], fitter[miss_locs, 1][i], fitter[miss_locs, 2][i], 
                #txt + '(' +str(i+1) +')', bbox=dict(facecolor='yellow', alpha=0.5), fontsize= 25)
                txt, bbox=dict(facecolor='white', alpha=0.5), fontsize= 25)
    ax.tick_params(labelsize = 25)
    plt.savefig(filename)


def generate_ground_truth(gt, mode):
    student_truth  = []
    for g in range(0, len(gt['stimuli'])):
        if gt['stimuli'][g] == 'Clouds':
            student_truth.append(gt['selection'][g]) 
    fdist = dict(FreqDist(student_truth))
    if mode == 'mean':
        val = np.mean(fdist.values())
    if mode == 'median':
        val = np.median(fdist.values())
        
    words = np.array(fdist.keys())
    freqs = np.array(fdist.values())
    new_words = words[freqs > val]
    
    return new_words

word_vectors = sys.argv[1]
document = sys.argv[2]
stopwords = sys.argv[3]
ground_truth = sys.argv[4]
mode = sys.argv[5]
method = sys.argv[6]
viz = sys.argv[7]
viz_name = sys.argv[8]

#word_vectors = 'GoogleNews-vectors-negative300.bin'
random_vect = np.random.rand(300)
#diction = gensim.models.KeyedVectors.load_word2vec_format('C:\Users\prudh\Quora\GoogleNews-vectors-negative300.bin', binary= True)
direc = "C:\Users\prudh\Quora\\" + word_vectors
diction = gensim.models.KeyedVectors.load_word2vec_format(direc, binary = True)
#os.chdir('NLP_2')
#text= open('Clouds.txt').read()
text= open(document).read()
stopwords = open(stopwords).read().splitlines()

direc = "C:\Users\prudh\NLP_2\New folder\Keyword stimuli and data\\"
#gt = pd.read_csv("C:\Users\prudh\NLP_2\New folder\Keyword stimuli and data\Keyword-Selection.csv")
gt = pd.read_csv(direc + ground_truth)

new_words = generate_ground_truth(gt, mode)
lem_new = lemmatization(new_words)
sepe = pre_stripper(text.split())
candidate_words = candidate_phrase_extraction(stopwords, sepe)
lemmatized = lemmatization(candidate_words)
mapper = distrib_represent_conversion(lemmatized, diction)
no_of_keywords = len(lem_new) + 2
if method == 'chain':
    keyw = close_topic_clustering(mapper, no_of_keywords)
if method == 'cluster':
    keyw, kdict = keyword_cluster(mapper, no_of_keywords)
if method == 'cluster_chain':
    keyw, kdict = keyword_cluster(mapper, no_of_keywords)
    keyw = cluster_chain_evaluation(mapper, kdict, len(lem_new) + 1)

prec, rec = evaluate(lem_new, keyw)
f_score = score(lem_new, keyw)
print list(keyw)
print "Precision: " +str(prec) + " Recall: " +str(rec)+ " F_score: " +str(f_score)

fitter=representation_reduction(mapper)
if viz == 'predictions':
    vizualize_predictions(mapper, keyw, lem_new, fitter, viz_name)
if viz == 'hits':
    vizualize_hits_and_misses(mapper, keyw, lem_new, fitter, viz_name)