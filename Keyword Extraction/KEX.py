# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 13:22:53 2017

@author: prudh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 18:42:28 2017

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
import pickle
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

def candidate_word_extractor(stopwords, seperated):
    words = []
    for s in seperated:
        if s.lower() not in stopwords and s.isalnum() == True:
            words.append(s)
            #phrases.append(current_word)
        if s.lower() in stopwords or s.isalnum() == False:
            pass
            #current_word = ''
    return words

def candidate_phrase_extraction(stopwords, seperated):
    phrases = []
    current_word = ''
    for s in seperated:
        if s.lower() not in stopwords and s.isalnum() == True:
            current_word+= s + ' '
        if s.lower() in stopwords or s.isalnum() == False:
            phrases.append(current_word)
            current_word = ''
    
    new_phrases = []
    for p in phrases:
        if len(p.split()) == len(np.unique(p.split())):
            new_phrases.append(p)
    return new_phrases

def stripper(phrases):
    candidate_phrases = []
    for p in phrases:
        if p != '':
            candidate_phrases.append(p.strip())
    return candidate_phrases

def phrase_filtering(candidate_phrases, thresholder, diction):
    clean_phrases = []
    for c in candidate_phrases:
        if len(c.split()) > 1:
            splitted_phrase = c.split()
            current_phrase = splitted_phrase[0] + ' ' 
            for s in range(0, len(splitted_phrase)-1):
                w1, w2 = [], []
                if splitted_phrase[s] in diction:
                    w1 = diction[splitted_phrase[s]]
                else:
                    w1 = random_vect
                if splitted_phrase[s+1] in diction:
                    w2 = diction[splitted_phrase[s+1]]
                else:
                    w2 = random_vect
                distance = cosine(w1, w2)
                if distance > thresholder:
                    clean_phrases.append(splitted_phrase[s+1])
                if distance < thresholder:
                    current_phrase += splitted_phrase[s+1] + ' '  
            clean_phrases.append(current_phrase)
        else:
            clean_phrases.append(c)
    return clean_phrases

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
            summer = summer/len(c.split())
            mapper[c] = summer
    return mapper

def clustering(mapper, k_count):
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

    '''
    check_list = dict()
    for d in distance_dict:
        check_list[d] = np.sum(distance_dict[d])/k_count

    sorted_list = sorted(check_list.items(), key=operator.itemgetter(1))
    best_key = sorted_list[0][0]
    keywords = [best_key] + list(keyword_dict[best_key])
    '''
    #return keywords, keyword_dict
    return keyword_dict

def cosine_broad(A, B):
    num = np.dot(A, B.T)
    denA = np.linalg.norm(A)
    denB = np.linalg.norm(B, axis = 1)
    den = np.dot(denA, denB)
    return 1- num/den


def convert_list_to_dict(lister):
    chain_dict = dict()
    for c in lister:
        chain_dict[c[0]] = c[1:]
    return chain_dict

def convert_dict_to_list(dictionary):
    lister = []
    for a in dictionary:
        lister.append([a] + list(dictionary[a]))
    return lister

def chain_scoring(chains):
    if type(chains) != list:
        chains = convert_dict_to_list(chains)
    chain_scores= dict()
    for a in range(0, len(chains)):
        current_chain = chains[a]
        summer =0
        for c in range(0, len(current_chain)-1):
            summer += cosine(mapper[current_chain[c]], mapper[current_chain[c+1]])
        chain_scores[chains[a][0]] = summer
    return chain_scores

def skip_agglomerative_distance(clusters):
    if type(clusters) != list:
        clusters = convert_dict_to_list(clusters)
    dists = dict()  
    for ch in range(0, len(clusters)):
        current_cluster = clusters[ch]
        distance = 0
        current_word = mapper[current_cluster[0]] 
        for c in range(0, len(current_cluster)-1):   
            current_word = mapper[current_cluster[0]] +  mapper[current_cluster[c+1]]
        dists[current_cluster[0]] = distance
    sort_dists = sorted(dists.items(), key = operator.itemgetter(1))
    return sort_dists[0][0]
    #key_clus_cent = [clus_cent_cons] + list(skip_clusters[clus_cent_cons])
    

def cluster_center_consideration(clusters, mapper):
    mapper =copy.deepcopy(mapper)
    if type(clusters) != list:
        clusters = convert_dict_to_list(clusters)
    dists = dict()  
    for ch in range(0, len(clusters)):
        current_cluster = clusters[ch]
        distance = 0
        current_word = mapper[current_cluster[0]] 
        for c in range(0, len(current_cluster)-1):   
            current_word = mapper[current_cluster[0]] +  mapper[current_cluster[c+1]]
        dists[current_cluster[0]] = distance
    sort_dists = sorted(dists.items(), key = operator.itemgetter(1))
    return sort_dists[0]

def baseline_score(di, lemmatized, no_of_keywords):
    tfidf_dict = dict()
    for le in lemmatized:
        if le in di:
            tfidf_dict[le] = di[le]

    sorted_tdidf = sorted(tfidf_dict.items(), key = operator.itemgetter(1))
    s = sorted_tdidf[::-1][:no_of_keywords]
    #print s
    keyw = []
    for so in s:
        keyw.append(so[0])
    
    return keyw

def skip_agglomerative_method(mapper, count):
    
    items = mapper.keys()
    dist_dict = {}
    word_dict = {}
    for i in range(0, len(items)):
        ind = []
        words = []
        distance = []

        center = items[i]
        
        ind.append(i)
        words.append(center)
        
        values = np.array(mapper.values())
        first = cosine_broad(mapper[center], values)
        first[first == 0] = 100.0
        min_index = np.argmin(first)
        dist = np.min(first)
        ind.append(min_index)
        distance.append(dist)
        if items[min_index] not in words:
            words.append(items[min_index])

        while len(words) < count:
            agg = mapper[center] + mapper[words[-1]]
            next_word_dist = cosine_broad(agg, values)
            next_word_dist[ind] = 100.0
            next_word_dist[next_word_dist < 0.3] = 100.0
            min_index = np.argmin(next_word_dist)
            dist=  np.min(next_word_dist)
            distance.append(dist)
            ind.append(min_index)
            predicted_word = items[min_index]
            words.append(items[min_index])
            #print center, words, np.mean(distance)
        dist_dict[center] = np.mean(distance)
        word_dict[center] = words[1:]
    #return dist_dict, word_dict
    return word_dict

def word_to_word_distance(all_clusters):
    chain_scores = chain_scoring(all_clusters)
    sorted_chain_scores = sorted(chain_scores.items(), key = operator.itemgetter(1))
    key = sorted_chain_scores[0][0]
    key_chain_eval = [key] + list(all_clusters[key])
    return key_chain_eval

def find_closest_word(word, mapper, diction):
    if len(word.split()) == 1:
        if  word in diction:
            word_vector = diction[word]
        else:
            word_vector = random_vect
    else:
        word_vector = 0
        for w in word.split():
            if w in diction:
                word_vector+= diction[w]
        word_vector/= len(word.split())
                
    distances = cosine_broad(word_vector, np.array(mapper.values()))
    return mapper.keys()[np.argmin(distances)]



import sys
arguments = sys.argv[:]
word_vectors = "GoogleNews-vectors-negative300.bin"
document = "docs\\" + arguments[1]#"animal_groups.txt" #survive.txt"
stopwords = "stopwords_en.txt"

random_vect = np.random.rand(300)
#diction = gensim.models.KeyedVectors.load_word2vec_format('C:\Users\prudh\Quora\GoogleNews-vectors-negative300.bin', binary= True)
direc = word_vectors
diction = gensim.models.KeyedVectors.load_word2vec_format(direc, binary = True)

text= open(document).read()
stopwords = open(stopwords).read().splitlines()

open_name = open("tf_idf_dict.txt", "rb")
di = pickle.load(open_name)

#mode_number = 1
mode_number = int(arguments[2])


def average_diameter(clusters):
    distance_dict = {}
    for c in clusters:
        distance_dict[c] = np.mean(cosine_broad(mapper[c], np.array(mapper.values())))
    #Find min distance
    min_index = np.argmin(distance_dict.values())
    center = distance_dict.keys()[min_index]
    return center

def metric_chooser(clusters, distance_metric):
    if distance_metric == "avg":
        cluster_center = average_diameter(clusters)
        pre_tf_scored_words = [cluster_center] + list(clusters[cluster_center])
    if distance_metric == "w2w":
        pre_tf_scored_words = word_to_word_distance(clusters)
    if distance_metric == "skip":
        cluster_center = skip_agglomerative_distance(clusters)
        pre_tf_scored_words = [cluster_center] + list(clusters[cluster_center])
    return pre_tf_scored_words

def relative_keywords(text):
    sepe = pre_stripper(text.split())
    candidate_words = candidate_word_extractor(stopwords, sepe)
    lemmatized = stripper(lemmatization(candidate_words))
    return lemmatized

def absolute_keywords(text):
    sepe = pre_stripper(text.split())
    candidate_words = candidate_phrase_extraction(stopwords, sepe)
    candidate_words = stripper(candidate_words)
    phrases = phrase_filtering(candidate_words, 0.85, diction)
    lemmatized = stripper(lemmatization(phrases))
    return lemmatized
    
if mode_number == 1:
    if len(arguments) > 3:    
        distance_metric = arguments[3]
    else:
        distance_metric = "w2w"
    
    lemmatized = relative_keywords(text)
    mapper = distrib_represent_conversion(lemmatized, diction)
    no_of_keywords = int(np.floor(len(lemmatized)/1.5))
    count = len(lemmatized)/2

    clusters = clustering(mapper, no_of_keywords)
    pre_tf_scored_words = metric_chooser(clusters, distance_metric)
    tf_scored_words = baseline_score(di, pre_tf_scored_words, count)
    print distance_metric
    print tf_scored_words

if mode_number == 2:
    if len(arguments) > 3:    
        distance_metric = arguments[3]
    else:
        distance_metric = "skip"

    lemmatized = relative_keywords(text)
    mapper = distrib_represent_conversion(lemmatized, diction)
    no_of_keywords = int(np.floor(len(lemmatized)/1.5))
    count = len(lemmatized)/2

    clusters = skip_agglomerative_method(mapper, no_of_keywords)
    pre_tf_scored_words = metric_chooser(clusters, distance_metric)

    tf_scored_words = baseline_score(di, pre_tf_scored_words, count)
    print distance_metric
    print tf_scored_words

if mode_number == 3:
    no_of_keywords = int(arguments[3])
    if len(arguments) > 4:    
        distance_metric = arguments[4]
    else:
        distance_metric = "w2w"

    lemmatized = absolute_keywords(text)
    mapper = distrib_represent_conversion(lemmatized, diction)
    clusters = clustering(mapper, no_of_keywords)
    pre_tf_scored_words = metric_chooser(clusters, distance_metric)
    print distance_metric
    print pre_tf_scored_words    

if mode_number == 4:
    no_of_keywords = int(arguments[3])
    if len(arguments) > 4:    
        distance_metric = arguments[4]
    else:
        distance_metric = "skip"
    lemmatized = absolute_keywords(text)
    mapper = distrib_represent_conversion(lemmatized, diction)
    clusters = skip_agglomerative_method(mapper, no_of_keywords)
    
    pre_tf_scored_words = metric_chooser(clusters, distance_metric)
    print distance_metric
    print pre_tf_scored_words

if mode_number == 5:
        
    no_of_keywords = int(arguments[3])
    lemmatized = absolute_keywords(text)
    mapper = distrib_represent_conversion(lemmatized, diction)
    clusters = clustering(mapper, no_of_keywords+1)
    
    if len(arguments) > 4:    
        related_word = arguments[4]    
        if related_word not in clusters:
            closest_word = find_closest_word(related_word, mapper, diction)
            print [closest_word] + list(clusters[closest_word])        
        if related_word in clusters:
            print [related_word] + list(clusters[related_word])
    else:
        pre_tf_scored_words = metric_chooser(clusters, "w2w")
        print pre_tf_scored_words    
    

if mode_number == 6:
    no_of_keywords = int(arguments[3])
    
    lemmatized = absolute_keywords(text)
    mapper = distrib_represent_conversion(lemmatized, diction)
    clusters = skip_agglomerative_method(mapper, no_of_keywords+1)
        
    if len(arguments) > 4:    
        related_word = arguments[4]    
        if related_word not in clusters:
            closest_word = find_closest_word(related_word, mapper, diction)
            print [closest_word] + list(clusters[closest_word])        
        if related_word in clusters:
            print [related_word] + list(clusters[related_word])
    else:
        pre_tf_scored_words = metric_chooser(clusters, "skip")
        print pre_tf_scored_words    
