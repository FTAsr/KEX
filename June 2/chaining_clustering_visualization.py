# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 13:40:40 2017

@author: prudh
"""

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
np.random.seed(2)


random_vect = np.random.rand(300)
diction = gensim.models.KeyedVectors.load_word2vec_format('C:\Users\prudh\Quora\GoogleNews-vectors-negative300.bin', binary= True)

'''
This method takes a split file as an input and strips
it down to its special characters.

For the process of candidate phrase extraction,
this method adds an extra space after a word accompanying 
a special character.

For example, if the token is 'easily,', this strips and
replaces it with a space special character, so the 
resultant would be 'easily ,'. Now when it is split, the 
word and the specical character come out as two different
entities.

This method returns all the words along with the special 
characters.
'''
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



# Phrase extraction
'''
The method takes the pre-stripped tokens and uses 
them to create candidate phrases.

The creation of candidate clean phrase follows
a similar process of RAKE(Rapid automatic keyword extraction for 
information retrieval and analysis US 8131735 B2). Whenever a stopword
or a special character arrives, the group of words between these constraints
is considered to be a candidate phrase.

This function takes a stopword list, and the split words
which are pre-processed in the function (pre_stripper) and returns
a set of candidate phrases.
'''
def candidate_phrase_extraction(stopwords, seperated):
    phrases = []
    current_word = ''
    for s in seperated:
        if s.lower() not in stopwords and s.isalnum() == True:
            current_word+= s + ' '
        if s.lower() in stopwords or s.isalnum() == False:
            phrases.append(current_word)
            current_word = ''
    return phrases

# Clearing empty spots
'''
This is a small helper function that removes
trailing spaces from the extracted candidate
keywords
'''
def stripper(phrases):
    candidate_phrases = []
    for p in phrases:
        if p != '':
            candidate_phrases.append(p.strip())
    return candidate_phrases

#Split candidate phrases
'''
This method is used for filtering the candidate phrases.

The main purpose of this function is to find whether a given
candidate phrases is an eligible phrase. A phrase can be described,
in this context as a sequence of words that represent a cohesive 
meaning. 

Therefore, for each phrase, the words are split and are compared
to one another. This comparison is done using cosine distance and 
a threshold is used.

If threshold is low: less phrases, more words
if threshold is high: less words, more phrases
if threshold is very high: less words, more phrases with unneccesary connotations 

The idea is to find a good threshold that lies between both the cases.

Tis method takes the candidate phrases, a threshold value,
and a distributional representation dictionary as argument and
returns a set of filtered phrases and words.
'''
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

'''
Lemmatization converts a word into its most
common form.

This use of this is to convert plural forms of words
to singular forms on the lower level.

This takes the filtered phrases as input and returns 
them with there lemmatized form, if there exists one.
'''
def lemmatization(clean_phrases):
    lm = WordNetLemmatizer()
    cleanest_phrases = []
    for c in clean_phrases:
        cleanest_phrases.append(np.str(lm.lemmatize(c)))
    clean_phrases = copy.deepcopy(cleanest_phrases)
    clean_phrases = np.unique(clean_phrases)
    return clean_phrases


#Convert phrases into distributional representations
'''
This method takes the lemmatized phrases and words
and converts them into a dictionary with their respective
distributional vectors.

This takes two arguments as input - a list of phrases
and words, and the word2vec or GloVe dictionary.

For phrases, it is split into seperate words, and those
are added and normalized to get a single vector.
'''
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

#Broadcastable cosine function
'''
This method was written to make cosine
distance broadcastable. 

For clustering, this helps make this run
fast.

This takes two vectors, or one vector and 
an array of vectors as input, and returns
an array of cosine distances. 
'''
def cosine_broad(A, B):
    num = np.dot(A, B.T)
    denA = np.linalg.norm(A)
    denB = np.linalg.norm(B, axis = 1)
    den = np.dot(denA, denB)
    return 1- num/den

#chaining method
'''
This method implements the chaining algorithm.

This function takes three arguments:
    k_count - Number of keywords the user wants to extract
    w2v_matrix - the values part of the mapper function, that 
                 contains a matrix of word vectors
    words_original - the list of words and phrases taken into
                     account (or) keys in the mapper dictionary
'''
def close_topic_clustering(k_count, w2v_matrix, words_original):
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
    
    return keywords, locations, iter_words

#chaining method
#keyword collector
#Cluster method

#Clustering method
'''
This method implements the clustering algorithm.

This function takes two arguments:
    mapper - a dictionary that maps words/phrases to
             its word vector
    k_count - the number of keywords to be extracted
'''
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


#chaining evaluation for clustering results
'''
This function takes the results of the clustering 
algorithm and applies a chaining evaluation to chose the
best set of keywords.
'''
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

'''
A simple evaluator function
that calculates the number of hits,
misses, and maybes.
'''
def evaluate(gt, keywords):
    hits = np.intersect1d(gt, keywords)
    union = (set(keywords).symmetric_difference(gt))
    misses = np.intersect1d(gt,list(union))
    halfs= set(union).symmetric_difference(misses)
    
    return hits, misses, halfs

'''
A simple score function that scores 
the effectiveness of the keywords predicted
using the counts from evaluation function
'''
def score(gt, keywords):
    hits, misses, halfs = evaluate(gt, keywords)
    n_hits, n_misses, n_halfs = len(hits), len(misses), len(halfs)
    scorer = (1.0 * n_hits) + (0.5 * n_halfs)  - (1.0* n_misses)# 
    scorer = scorer/len(keywords)
    return scorer

os.chdir('NLP_2')
files = open('labeled_data_12.txt').read().splitlines()
#files = open('287.txt').read()
stopwords = open('stopwords_en.txt').read().splitlines()
s1, s2, s3 = [], [], []
for i in range(0, len(files)-1):
    current_file = files[i]
    seperated = current_file.split()
    gt = files[i+1].split(',')
    gt = stripper(gt)
    thresholder = 0.97
    if i%2 == 0:
        k = len(gt) + 1 
        seperated = pre_stripper(seperated)            
        phrases =  candidate_phrase_extraction(stopwords, seperated)
        candidate_phrases = stripper(phrases)
        filtered_phrases = phrase_filtering(candidate_phrases, thresholder, diction)       
        clean_phrases = stripper(filtered_phrases)
        clean_phrases = lemmatization(clean_phrases)
        mapper = distrib_represent_conversion(clean_phrases, diction)
        keyword_chaining, loc, chaining_dict = close_topic_clustering(k, np.array(mapper.values()), mapper.keys())
        keywords_cluster, keyword_dict = keyword_cluster(mapper, k)
        print keyword_dict
        keywords_cluster_chain = cluster_chain_evaluation(mapper, keyword_dict, k)

        #print keyword_chaining
        s1.append(score(gt, keyword_chaining))
        s2.append(score(gt, keywords_cluster))
        s3.append(score(gt, keywords_cluster_chain))

print np.mean(s1)
print np.mean(s2)
print np.mean(s3)


'''
The code below is for the purpose of visualization
'''

'''
The method take the distributional representation
matrix and applies dimensionality reduction to convert it into
a 3-dimensional space.
'''
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
        print locs
        locs.append(np.where(np.array(mapper.keys()) == k)[0][0])
    return locs

'''
This visualizes the keywords and non-keywords in
a 3-d space.
'''
def vizualize(mapper, keywords, fitter, filename):
    words_original = mapper.keys() 
    locations = location_finder(mapper, keywords)
    fig = plt.figure(figsize = (20, 20))
    #ax = fig.add_subplot(111, projection = '3d')
    ax = Axes3D(fig)
    
    for l in range(0, len(mapper.keys())):
        if l not in locations:
            ax.scatter(fitter[l, 0], fitter[l, 1], fitter[l, 2])
    for i, txt in enumerate(words_original):
        if txt not in keywords:
            ax.text(fitter[:, 0][i], fitter[:, 1][i], fitter[:, 2][i], txt, fontsize= 25)

    ax.text(fitter[locations[0], 0], fitter[locations[0], 1], fitter[locations[0], 2], 
            words_original[locations[0]] + '(' +str(1) +')', bbox=dict(facecolor='yellow', alpha=0.5),
                          fontsize= 25)
    
    ax.scatter(fitter[locations, 0], fitter[locations, 1], fitter[locations, 2])
    for i, txt in enumerate(keywords):
        ax.text(fitter[locations, 0][i], fitter[locations, 1][i], fitter[locations, 2][i], 
                txt + '(' +str(i+1) +')', bbox=dict(facecolor='red', alpha=0.5), fontsize= 25)

    ax.text(fitter[locations[0], 0], fitter[locations[0], 1], fitter[locations[0], 2], 
            words_original[locations[0]] + '(' +str(1) +')', bbox=dict(facecolor='yellow', alpha=0.95), 
                          fontsize= 25)
    
    ax.tick_params(labelsize = 25)
    plt.savefig(filename)



'''
Saves visualizations of all the keyword sets
'''
def vis_saver(mapper, keyword_dict, file_name):
    for i in range(0, len(mapper.keys())):
        current_key = keyword_dict.keys()[i]
        keyed_values = [current_key] + list(keyword_dict[current_key])
        vizualize(mapper, keyed_values, fitter, file_name + '_' +  current_key + ".png")


def chain(chaining_dict):
    chain_dict = dict()
    for c in chaining_dict:
        chain_dict[c[0]] = c[1:]
    return chain_dict

fitter=representation_reduction(mapper)


vis_saver(mapper, chain(chaining_dict), 'nucleotides_chain')



