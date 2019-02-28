# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:39:33 2019

@author: User
"""

import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import glob, os
import pickle
import math
import Counter
import jellyfish

def feature_word_overlap(df, directory):
    df_return = pd.DataFrame()
    shared_words = df.apply(lambda row:len(set(row['q1_tokens']) & set(row['q2_tokens']))/len(set(row['q1_tokens'] + row['q2_tokens'])), axis=1)
    df_return['shared_words'] = shared_words
    filename = directory + '/feature_word_overlap.csv'
    df_return.to_csv(filename, index=False)
    print ('written ' + filename)
    return True

def feature_number_of_words(df, directory):
    df_return = pd.DataFrame()
    df_return['diff_num_words'] = df.apply(lambda x: len(x['q1_tokens']) - len(x['q2_tokens']), axis=1)
    filename = directory + '/feature_word_number_of_words.csv'
    df_return.to_csv(filename, index=False)
    print ('written ' + filename)

def feature_average_token_length(df, directory):
    df_return = pd.DataFrame()
    df_return['diff_num_words'] = df.apply(lambda x: np.mean([len(word) for word in x['q1_tokens']]) - np.mean([len(word) for word in x['q2_tokens']]), axis=1)
    df_return.to_csv(directory +'/feature_average_token_length.csv', index=False)

def feature_levenshtein_distance(df, directory):
    # Packages considered: 
    # https://pypi.org/project/jellyfish/ - selected as updated most recently and 1k Github stars
    # https://pypi.org/project/editdistance/
    # https://pypi.org/project/python-Levenshtein/0.12.0/

    df_return = pd.DataFrame()
    df_return['levenshtein'] = df.apply(lambda x: jellyfish.levenshtein_distance(''.join(x['q1_tokens']),''.join(x['q2_tokens'])), axis=1)
    df_return.to_csv(directory + '/feature_levenshtein_distance.csv', index=False)





def _naive_cosine(vec1, vec2):
    # From https://stackoverflow.com/questions/15173225/calculate-cosine-similarity-given-2-sentence-strings
    vec1 = Counter(vec1)
    vec2 = Counter(vec2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator



def feature_cosine_similarity(df, directory):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python
    df_return = pd.DataFrame() 
    df_return['cosine_similarity'] = df.apply(lambda x: _naive_cosine(x['q1_tokens'], x['q2_tokens']), axis=1)
    df_return.to_csv(directory + '/feature_cosine_similarity.csv', index=False)