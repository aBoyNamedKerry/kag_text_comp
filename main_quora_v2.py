# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:34:03 2019

@author: User
"""
import nltk
import pandas as pd
import numpy as np
import jellyfish
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import glob, os
import pickle
import re, math
from collections import Counter

tokenize_train = False
tokenize_test = True
build_train_features = True
build_model = True
build_test_features = True
execute_model = True

def tokenize(df, name):
    df['q1_tokens'] = df_train.apply(lambda row: nltk.word_tokenize(row['question1']), axis=1)
    df['q2_tokens'] = df_train.apply(lambda row: nltk.word_tokenize(row['question2']), axis=1)
    
    df.to_csv('data/' + name + '_tokenized.csv')
    return df
    
def feature_word_overlap(df):
    df_return = pd.DataFrame()
    shared_words = df.apply(lambda row:len(set(row['q1_tokens']) & set(row['q2_tokens']))/len(set(row['q1_tokens'] + row['q2_tokens'])), axis=1)
    df_return['shared_words'] = shared_words
    df_return.to_csv('features_train/feature_word_overlap.csv', index=False)
    print ('written feature_word_overlap to csv')
    return True

def feature_number_of_words(df):
    df_return = pd.DataFrame()
    df_return['diff_num_words'] = df.apply(lambda x: len(x['q1_tokens']) - len(x['q2_tokens']), axis=1)
    df_return.to_csv('features_train/feature_number_of_words.csv', index=False)

def feature_average_token_length(df):
    df_return = pd.DataFrame()
    df_return['diff_num_words'] = df.apply(lambda x: np.mean([len(word) for word in x['q1_tokens']]) - np.mean([len(word) for word in x['q2_tokens']]), axis=1)
    df_return.to_csv('features_train/feature_average_token_length.csv', index=False)

def feature_levenshtein_distance(df):
    # Packages considered: 
    # https://pypi.org/project/jellyfish/ - selected as updated most recently and 1k Github stars
    # https://pypi.org/project/editdistance/
    # https://pypi.org/project/python-Levenshtein/0.12.0/

    df_return = pd.DataFrame()
    df_return['diff_num_words'] = df.apply(lambda x: jellyfish.levenshtein_distance(''.join(x['q1_tokens']),''.join(x['q1_tokens'])), axis=1)
    df_return.to_csv('features_train/feature_levenshtein_distance.csv', index=False)





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



def feature_cosine_similarity(df):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python
    df_return = pd.DataFrame() 
    df_return['cosine_similarity'] = df.apply(lambda x: _naive_cosine(x['q1_tokens'], x['q2_tokens']), axis=1)
    def_return.to_csv('features_train/feature_cosine_similarity.csv', index=False)

def create_features(df_input):
    # df_input should be training or test data whihc has been tokenized and has columns q1_tokens and q2_tokens
    feature_word_overlap(df_input)
    return True

def get_features_from_csv(directory_name):
    
    df_features_all = pd.DataFrame()
    os.chdir(directory_name)
    for file in glob.glob("*.csv"):
 
        df_temp_features = pd.read_csv(file)
        print ('reading file ' + directory_name + file + ' from csv')
        for column in df_temp_features:
            df_features_all[column] = df_temp_features[column]
            print('added feature ' + column)
    os.chdir('..')        
    return df_features_all

if build_features:
    if tokenize_train:
        df_train = pd.read_csv('data/train.csv')
    
        df_train = df_train.fillna('noentry')
        df_train = tokenize(df_train, 'train')
    else:
        # read from csv already tokenized
        df_train = pd.read_csv('data/train_tokenized.csv', encoding = "ISO-8859-1")

    df_train = df_train[0:100]
    df_train_features = create_features(df_train)

if build_model:

    df_train_features = get_features_from_csv('features_train')
    
    X = df_train_features.values
    y = df_train['is_duplicate'].values
    
    #print(X)
    #print(y)
    
    clf = tree.DecisionTreeClassifier(max_depth = 10)
    #clf = RandomForestClassifier(n_estimators=3, max_depth=5)
      
    scores = cross_val_score(clf, X, y, cv=5, scoring = 'accuracy') #scoring = 'neg_log_loss'
   
    print(scores)
       
    clf.fit(X,y)
    
    # save the model to disk
    modelfilename = 'finalized_model.pkl'
    pickle.dump(clf, open(modelfilename, 'wb'))
 

def main(): 
    if build_test_features:
        if tokenize_test:
            df_test = pd.read_csv('data/test.csv')
        
            df_test = df_test.fillna('noentry')
            df_test = tokenize(df_test, 'test')
        else:
            # read from csv already tokenized
            df_test = pd.read_csv('data/test_tokenized.csv', encoding = "ISO-8859-1")

    df_test = df_test[0:100]
    df_test_features = create_features(df_test)
        
    if execute_model:

        clf = pickle.load(open(modelfilename, 'rb'))
        df_test_features = get_features_from_csv('features_test')

        X = df_test_features.values
        probs = clf.predict_proba(X)
        print(probs)
          
if __name__=='__main__':
    main()
