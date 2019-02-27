# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:34:03 2019

@author: User
"""
import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

tokenize_train = False
tokenize_test = True
build_model = True
execute_model = True

def tokenize(df, name):
    df['q1_tokens'] = df_train.apply(lambda row: nltk.word_tokenize(row['question1']), axis=1)
    df['q2_tokens'] = df_train.apply(lambda row: nltk.word_tokenize(row['question2']), axis=1)
    
    df.to_csv('quora_data/' + name + '_tokenized.csv')
    return df
    
def feature_word_overlap(df):
    df_return = pd.DataFrame()
    shared_words = df.apply(lambda row:len(set(row['q1_tokens']) & set(row['q2_tokens']))/len(set(row['q1_tokens'] + row['q2_tokens'])), axis=1)
    df_return['shared_words'] = shared_words
    return df_return

def features(df_input):
    # df_input should be training or test data whihc has been tokenized and has columns q1_tokens and q2_tokens
    df_features = pd.DataFrame()
    
    df_features1 = feature_word_overlap(df_input)
    for column in df_features1:
        df_features[column] = df_features1[column]
    
    return df_features

if build_model:
    if tokenize_train:
        df_train = pd.read_csv('quora_data/train.csv')
    
        df_train = df_train.fillna('no entry')
        df_train = tokenize(df_train, 'train')
    else:
        # read from csv already tokenized
        df_train = pd.read_csv('quora_data/train_tokenized.csv', encoding = "ISO-8859-1")

    df_train_features = features(df_train)


    
    X = df_train_features.values
    y = df_train['is_duplicate'].values
    
    print(X)
    print(y)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    clf.fit(X, y)