# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:34:03 2019

@author: User
"""
import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import glob, os
import pickle


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
      
    