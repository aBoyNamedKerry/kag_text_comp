# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:34:03 2019

@author: User
"""
import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from sklearn import tree
import glob, os
import pickle
from features import *


tokenize_train = False
tokenize_test = False
build_train_features = False
build_model = True
build_test_features = False
execute_model = True

def tokenize(df, name):
    df['q1_tokens'] = df.apply(lambda row: nltk.word_tokenize(row['question1']), axis=1)
    df['q2_tokens'] = df.apply(lambda row: nltk.word_tokenize(row['question2']), axis=1)
    
    df.to_csv('data/' + name + '_tokenized.csv')
    return df
    

def create_features(df_input, directory):
    # df_input should be training or test data whihc has been tokenized and has columns q1_tokens and q2_tokens
   
    
    feature_word_overlap(df_input, directory)
    feature_number_of_words(df_input, directory)
    feature_average_token_length(df_input, directory)
    feature_levenshtein_distance(df_input, directory)
    feature_cosine_similarity(df_input, directory)
    return True

def get_features_from_csv(directory_name):
    
    df_features_all = pd.DataFrame()
    os.chdir(directory_name)
    for file in glob.glob("*.csv"):
 
        df_temp_features = pd.read_csv(file)
        df_temp_features.replace(to_replace='NA', value=0)
        print ('reading file ' + directory_name + file + ' from csv')
        for column in df_temp_features:
            df_features_all[column] = df_temp_features[column]
            print('added feature ' + column)
    os.chdir('..')        
    return df_features_all

if build_train_features:
    if tokenize_train:
        df_train = pd.read_csv('data/train.csv')
    
        df_train = df_train.fillna('noentry')
        df_train = tokenize(df_train, 'train')
    else:
        # read from csv already tokenized
        df_train = pd.read_csv('data/train_tokenized.csv', encoding = "ISO-8859-1")

    #df_train = df_train[0:100]
    create_features(df_train, 'features_train')

if build_model:
    if not build_train_features:
        df_train = pd.read_csv('data/train_tokenized.csv', encoding = "ISO-8859-1")
    df_train_features = get_features_from_csv('features_train')
    
    X = df_train_features.values
    y = df_train['is_duplicate'].values
    
    #print(X)
    #print(y)
    
    #clf = tree.DecisionTreeClassifier(max_depth = 4)
    #clf = RandomForestClassifier(n_estimators=50, min_samples_leaf=500) # -0.508
    clf1 = LogisticRegression(solver='lbfgs',
                              random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, min_samples_leaf=500, random_state=1)
    clf3 = GaussianNB()
    clf4 = LinearSVC()
    clf5 = AdaBoostClassifier()
    clf_z = VotingClassifier(estimators=[
            ('lr', clf1), ('rf', clf2), ('', clf3), ('svm', clf4)], voting='soft')

    
    # Try this on the model build - it will double the 
    #class_weight={0: 2, 1: 1}
    clf = RandomForestClassifier(n_estimators=50, min_samples_leaf=500, random_state=1)
    
    get_score=False
    if get_score:
        print("Running cross_val_score")
        scores = cross_val_score(clf, X, y, cv=2, scoring = 'neg_log_loss') #scoring = 'neg_log_loss',scoring = 'accuracy'
       
        print(scores)
       
    print('fitting model on all training data')
    clf.fit(X,y)
    
    try:
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
    
        for f in range(X.shape[1]):
            print("%d. feature %d %s (%f)" % (f + 1, indices[f], df_train_features.columns[indices[f]],importances[indices[f]]))
        
            
    except:
        print("Feature importances not available")
   
    

    # save the model to disk
    modelfilename = 'finalized_model.pkl'
    pickle.dump(clf, open(modelfilename, 'wb'))
    
    #overfitted, split into train/test!
    probs = clf.predict_proba(X)
    pred = clf.predict(X)
    #print(probs[:,1])
    df=pd.DataFrame()
    #df['X'] = df_train_features
    df['probs'] = probs[:,1]
    df['pred'] = pred
    df['actual'] = df_train['is_duplicate']    
    #print(df[0:30])

    
if build_test_features:
    if tokenize_test:
        df_test = pd.read_csv('data/test.csv', dtype={"test_id": str, "question1": str,  "question2": str})
        #df_test = df_test[0:100000]
        print(df_test)
        df_test = df_test.fillna('noentry')
        df_test = tokenize(df_test, 'test2')
        
    else:
        # read from csv already tokenized
        print ('read_test_tokenized')
        df_test = pd.read_csv('data/test2_tokenized.csv', dtype={"test_id": str, "question1": str,  "question2": str}, encoding = "ISO-8859-1")

    #df_test = df_test[0:100]
    #print(df_test)
    print('start creating test features')
    create_features(df_test, 'features_test')
    
if execute_model:
    if not build_test_features:
        df_test = pd.read_csv('data/test2_tokenized.csv', dtype={"test_id": str, "question1": str,  "question2": str}, encoding = "ISO-8859-1")
    df_test=df_test[0:2345796]
    clf = pickle.load(open(modelfilename, 'rb'))
    print('getting features')
    df_test_features = get_features_from_csv('features_test')
    df_test_features = df_test_features[0:2345796]
    X = df_test_features.values
    probs = clf.predict_proba(X)
    show_results=False
    if show_results:
        df=pd.DataFrame()
        df['X']=df_test_features
        df['probs'] = probs[:,1]
        print(df[0:100])
    df_submit = pd.DataFrame()
    df_submit['test_id'] = df_test['test_id']
    df_submit['is_duplicate'] = probs[:,1]
    df_submit = df_submit[0:2345796]
    df_submit.to_csv('submission.csv', index=False)
    