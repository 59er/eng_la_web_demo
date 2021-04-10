import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st 

df_edu = pd.read_csv('data/sample_data_pass_fail.csv')

df = df_edu.copy()
target = 'Pass/Fail'
encode = ['Class','Subject']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df,dummy], axis = 1)
    del df[col]

target_mapper = {'Fail':0, 'Pass':1}
def target_encode(val):
    return target_mapper[val]

df['Pass/Fail'] = df['Pass/Fail'].apply(target_encode)

X = df.drop(['Pass/Fail','Teacher','ID'],axis=1)
Y = df['Pass/Fail']

clf = RandomForestClassifier(n_estimators = 250, random_state = 0)
clf.fit(X, Y)

pickle.dump(clf, open('data/subject_pass_fail.pkl', 'wb'))
