import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import streamlit as st

df_edu = pd.read_csv('data/eng_sample_data_overview.csv')

df = df_edu.copy()
target = 'Score'
encode = ['Class','Subject']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df,dummy], axis = 1)
    del df[col]

X = df.drop(['Score','Teacher','ID'],axis=1)
Y = df['Score']

clf = RandomForestRegressor()
clf.fit(X, Y)

pickle.dump(clf, open('data/subject_score_prediction.pkl', 'wb'))