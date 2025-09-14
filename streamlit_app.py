import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# loading the csv data to a Pandas DataFrame
with st.expander('data'):
 st.write('**Raw Data**')
 heart_data = pd.read_csv('heart.csv')
 heart_data
 st.write('**X**')
 X = heart_data.drop(columns='target',axis=1)
 X
 st.write('**Y**')
 Y = heart_data['target']
 Y
with st.expander('data visualization'):
 #age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target

 st.bar_chart(data=None, *, x=age, y=fbs, x_label=None, y_label=None, color=#ffaa00, horizontal=False, stack=None, width=None, height=None, use_container_width=True)
 
