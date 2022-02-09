# -*- coding: utf-8 -*-
"""
This app allows to input new values for risk factors 
and predict if the pacient has diabetes in real-time
"""
#!/usr/bin/env python
# coding: utf-8

#%% Import Libraries

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#%% Import the dataset
df = pd.read_csv("diabetes.csv")

#%% Reproducing the model using the dataset from India

# standardize the variables of the dataset
scaler = StandardScaler()
scaler.fit(df.drop('Outcome',axis=1))
scaled_features = scaler.transform(df.drop('Outcome',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
   
# define Train Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Outcome'],
                                                    test_size=0.30,random_state=42)

# training KNN
knn = KNeighborsClassifier(n_neighbors=21)

knn.fit(X_train,y_train)

#%% Creating the app

# Title
st.write("""
Predicting Diabetes\n
App that uses machine learning to predict wheater a patient has diabetes\n
Source of trainning/testing dataset: PIMA - INDIA (Kaggle)       
""")

# Cabeçalho
st.subheader('Data information')

# Nome do usuário
user_input = st.sidebar.text_input('Input your name')

st.write('Patient: ', user_input)

# dados dos usuários com a função
def get_user_data():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 15, 1)
    glucose = st.sidebar.slider('Glucose', 0.0, 200.0, 110.0)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0.0, 122.0, 72.0)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0.0, 99.0, 20.0)
    insulin = st.sidebar.slider('Insulin', 0.0, 900.0, 30.0)
    bmi = st.sidebar.slider('Body Mass Index', 0.01, 70.0, 15.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 3.0, 0.0)
    age = st.sidebar.slider('Age', 15, 100, 21)
    
    # um dicionário recebe as informações acima
    user_data = {'Pregnancies': pregnancies,
                'Glucose': glucose,
                'Blood Pressure': blood_pressure,
                'Skin Thickness': skin_thickness,
                'Insuline': insulin,
                'Body Mass Index': bmi,
                'Diabetes Pedigree Function': dpf,
                'Age': age
                 }
    features = pd.DataFrame(user_data,index=[0])
  
    return features

user_input_variables = get_user_data()

# gráfico
graf = st.bar_chart(user_input_variables)
    
st.subheader('Data of the patient')
st.write(user_input_variables)

# standardize the new variables input by the user
user_input_variables_standard = scaler.transform(user_input_variables)

# Predict
prediction = knn.predict(user_input_variables_standard)


# Acurácia do modelo 
#st.subheader('Acuracia do modelo')
st.write(accuracy_score(y_test, knn.predict(X_test))*100)


st.subheader('Pretiction: ')
st.write(prediction)
'''0: no
1: yes
'''


