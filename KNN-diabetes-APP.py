# -*- coding: utf-8 -*-
"""
Here it is possible to input new values for risk factors 
and predict if the pacient has diabetes in real-time
"""
#!/usr/bin/env python
# coding: utf-8

#%% Import Libraries

import pandas as pd
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
App que utiliza machine learn para prever possível diabetes dos pacientes.\n
Fonte: PIMA - INDIA (Kaggle)       
""")

# Cabeçalho
st.subheader('Informações dos dados')

# Nome do usuário
user_input = st.sidebar.text_input('Digite seu nome')

st.write('Paciente: ', user_input)

# dados dos usuários com a função
def get_user_data():
    pregnancies = st.sidebar.slider('Gravidez', 0, 15, 1)
    glucose = st.sidebar.slider('Glicose', 0.0, 200.0, 110.0)
    blood_pressure = st.sidebar.slider('Pressão Sanguínea', 0.0, 122.0, 72.0)
    skin_thickness = st.sidebar.slider('Espessuara da pele', 0.0, 99.0, 20.0)
    insulin = st.sidebar.slider('Insulina', 0.0, 900.0, 30.0)
    bmi = st.sidebar.slider('Índice de massa corporal', 0.01, 70.0, 15.0)
    dpf = st.sidebar.slider('Histórico familiar de diabetes', 0.0, 3.0, 0.0)
    age = st.sidebar.slider('Idade', 15, 100, 21)
    
    # um dicionário recebe as informações acima
    user_data = {'Gravidez': pregnancies,
                'Glicose': glucose,
                'Blood pressure': blood_pressure,
                'Espessura da pele': skin_thickness,
                'Insulina': insulin,
                'Indice de massa corporal': bmi,
                'Historico familiar de diabetes': dpf,
                'Idade': age
                 }
    features = pd.DataFrame(user_data,index=[0])
  
    return features

user_input_variables = get_user_data()

# gráfico
graf = st.bar_chart(user_input_variables)
    
st.subheader('Dados do Usuário')
st.write(user_input_variables)

# standardize the new variables input by the user
user_input_variables_standard = scaler.transform(user_input_variables)

# Previsao
prediction = knn.predict(user_input_variables_standard)


# Acurácia do modelo 
#st.subheader('Acuracia do modelo')
st.write(accuracy_score(y_test, knn.predict(X_test))*100)


st.subheader('Previsão: ')
st.write(prediction)


