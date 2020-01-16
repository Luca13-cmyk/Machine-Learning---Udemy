#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:33:19 2020

@author: luca13-cmyk
"""

import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')

# Lida com valores invalidos (negativos) e coloca a media em seu lugar 
base.loc[base.age < 0, 'age'] = 40.92

# Divide a base de dados em previsores e classe               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# Troca valores faltantes pela media
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# Faz o calculo de escalonamento dos dados e aplica
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Efetua a divisao da base de treinamento e a base de testes 
from sklearn.model_selection import train_test_split
(previsores_treinamento, 
 previsores_teste, 
 classe_treinamento, 
 classe_teste) = train_test_split(previsores, 
                                 classe, 
                                 test_size=0.25, 
                                 random_state=0)

# Calculo KNN
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, 
                                     metric='minkowski',
                                     p = 2)

# Efetuando o treinamento
classificador.fit(previsores_treinamento, classe_treinamento)
# Efetuando o teste
previsoes = classificador.predict(previsores_teste)

# Verificando a media de acertos
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

import collections
collections.Counter(classe_teste)