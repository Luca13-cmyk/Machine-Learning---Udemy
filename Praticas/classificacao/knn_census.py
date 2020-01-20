#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:46:00 2020

@author: luca13-cmyk
"""

import pandas as pd

# Leitura do arquivo
base = pd.read_csv('census.csv')

# Divisao da base de dados
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
# Pre-processamento de dados
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Efetuando o dummy dos previsores
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()

# Efetuando o label encoder (transformando valores categoricos em numeros)
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# Efetuando o escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Dividindo a base de dados entre treinamento e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

# Efetuando o algoritmo KNN
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5,
                                     metric='minkowski', p=2)

# Efetuando os testes
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Gerando a porcentagem de acertos
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

# Verificando o Base Line (ZeroF)
# Pega o maior numero e divide pela total
import collections
print(collections.Counter(classe_teste))