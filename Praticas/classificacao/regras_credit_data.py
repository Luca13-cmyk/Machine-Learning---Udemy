#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:04:02 2020

@author: luca13-cmyk
"""

import Orange

base = Orange.data.Table('credit_data.csv')

# Verificando os campos dos dados
print(base.domain)

# Efutando a divisao de base de dados de treinamento e teste

base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

# Verificando o tamanho das bases
print(len(base_treinamento))
print(len(base_teste))

# Desenvolvendo as regras e efetuando o treinamento
cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base_treinamento)

for regras in classificador.rule_list:
    print(regras)
    

# Efetuando o teste de precisao
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento,
                                                     base_teste,
                                                     [classificador])
print(Orange.evaluation.CA(resultado))

