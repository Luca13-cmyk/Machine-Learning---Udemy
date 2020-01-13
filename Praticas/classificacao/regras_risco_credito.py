#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:34:57 2020

@author: luca13-cmyk
"""

import Orange

# Carregando os dados com Orange
base = Orange.data.Table('risco_credito.csv')

# Verificando os campos dos dados
print(base.domain)

# Usando o algoritmo para inducao de regras
cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base)

for regras in classificador.rule_list:
    print(regras)

# Efetuando a classificacao por regras
resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'], 
                           ['ruim', 'alta', 'adequada', '0_15']])

# Resultado
for i in resultado:
    print(base.domain.class_var.values[i])

#baixo
#alto