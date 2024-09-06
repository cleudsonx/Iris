# Data: 6-09-2024
# Versão: 1.0
# Descrição: agente de IA para programação em python, analise de dados, ciencia de dados
# Categoria: IA
# Tags: IA, Python, Analise de Dados, Ciencia de Dados
# URL:
# Salvar como: agente1.py
# ------------------------------------------------------------------------

# Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Definindo os nomes das colunas
colunas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Carregando o dataset com os nomes das colunas
df = pd.read_csv('https://github.com/cleudsonx/Iris/blob/master/iris_super.csv?raw=true', names=colunas)

# Visualizando as primeiras linhas do dataset
print(df.head())

# Verificando a quantidade de linhas e colunas
print(df.shape)

# Verificando os tipos de dados
print(df.dtypes)

# Verificando a presença de valores nulos
print(df.isnull().sum())

# Estatísticas descritivas
print(df.describe())

# Verificando a contagem de valores na coluna 'species'
print(df['species'].value_counts())