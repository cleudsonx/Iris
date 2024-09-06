# agente de IA para programação em python, analise de dados, ciencia de dados
# Autor: Cleudson Cavalcante
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

# Carregando o dataset
df = pd.read_csv('https://raw.githubusercontent.com/cleudsoncavalcante/IA/main/iris.csv')

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

# Verificando a distribuição das classes
print(df['species'].value_counts())

# Visualizando a distribuição das classes
sns.countplot(df['species'])
plt.show()

# Visualizando a distribuição das features
sns.pairplot(df, hue='species')
plt.show()

# Separando as features e a variável target
X = df.drop('species', axis=1)
y = df['species']

# Dividindo o dataset em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred))

# Salvando o modelo
import joblib
joblib.dump(model, 'modelo.pkl')

# Salvando o dataset de teste
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Carregando o modelo
model = joblib.load('modelo.pkl')

# Carregando o dataset de teste
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Visualizando a importância das features
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12,6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Visualizando a matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# Visualizando a curva ROC
from sklearn.metrics import roc_curve, roc_auc_score
y_pred_prob = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label='Iris-virginica')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Calculando a área sob a curva ROC
roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

# Visualizando a curva de aprendizado
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning curve')
plt.legend()
plt.show()

# Visualizando a curva de validação
from sklearn.model_selection import validation_curve
param_range = np.arange(1, 200, 2)
train_scores, test_scores = validation_curve(model, X, y, param_name='n_estimators', param_range=param_range, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, train_scores_mean, label='Training score')
plt.plot(param_range, test_scores_mean, label='Cross-validation score')
plt.xlabel('n_estimators')
plt.ylabel('Score')
plt.title('Validation curve')
plt.legend()
plt.show()

# Visualizando a curva de complexidade
from sklearn.model_selection import validation_curve
param_range = np.arange(1, 11)
train_scores, test_scores = validation_curve(model, X, y, param_name='max_depth', param_range=param_range, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, train_scores_mean, label='Training score')
plt.plot(param_range, test_scores_mean, label='Cross-validation score')
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.title('Complexity curve')
plt.legend()
plt.show()

# Visualizando a curva de precisão-recall
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob, pos_label='Iris-virginica')
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()

# Calculando a área sob a curva de precisão-recall
from sklearn.metrics import auc
auc(recall, precision)

# Visualizando a curva de threshold
plt.plot(thresholds, precision[1:], label='Precision')
plt.plot(thresholds, recall[1:], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold curve')
plt.legend()
plt.show()

# Visualizando a curva de F1
f1 = 2 * (precision * recall) / (precision + recall)
plt.plot(thresholds, f1[1:], label='F1')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('F1 curve')
plt.legend()
plt.show()

# Visualizando a curva de F1
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, pos_label='Iris-virginica')

# Visualizando a curva de F1
from sklearn.metrics import fbeta_score
fbeta_score(y_test, y_pred, beta=0.5, pos_label='Iris-virginica')

# Visualizando a curva de F1
from sklearn.metrics import fbeta_score
fbeta_score(y_test, y_pred, beta=2, pos_label='Iris-virginica')


