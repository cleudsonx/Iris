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
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, fbeta_score
import joblib

# Definindo os nomes das colunas
colunas = ['comprimento_sepala', 'largura_sepala', 'comprimento_petala', 'largura_petala', 'especie']

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

# Verificando a contagem de valores na coluna 'especie'
print(df['especie'].value_counts())

# Visualizando a distribuição das classes
sns.countplot(df['especie'])
plt.show()

# Visualizando a distribuição das features
sns.pairplot(df, hue='especie')
plt.show()

# Separando as features e a variável target
X = df.drop('especie', axis=1)
y = df['especie']

# Dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo com barra de progresso
model = RandomForestClassifier()
for _ in tqdm(range(1), desc="Treinando o modelo"):
    model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
print(classification_report(y_test, y_pred))

# Salvando o modelo
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
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# Visualizando a curva ROC
y_pred_prob = model.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}

for i, classe in enumerate(model.classes_):
    fpr[classe], tpr[classe], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=classe)
    roc_auc[classe] = roc_auc_score(y_test == classe, y_pred_prob[:, i])

# Plotando a curva ROC para cada classe
for classe in model.classes_:
    plt.plot(fpr[classe], tpr[classe], label=f'ROC curve (area = {roc_auc[classe]:.2f}) for class {classe}')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="best")
plt.show()

# Calculando a área sob a curva ROC para cada classe
for classe in model.classes_:
    print(f'ROC AUC score for class {classe}: {roc_auc[classe]:.2f}')

# Visualizando a curva de aprendizado
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

# Visualizando a curva de validação para max_depth
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
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob[:, 1], pos_label='Iris-virginica')
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()

# Calculando a área sob a curva de precisão-recall
print(auc(recall, precision))

# Visualizando a curva de threshold
plt.plot(thresholds, precision[1:], label='Precision')
plt.plot(thresholds, recall[1:], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold curve')
plt.legend()
plt.show()

# Visualizando a curva de F1
f1 = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)
plt.plot(thresholds, f1[1:], label='F1')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('F1 curve')
plt.legend()
plt.show()

# Visualizando a métrica F1
print(f1_score(y_test, y_pred, average='macro'))

# Visualizando a métrica F-beta
print(fbeta_score(y_test, y_pred, beta=0.5, average='macro'))

# Visualizando a métrica F-beta
print(fbeta_score(y_test, y_pred, beta=2, average='macro'))