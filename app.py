from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Carregar o modelo treinado
model = joblib.load('modelo.pkl')

# Rota para a página principal
@app.route('/')
def home():
    return render_template('index.html')

# Rota para predições
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

# Rota para a importância das features
@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    importances = model.feature_importances_
    features = ['comprimento_sepala', 'largura_sepala', 'comprimento_petala', 'largura_petala']
    importance_data = [{'feature': feature, 'importance': importance} for feature, importance in zip(features, importances)]
    return jsonify(importance_data)

if __name__ == '__main__':
    app.run(debug=True)