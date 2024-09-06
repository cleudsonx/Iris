import os
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo treinado
model = joblib.load('modelo.pkl')

# Função para gerar o gráfico de importância das features
def plot_feature_importance(model):
    importances = model.feature_importances_
    features = ['comprimento_sepala', 'largura_sepala', 'comprimento_petala', 'largura_petala']
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='skyblue')
    plt.xlabel('Importância')
    plt.ylabel('Features')
    plt.title('Importância das Features')
    
    # Garantir que o diretório 'static' exista
    if not os.path.exists('static'):
        os.makedirs('static')
    
    plt.savefig('static/feature_importance.png')
    plt.close()

# Gerar o gráfico ao iniciar o servidor
plot_feature_importance(model)

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

# Rota para servir a imagem de importância das features
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Rota para servir a imagem de importância das features
@app.route('/feature-importance')
def feature_importance():
    return send_from_directory('static', 'feature_importance.png')

if __name__ == '__main__':
    app.run(debug=True)