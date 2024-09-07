import os
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo treinado
try:
    model = joblib.load('modelo.pkl')
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# Função para gerar o gráfico de importância das features
def plot_feature_importance(model):
    try:
        # Obter a importância das features do modelo
        importances = model.feature_importances_
        features = ['comprimento_sepala', 'largura_sepala', 'comprimento_petala', 'largura_petala']
        
        # Criar o gráfico de barras horizontal
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances, color='skyblue')
        plt.xlabel('Importância')
        plt.ylabel('Features')
        plt.title('Importância das Features')
        
        # Garantir que o diretório 'static' exista
        if not os.path.exists('static'):
            os.makedirs('static')
            print("Diretório 'static' criado.")
        
        # Salvar o gráfico no diretório 'static' com a extensão .jpeg
        plt.savefig('static/feature_importance.jpeg', format='jpeg')
        plt.close()
        print("Imagem feature_importance.jpeg gerada e salva no diretório 'static'.")
    except Exception as e:
        print(f"Erro ao gerar o gráfico de importância das features: {e}")

# Gerar o gráfico ao iniciar o servidor
plot_feature_importance(model)

# Rota para a página principal
@app.route('/')
def home():
    print("Rota '/' acessada.")
    return render_template('index.html')

# Rota para predições
@app.route('/predict', methods=['POST'])
def predict():
    print("Rota '/predict' acessada.")
    data = request.get_json(force=True)
    print(f"Dados recebidos para predição: {data}")
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    print(f"Predições geradas: {predictions.tolist()}")
    return jsonify(predictions.tolist())

# Rota para servir arquivos estáticos
@app.route('/static/<path:filename>')
def static_files(filename):
    print(f"Rota '/static/{filename}' acessada.")
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("Iniciando o servidor Flask...")
    app.run(debug=True)