import streamlit as st  # Importar a biblioteca Streamlit
import joblib           # Importar a biblioteca Joblib
import pandas as pd     # Importar a biblioteca Pandas

# Carregar o modelo treinado
try:
    model = joblib.load('modelo.pkl')
    st.write("Modelo carregado com sucesso.")
except Exception as e:
    st.write(f"Erro ao carregar o modelo: {e}")

# Lista global para armazenar as últimas cinco predições
if 'ultimas_predicoes' not in st.session_state:
    st.session_state.ultimas_predicoes = []

# Função para fazer predições
def fazer_predicao(dados):
    df = pd.DataFrame([dados])
    predicao = model.predict(df)[0]
    return predicao

# Interface do usuário
st.title("Predição com Modelo de IA")
st.write("Este modelo foi treinado para prever a espécie de uma flor Iris com base nas medidas de suas sépalas e pétalas.")

# Formulário para entrada de dados
comprimento_sepala = st.number_input("Comprimento da Sépala", min_value=0.0, step=0.0)
largura_sepala = st.number_input("Largura da Sépala", min_value=0.0, step=0.1)
comprimento_petala = st.number_input("Comprimento da Pétala", min_value=0.0, step=0.1)
largura_petala = st.number_input("Largura da Pétala", min_value=0.0, step=0.1)

if st.button("Fazer Predição"):
    dados = {
        'comprimento_sepala': comprimento_sepala,
        'largura_sepala': largura_sepala,
        'comprimento_petala': comprimento_petala,
        'largura_petala': largura_petala
    }
    predicao = fazer_predicao(dados)
    st.write(f"Predição Atual: {predicao}")

    # Selecionar a imagem correspondente à predição
    if predicao == 'Iris-setosa':
        image_url = 'static/iris_setosa.jpeg'
    elif predicao == 'Iris-versicolor':
        image_url = 'static/iris_versicolor.jpeg'
    elif predicao == 'Iris-virginica':
        image_url = 'static/iris_virginica.jpeg'
    else:
        image_url = None

    if image_url:
        st.image(image_url, caption=f"Imagem da espécie prevista: {predicao}")

    # Armazenar as últimas cinco predições
    st.session_state.ultimas_predicoes.append({'predicao': predicao, 'image_url': image_url})
    if len(st.session_state.ultimas_predicoes) > 5:
        st.session_state.ultimas_predicoes.pop(0)

# Exibir as últimas cinco predições na ordem da mais atual para a mais antiga
st.write("Últimas 5 Predições:")
cols = st.columns(5)
for i, pred in enumerate(reversed(st.session_state.ultimas_predicoes)):
    with cols[i]:
        st.image(pred['image_url'], caption=f"{pred['predicao']}", width=100)
        