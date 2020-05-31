import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv('../dados.csv')

# função para treinar o modelo
def train_model():
    df = get_data()
    X = df.drop(['MEDV'], axis=1)
    y = df.MEDV
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
    rf_regressor.fit(X,y)
    return rf_regressor

# criando um df
df = get_data()

# treinando o modelo
model = train_model()

# título
st.title("Data App - Prevendo valores de imóveis")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de ML para o prooblema de predição de valores de imóveis de Boston")

# verificando o dataset
st.subheader("Selecionando apenas um pequeno conjunto de atributos")

# atributos para serem exibidos por padrão
default_cols = ['RM', 'PTRATIO', 'LSTAT', 'MEDV']

# definindo atributos a partir do multiselect
cols = st.multiselect("Atributos", df.columns.tolist(), default=default_cols)

# exibindo os 10 primeiros registros do df
st.dataframe(df[cols].head(10))

st.subheader("Distribuição de imóveis por preço")

# definindo a faixa de valores
faixa_valores = st.slider("Faixa de preço", float(df.MEDV.min()), 150., (10.0, 100.0))

# filtrando os dados
dados = df[df.MEDV.between(left=faixa_valores[0], right=faixa_valores[1])]

# plot a distribuição dos dados
f = px.histogram(dados, x='MEDV', nbins=100, title='Distribuição de Preços')
f.update_xaxes(title='MEDV')
f.update_yaxes(title='Total Imóveis')
st.plotly_chart(f)

st.sidebar.subheader("Defina os atributos do imóvel para predição")

# mapeando dados do usuário para cada atributo
crim = st.sidebar.number_input("Taxa de criminalidade", value=df.CRIM.mean())
indus = st.sidebar.number_input("Proporção de hectares de negócio", value=df.INDUS.mean())
chas = st.sidebar.selectbox("Faz limite com o rio?", ('Sim', "Não"))

# transformando o dado de entrada em valor binario
chas = 1 if chas == 'Sim' else 0

nox = st.sidebar.number_input("Concentração de óxido nítrico", value=df.NOX.mean())

rm = st.sidebar.number_input("Número de quartos", value=1)

ptratio = st.sidebar.number_input("Índice de aluns para professores", value=df.PTRATIO.mean())

b = st.sidebar.number_input("Proporção de pessoas com descendência afro-americano", value=df.B.mean())

lstat = st.sidebar.number_input("Porcentagem de status baixo", value=df.LSTAT.mean())

btn_predict = st.sidebar.button('Realizar predição')

# verifica se o botão foi acionado
if btn_predict:
    result = model.predict([[crim, indus, chas, nox, rm, ptratio, b, lstat]])
    st.subheader("O valor previsto para o imóvel é: ")
    result = "US $ " + str(round(result[0]*10,2))
    st.write(result)





