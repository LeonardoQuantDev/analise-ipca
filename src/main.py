# Analisador e Previsor de Indicadores Econômicos
# Autor: [Seu Nome]
# Descrição: Coleta de dados, análise exploratória e previsão com Regressão Linear.

# -----------------------
# 📦 Importando bibliotecas
# -----------------------

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------
# 📥 Função de coleta de dados via SGS API
# -----------------------

def get_sgs_data(series_id, start_date='2010-01-01'):
    url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs. {series_id}/dados?formato=json&dataInicial={start_date}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df['valor'] = df['valor'].astype(float)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    return df

# -----------------------
# ✅ Coletando dados: IPCA - série 433
# -----------------------

ipca_df = get_sgs_data(series_id=433)

# -----------------------
# 📊 Análise exploratória
# -----------------------

plt.figure(figsize=(12,6))
sns.lineplot(x='data', y='valor', data=ipca_df)
plt.title('IPCA - Variação Mensal')
plt.xlabel('Data')
plt.ylabel('Valor (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

print("Estatísticas descritivas do IPCA:")
print(ipca_df['valor'].describe())

# -----------------------
# 🔢 Preparação dos dados para Machine Learning
# -----------------------

ipca_df['meses'] = (ipca_df['data'].dt.to_period('M') - ipca_df['data'].dt.to_period('M').min()).apply(lambda x: x.n)
X = ipca_df[['meses']]
y = ipca_df['valor']

# -----------------------
# 🤖 Treinando modelo de Regressão Linear
# -----------------------

model = LinearRegression()
model.fit(X, y)

# Previsões
y_pred = model.predict(X)

# -----------------------
# 📈 Visualizando a previsão
# -----------------------

plt.figure(figsize=(12,6))
plt.plot(ipca_df['data'], y, label='Valores Reais')
plt.plot(ipca_df['data'], y_pred, color='red', linestyle='--', label='Regressão Linear')
plt.title('Previsão de IPCA com Regressão Linear')
plt.xlabel('Data')
plt.ylabel('Valor (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------
# 📊 Avaliação do modelo
# -----------------------

mse = mean_squared_error(y, y_pred)
print(f'\nErro Quadrático Médio (MSE): {mse:.4f}')

print(f'Coeficiente Angular: {model.coef_[0]:.6f}')
print(f'Intercepto: {model.intercept_:.6f}')
