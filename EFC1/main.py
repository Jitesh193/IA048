""""
EFC 1 - Regressao Linear
Aluno: Jitesh Ashok Manilal Vassaram
RA: 175867
"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import funcoes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('../EFC1/air_traffic.csv')


df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))


df['Flt'] = df['Flt'].str.replace(',', '', regex=False).astype(float)

df = df[['Flt', 'date']]


# print(df.head(20))

# Questao 1
t1 = pd.Timestamp('2008-08-01')
t2 = pd.Timestamp('2019-12-01')
t3 = pd.Timestamp('2023-09-01')
#
# plt.figure(figsize=(10, 6))
# plt.plot(df['date'],df['Flt'], label='Numero total de voos')
# # ax = df['Flt'].plot()
# plt.axvline(t1, color='b', label='Jan/2003 - Ago/2008')
# plt.axvline(t2, color='r', label='Set/2008 - Dez/2019')
# plt.axvline(t3, color='g', label='Jan/2020 - Set/2023')
# plt.xlabel('Data')
# plt.ylabel('Numero de voos (em milhoes)')
# plt.legend()
# plt.grid(visible=True)
# plt.show()

#    Questao 2

" Treinamento - divisao do conjunto de validacao (validacao comeca a partir de Ago/2011) "

index1 = pd.Index(df.date).get_loc('2014-11-01')    # Limite entre dados de treinamento e dados de validacao
index2 = pd.Index(df.date).get_loc('2019-12-01')    # Limite dos dados de validacao e dados de teste
# df['time'] = (df.index.values + 0.5) / 12           # Conversao de escala de tempo, com ajuste de ponto medio,

train = df.iloc[0:index1+1].copy()
valid = df.iloc[index1+1:index2+1].copy()
test = df.iloc[index2+1:].copy()

print(train)
print('---------------------------------------')
print(valid)
print('---------------------------------------')
print(test)


"Exemplo K = 3"

y = list(train['Flt'])
y_valid = list(valid['Flt'])
num = len(train)
num_valid = len(valid)
rmse_val = []           # Lista com os valores RMSE da validacao para cada valor de k
# k = 3

for k in range(1, 25):

    train_x, train_y = funcoes.treino(y, num, k)

    # print(f'Os dados de treino x para k=3 sao {train_x}')
    # print(f'Os dados de treino y para k=3 sao {train_y}')

    model = LinearRegression().fit(train_x, train_y)
    # coeffs = [model.coef_, model.intercept_]

    # print(f'os coeficientes sao: {model.coef_}')
    # print(f'O coeficiente linear eh: {model.intercept_} \n')

    "Parte da validacao"
    # print('Dados de validacao:')
    # print(valid)

    # y_valid = list(valid['Flt'])

    # num_valid = len(valid)

    valid_x, valid_y = funcoes.validacao(y, y_valid, num_valid, k)

    # print(valid_x)
    # print(valid_y)

    # linear = model.predict(train_x)

    # print(f'A saida estimada eh de: {linear}')

    linear2 = model.predict(valid_x)

    MSE = np.square(np.subtract(y_valid, linear2)).mean()
    RMSE = math.sqrt(MSE)

    rmse_val.append(RMSE)

# plt.figure(figsize=(10,6))
# plt.plot(1:25)


# plt.figure(figsize=(10, 6))
# plt.scatter(train.date[1:], y[1:], s=20, c="blue", alpha=0.5, label="Training Flight Data")
# plt.plot(train.date[1:],y[1:])
# plt.plot(train.date[1:], linear, color="red", linewidth=2.5, label="Linear Fit")
# plt.scatter(valid.date, y_valid, s=20, c="green", alpha=0.5, label="validation Flight Data ")
# plt.plot(valid.date,y_valid)
# plt.plot(valid.date, linear2, color="black", linewidth=2.5, label="Linear validation")
# plt.xlabel('Time')
# plt.ylabel('Flight')
# plt.title('Linear Model Fit + validation')
# plt.legend()
# plt.grid(True)
# plt.show()
