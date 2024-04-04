""""
EFC 1 - Regressao Linear - item c
Aluno: Jitesh Ashok Manilal Vassaram
RA: 175867
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import funcoes
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


df = pd.read_csv('../EFC1/air_traffic.csv')


df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))


df['Flt'] = df['Flt'].str.replace(',', '', regex=False).astype(float)

df = df[['Flt', 'date']]

#    Questao 2

" Treinamento - divisao do conjunto de validacao (validacao comeca a partir de Ago/2011) "

index1 = pd.Index(df.date).get_loc('2019-12-01')    # Limite entre dados de treinamento e dados de validacao
index2 = pd.Index(df.date).get_loc('2021-12-01')    # Limite dos dados de validacao e dados de teste


train = df.iloc[0:index1+1].copy()
valid = df.iloc[index1+1:index2+1].copy()
test = df.iloc[index2+1:].copy()

print(train)
print('---------------------------------------')
print(valid)
print('---------------------------------------')
print(test)


# Fazendo treinamento dos dados novamente com o K otimo
train2 = df.iloc[0:index2+1].copy()
y2 = list(train2['Flt'])
num_test2 = len(train2)
k_otimo = rmse_val.index(min(rmse_val))+1

train2_x, train2_y = funcoes.treino(y2, num_test2, k_otimo)

model2 = LinearRegression().fit(train2_x, train2_y)

y_test = list(test['Flt'])
num_test = len(test)

test_x, test_y = funcoes.validacao(y_valid, y_test, num_test, k_otimo)

# print(f'A serie temporal de teste eh: {test_x}')

linear3 = model2.predict(test_x)

rmse_val2 = root_mean_squared_error(y_test, linear3)

print(f'O RMSE para o conjunto de Teste eh: {rmse_val2:.5f}')

mape_val = mean_absolute_percentage_error(y_test, linear3)
print(f'O MAPE para o conjunto de Teste eh: {mape_val:.5f}')


plt.figure(figsize=(10,6))
plt.plot(rmse_val)
plt.xlabel('Valores de k')
plt.ylabel('RMSE')
plt.grid(True)
plt.show()

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


plt.figure(figsize=(10, 6))
plt.scatter(test.date, y_test, s=20, c="green", alpha=0.5, label="Test Flight Data ")
plt.plot(test.date,y_test)
plt.plot(test.date, linear3, color="black", linewidth=2.5, label="Estimated Flights")
plt.xlabel('Time')
plt.ylabel('Flight')
plt.title('Linear Model test')
plt.legend()
plt.grid(True)
plt.show()

index4 = pd.Index(test.date).get_loc('2022-01-01')


rmse_test2 = root_mean_squared_error(y_test[index4:], linear3[index4:])
mape_val2 = mean_absolute_percentage_error(y_test[index4:], linear3[index4:])

print(f'O RMSE para o conjunto de Teste (2022-2023) eh: {rmse_test2:.5f}')
print(f'O MAPE para o conjunto de Teste (2022-2023) eh: {mape_val2:.5f}')


plt.figure(figsize=(10, 6))
plt.scatter(test.date.iloc[index4:], y_test[index4:], s=20, c="green", alpha=0.5, label="Test Flight Data (2022-2023)")
plt.plot(test.date.iloc[index4:], y_test[index4:])
plt.scatter(test.date.iloc[index4:], linear3[index4:], s=20, c="black", alpha=0.5, label="Estimated Flights (2022-2023)")
plt.plot(test.date.iloc[index4:], linear3[index4:], color="black", linewidth=2.5, label="Estimated Flights (2022-2023)")
plt.xlabel('Time')
plt.ylabel('Flight')
plt.title('Linear Model test')
plt.legend()
plt.grid(True)
plt.show()
