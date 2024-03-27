""""
EFC 1 - Regressao Linear
Aluno: Jitesh Ashok Manilal Vassaram
RA: 175867
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# df = pd.read_csv("D:/Usuario/Desktop/Backup/Jitesh/IA048/EFC1/air_traffic.csv")
df = pd.read_csv("D:/Jitesh/Unicamp/IA048/EFC1/air_traffic.csv")

df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))


df['Flt'] = df['Flt'].str.replace(',', '', regex=False).astype(float)

df = df[['Flt', 'date']]

# Questao 1
t1 = pd.Timestamp('2008-08-01')
t2 = pd.Timestamp('2019-12-01')
t3 = pd.Timestamp('2023-09-01')

plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['Flt'], label='Numero total de voos')
plt.axvline(t1, color='b', label='Jan/2003 - Ago/2008')
plt.axvline(t2, color='r', label='Set/2008 - Dez/2019')
plt.axvline(t3, color='g', label='Jan/2020 - Set/2023')
plt.xlabel('Data')
plt.ylabel('Numero de voos (em milhoes)')
plt.legend()
plt.grid(visible=True)
plt.show()

# Questao 2

" Treinamento - divisao do conjunto de validacao (validacao comeca a partir de Ago/2011) "

index1 = pd.Index(df.date).get_loc('2011-07-01')    # Limite entre dados de treinamento e dados de validacao
index2 = pd.Index(df.date).get_loc('2019-12-01')    # Limite dos dados de validacao e dados de teste
df['time'] = (df.index.values + 0.5) / 12           # Conversao de escala de tempo, com ajuste de ponto medio,

train = df.iloc[0:index1+1].copy()
valid = df.iloc[index1+1:index2+1].copy()
test = df.iloc[index2+1:].copy()

# print(train)
# print('---------------------------------------')
# print(valid)
# print('---------------------------------------')
# print(test)

# ind = (0.125*12)-0.5

"Exemplo K = 2"

print(train[-3:])   # Slicing
print(train.time[-2:])
x = list(train['time'])
y = list(train['Flt'])

# print(x)

num = len(train)
train_x = []
train_y = []
for i in range(num):

    # if i == 100:
    #     print('pare aqui')
    if num-3-i < 0 & num - i >= 0:
        train_x.append([0]*(i+3-num) + x[0:num-i])
    else:
        train_x.append(x[num - 3 - i:num - i])
    train_y.append(y[num-1-i])

print(f'Os dados de treino x para k=2 sao {train_x}')
print(f'Os dados de treino y para k=2 sao {train_y}')

model = LinearRegression().fit(train_x, train_y)
coeffs = [model.coef_[0], model.intercept_]

# print(f"O modelo linear para K = 1 eh: F(x) = {coeffs[0]:.3f}*x + {coeffs[1]} ")
print(f'os coeficientes sao: {model.coef_}')
print(f'O coeficiente linear eh: {model.intercept_} \n')


"Parte da validacao"
print('Dados de validacao:')
print(valid)

x_valid = list(valid['time'])
y_valid = list(valid['Flt'])

num_valid = len(valid)

valid_x = []
valid_y = []
for i in range(num_valid):

    # if i == 100:
    #     print('pare aqui')
    if num_valid-3-i < 0 & num_valid - i >= 0:
        valid_x.append(x[num_valid-3-i:] + x_valid[0:num_valid-i])
    else:
        valid_x.append(x_valid[num_valid - 3 - i:num_valid - i])
    valid_y.append(y_valid[num_valid-1-i])

print(valid_x)
# print(valid_y)


linear = model.predict(train_x)
linear2 = model.predict(valid_x)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=20, c="blue", alpha=0.5, label="Training Flight Data")
plt.plot(x, linear, color="red", linewidth=2.5, label="Linear Fit")
plt.scatter(x_valid, y_valid, s=20, c="green", alpha=0.5, label="validation Flight Data ")
plt.plot(x_valid, linear2, color="black", linewidth=2.5, label="Linear validation")
plt.xlabel('Time')
plt.ylabel('Flight')
plt.title('Linear Model Fit + validation')
plt.legend()
plt.grid(True)
plt.show()
