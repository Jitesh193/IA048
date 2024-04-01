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

df = pd.read_csv('../EFC1/air_traffic.csv')


df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))


df['Flt'] = df['Flt'].str.replace(',', '', regex=False).astype(float)

df = df[['Flt', 'date']]

df = df.set_index('date')

df = df.to_period()


print(df.head(20))

# Questao 1
t1 = pd.Timestamp('2008-08-01')
t2 = pd.Timestamp('2019-12-01')
t3 = pd.Timestamp('2023-09-01')

plt.figure(figsize=(10, 6))
# plt.plot(df['Flt'], label='Numero total de voos')
ax = df['Flt'].plot()
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

index1 = pd.Index(df.date).get_loc('2014-11-01')    # Limite entre dados de treinamento e dados de validacao
# index1 = pd.Index(df.date).get_loc('2005-06-01')    # Limite entre dados de treinamento e dados de validacao
index2 = pd.Index(df.date).get_loc('2019-12-01')    # Limite dos dados de validacao e dados de teste
df['time'] = (df.index.values + 0.5) / 12           # Conversao de escala de tempo, com ajuste de ponto medio,
# df['time'] = np.arange(len(df.index))           # Conversao de escala de tempo, com ajuste de ponto medio,


train = df.iloc[0:index1+1].copy()
valid = df.iloc[index1+1:index2+1].copy()
test = df.iloc[index2+1:].copy()

print(df.iloc[29])


print(train)
print('---------------------------------------')
print(valid)
print('---------------------------------------')
print(test)

# ind = (0.125*12)-0.5

"Exemplo K = 3"

# # # print(train[-3:])   # Slicing
# # # print(train.time[-2:])
# x = list(train['time'])
# y = list(train['Flt'])
# #
# # # print(x)
# #
# num = len(train)
# train_x = []
# train_y = []
# k = 3
#
#
# for i in range(num):
#
#     # if i == 100:
#     #     print('pare aqui')
#     if num-k-i-1 < 0 & num - i - 1 >= 0:
#         train_x.append([0]*(i+k+1-num) + x[0:num-i-1])
#     else:
#         train_x.append(x[num - k - i - 1:num - i-1])
#     train_x[i].reverse()
#     train_y.append(y[num-1-i])
# train_x.reverse()
# train_y.reverse()
# print(f'Os dados de treino x para k=10 sao {train_x}')
# print(f'Os dados de treino y para k=10 sao {train_y}')
#
# model = LinearRegression().fit(train_x,train_y)
# coeffs = [model.coef_, model.intercept_]
#
# # print(f"O modelo linear para K = 1 eh: F(x) = {coeffs[0]:.3f}*x + {coeffs[1]} ")
# print(f'os coeficientes sao: {model.coef_}')
# print(f'O coeficiente linear eh: {model.intercept_} \n')
#
#
# "Parte da validacao"
# print('Dados de validacao:')
# print(valid)
#
# x_valid = list(valid['time'])
# y_valid = list(valid['Flt'])
#
# num_valid = len(valid)
#
# # print(num_valid)
# # print(x[-3+0:])
# # print(x_valid[:1])
# # print(x[-3+1:] + x_valid[:1])
# # print(y_valid[1])
# valid_x = []
# valid_y = []
# # for i in range(num_valid):
# #
# #     # if i == 100:
# #     #     print('pare aqui')
# #     if num_valid-3-i < 0 & num_valid - i >= 0:
# #         valid_x.append(x[num_valid-3-i:] + x_valid[0:num_valid-i])
# #     else:
# #         valid_x.append(x_valid[num_valid - 3 - i:num_valid - i])
# #     valid_y.append(y_valid[num_valid-1-i])
# #
#
# for i in range(num_valid):
#
#     # if i == 100:
#     #     print('pare aqui')
#     if i < k:
#         valid_x.append(x[-k+i:] + x_valid[:i])
#     elif i == k:
#         valid_x.append(x_valid[:i])
#     else:
#         valid_x.append(x_valid[i-k:i])
#     valid_x[i].reverse()
#     valid_y.append(y_valid[i])
#
# print(valid_x)
# print(valid_y)
#
# # print(model.predict(np.array(train_x[3]).reshape(1,-1)))
#
# linear = []
# for j in range(num):
#     linear.append(model.predict(np.array(train_x[j]).reshape(1, -1)))
#
# print(linear)
# linear2 = []
# for i in range(num_valid):
#     linear2.append(model.predict(np.array(valid_x[i]).reshape(1, -1)))
#
# print(linear2)
# # print(len(coeffs[0]))
#
# #
# # # print('------------------------------------------------')
# # # print(coeffs[0])
# # # print(valid_x[60])
# # # y_pred = []
# # # y1 = np.matmul(coeffs[0],valid_x[60]) + coeffs[1]
# # # y_pred.append(y1)
# # # print(y_pred)
# # # print(linear2[60])
# # # for j in range(num_valid):
# # #     y_pred.append(np.matmul(coeffs[0], valid_x[j])+coeffs[1])
# #
# #
# plt.figure(figsize=(10, 6))
# plt.scatter(train.date, y, s=20, c="blue", alpha=0.5, label="Training Flight Data")
# plt.plot(train.date,y)
# plt.plot(train.date, linear, color="red", linewidth=2.5, label="Linear Fit")
# plt.scatter(valid.date, y_valid, s=20, c="green", alpha=0.5, label="validation Flight Data ")
# plt.plot(valid.date,y_valid)
# plt.plot(valid.date, linear2, color="black", linewidth=2.5, label="Linear validation")
# # plt.plot(valid.date, y_pred, '--',color="yellow", linewidth=2.5, label="Linear validation")
# plt.xlabel('Time')
# plt.ylabel('Flight')
# plt.title('Linear Model Fit + validation')
# plt.legend()
# plt.grid(True)
# plt.show()
