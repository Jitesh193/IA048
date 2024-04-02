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
df['time'] = (df.index.values + 0.5) / 12           # Conversao de escala de tempo, com ajuste de ponto medio,
# df['time'] = np.arange(len(df.index))           # Conversao de escala de tempo, com ajuste de ponto medio,


train = df.iloc[0:index1+1].copy()
valid = df.iloc[index1+1:index2+1].copy()
test = df.iloc[index2+1:].copy()


train['lag'] = train['Flt'].shift(1)

print(train)
X = train.loc[:,['lag']]
X.fillna(0,inplace=True)

print(X)

y = train.loc[:, 'Flt']

print(f'valor de y antes: {y}')

# y, X = y.align(X, join='inner')

print(X)
print(y)

# print(df.iloc[29])


print(train)
print('---------------------------------------')


model = LinearRegression().fit(X,y)

print(f'os coeficientes sao: {model.coef_}')
print(f'O coeficiente linear eh: {model.intercept_} \n')

linear = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(train.date, y, s=20, c="blue", alpha=0.5, label="Training Flight Data")
plt.plot(train.date,y)
plt.plot(train.date, linear, color="red", linewidth=2.5, label="Linear Fit")
plt.xlabel('Time')
plt.ylabel('Flight')
plt.title('Linear Model Fit + validation')
plt.legend()
plt.grid(True)
plt.show()


"Cemiterio do codigo main.py"

# train['lag'] = train['Flt'].shift(3)
#
# print(train)
# X = train.loc[:,['lag']]
# X.fillna(0,inplace=True)
#
# print(X)

# print(df.iloc[29])


# A = np.matmul(np.array(train_x).T,np.array(train_x))
# B = np.linalg.inv(A)
# C = np.matmul(B,np.array(train_x).T)
# W = np.matmul(C,np.array(train_y).T)
#
# print(W)

# coeffs = []
# coeffs.append(W[1:])
# print(coeffs)
# linear = []
#
# # np.matmul(np.array(coeffs).T,np.array(train_x)) + W[0]
# for j in range(num):
#     linear.append(np.matmul(np.array(W).T, train_x[j]))



# print(num_valid)
# print(x[-3+0:])
# print(x_valid[:1])
# print(x[-3+1:] + x_valid[:1])
# print(y_valid[1])

# for i in range(num_valid):
#
#     # if i == 100:
#     #     print('pare aqui')
#     if num_valid-3-i < 0 & num_valid - i >= 0:
#         valid_x.append(x[num_valid-3-i:] + x_valid[0:num_valid-i])
#     else:
#         valid_x.append(x_valid[num_valid - 3 - i:num_valid - i])
#     valid_y.append(y_valid[num_valid-1-i])
#

# print(model.predict(np.array(train_x[3]).reshape(1,-1)))
#
# linear = model.predict(np.array(train_x))

# # print(linear2)
# # print(len(coeffs[0]))
#
#
# # print('------------------------------------------------')
# # print(coeffs[0])
# # print(valid_x[60])
# # y_pred = []
# # y1 = np.matmul(coeffs[0],valid_x[60]) + coeffs[1]
# # y_pred.append(y1)
# # print(y_pred)
# # print(linear2[60])
# # for j in range(num_valid):
# #     y_pred.append(np.matmul(coeffs[0], valid_x[j])+coeffs[1])
#
#

# linear = []
# for j in range(num-1):
#     linear.append(model.predict(np.array(train_x[j]).reshape(1, -1)))


# linear2 = []
# for i in range(num_valid):
#     linear2.append(model.predict(np.array(valid_x[i]).reshape(1, -1)))


# treino = pd.DataFrame()
# treino['tempos'] = train_x
# treino['voo'] = train_y
#
# print(treino)


# train_x = []
# train_y = []
# for i in range(num):
#
#     # if i == 100:
#     #     print('pare aqui')
#     if num-k-i-1 < 0 & num - i - 1 >= 0:
#         train_x.append([0]*(i+k+1-num) + y[0:num-i-1])
#     else:
#         train_x.append(y[num - k - i - 1:num - i-1])
#     train_x[i].reverse()
#     train_y.append(y[num-1-i])
#
#
# train_x.reverse()
# train_y.reverse()


# valid_x = []
# valid_y = []

# for i in range(num_valid):
#
#     # if i == 100:
#     #     print('pare aqui')
#     if i < k:
#         valid_x.append(y[-k+i:] + y_valid[:i])
#     elif i == k:
#         valid_x.append(y_valid[:i])
#     else:
#         valid_x.append(y_valid[i-k:i])
#     valid_x[i].reverse()
#     valid_y.append(y_valid[i])
