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
