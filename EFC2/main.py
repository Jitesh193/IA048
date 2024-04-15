""""
EFC 2 - Classificação
Aluno: Jitesh Ashok Manilal Vassaram
RA: 175867
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


train_x = pd.read_csv("X_train.txt", sep='\s+', header=None)

# print(train_x.head(20))

train_y = pd.read_csv("y_train.txt", sep='\s+', header=None)

# print(train_y.head(20))


X = train_x[train_x.columns[0:]].values

y = train_y.values
y=np.ravel(y)
print(y)

softReg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
softReg.fit(X,y)

print(f'O coeficiente linear eh: {softReg.intercept_}')
print(f'O coeficiente angular eh: {softReg.coef_}')
