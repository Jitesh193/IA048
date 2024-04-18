""""
EFC 2 - Classificação
Aluno: Jitesh Ashok Manilal Vassaram
RA: 175867
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score,recall_score
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

train_x = pd.read_csv("X_train.txt", sep='\s+', header=None)

# print(train_x.head(20))

train_y = pd.read_csv("y_train.txt", sep='\s+', header=None)

# print(train_y.head(20))

test_x = pd.read_csv("X_test.txt", sep='\s+', header=None)
test_y = pd.read_csv("y_test.txt", sep='\s+', header=None)

y_test = test_y.values
y_test = np.ravel(y_test)


X = train_x[train_x.columns[0:]].values

y = train_y.values
# y = np.ravel(y)

# print(X)
print('--'*50)
# print(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

y_train = np.ravel(y_train)
y_val = np.ravel(y_val)

# print(X_train)
# print(y_train)
# print(X_val)
# print(y_val)


# Validacao
max_iter = 25
Rec = np.zeros((max_iter, 6))
bacc = []

for k in range(1, max_iter+1):

    kNNReg = KNeighborsClassifier(n_neighbors=k, weights='distance')
    kNNReg.fit(X_train, y_train)

    y_hat = kNNReg.predict(X_val)

    # Metricas de Avaliação Global
    recall = recall_score(y_val,y_hat,average=None)
    Rec[k-1, :] = recall
    bA = np.sum(recall)/(len(recall))

    bacc.append(bA)

k_otimo = bacc.index(max(bacc))+1

print(f'Os valores das Acurácia Balanceada são: \n {bacc} \n')
print(f'O valor do k que gera o maior valor da Acurácia Balanceada é: {k_otimo}')
print('--'*50)
print(f'Os valores de recall para cada valor de k é: \n {Rec}')
print('--'*50)

# Re-treinamento com o k ótimo

y = np.ravel(y)

kNNReg = KNeighborsClassifier(n_neighbors=k_otimo)
kNNReg.fit(X, y)

y_hat2 = kNNReg.predict(test_x)

C = confusion_matrix(y_test, y_hat2)
confusionMatrix = pd.DataFrame(data=C, index=['1, true', '2, true', '3, true', '4, true', '5, true', '6, true'], columns=['1, predicted', '2, predicted', '3, predicted', '4, predicted', '5, predicted', '6, predicted'])
confusionMatrix.loc['sum'] = confusionMatrix.sum()
confusionMatrix['sum'] = confusionMatrix.sum(axis=1)
print(confusionMatrix)
print('--'*50)
print(f'A acuracia balanceada foi de: {balanced_accuracy_score(y_test,y_hat2)}')
print('--'*50)
print(f'O recall para o conjunto de teste é: \n {recall_score(y_test,y_hat2,average=None)}')
