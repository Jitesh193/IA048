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

# Carregamento Dados de treinamento
ACC_xTr = pd.read_csv("Inertial_Signals_train/body_acc_x_train.txt", sep='\s+', header=None)
ACC_yTr = pd.read_csv("Inertial_Signals_train/body_acc_y_train.txt", sep='\s+', header=None)
ACC_zTr = pd.read_csv("Inertial_Signals_train/body_acc_z_train.txt", sep='\s+', header=None)

GYR_xTr = pd.read_csv("Inertial_Signals_train/body_gyro_x_train.txt", sep='\s+', header=None)
GYR_yTr = pd.read_csv("Inertial_Signals_train/body_gyro_y_train.txt", sep='\s+', header=None)
GYR_zTr = pd.read_csv("Inertial_Signals_train/body_gyro_z_train.txt", sep='\s+', header=None)

# print(f'ACC X: \n {ACC_xTr}')
# print('--'*50)
# print(f'ACC Y: \n {ACC_yTr}')
# print('--'*50)
# print(f'ACC Z: \n {ACC_zTr}')
# print('--'*50)
# print(f'GYR X: \n {GYR_xTr}')
# print('--'*50)
# print(f'GYR Y: \n {GYR_yTr}')
# print('--'*50)
# print(f'GYR Z: \n {GYR_zTr}')
# print('--'*50)

ACC_xTr = ACC_xTr[ACC_xTr.columns[0:]].values
ACC_yTr = ACC_yTr[ACC_yTr.columns[0:]].values
ACC_zTr = ACC_zTr[ACC_zTr.columns[0:]].values

GYR_xTr = GYR_xTr[GYR_xTr.columns[0:]].values
GYR_yTr = GYR_yTr[GYR_yTr.columns[0:]].values
GYR_zTr = GYR_zTr[GYR_zTr.columns[0:]].values


x_train = np.concatenate((ACC_xTr, ACC_yTr, ACC_zTr, GYR_xTr, GYR_yTr, GYR_zTr), axis=1)
# print(x_train)
y_train = pd.read_csv("y_train.txt", sep='\s+', header=None)

# print(x_train)

# Carregamento Dados de teste
ACC_xTe = pd.read_csv("Inertial_Signals_test/body_acc_x_test.txt", sep='\s+', header=None)
ACC_yTe = pd.read_csv("Inertial_Signals_test/body_acc_y_test.txt", sep='\s+', header=None)
ACC_zTe = pd.read_csv("Inertial_Signals_test/body_acc_z_test.txt", sep='\s+', header=None)

GYR_xTe = pd.read_csv("Inertial_Signals_test/body_gyro_x_test.txt", sep='\s+', header=None)
GYR_yTe = pd.read_csv("Inertial_Signals_test/body_gyro_y_test.txt", sep='\s+', header=None)
GYR_zTe = pd.read_csv("Inertial_Signals_test/body_gyro_z_test.txt", sep='\s+', header=None)

# print(f'ACC X: \n {ACC_xTe}')
# print('--'*50)
# print(f'ACC Y: \n {ACC_yTe}')
# print('--'*50)
# print(f'ACC Z: \n {ACC_zTe}')
# print('--'*50)
# print(f'GYR X: \n {GYR_xTe}')
# print('--'*50)
# print(f'GYR Y: \n {GYR_yTe}')
# print('--'*50)
# print(f'GYR Z: \n {GYR_zTe}')
# print('--'*50)

ACC_xTe = ACC_xTe[ACC_xTe.columns[0:]].values
ACC_yTe = ACC_yTe[ACC_yTe.columns[0:]].values
ACC_zTe = ACC_zTe[ACC_zTe.columns[0:]].values

GYR_xTe = GYR_xTe[GYR_xTe.columns[0:]].values
GYR_yTe = GYR_yTe[GYR_yTe.columns[0:]].values
GYR_zTe = GYR_zTe[GYR_zTe.columns[0:]].values

x_test = np.concatenate((ACC_xTe, ACC_yTe, ACC_zTe, GYR_xTe, GYR_yTe, GYR_zTe), axis=1)
y_test = pd.read_csv("y_test.txt", sep='\s+', header=None)

# Separacao Conjunto de Treinamento e Validação
X, X_val, y, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

y = np.ravel(y)
y_val = np.ravel(y_val)


# Validacao:

max_iter = 25
Rec = np.zeros((max_iter, 6))
bacc = []

for k in range(1, max_iter+1):

    kNNReg = KNeighborsClassifier(n_neighbors=k, weights='distance')
    kNNReg.fit(X, y)

    y_hat = kNNReg.predict(X_val)

    # Metricas de Avaliação Global
    recall = recall_score(y_val, y_hat, average=None)
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

y2 = np.ravel(y_train)

kNNReg = KNeighborsClassifier(n_neighbors=k_otimo)
kNNReg.fit(x_train, y2)

y_hat2 = kNNReg.predict(x_test)

C = confusion_matrix(y_test, y_hat2)
confusionMatrix = pd.DataFrame(data=C, index=['1, true', '2, true', '3, true', '4, true', '5, true', '6, true'], columns=['1, predicted', '2, predicted', '3, predicted', '4, predicted', '5, predicted', '6, predicted'])
confusionMatrix.loc['sum'] = confusionMatrix.sum()
confusionMatrix['sum'] = confusionMatrix.sum(axis=1)
print(confusionMatrix)
print('--'*50)
print(f'A acuracia balanceada foi de: {balanced_accuracy_score(y_test,y_hat2)}')
print('--'*50)
print(f'O recall para o conjunto de teste é: \n {recall_score(y_test,y_hat2,average=None)}')
