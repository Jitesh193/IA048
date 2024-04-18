""""
EFC 2 - Classificação
Aluno: Jitesh Ashok Manilal Vassaram
RA: 175867
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score,classification_report,recall_score
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


train_x = pd.read_csv("X_train.txt", sep='\s+', header=None)

# print(train_x.head(20))

train_y = pd.read_csv("y_train.txt", sep='\s+', header=None)

# print(train_y.head(20))

test_x = pd.read_csv("X_test.txt", sep='\s+', header=None)
test_y = pd.read_csv("y_test.txt", sep='\s+', header=None)


X = train_x[train_x.columns[0:]].values

y = train_y.values
y = np.ravel(y)


softReg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=250)
softReg.fit(X, y)

y_hat = softReg.predict(test_x)

C = confusion_matrix(test_y, y_hat)
confusionMatrix = pd.DataFrame(data=C, index=['1, true', '2, true', '3, true', '4, true', '5, true', '6, true'], columns=['1, predicted', '2, predicted', '3, predicted', '4, predicted', '5, predicted', '6, predicted'])
confusionMatrix.loc['sum'] = confusionMatrix.sum()
confusionMatrix['sum'] = confusionMatrix.sum(axis=1)
print(confusionMatrix)
print('--'*50)

# metrica global de avaliacao: Acuracia balanceada e recall

print(f'A acuracia balanceada foi de: {balanced_accuracy_score(test_y,y_hat)}')
print('--'*50)

recall = recall_score(test_y,y_hat,average=None)
bacc = np.sum(recall)/(len(recall))
print(f'O recall para o conjunto de teste é: \n {recall_score(test_y,y_hat,average=None)}')
# print(bacc)
