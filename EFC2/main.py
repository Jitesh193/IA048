""""
EFC 2 - Classificação
Aluno: Jitesh Ashok Manilal Vassaram
RA: 175867
"""

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


df = pd.read_csv("X_train.txt", sep='\s+', header=None)

print(df)
