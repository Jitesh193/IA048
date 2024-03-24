""""
EFC 1 - Regressao Linear
Aluno: Jitesh Ashok Manilal Vassaram
RA: 175867
"""

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("D:/Usuario/Desktop/Backup/Jitesh/IA048/EFC1/air_traffic.csv")

print(df.head())

# Dados do cabecalho:
# Ano e mes; Dom_pax (Passageiros domesticos); Int_pax (passageiros internacionais); Pax (passageiros totais);
# Dom_flt (voos domesticos); Int_flt (voos internacionais); Flt (voos totais); Dom_RPM (Revenue Passenger-miles
# (Domestic)); Int_RPM (Revenue Passenger-miles (International))

df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

# print(df.head())
print(type(df.loc[0, 'Flt']))
total_flt = df['Flt']
voos = []

for i in range(0, len(total_flt)):
    voos.append(float(total_flt[i].replace(',', '.')))

print(type(voos))
print(type(voos[0]))

t = df['date']
print(t)

plt.figure(figsize=(10, 6))
plt.plot(t, voos, label='Numero total de voos')
plt.xlabel('Time')
plt.ylabel('Numero de voos (em mil)')
plt.legend()
plt.grid(True)
plt.show()

