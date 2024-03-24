""""
EFC 1 - Regressao Linear
Aluno: Jitesh Ashok Manilal Vassaram
RA: 175867
"""

import pandas as pd
import matplotlib.pyplot as plt

# Dados do cabecalho:
# Ano e mes; Dom_pax (Passageiros domesticos); Int_pax (passageiros internacionais); Pax (passageiros totais);
# Dom_flt (voos domesticos); Int_flt (voos internacionais); Flt (voos totais); Dom_RPM (Revenue Passenger-miles
# (Domestic)); Int_RPM (Revenue Passenger-miles (International))

df = pd.read_csv("D:/Usuario/Desktop/Backup/Jitesh/IA048/EFC1/air_traffic.csv")

df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

total_flt = df['Flt']
voos = []

for i in range(0, len(total_flt)):
    voos.append(float(total_flt[i].replace(',', '.')))

t = df['date']

df_new = pd.DataFrame({'X': voos, 'Y': t})

print(df_new.head(10))
print('-----------------------------------------')
print(df_new.Y)

plt.figure(figsize=(10, 6))
plt.plot(t, voos, label='Numero total de voos')
plt.xlabel('Time')
plt.ylabel('Numero de voos (em mil)')
plt.legend()
plt.grid(True)
plt.show()

