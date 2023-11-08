import pandas as pd
import matplotlib.pyplot as plt

file = '../../DataSets/insurance.csv'

df_raw=pd.read_csv(file,sep='delimiter', header=None,  engine='python')

# Obteniendo valores y renombrando las columnas.
df = df_raw.drop([0, 1, 2, 3], axis=0).reset_index(drop=True).rename(columns={0:'Reclamos'})
df = df.Reclamos.str.split(',',expand=True).rename(columns={0:'Reclamos', 1:'Pagos'})
df.head()

df.info()

# Convertir las columnas a valores numéricos.
df.Reclamos = pd.to_numeric(df.Reclamos, errors='coerce')
df.Pagos = pd.to_numeric(df.Pagos, errors='coerce')

df.hist()   # Histograma.

y1 = df.Reclamos
y2 = df.Pagos

# Gráfica Reclamos - Pagos.
plt.rcParams["figure.figsize"] = (20,3)

plt.plot(y1, linewidth = '2', marker = '*', color = 'red')
plt.plot(y2, linewidth = '2', marker = '*', color = 'red')

plt.xlabel("Reclamo")
plt.ylabel("Pago")

plt.grid()
plt.show()