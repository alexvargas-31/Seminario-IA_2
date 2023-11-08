import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

file = "../../DataSets/insurance.csv"

df_raw = pd.read_csv(file,sep='delimiter', header=None,  engine='python')

# Obteniendo valores y renombrando las columnas.
df = df_raw.drop([0, 1, 2, 3], axis=0).reset_index(drop=True).rename(columns={0:'Reclamos'})
df = df.Reclamos.str.split(',',expand=True).rename(columns={0:'Reclamos', 1:'Pagos'})

x = df['Reclamos'].values
y = df['Pagos'].values

print(df) 

plt.scatter(x, y, color="blue")
plt.show()

model = GaussianNB()

for i in range(len(y)):
    if y[i] < 120:
        plt.scatter(x[i], y[i], color="blue")
    else:
        plt.scatter(x[i], y[i], color="red")
plt.show()