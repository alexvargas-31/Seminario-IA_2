import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

file = '../../DataSets/insurance.csv'

df_raw=pd.read_csv(file,sep='delimiter', header=None,  engine='python')

# Obteniendo valores y renombrando las columnas.
dataset = df_raw.drop([0, 1, 2, 3], axis=0).reset_index(drop=True).rename(columns={0:'Reclamos'})
dataset = dataset.Reclamos.str.split(',',expand=True).rename(columns={0:'Reclamos', 1:'Pagos'})

X_l = dataset.iloc[:, 0:-1].values
y_p = dataset.iloc[:, -1].values

y_p = y_p.reshape(-1,1)

StdS_X = StandardScaler()
StdS_y = StandardScaler()
X_l = StdS_X.fit_transform(X_l)
y_p = StdS_y.fit_transform(y_p)

plt.scatter(X_l, y_p, color = 'green')
plt.title('Gráfico Scatter')
plt.xlabel('Número de reclamos')
plt.ylabel('Pagos')
plt.show()

regressor = SVR(kernel = 'rbf')
regressor.fit(X_l, y_p)

A=regressor.predict(StdS_X.transform([[6.5]]))
A = A.reshape(-1,1)
print(A)

A_pred = StdS_y.inverse_transform(A)
print(A_pred)

B_pred = StdS_y.inverse_transform(regressor.predict(StdS_X.transform([[6.5]])).reshape(-1,1))
print(B_pred)

plt.scatter(StdS_X.inverse_transform(X_l), StdS_y.inverse_transform(y_p), color = 'green')
plt.plot(StdS_X.inverse_transform(X_l), StdS_y.inverse_transform(regressor.predict(X_l).reshape(-1,1)), color = 'red')
plt.title('Modelo: Máquinas Vector Soporte')
plt.xlabel('Número de reclamos')
plt.ylabel('Pagos')
plt.show()