import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

file = '../../DataSets/insurance.csv'

df_raw=pd.read_csv(file,sep='delimiter', header=None,  engine='python')

# Obteniendo valores y renombrando las columnas.
df = df_raw.drop([0, 1, 2, 3], axis=0).reset_index(drop=True).rename(columns={0:'Reclamos'})
df = df.Reclamos.str.split(',',expand=True).rename(columns={0:'Reclamos', 1:'Pagos'})

df.info()

x = df['Reclamos'].values
y = df['Pagos'].values

plt.scatter(x, y)
plt.xlabel('Number of claims', fontsize=20)
plt.ylabel('Total payment', fontsize=20)
plt.title('Scatter Plot', fontsize=25)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)

fig,ax=plt.subplots(figsize=(10, 10))
k_list=np.arange(1, 44, 1)
knn_dict={} # To store k and mse pairs

for i in k_list:
#Knn Model Creation
    knn = KNeighborsRegressor(n_neighbors=int(i))
    x_train = x_train.reshape(-1, 1)
    model_knn = knn.fit(x_train, y_train)
    x_test = x_test.reshape(-1, 1)

    y_knn_pred = model_knn.predict(x_test)

#Storing MSE
    mse=mean_squared_error(y_test, y_knn_pred)
    knn_dict[i]=mse

#Plotting the results
ax.plot(knn_dict.keys(),knn_dict.values())
ax.set_xlabel('K-VALUE', fontsize = 20)
ax.set_ylabel('MSE', fontsize = 20)
ax.set_title('ELBOW PLOT', fontsize = 28)
plt.show()

print(mean_squared_error(y_test,y_knn_pred))