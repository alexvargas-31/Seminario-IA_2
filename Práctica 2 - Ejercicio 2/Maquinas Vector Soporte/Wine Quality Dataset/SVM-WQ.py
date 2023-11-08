import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data_import=pd.read_csv('../../DataSets/winequality-red.csv')

scaler=StandardScaler()  # creating instance of StandardScaler
data_import.shape
data_import.head()
data_import.tail()
data_import.info()
data_import.describe()
data_import.isnull().sum()

corr = data_import[data_import.columns].corr()
sns.heatmap(corr, cmap="plasma", annot = True)
plt.title('Mapa de calor de los parámetros de correlación')
plt.show()

data_import.hist(figsize=(10,10),color='red')
plt.show()

sns.pairplot(data_import, hue = 'quality')
plt.show()