import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("../../Datasets/winequality-white.csv", sep=';')
print(data.head())
print(data.corr)
print(data.columns)
print(data.info())
print(data['quality'].unique())

print(Counter(data['quality']))

sns.countplot(x='quality', data=data)
plt.show()

print(data.describe())

reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
data['Reviews'] = reviews

print(data.columns)

print(data['Reviews'].unique())

print(Counter(data['Reviews']))

x = data.iloc[:,:11]
y = data['Reviews']

print(x.head(10))
print(y.head(10))

sc = StandardScaler()
x = sc.fit_transform(x)

print(x)

pca = PCA()
x_pca = pca.fit_transform(x)

plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.grid()
plt.show()

pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)

print(x_new)

x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

nb = GaussianNB()
nb.fit(x_train,y_train)
nb_predict=nb.predict(x_test)

nb_conf_matrix = confusion_matrix(y_test, nb_predict)
nb_acc_score = accuracy_score(y_test, nb_predict)
print(nb_conf_matrix)
print(nb_acc_score*100)

plt.plot(x_test, y_test)
plt.show()