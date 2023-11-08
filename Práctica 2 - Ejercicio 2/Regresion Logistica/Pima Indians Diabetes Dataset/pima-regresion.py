import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Cargando el dataset.
dataset = pd.read_csv('../../DataSets/diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values.astype(float)

# Asignando la información del dataset al set de entrenamiento y al set de prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Escalado de funciones.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Aplicar regresión logística al set de entrenamiento.
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Resultados de la predicción del set de prueba.
y_pred = classifier.predict(X_test)

# Matriz de confusión.
cm = confusion_matrix(y_test, y_pred)

# Exactitud.
true_pred = np.add(cm[0,0], cm[1,1])
false_pred = np.add(cm[0,1], cm[1,0])
accuracy_denominator = np.add(true_pred, false_pred)
accuracy_division = np.divide(true_pred, accuracy_denominator)
final_accuracy = np.multiply(accuracy_division, 100)
plt.rcParams["figure.figsize"] = (10, 10)

print("Accuracy in percentage is: ", final_accuracy, "%")

plt.plot(X, linewidth = '0', marker = '.')

plt.xlabel("insulina")
plt.ylabel("glucosa")

plt.grid()
plt.show()