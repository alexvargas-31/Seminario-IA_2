import numpy as npy
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

def dataSplit():
    """
    This method do train test split
    :return: trainX, testX, trainY, testY
    """

    splitValue = 0.8
    randomState = 1
    #split data
    dataForTrain = data.sample(frac=splitValue, random_state=randomState, replace=True)
    dfForTrain = pd.DataFrame(dataForTrain)
    trainX = dataForTrain[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol']].values
    # extracting quality labels
    trainY = dataForTrain['quality'].values


    # Select rest of data for testing
    drop_part = dfForTrain.index
    dataForTest = data.drop(drop_part)
    # dfForTest = pd.DataFrame(dataForTest)
    testX = dataForTest[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                           'pH', 'sulphates', 'alcohol']].values
    # extracting quality labels
    testY = dataForTest['quality'].values


    return trainX, testX, trainY, testY

# Cargar dataset.
readData = "../../DataSets/winequality-white.csv"
data = pd.read_csv(readData, sep=';')
dataFrame = pd.DataFrame(data)

# Gráficas.
sns.set()
fig = data.hist(figsize=(10,10), color='blue', xlabelsize=6, ylabelsize=6)
[x.title.set_size(8) for x in fig.ravel()]
plt.show()

# La variable objetivo fue actualizada después del cambio (3 a 6 no es buen vino, 6 a 8 es un buen vino).
data["quality"] = 1 * (data["quality"] >= 6)
qualityEqualsOne = data['quality'] == 1
qualityEqualsZero = data['quality'] == 0

# Matriz de correlación.
plt.figure(figsize=(9, 9))
correlation = data.corr()
heatmap = sns.heatmap(correlation, annot=True, cmap='viridis')
plt.show()

# Matriz de correlación usando scatterplot.
sm = scatter_matrix(dataFrame, figsize=(6, 6), diagonal='kde')
[s.xaxis.label.set_rotation(40) for s in sm.reshape(-1)]            # Cambiar la rotación de la etiqueta.
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]             # Cambiar la rotación de la etiqueta.
[s.get_yaxis().set_label_coords(-0.6,0.5) for s in sm.reshape(-1)]  # Offset para acomodar la figura.
[s.set_xticks(()) for s in sm.reshape(-1)]                          # Ocultar puntos.
[s.set_yticks(()) for s in sm.reshape(-1)]                          # Ocultar puntos.
plt.show()

trainDfColumnNumber = len(dataFrame.columns) - 1
weight = npy.array(npy.random.rand(trainDfColumnNumber))

# Entrenamiento de regresión logística.
logisticRegression = LogisticRegression(weight)