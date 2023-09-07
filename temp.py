# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

# Definición de la función de activación (función escalón)
def activation_function(x):
    return 1 if x >= 0 else 0

# Clase para el Perceptrón
class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)  # Inicializamos pesos de manera aleatoria
        self.bias = np.random.rand()
        
    def train(self, X, y, learning_rate, max_epochs, stop_criterion):
        epoch = 0
        while True:
            error_count = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += learning_rate * error * X[i]
                self.bias += learning_rate * error
                if error != 0:
                    error_count += 1
            epoch += 1
            if stop_criterion == 'epochs' and epoch >= max_epochs:
                break
            elif stop_criterion == 'error' and error_count == 0:
                break
    
    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return activation_function(weighted_sum)

# Función para cargar patrones de entrenamiento desde un archivo de texto
def load_training_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# Función para probar el perceptrón entrenado en datos reales
def test_perceptron(perceptron, test_data):
    for inputs in test_data:
        prediction = perceptron.predict(inputs)
        print(f"Entradas: {inputs}, Predicción: {prediction}")

# Lectura de patrones de entrenamiento desde un archivo de texto
training_file = 'training_data.txt'
X_train, y_train = load_training_data(training_file)

# Crear un perceptrón con la cantidad de entradas adecuada
num_inputs = X_train.shape[1]
perceptron = Perceptron(num_inputs)

# Selección de criterio de finalización del entrenamiento, número de épocas y tasa de aprendizaje
stop_criterion = input("Seleccione el criterio de finalización del entrenamiento ('epochs' o 'error'): ")
max_epochs = int(input("Ingrese el número máximo de épocas de entrenamiento: "))
learning_rate = float(input("Ingrese la tasa de aprendizaje: "))

# Entrenar el perceptrón
perceptron.train(X_train, y_train, learning_rate, max_epochs, stop_criterion)

# Prueba del perceptrón entrenado en datos reales
test_file = 'test_data.txt'
X_test, _ = load_training_data(test_file)
print("Resultados de prueba:")
test_perceptron(perceptron, X_test)

