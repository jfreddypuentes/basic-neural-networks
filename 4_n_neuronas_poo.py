'''
DEfinicion de capas y neuronas de forma dinamica a traves de POO.
Se definen 2 capas de neuroas.
Capa 1: 5 neuronas, 3 samples, cada sample con 4 features. Resultado: matriz con 5 features (columnas) y 3 filas (samples).
Capa 2: 2 neuronas, 3 samples, cada sample con 5 features. Resultado: Matriz con 2 features (columnas) y 3 filas (samples).
'''
import numpy as np 

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print('Result layer 1:')
print(layer1.output)

layer2.forward(layer1.output)
print('Result layer 2:')
print(layer2.output)