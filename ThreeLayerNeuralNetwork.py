import numpy as np


class ThreeLayerNeuralNetwork:
    X = None
    y = None
    num_features = None
    num_labels = None
    weights1 = None
    weights2 = None
    weights3 = None
    bias1 = None
    bias2 = None
    bias3 = None

    def __init__(self, X, y, hlayer1_num_neurons, hlayer2_num_neurons):
        self.X = X
        self.num_features = X.shape[1]
        self.output_space_size = 2
        self.weights1 = np.random.rand(hlayer1_num_neurons, self.num_features)
        self.weights2 = np.random.rand(hlayer2_num_neurons, hlayer1_num_neurons)
        self.weights3 = np.random.rand(self.output_space_size, hlayer2_num_neurons)

    def feedforward(self):
        hlayer1_output = np.dot(self.X, self.weights1.T)
        hlayer1_output = self.ReLU(hlayer1_output)
        hlayer2_output = np.dot(hlayer1_output, self.weights2.T)
        hlayer2_output = self.ReLU(hlayer2_output)
        output = np.dot(hlayer2_output, self.weights3.T)
        output = self.ReLU(output)
        return output

    @staticmethod
    def ReLU(z):
        z[z < 0] = 0  # numpy fancy-indexing
        return z
