#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class for neural networks with 2 hidden layers.
References:
* https://towardsdatascience.com/an-introduction-to-neural-networks-with-implementation-from-scratch-using-python-da4b6a45c05b
"""

import numpy as np


class ThreeLayerNeuralNetwork:
    layers = 3
    X = None
    y = None
    num_features = None
    num_labels = None
    weights1 = None
    weights2 = None
    weights3 = None

    def __init__(self, X, y, hlayer1_num_neurons, hlayer2_num_neurons, lr=0.005):
        self.lr = lr
        self.X = X
        self.y = y
        self.num_features = X.shape[1]
        self.weights1 = np.random.randn(hlayer1_num_neurons, self.num_features)
        self.weights2 = np.random.randn(hlayer2_num_neurons, hlayer1_num_neurons)
        self.weights3 = np.random.randn(y.shape[1], hlayer2_num_neurons)
        # model configuration data
        self.hidden_layer1_size = hlayer1_num_neurons
        self.hidden_layer2_size = hlayer2_num_neurons
        self.num_neurons = hlayer1_num_neurons + hlayer2_num_neurons

    def train(self, epochs=1000):
        for i in range(epochs):
            forward_pass_vals = self.forward_pass(self.X.T)
            grads = self.get_grads(forward_pass_vals)
            self.update_weights(grads)

    def get_error(self, X, y):
        forward_pass_vals = self.forward_pass(X.T)
        predictions = forward_pass_vals["layer3_out"].T
        error = self.cost_function(y_pred=predictions, y=y)
        return error

    def forward_pass(self, X):
        vals = {}
        vals["layer1_z"] = np.dot(self.weights1, X)
        vals["layer1_out"] = self.ReLU(vals["layer1_z"])
        vals["layer2_z"] = np.dot(self.weights2, vals["layer1_out"])
        vals["layer2_out"] = self.ReLU(vals["layer2_z"])
        vals["layer3_z"] = np.dot(self.weights3, vals["layer2_out"])
        vals["layer3_out"] = self.ReLU(vals["layer3_z"])
        return vals

    @staticmethod
    def ReLU(z):
        z[z < 0] = 0
        return z

    def cost_function(self, y_pred, y):
        return 1 / (2 * len(y)) * np.sum(np.square(y_pred - y))

    def update_weights(self, grads):
        self.weights1 -= self.lr * grads["weights1"]
        self.weights2 -= self.lr * grads["weights2"]
        self.weights3 -= self.lr * grads["weights3"]

    def get_grads(self, forward_pass_vals):
        grads = {}
        m_inv = 1 / len(self.y)
        y = self.y.T
        X = self.X.T
        # layer 3
        dA = m_inv * (forward_pass_vals["layer3_out"] - y)
        dZ = dA
        grads["weights3"] = m_inv * np.dot(dZ, forward_pass_vals['layer2_out'].T)
        # layer 2
        dA = np.dot(self.weights3.T, dZ)
        dZ = np.multiply(dA, np.where(forward_pass_vals["layer2_out"] > 0, 1, 0))  # ReLU derivative
        grads["weights2"] = m_inv * np.dot(dZ, forward_pass_vals['layer1_out'].T)
        # layer 1
        dA = np.dot(self.weights2.T, dZ)
        dZ = np.multiply(dA, np.where(forward_pass_vals["layer1_out"] > 0, 1, 0))
        grads["weights1"] = m_inv * np.dot(dZ, X.T)
        return grads
