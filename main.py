"""
References:
* https://towardsdatascience.com/an-introduction-to-neural-networks-with-implementation-from-scratch-using-python-da4b6a45c05b
"""
import numpy as np

from Preprocessor import Preprocessor
from ThreeLayerNeuralNetwork import ThreeLayerNeuralNetwork

HL1_SIZE = 3
HL2_SIZE = 3


data = np.loadtxt(fname="data_banknote_authentication.txt", delimiter=',')
X_train, y_train, X_val, y_val, X_test, y_test = Preprocessor.split_data(data, seed=878, train_split=0.5, val_split=0.25)
net = ThreeLayerNeuralNetwork(X_train, y_train, hlayer1_num_neurons=HL1_SIZE, hlayer2_num_neurons=HL2_SIZE)
for i in range(10000):
    forward_pass_vals = net.forward_pass()
    grads = net.get_grads(forward_pass_vals)
    net.update_weights(grads=grads)
    if i % 1000 == 0:
        predictions = forward_pass_vals["layer3_out"].T
        cost = round(net.mse_cost_function(predictions), 3)
        print(f"iter-{i} mse: {cost}")
