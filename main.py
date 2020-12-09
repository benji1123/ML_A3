#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vary hidden-layer neurons in [2, 3, 4].
Choose the best configuration with validation-set.
    For each config, take the average of 3 trials.
If 2 networks perform the same, choose the one with least neurons.
Evaluate the top model's test-error.
"""
from math import inf
import numpy as np

import get_plots as helper
from Preprocessor import Preprocessor
from ThreeLayerNeuralNetwork import ThreeLayerNeuralNetwork

TRIALS = 3
EPOCHS = 10000
NUM_FEATURES_TO_TEST = [4, 3, 2]
HIDDEN_LAYER_SIZES_TO_TEST = [2, 3, 4]


data = np.loadtxt(fname="data_banknote_authentication.txt", delimiter=',')
X_train, y_train, X_val, y_val, X_test, y_test = Preprocessor.split_data(
    data, train_split=0.5, val_split=0.25)


# vary the number of features
for num_features in NUM_FEATURES_TO_TEST:
    # return the optimal model for the given data
    optimal_avg_error = inf
    optimal_model = None
    print(f"\n\n\nNUM FEATURES={num_features}")
    # experiment with hidden layer sizes
    for hidden_layer1_size in HIDDEN_LAYER_SIZES_TO_TEST:
        for hidden_layer2_size in HIDDEN_LAYER_SIZES_TO_TEST:
            print(f"\nhl1={hidden_layer1_size} hl2={hidden_layer2_size}\n-------------")
            avg_error_of_config = 0
            for trial in range(TRIALS):
                model = ThreeLayerNeuralNetwork(
                    X_train[:, :num_features],
                    y_train,
                    hidden_layer1_size,
                    hidden_layer2_size)
                model.train(epochs=EPOCHS)
                error = model.get_error(
                    X=X_val[:, :num_features],
                    y=y_val)
                print(f"trial={trial} mse_error={error}")
                avg_error_of_config += error * (1 / TRIALS)
            print(f"avg error is {avg_error_of_config}")
            # check if this model is better
            if avg_error_of_config < optimal_avg_error:
                optimal_avg_error = avg_error_of_config
                optimal_model = model
            # if error is equal, favour the smaller model
            elif avg_error_of_config == optimal_avg_error:
                optimal_model = optimal_model if optimal_model.num_neurons < model.num_neurons else model

    # print test_error of optimal model
    test_error = optimal_model.get_error(
        X=X_test[:, :num_features],
        y=y_test)
    misclassification_rate = helper.get_misclassification_rate(
        optimal_model, X_test[:, :num_features], y_test)
    print(f"\n\nOptimal model\nhidden_layer1={optimal_model.hidden_layer1_size} "
          f"hidden_layer2={optimal_model.hidden_layer2_size}\n"
          f"test_error={test_error} miss-rate={misclassification_rate}")

# Generate Plots
D = 2
_X_train = X_train[:, :D]
_X_val = X_val[:, :D]
_X_test = X_test[:, :D]
for hidden_layer1_size in HIDDEN_LAYER_SIZES_TO_TEST:
    for hidden_layer2_size in HIDDEN_LAYER_SIZES_TO_TEST:
        net = ThreeLayerNeuralNetwork(
            _X_train,
            y_train,
            hidden_layer1_size,
            hidden_layer2_size)
        helper.generate_error_plots(
            f"train & val error, misclassification vs epochs | h1={hidden_layer1_size} h2={hidden_layer2_size}",
            net,
            _X_train,
            y_train,
            _X_val,
            y_val,
            _X_test,
            y_test)
