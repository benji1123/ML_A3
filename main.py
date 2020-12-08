"""
Vary hidden-layer neurons in [2, 3, 4].
Choose the best configuration with validation-set.
    For each config, take the average of 3 trials.

If 2 networks perform the same, choose the one with least neurons.
Evaluate the top model's test-error.
"""
from math import inf
import numpy as np
from Preprocessor import Preprocessor
from ThreeLayerNeuralNetwork import ThreeLayerNeuralNetwork

TRIALS = 3
EPOCHS = 10000


data = np.loadtxt(fname="data_banknote_authentication.txt", delimiter=',')
X_train, y_train, X_val, y_val, X_test, y_test = Preprocessor.split_data(data, train_split=0.5, val_split=0.25)

optimal_avg_error = inf
optimal_model = None

# find optimal network configuration
for hidden_layer1_size in [2, 3, 4]:
    for hidden_layer2_size in [2, 3, 4]:
        print(f"\nhl1={hidden_layer1_size} hl2={hidden_layer2_size}\n-------------")
        avg_error = 0
        for trial in range(1, TRIALS+1):
            # initial weights are different for each trial
            model = ThreeLayerNeuralNetwork(X_train, y_train, hidden_layer1_size, hidden_layer2_size)
            model.train(epochs=EPOCHS)
            trial_error = model.get_error(X=X_val, y=y_val)  # using validation data
            print(f"trial={trial} mse_error={trial_error}")
            avg_error += trial_error * (1 / TRIALS)
        print(f"avg error is {avg_error}")
        # update optimal model
        if avg_error < optimal_avg_error:
            optimal_avg_error = avg_error
            optimal_model = model
        # prefer smaller network if error is equal
        elif avg_error == optimal_avg_error:
            optimal_model = optimal_model if optimal_model.num_neurons < model.num_neurons else model

# print test_error of optimal model
test_error = optimal_model.get_error(X=X_test, y=y_test)
print(f"\n\nOptimal model\nhidden_layer1={optimal_model.hidden_layer1_size} "
      f"hidden_layer2={optimal_model.hidden_layer2_size}\n"
      f"test_error={test_error}")
