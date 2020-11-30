"""
Data
    - 4 features
    - boolean labels (0, 1)
- split data:
    -test (25%)
    -validation (25%)
    -training (50%)
- standardize features

Neural Network
- lr = 0.005
- 2 hidden layers
    -alter num-units between: 2, 3, 4 for each layer
- ReLu activation function at hidden units
- train models with early stopping (100 epochs -- 1 epoch is one pass-through of training set)

"""
import numpy as np
from sklearn.preprocessing import StandardScaler

SEED = 878


data = np.loadtxt(fname="data_banknote_authentication.txt", delimiter=',')
np.random.seed(SEED)
np.random.shuffle(data)

# split dataset: training, validation, test
length = data.shape[0]
train_data = data[:length//2]
val_data = data[length//2:(length//2) + (length//2)//2]
test_data = data[(length//2) + (length//2)//2:]

# separate features and labels
X_train, y_train = train_data[:, :4], train_data[:, 4]
X_val, y_val = val_data[:, :4], train_data[:, 4]
X_test, y_test = test_data[:, :4], test_data[:, 4]

# normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.fit_transform(X_val)
X_test = sc.fit_transform(X_test)
