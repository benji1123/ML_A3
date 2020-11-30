import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    @staticmethod
    def split_data(data, seed, train_split=0.5, val_split=0.25):
        np.random.seed(seed)
        np.random.shuffle(data)

        # split dataset: training, validation, test
        length = data.shape[0]
        train_data = data[:int(length * train_split)]
        val_data = data[int(length * train_split):int(length * train_split) + int(length * val_split)]
        test_data = data[int(length * train_split) + int(length * val_split):]

        # separate features and labels
        num_features = data.shape[1] - 1
        X_train, y_train = train_data[:, :num_features], train_data[:, num_features]
        X_val, y_val = val_data[:, :num_features], train_data[:, num_features]
        X_test, y_test = test_data[:, :num_features], test_data[:, num_features]
        X_train, X_val, X_test = Preprocessor.normalize(X_train, X_val, X_test)
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def normalize(X_train, X_val, X_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        X_test = sc.transform(X_test)
        return X_train, X_val, X_test
