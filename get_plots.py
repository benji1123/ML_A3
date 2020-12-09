#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:54:40 2020

@author: benjaminli
"""
import matplotlib.pyplot as plt
import numpy as np



def generate_error_plots(
        title, net, X_train, y_train, X_val, y_val, X_test, y_test, epochs=1000):
    train_errors, val_errors, misclassification_rates = get_data_per_epoch(
        net, X_train, y_train, X_val, y_val, X_test, y_test, epochs)
    plt.figure()
    plt.plot(train_errors)
    plt.plot(val_errors)
    plt.plot(misclassification_rates)
    plt.title(title)
    

def get_data_per_epoch(
        net, X_train, y_train, X_val, y_val, X_test, y_test, epochs=1000):
    train_errors = []
    val_errors = []
    misclassification_rates = []
    for i in range(epochs):
        # misclassification rate
        rate = get_misclassification_rate(net, X_test, y_test)
        misclassification_rates.append(rate)
        # validation error
        error = net.get_error(X_val, y_val)
        val_errors.append(error)
        # training error
        train_values = net.forward_pass(X_train.T)
        error = net.get_error(X_train, y_train)
        train_errors.append(error)
        # backpropogate
        grads = net.get_grads(train_values)
        net.update_weights(grads)
    return train_errors, val_errors, misclassification_rates


def get_misclassification_rate(net, X, y):
    forward_pass_vals = net.forward_pass(X.T)
    y_pred = forward_pass_vals["layer3_out"].T
    misclassifications = np.sum(np.rint(y_pred) != y)
    return misclassifications / y.shape[0]
    