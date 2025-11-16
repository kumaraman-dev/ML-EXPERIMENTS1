"""
Experiment 9 – Neural Network From Scratch
Clean Python File for Overleaf Submission
Author: (Your Name)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random

# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# ----------------------------------------------------
# Forward Propagation
# ----------------------------------------------------

def forward_propagation(X, parameters, activation="relu"):
    caches = {}
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        Z = np.dot(W, A) + b
        A = relu(Z)
        caches["Z" + str(l)] = Z
        caches["A" + str(l - 1)] = A

    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    ZL = np.dot(WL, A) + bL
    AL = sigmoid(ZL)

    caches["Z" + str(L)] = ZL
    caches["A" + str(L - 1)] = A

    return AL, caches


# ----------------------------------------------------
# Backward Propagation
# ----------------------------------------------------

def backward_propagation(AL, Y, parameters, caches):
    grads = {}
    L = len(parameters) // 2
    m = Y.shape[1]

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    ZL = caches["Z" + str(L)]
    dZL = dAL * sigmoid_derivative(sigmoid(ZL))
    grads["dW" + str(L)] = (1 / m) * np.dot(dZL, caches["A" + str(L - 1)].T)
    grads["db" + str(L)] = (1 / m) * np.sum(dZL, axis=1, keepdims=True)

    dA_prev = np.dot(parameters["W" + str(L)].T, dZL)

    for l in reversed(range(1, L)):
        Z = caches["Z" + str(l)]
        dZ = dA_prev * relu_derivative(Z)

        grads["dW" + str(l)] = (1 / m) * np.dot(dZ, caches["A" + str(l - 1)].T)
        grads["db" + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = np.dot(parameters["W" + str(l)].T, dZ)

    return grads


# ----------------------------------------------------
# Update Parameters
# ----------------------------------------------------

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters


# ----------------------------------------------------
# Model Training
# ----------------------------------------------------

def model(X, Y, layer_dims, learning_rate=0.01, epochs=1000):
    parameters = initialize_parameters(layer_dims)
    losses = []

    for i in range(epochs):
        AL, caches = forward_propagation(X, parameters)
        loss = -(1/Y.shape[1]) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        grads = backward_propagation(AL, Y, parameters, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            print(f"Epoch {i} — Loss: {loss:.4f}")
            losses.append(loss)

    return parameters, losses


# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    return (AL > 0.5).astype(int)


# ----------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------

def main():
    print("Experiment 9 – Neural Network From Scratch")
    print("Paste your dataset loading and preprocessing code here.")
    print("Then call: model(X_train, y_train, layer_dims=[n_x, h1, h2, ..., n_y])")


if __name__ == "__main__":
    main()
