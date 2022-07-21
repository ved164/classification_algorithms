import numpy as np

class Adaline:
    """Adaptive Linear Neuron Classifier.

Parameters:
--------------
eta : float 
    Learning rate (between 0.0 and 10.0
n_iter : int
    passes over the training set
random_state : int
    random number generator seed for random weight initialization

Attributes:
---------------

w_ : 1d - array
    weights after fitting
b_ : Scalar
    bias unit after fitting
losses_ : list
    MSE loss function values in each epoch
    """

def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

def fit(self, x, y):
    """Fit training data.

    Parameters:
    -------------

    x : {array-like}, shape  [n_examples, n_features]
        Training vectors, where n_examples 
        is the number of examples and n_features is the number of features.

    y : array_like, shape = [n_examples]
        Target values
    
    
    Returns:
    -------------

    self : Object
    """
