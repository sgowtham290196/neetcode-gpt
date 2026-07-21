import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        output = {}
        # Layer 1
        z1 = np.dot(W1,x) + b1

        #RelU
        y_hat1 = np.maximum(0,z1)

        # Layer 2
        y_hat2 = np.dot(W2,y_hat1) + b2

        # Loss
        error = y_hat2 - y_true
        loss = np.mean(np.power(error,2))
        output["loss"] = np.round(loss,5)
        dpred2 = 2 * error/ len(error)

        #Backpropogration
        dW1 = np.zeros_like(W1)
        dW2 = np.outer(dpred2, y_hat1)
        db2 = dpred2

        n = len(z1)
        da1 = np.dot(np.array(W2).T, dpred2)
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x)
        db1 = dz1
        output["dW1"] = np.round(dW1,5)
        output["db1"] = np.round(db1,5)
        output["dW2"] = np.round(dW2,5)
        output["db2"] = np.round(db2,5)
        return output
