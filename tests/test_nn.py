import neural_network as nn
import numpy as np

def test_MSE_loss():
    # given 
    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])

    # then 
    assert nn.MSE_loss(y_true, y_pred) == 0.5