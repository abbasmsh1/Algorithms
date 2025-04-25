import numpy as np

def sigmoid(x):
        """
        Compute the sigmoid function with numerical stability.

        Args:
            x (numpy.ndarray or float): Input value(s).

        Returns:
            numpy.ndarray or float: Sigmoid of the input, in the range (0, 1).
        """
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))