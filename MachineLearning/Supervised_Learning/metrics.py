import numpy as np
def accuracy(Y, Y_pred, threshold=0.5):
        """
        Compute the classification accuracy of the model.

        Accuracy is calculated as the proportion of correct predictions (matching true labels)
        to the total number of samples, using the specified threshold to convert probabilities
        to binary labels.

        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            Y (numpy.ndarray): True labels (0 or 1), shape (m,) or (m, 1).
            threshold (float): Threshold for converting probabilities to binary labels. Default is 0.5.

        Returns:
            float: Accuracy score in the range [0, 1].
        """
        Y = Y.reshape(-1, 1)  # Ensure Y is (m, 1) for comparison
        return np.mean(Y_pred == Y)
    
def precision(Y, Y_pred, threshold=0.5):
        """
        Compute the precision of the model.

        Precision is calculated as the proportion of true positive predictions to the total number
        of positive predictions (true positives + false positives), using the specified threshold
        to convert probabilities to binary labels.

        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            Y (numpy.ndarray): True labels (0 or 1), shape (m,) or (m, 1).
            threshold (float): Threshold for converting probabilities to binary labels. Default is 0.5.

        Returns:
            float: Precision score in the range [0, 1].
        """
        Y = Y.reshape(-1, 1)  # Ensure Y is (m, 1) for comparison
        true_positives = np.sum((Y_pred == 1) & (Y == 1))
        predicted_positives = np.sum(Y_pred == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
def MSE(Y,Y_pred):
        """
        Compute the Mean Squared Error (MSE) of the model.

        MSE is calculated as the average of the squared differences between true labels and predicted labels.

        Args:
            Y (numpy.ndarray): True labels, shape (m,) or (m, 1).
            Y_pred (numpy.ndarray): Predicted labels, shape (m,) or (m, 1).

        Returns:
            float: Mean Squared Error.
        """
        Y = Y.reshape(-1, 1)  # Ensure Y is (m, 1) for comparison
        return np.mean((Y - Y_pred) ** 2)
    