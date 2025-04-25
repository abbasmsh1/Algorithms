import numpy as np
from preprocessing import *
from utils import *


class LinearRegression:
    """
    A simple implementation of Linear Regression using gradient descent.

    This class fits a linear model to predict continuous outputs by minimizing the mean squared error
    (MSE) between predicted and actual values. It supports training with customizable learning rate and
    number of epochs. If enabled, features are normalized to [0, 1] using the `normalize_data` function
    from the preprocessing module, with min and max values stored during training for consistent
    prediction.

    Attributes:
        W (numpy.ndarray): Weights of the linear model, initialized during fit.
        b (numpy.ndarray): Bias term of the linear model, initialized during fit.
        min_val (numpy.ndarray): Minimum values of training features for normalization, computed during fit if normalize=True.
        max_val (numpy.ndarray): Maximum values of training features for normalization, computed during fit if normalize=True.
    """
    def __init__(self, normalize=True):
        """
        Initialize the LinearRegression model.

        Args:
            normalize (bool): If True, apply min-max normalization using `normalize_data`. Default is True.
        """
        self.W = None
        self.b = None
        self.min_val = None
        self.max_val = None
        self.normalize = normalize

    def fit(self, X, Y, lr=0.01, epochs=1000, verbose=100):
        """
        Train the linear regression model using gradient descent.

        Args:
            X (numpy.ndarray): Training features, shape (m, n) where m is samples, n is features.
            Y (numpy.ndarray): Target values, shape (m,) or (m, 1).
            lr (float): Learning rate for gradient descent. Default is 0.01.
            epochs (int): Number of training iterations. Default is 1000.
            verbose (int): Print loss every `verbose` epochs; if None, no printing. Default is 100.
        """
        # Validate inputs
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1, 1)  # Ensure Y is (m, 1)
        m, n = X.shape
        if m != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")

        # Normalize features
        if self.normalize:
            X_norm, self.min_val, self.max_val = normalize_data(X)
        else:
            X_norm = X

        # Initialize weights and bias
        self.W = np.random.randn(n, 1) * 0.01  # Small random weights
        self.b = np.zeros((1, 1))

        # Gradient descent
        for i in range(epochs):
            # Predictions
            Y_pred = np.dot(X_norm, self.W) + self.b

            # Loss (MSE)
            loss = np.mean((Y - Y_pred) ** 2)

            # Gradients
            error = Y - Y_pred
            dW = -2 * np.dot(X_norm.T, error) / m
            db = -2 * np.mean(error)

            # Update weights and bias
            self.W -= lr * dW
            self.b -= lr * db

            # Print loss if verbose
            if verbose and i % verbose == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Predict continuous values for input data.

        Args:
            X (numpy.ndarray): Input features, shape (m, n).

        Returns:
            numpy.ndarray: Predicted values, shape (m, 1).
        """
        X = np.asarray(X)
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Model must be fitted with normalize=True to predict with normalization.")
            X_norm = normalize_data(X, min_val=self.min_val, max_val=self.max_val)
        else:
            X_norm = X
        return np.dot(X_norm, self.W) + self.b

    def fit_predict(self, X, Y, lr=0.01, epochs=1000, verbose=100):
        """
        Fit the model and return predictions on the training data.

        Args:
            X (numpy.ndarray): Training features, shape (m, n).
            Y (numpy.ndarray): Target values, shape (m,) or (m, 1).
            lr (float): Learning rate for gradient descent. Default is 0.01.
            epochs (int): Number of training iterations. Default is 1000.
            verbose (int): Print loss every `verbose` epochs; if None, no printing. Default is 100.

        Returns:
            numpy.ndarray: Predicted values, shape (m, 1).
        """
        self.fit(X, Y, lr, epochs, verbose)
        return self.predict(X)

    def score(self, X, Y):
        """
        Compute the R-squared score for the model.

        The R-squared score measures the proportion of variance in the target explained by the model.
        It ranges from -inf to 1, where 1 indicates perfect prediction.

        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            Y (numpy.ndarray): True target values, shape (m,) or (m, 1).

        Returns:
            float: R-squared score.
        """
        Y = np.asarray(Y).reshape(-1, 1)
        Y_pred = self.predict(X)
        ss_total = np.sum((Y - np.mean(Y)) ** 2)
        ss_residual = np.sum((Y - Y_pred) ** 2)
        return 1 - ss_residual / ss_total if ss_total != 0 else 0
    
    
    
class LogisticRegression:
    """
    A simple implementation of Logistic Regression using gradient descent for binary classification.

    This class fits a logistic model to predict binary outcomes (0 or 1) by minimizing binary cross-entropy
    loss. It uses an external sigmoid function to produce probabilities and supports training with
    customizable learning rate and number of epochs. If enabled, features are scaled to [0, 1] using the
    `normalize_data` function, with min and max values stored during training for consistent prediction.

    Attributes:
        W (numpy.ndarray): Weights of the logistic model, initialized during fit.
        b (numpy.ndarray): Bias term of the logistic model, initialized during fit.
        min_val (numpy.ndarray): Minimum values of training features for normalization, computed during fit if normalize=True.
        max_val (numpy.ndarray): Maximum values of training features for normalization, computed during fit if normalize=True.
    """
    def __init__(self, normalize=True):
        """
        Initialize the LogisticRegression model.

        Args:
            normalize (bool): If True, apply min-max normalization using `normalize_data`. Default is True.
        """
        self.W = None
        self.b = None
        self.min_val = None
        self.max_val = None
        self.normalize = normalize

    def fit(self, X, Y, lr=0.01, epochs=1000, verbose=100):
        """
        Train the logistic regression model using gradient descent.

        Args:
            X (numpy.ndarray): Training features, shape (m, n) where m is samples, n is features.
            Y (numpy.ndarray): Target values (0 or 1), shape (m,) or (m, 1).
            lr (float): Learning rate for gradient descent. Default is 0.01.
            epochs (int): Number of training iterations. Default is 1000.
            verbose (int): Print loss every `verbose` epochs; if None, no printing. Default is 100.
        """
        # Validate inputs
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1, 1)  # Ensure Y is (m, 1)
        m, n = X.shape
        if m != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")
        if not np.all(np.isin(Y, [0, 1])):
            raise ValueError("Target values must be 0 or 1 for binary classification.")

        # Normalize features
        if self.normalize:
            X_norm, self.min_val, self.max_val = normalize_data(X)
        else:
            X_norm = X

        # Initialize weights and bias
        self.W = np.random.randn(n, 1) * 0.01  # Small random weights
        self.b = np.zeros((1, 1))

        # Gradient descent
        for i in range(epochs):
            # Forward pass
            Z = np.dot(X_norm, self.W) + self.b
            Y_pred = sigmoid(Z)

            # Binary cross-entropy loss
            epsilon = 1e-15  # Prevent log(0)
            loss = -np.mean(Y * np.log(Y_pred + epsilon) + (1 - Y) * np.log(1 - Y_pred + epsilon))

            # Gradients
            error = Y_pred - Y
            dW = np.dot(X_norm.T, error) / m
            db = np.mean(error)

            # Update weights and bias
            self.W -= lr * dW
            self.b -= lr * db

            # Print loss if verbose
            if verbose and i % verbose == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels or probabilities for input data.

        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            threshold (float): Threshold for binary classification. If None, return probabilities.

        Returns:
            numpy.ndarray: Predicted probabilities or binary labels (0 or 1) based on threshold.
        """
        X = np.asarray(X)
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Model must be fitted with normalize=True to predict with normalization.")
            X_norm = normalize_data(X, min_val=self.min_val, max_val=self.max_val)
        else:
            X_norm = X
        Z = np.dot(X_norm, self.W) + self.b
        probs = sigmoid(Z)
        if threshold is not None:
            return (probs >= threshold).astype(int)
        return probs

    def fit_predict(self, X, Y, lr=0.01, epochs=1000, threshold=0.5, verbose=100):
        """
        Fit the model and return predictions on the training data.

        Args:
            X (numpy.ndarray): Training features, shape (m, n).
            Y (numpy.ndarray): Target values (0 or 1), shape (m,) or (m, 1).
            lr (float): Learning rate for gradient descent. Default is 0.01.
            epochs (int): Number of training iterations. Default is 1000.
            threshold (float): Threshold for binary classification. Default is 0.5.
            verbose (int): Print loss every `verbose` epochs; if None, no printing. Default is 100.

        Returns:
            numpy.ndarray: Predicted binary labels (0 or 1) or probabilities.
        """
        self.fit(X, Y, lr, epochs, verbose)
        return self.predict(X, threshold)


class SVM:
    """
    A simple implementation of Support Vector Machine (SVM) for binary classification using gradient descent.
    
    This class implements a linear SVM that optimizes the hinge loss with L2 regularization. It supports
    training with customizable learning rate, regularization parameter, and number of epochs. If enabled,
    features are normalized to [0, 1] using the `normalize_data` function, with min and max values stored
    during training for consistent prediction.
    
    Attributes:
        W (numpy.ndarray): Weights of the SVM model, initialized during fit.
        b (float): Bias term of the SVM model, initialized during fit.
        min_val (numpy.ndarray): Minimum values of training features for normalization, computed during fit if normalize=True.
        max_val (numpy.ndarray): Maximum values of training features for normalization, computed during fit if normalize=True.
        C (float): Regularization parameter that controls the trade-off between maximizing the margin and minimizing the hinge loss.
    """
    
    def __init__(self, C=1.0, normalize=True):
        """
        Initialize the SVM model.
        
        Args:
            C (float): Regularization parameter. Default is 1.0. Higher values penalize misclassification more.
            normalize (bool): If True, apply min-max normalization using `normalize_data`. Default is True.
        """
        self.W = None
        self.b = None
        self.min_val = None
        self.max_val = None
        self.normalize = normalize
        self.C = C  # Regularization parameter
    
    def fit(self, X, Y, lr=0.01, epochs=1000, verbose=100):
        """
        Train the SVM model using gradient descent.
        
        Args:
            X (numpy.ndarray): Training features, shape (m, n) where m is samples, n is features.
            Y (numpy.ndarray): Target values (-1 or 1), shape (m,) or (m, 1).
            lr (float): Learning rate for gradient descent. Default is 0.01.
            epochs (int): Number of training iterations. Default is 1000.
            verbose (int): Print loss every `verbose` epochs; if None, no printing. Default is 100.
        """
        # Validate inputs
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1, 1)  # Ensure Y is (m, 1)
        m, n = X.shape
        if m != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")
        
        # Convert 0/1 labels to -1/1 if needed
        if np.all(np.isin(Y, [0, 1])):
            Y = 2 * Y - 1  # Convert 0/1 to -1/1
        
        if not np.all(np.isin(Y, [-1, 1])):
            raise ValueError("Target values must be -1 or 1 (or 0 or 1, which will be converted) for SVM classification.")
        
        # Normalize features
        if self.normalize:
            X_norm, self.min_val, self.max_val = normalize_data(X)
        else:
            X_norm = X
        
        # Initialize weights and bias
        self.W = np.zeros((n, 1))  # Initialize weights to zero
        self.b = 0.0
        
        # Gradient descent
        for i in range(epochs):
            # Compute margin (decision function)
            margins = Y * (np.dot(X_norm, self.W) + self.b)
            
            # Compute hinge loss
            hinge_loss = np.maximum(0, 1 - margins)
            loss = np.mean(hinge_loss) + 0.5 * (1/self.C) * np.sum(self.W**2)  # L2 regularization
            
            # Compute gradients
            dW = np.zeros((n, 1))
            db = 0
            
            # Gradients for misclassified points (those with margin < 1)
            misclassified = (margins < 1).flatten()
            if np.any(misclassified):
                X_mis = X_norm[misclassified]
                Y_mis = Y[misclassified]
                
                # Gradient of the hinge loss
                dW += -(1/self.C) * self.W + (1/m) * np.dot(X_mis.T, Y_mis)
                db += (1/m) * np.sum(Y_mis)
            else:
                # Only regularization gradient if all points correctly classified
                dW += -(1/self.C) * self.W
            
            # Update weights and bias
            self.W += lr * dW
            self.b += lr * db
            
            # Print loss if verbose
            if verbose and i % verbose == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        Predict binary class labels for input data.
        
        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            
        Returns:
            numpy.ndarray: Predicted binary labels (-1 or 1), shape (m, 1).
        """
        X = np.asarray(X)
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Model must be fitted with normalize=True to predict with normalization.")
            X_norm = normalize_data(X, min_val=self.min_val, max_val=self.max_val)
        else:
            X_norm = X
        
        # Decision function: positive side of hyperplane is class 1, negative is class -1
        decision = np.dot(X_norm, self.W) + self.b
        return np.sign(decision)
    
    def predict_proba(self, X):
        """
        Compute distance from the decision boundary as a proxy for probability.
        
        This is not a true probability, but rather a distance-based score that can be
        used for relative confidence. Higher absolute values indicate higher confidence.
        
        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            
        Returns:
            numpy.ndarray: Decision function values, shape (m, 1).
        """
        X = np.asarray(X)
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Model must be fitted with normalize=True to predict with normalization.")
            X_norm = normalize_data(X, min_val=self.min_val, max_val=self.max_val)
        else:
            X_norm = X
        
        # Return raw decision function values
        return np.dot(X_norm, self.W) + self.b
    
    def fit_predict(self, X, Y, lr=0.01, epochs=1000, verbose=100):
        """
        Fit the model and return predictions on the training data.
        
        Args:
            X (numpy.ndarray): Training features, shape (m, n).
            Y (numpy.ndarray): Target values (-1 or 1), shape (m,) or (m, 1).
            lr (float): Learning rate for gradient descent. Default is 0.01.
            epochs (int): Number of training iterations. Default is 1000.
            verbose (int): Print loss every `verbose` epochs; if None, no printing. Default is 100.
            
        Returns:
            numpy.ndarray: Predicted binary labels (-1 or 1), shape (m, 1).
        """
        self.fit(X, Y, lr, epochs, verbose)
        return self.predict(X)
    
    def score(self, X, Y):
        """
        Compute the accuracy score for the model.
        
        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            Y (numpy.ndarray): True target values (-1 or 1), shape (m,) or (m, 1).
            
        Returns:
            float: Accuracy score (proportion of correctly classified instances).
        """
        Y = np.asarray(Y).reshape(-1, 1)
        
        # Convert 0/1 labels to -1/1 if needed
        if np.all(np.isin(Y, [0, 1])):
            Y = 2 * Y - 1  # Convert 0/1 to -1/1
            
        Y_pred = self.predict(X)
        return np.mean(Y_pred == Y)


class NaiveBayes:
    """
    A simple implementation of Naive Bayes classifier for both binary and multi-class classification.
    
    This class implements the Gaussian Naive Bayes algorithm, which assumes features follow a normal
    distribution. It is suitable for continuous data and works well even with small training datasets.
    The class computes the mean and variance of each feature for each class during training, then uses
    these to calculate class conditional probabilities during prediction.
    
    Attributes:
        classes (numpy.ndarray): Unique class labels identified during training.
        class_priors (dict): Prior probabilities for each class.
        means (dict): Mean of each feature for each class.
        variances (dict): Variance of each feature for each class.
        epsilon (float): Small value added to variances to prevent division by zero.
    """
    
    def __init__(self, epsilon=1e-9):
        """
        Initialize the Naive Bayes classifier.
        
        Args:
            epsilon (float): Small value to add to variances to prevent division by zero. Default is 1e-9.
        """
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.variances = {}
        self.epsilon = epsilon  # To prevent division by zero in variance
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        
        Args:
            X (numpy.ndarray): Training features, shape (m, n) where m is samples, n is features.
            y (numpy.ndarray): Target class labels, shape (m,).
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()  # Ensure y is flattened
        
        m, n = X.shape
        
        # Get unique classes
        self.classes = np.unique(y)
        
        # Calculate class priors and feature statistics for each class
        for c in self.classes:
            # Get samples of this class
            X_c = X[y == c]
            
            # Prior probability P(y)
            self.class_priors[c] = X_c.shape[0] / m
            
            # Calculate mean and variance for each feature
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + self.epsilon  # Add epsilon to prevent division by zero
    
    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate the Gaussian likelihood P(x|y) for a feature.
        
        Args:
            x (float): Feature value.
            mean (float): Mean of the feature for a class.
            var (float): Variance of the feature for a class.
            
        Returns:
            float: Gaussian likelihood probability.
        """
        exponent = np.exp(-0.5 * ((x - mean) ** 2) / var)
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    
    def _calculate_class_probability(self, x, c):
        """
        Calculate the class conditional probability P(x|y) for all features.
        
        Args:
            x (numpy.ndarray): Input sample, shape (n,).
            c: Class label.
            
        Returns:
            float: Log probability of the sample belonging to the class.
        """
        # Using log probabilities to avoid numerical underflow
        log_prob = np.log(self.class_priors[c])
        
        for i in range(len(x)):
            likelihood = self._calculate_likelihood(x[i], self.means[c][i], self.variances[c][i])
            # Add epsilon to prevent log(0)
            log_prob += np.log(likelihood + self.epsilon)
            
        return log_prob
    
    def predict(self, X):
        """
        Predict class labels for input data.
        
        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            
        Returns:
            numpy.ndarray: Predicted class labels, shape (m,).
        """
        X = np.asarray(X)
        m = X.shape[0]
        predictions = np.zeros(m)
        
        for i in range(m):
            # Calculate probability for each class
            class_probs = {c: self._calculate_class_probability(X[i], c) for c in self.classes}
            
            # Choose class with highest probability
            predictions[i] = max(class_probs.items(), key=lambda x: x[1])[0]
            
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for input data.
        
        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            
        Returns:
            numpy.ndarray: Predicted probabilities for each class, shape (m, len(classes)).
        """
        X = np.asarray(X)
        m = X.shape[0]
        probs = np.zeros((m, len(self.classes)))
        
        for i in range(m):
            # Calculate log probability for each class
            log_probs = {c: self._calculate_class_probability(X[i], c) for c in self.classes}
            
            # Convert log probabilities to actual probabilities
            # First, shift log probs to avoid numerical issues
            log_prob_values = np.array(list(log_probs.values()))
            max_log_prob = np.max(log_prob_values)
            exp_probs = np.exp(log_prob_values - max_log_prob)
            
            # Normalize to get probabilities that sum to 1
            norm_probs = exp_probs / np.sum(exp_probs)
            
            # Assign probabilities to output array
            for j, c in enumerate(self.classes):
                probs[i, j] = norm_probs[j]
                
        return probs
    
    def fit_predict(self, X, y):
        """
        Fit the model and return predictions on the training data.
        
        Args:
            X (numpy.ndarray): Training features, shape (m, n).
            y (numpy.ndarray): Target class labels, shape (m,).
            
        Returns:
            numpy.ndarray: Predicted class labels, shape (m,).
        """
        self.fit(X, y)
        return self.predict(X)
    
    def score(self, X, y):
        """
        Compute the accuracy score for the model.
        
        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            y (numpy.ndarray): True target class labels, shape (m,).
            
        Returns:
            float: Accuracy score (proportion of correctly classified instances).
        """
        y = np.asarray(y).flatten()
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MultinomialNaiveBayes:
    """
    A Multinomial Naive Bayes classifier primarily for text classification and discrete count data.
    
    This implementation is suitable for classification with discrete features (e.g., word counts for text
    classification). It computes the conditional probability for each class based on the frequency of each
    feature. A smoothing parameter (alpha) is used to handle the zero-frequency problem.
    
    Attributes:
        classes (numpy.ndarray): Unique class labels identified during training.
        class_priors (dict): Prior probabilities for each class.
        feature_prob (dict): Conditional probability of each feature given each class.
        alpha (float): Smoothing parameter for handling the zero-frequency problem.
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize the Multinomial Naive Bayes classifier.
        
        Args:
            alpha (float): Smoothing parameter (Laplace/Lidstone smoothing). Default is 1.0.
        """
        self.classes = None
        self.class_priors = {}
        self.feature_prob = {}
        self.alpha = alpha
    
    def fit(self, X, y):
        """
        Train the Multinomial Naive Bayes classifier.
        
        Args:
            X (numpy.ndarray): Training features, shape (m, n) where m is samples, n is features.
                Features should be non-negative counts or frequencies.
            y (numpy.ndarray): Target class labels, shape (m,).
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        if np.any(X < 0):
            raise ValueError("Input X must contain non-negative values only.")
        
        m, n = X.shape
        
        # Get unique classes
        self.classes = np.unique(y)
        
        # Calculate class priors and feature conditional probabilities
        for c in self.classes:
            # Get samples of this class
            X_c = X[y == c]
            
            # Prior probability P(y)
            self.class_priors[c] = X_c.shape[0] / m
            
            # Feature counts for this class (sum each feature across all samples)
            feature_counts = np.sum(X_c, axis=0)
            
            # Calculate conditional probability with smoothing
            # P(feature|class) = (count + alpha) / (total_count + alpha*n_features)
            total_count = np.sum(feature_counts)
            self.feature_prob[c] = (feature_counts + self.alpha) / (total_count + self.alpha * n)
    
    def predict(self, X):
        """
        Predict class labels for input data.
        
        Args:
            X (numpy.ndarray): Input features, shape (m, n). Features should be non-negative counts or frequencies.
            
        Returns:
            numpy.ndarray: Predicted class labels, shape (m,).
        """
        X = np.asarray(X)
        
        if np.any(X < 0):
            raise ValueError("Input X must contain non-negative values only.")
        
        m = X.shape[0]
        predictions = np.zeros(m)
        
        for i in range(m):
            # Calculate log probability for each class
            log_probs = {}
            for c in self.classes:
                # Start with log prior
                log_probs[c] = np.log(self.class_priors[c])
                
                # Add log conditional probabilities for each feature
                for j in range(len(X[i])):
                    if X[i, j] > 0:  # Only consider non-zero features
                        log_probs[c] += X[i, j] * np.log(self.feature_prob[c][j])
            
            # Choose class with highest probability
            predictions[i] = max(log_probs.items(), key=lambda x: x[1])[0]
            
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for input data.
        
        Args:
            X (numpy.ndarray): Input features, shape (m, n). Features should be non-negative counts or frequencies.
            
        Returns:
            numpy.ndarray: Predicted probabilities for each class, shape (m, len(classes)).
        """
        X = np.asarray(X)
        
        if np.any(X < 0):
            raise ValueError("Input X must contain non-negative values only.")
        
        m = X.shape[0]
        probs = np.zeros((m, len(self.classes)))
        
        for i in range(m):
            # Calculate log probability for each class
            log_probs = {}
            for c in self.classes:
                # Start with log prior
                log_probs[c] = np.log(self.class_priors[c])
                
                # Add log conditional probabilities for each feature
                for j in range(len(X[i])):
                    if X[i, j] > 0:  # Only consider non-zero features
                        log_probs[c] += X[i, j] * np.log(self.feature_prob[c][j])
            
            # Convert log probabilities to actual probabilities
            log_prob_values = np.array(list(log_probs.values()))
            max_log_prob = np.max(log_prob_values)
            exp_probs = np.exp(log_prob_values - max_log_prob)
            
            # Normalize to get probabilities that sum to 1
            norm_probs = exp_probs / np.sum(exp_probs)
            
            # Assign probabilities to output array
            for j, c in enumerate(self.classes):
                probs[i, j] = norm_probs[j]
                
        return probs
    
    def fit_predict(self, X, y):
        """
        Fit the model and return predictions on the training data.
        
        Args:
            X (numpy.ndarray): Training features, shape (m, n).
            y (numpy.ndarray): Target class labels, shape (m,).
            
        Returns:
            numpy.ndarray: Predicted class labels, shape (m,).
        """
        self.fit(X, y)
        return self.predict(X)
    
    def score(self, X, y):
        """
        Compute the accuracy score for the model.
        
        Args:
            X (numpy.ndarray): Input features, shape (m, n).
            y (numpy.ndarray): True target class labels, shape (m,).
            
        Returns:
            float: Accuracy score (proportion of correctly classified instances).
        """
        y = np.asarray(y).flatten()
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class KNN:
    """
    A k-Nearest Neighbors classifier implementation.
    
    This class implements both classification and regression using the k-nearest neighbors 
    algorithm. For classification, the class label is determined by majority vote of the 
    k nearest neighbors. For regression, the target value is the average of the k nearest
    neighbors. The algorithm uses Euclidean distance by default, but can be configured to
    use other distance metrics.
    
    Attributes:
        k (int): Number of neighbors to use.
        X_train (numpy.ndarray): Training feature data, stored during fit.
        y_train (numpy.ndarray): Training target data, stored during fit.
        distance_metric (str): Distance metric to use, one of 'euclidean', 'manhattan', or 'minkowski'.
        p (float): Power parameter for Minkowski distance. Only used when distance_metric='minkowski'.
        weights (str): Weight function used in prediction. One of 'uniform' or 'distance'.
    """
    
    def __init__(self, k=5, distance_metric='euclidean', p=2, weights='uniform'):
        """
        Initialize the KNN classifier/regressor.
        
        Args:
            k (int): Number of neighbors to use. Default is 5.
            distance_metric (str): Distance metric to use, one of 'euclidean', 'manhattan', or 'minkowski'.
                Default is 'euclidean'.
            p (float): Power parameter for Minkowski distance. Only used when distance_metric='minkowski'.
                Default is 2 (equivalent to Euclidean distance).
            weights (str): Weight function used in prediction. One of 'uniform' (all points weighted equally)
                or 'distance' (points weighted by inverse of distance). Default is 'uniform'.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.distance_metric = distance_metric
        self.p = p
        self.weights = weights
        
    def fit(self, X, y):
        """
        Fit the k-nearest neighbors classifier/regressor.
        
        This method simply stores the training data for later use during prediction.
        No actual model is trained since KNN is a lazy learning algorithm.
        
        Args:
            X (numpy.ndarray): Training feature data, shape (n_samples, n_features).
            y (numpy.ndarray): Training target data, shape (n_samples,) or (n_samples, n_targets).
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        
        # Ensure y is 2D if it's a single column (for multi-output support)
        if len(self.y_train.shape) == 1:
            self.y_train = self.y_train.reshape(-1, 1)
            
    def _calculate_distances(self, x):
        """
        Calculate distances from a single point to all training points.
        
        Args:
            x (numpy.ndarray): A single sample point, shape (n_features,).
            
        Returns:
            numpy.ndarray: Array of distances to each training point.
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance: sqrt(sum((x - y)^2))
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        elif self.distance_metric == 'manhattan':
            # Manhattan distance: sum(abs(x - y))
            return np.sum(np.abs(self.X_train - x), axis=1)
        
        elif self.distance_metric == 'minkowski':
            # Minkowski distance: (sum(abs(x - y)^p))^(1/p)
            return np.power(np.sum(np.power(np.abs(self.X_train - x), self.p), axis=1), 1/self.p)
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
    def _predict_single(self, x):
        """
        Predict for a single sample.
        
        Args:
            x (numpy.ndarray): A single sample point, shape (n_features,).
            
        Returns:
            numpy.ndarray: Predicted value(s) for the input sample.
        """
        # Calculate distances to all training points
        distances = self._calculate_distances(x)
        
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[nearest_indices]
        
        # Calculate weights
        if self.weights == 'uniform':
            weights = np.ones_like(distances[nearest_indices])
        elif self.weights == 'distance':
            # Use inverse of distances as weights (handle zero distances)
            weights = 1.0 / (distances[nearest_indices] + 1e-10)
        else:
            raise ValueError(f"Unknown weight function: {self.weights}")
            
        # Check if we're dealing with regression or classification
        if k_nearest_labels.dtype.kind in ['i', 'u', 'b']:  # Integer, unsigned int, or boolean
            # Classification: weighted voting
            unique_classes = np.unique(k_nearest_labels)
            class_votes = {}
            
            for i, cls in enumerate(unique_classes):
                # Sum weights for each class
                mask = (k_nearest_labels == cls)
                if mask.ndim > 1:  # Handle multi-output case
                    mask = mask.any(axis=1)
                class_votes[cls] = np.sum(weights[mask])
                
            # Return class with highest weighted vote
            return max(class_votes.items(), key=lambda x: x[1])[0]
            
        else:  # Float or other type
            # Regression: weighted average
            return np.average(k_nearest_labels, weights=weights, axis=0)
            
    def predict(self, X):
        """
        Predict the class labels or target values for the provided data.
        
        Args:
            X (numpy.ndarray): Test samples, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted class labels or target values.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        X = np.asarray(X)
        predictions = np.array([self._predict_single(x) for x in X])
        
        # If the predictions are 1D, return a flattened array
        if predictions.shape[1] == 1:
            return predictions.flatten()
        return predictions
        
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels (for classification)
        or the coefficient of determination R^2 (for regression).
        
        Args:
            X (numpy.ndarray): Test samples, shape (n_samples, n_features).
            y (numpy.ndarray): True labels for X, shape (n_samples,) or (n_samples, n_targets).
            
        Returns:
            float: Mean accuracy or R^2 score.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        
        # Check if we're dealing with regression or classification
        if y.dtype.kind in ['i', 'u', 'b']:  # Integer, unsigned int, or boolean
            # Classification: return accuracy
            return np.mean(y_pred == y)
        else:
            # Regression: return R^2 score
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            return 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
    def predict_proba(self, X):
        """
        Return probability estimates for the test data X (only for classification).
        
        This method calculates probability estimates based on the proportion of each class
        among the k nearest neighbors, optionally weighted by distance.
        
        Args:
            X (numpy.ndarray): Test samples, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Probability estimates for each class, shape (n_samples, n_classes).
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        X = np.asarray(X)
        unique_classes = np.unique(self.y_train)
        n_classes = len(unique_classes)
        
        # Create a mapping from class labels to indices
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        # Initialize probability array
        probas = np.zeros((X.shape[0], n_classes))
        
        for i, x in enumerate(X):
            # Calculate distances to all training points
            distances = self._calculate_distances(x)
            
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]
            
            # Calculate weights
            if self.weights == 'uniform':
                weights = np.ones_like(distances[nearest_indices])
            elif self.weights == 'distance':
                # Use inverse of distances as weights (handle zero distances)
                weights = 1.0 / (distances[nearest_indices] + 1e-10)
            
            # Sum weights for each class
            for idx in nearest_indices:
                cls = self.y_train[idx].item() if self.y_train[idx].size == 1 else self.y_train[idx][0]
                class_idx = class_to_idx[cls]
                weight = weights[np.where(nearest_indices == idx)[0][0]]
                probas[i, class_idx] += weight
                
            # Normalize to get probabilities
            if np.sum(probas[i]) > 0:
                probas[i] = probas[i] / np.sum(probas[i])
                
        return probas


class DecisionTree:
    """
    A Decision Tree implementation for both classification and regression.
    
    This class implements decision trees using the CART (Classification and Regression Trees)
    algorithm. It can be used for both classification and regression tasks. For classification,
    it uses Gini impurity to split nodes. For regression, it uses mean squared error (MSE).
    
    The implementation supports customizable parameters like maximum depth, minimum samples
    for splitting, and minimum samples for a leaf node.
    
    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        task (str): The type of task, either 'classification' or 'regression'.
        root (Node): The root node of the decision tree after fitting.
    """
    
    class Node:
        """
        A node in the decision tree.
        
        Attributes:
            feature_idx (int): Index of the feature used for splitting.
            threshold (float): Threshold value for the feature to split on.
            left (Node): Left child node (samples where feature <= threshold).
            right (Node): Right child node (samples where feature > threshold).
            value (float or dict): For leaf nodes, the predicted value (regression) or class distribution (classification).
            is_leaf (bool): Whether this node is a leaf node.
        """
        def __init__(self, value=None):
            self.feature_idx = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = value
            self.is_leaf = False
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize the DecisionTree.
        
        Args:
            max_depth (int, optional): Maximum depth of the tree. None means unlimited.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.task = None
        self.n_classes = None
        self.class_labels = None
        
    def fit(self, X, y):
        """
        Build the decision tree.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_features).
            y (numpy.ndarray): Target values, shape (n_samples,).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Determine the task (classification or regression)
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10:
            self.task = 'regression'
        else:
            self.task = 'classification'
            self.class_labels = np.unique(y)
            self.n_classes = len(self.class_labels)
        
        # Build the tree
        self.root = self._grow_tree(X, y, depth=0)
        
    def _grow_tree(self, X, y, depth):
        """
        Recursively grow the decision tree.
        
        Args:
            X (numpy.ndarray): Feature data for current node.
            y (numpy.ndarray): Target data for current node.
            depth (int): Current depth in the tree.
            
        Returns:
            Node: A decision tree node.
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf or
            np.all(y == y[0])):
            
            # Create a leaf node
            leaf = self.Node()
            leaf.is_leaf = True
            
            if self.task == 'classification':
                # For classification, store class distribution
                leaf.value = {}
                unique_classes, counts = np.unique(y, return_counts=True)
                for i, cls in enumerate(unique_classes):
                    leaf.value[cls] = counts[i] / n_samples
            else:
                # For regression, store mean value
                leaf.value = np.mean(y)
                
            return leaf
        
        # Find the best split
        best_feature_idx, best_threshold = self._find_best_split(X, y)
        
        if best_feature_idx is None:
            # If no good split is found, create a leaf node
            leaf = self.Node()
            leaf.is_leaf = True
            
            if self.task == 'classification':
                leaf.value = {}
                unique_classes, counts = np.unique(y, return_counts=True)
                for i, cls in enumerate(unique_classes):
                    leaf.value[cls] = counts[i] / n_samples
            else:
                leaf.value = np.mean(y)
                
            return leaf
        
        # Split the data
        left_indices = X[:, best_feature_idx] <= best_threshold
        right_indices = ~left_indices
        
        # Check if split results in sufficient samples in each node
        if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
            # If not, create a leaf node
            leaf = self.Node()
            leaf.is_leaf = True
            
            if self.task == 'classification':
                leaf.value = {}
                unique_classes, counts = np.unique(y, return_counts=True)
                for i, cls in enumerate(unique_classes):
                    leaf.value[cls] = counts[i] / n_samples
            else:
                leaf.value = np.mean(y)
                
            return leaf
        
        # Create a new decision node
        node = self.Node()
        node.feature_idx = best_feature_idx
        node.threshold = best_threshold
        
        # Recursively grow the left and right subtrees
        node.left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        node.right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return node
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        
        Args:
            X (numpy.ndarray): Feature data.
            y (numpy.ndarray): Target data.
            
        Returns:
            tuple: (best_feature_idx, best_threshold) or (None, None) if no good split is found.
        """
        n_samples, n_features = X.shape
        
        # If all labels are the same, no need to split further
        if np.all(y == y[0]):
            return None, None
        
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        
        # Calculate parent impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Try each feature
        for feature_idx in range(n_features):
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # If there's only one value, we can't split on this feature
            if len(thresholds) == 1:
                continue
            
            # Try each threshold (midpoint between consecutive values)
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2
                
                # Split the data
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                # Check if split results in sufficient samples in each node
                if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
                    continue
                
                # Calculate impurity for each child
                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])
                
                # Calculate the weighted sum of child impurities
                n_left = np.sum(left_indices)
                n_right = np.sum(right_indices)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                
                # Calculate the information gain
                gain = parent_impurity - weighted_impurity
                
                # Update best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold
    
    def _calculate_impurity(self, y):
        """
        Calculate the impurity of a node.
        
        For classification, we use Gini impurity.
        For regression, we use variance (MSE).
        
        Args:
            y (numpy.ndarray): Target values in the node.
            
        Returns:
            float: Impurity value.
        """
        if self.task == 'classification':
            # Gini impurity
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return 1 - np.sum(probabilities ** 2)
        else:
            # Mean squared error (variance)
            return np.var(y) if len(y) > 0 else 0
            
    def predict_single(self, x, node=None):
        """
        Predict the class or value for a single sample.
        
        Args:
            x (numpy.ndarray): A single sample.
            node (Node, optional): Current node in the traversal. Defaults to the root.
            
        Returns:
            The predicted class (classification) or value (regression).
        """
        if node is None:
            node = self.root
            
        # If leaf node, return the prediction
        if node.is_leaf:
            if self.task == 'classification':
                # Return the class with the highest probability
                return max(node.value, key=node.value.get)
            else:
                # Return the mean value
                return node.value
                
        # Otherwise, traverse the tree
        if x[node.feature_idx] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)
            
    def predict(self, X):
        """
        Predict classes or values for all samples in X.
        
        Args:
            X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted classes or values.
        """
        X = np.asarray(X)
        return np.array([self.predict_single(x) for x in X])
        
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X (only for classification).
        
        Args:
            X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Class probabilities, shape (n_samples, n_classes).
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
            
        X = np.asarray(X)
        proba = np.zeros((X.shape[0], self.n_classes))
        
        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
                    
            # At leaf node, get class probabilities
            for j, cls in enumerate(self.class_labels):
                proba[i, j] = node.value.get(cls, 0)
                
        return proba
        
    def score(self, X, y):
        """
        Return the accuracy (classification) or R^2 score (regression) on the given test data and labels.
        
        Args:
            X (numpy.ndarray): Test samples, shape (n_samples, n_features).
            y (numpy.ndarray): True labels for X.
            
        Returns:
            float: Accuracy or R^2 score.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            # Classification: return accuracy
            return np.mean(y_pred == y)
        else:
            # Regression: return R^2 score
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            return 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
    def print_tree(self, node=None, indent=''):
        """
        Print the decision tree structure.
        
        Args:
            node (Node, optional): Current node. Defaults to the root.
            indent (str, optional): Indentation string. Defaults to ''.
        """
        if node is None:
            node = self.root
            print("Decision Tree:")
            
        if node.is_leaf:
            if self.task == 'classification':
                print(f"{indent}Leaf: {node.value}")
            else:
                print(f"{indent}Leaf: {node.value:.4f}")
        else:
            print(f"{indent}Feature {node.feature_idx} <= {node.threshold:.4f}")
            print(f"{indent}Left ->")
            self.print_tree(node.left, indent + '  ')
            print(f"{indent}Right ->")
            self.print_tree(node.right, indent + '  ')


class RandomForest:
    """
    A Random Forest implementation for both classification and regression.
    
    This class implements Random Forest using an ensemble of DecisionTree models. Each tree
    is trained on a bootstrap sample of the data with a random subset of features considered
    for each split. For classification, predictions are made by majority voting. For regression,
    predictions are the average of the trees' predictions.
    
    The implementation supports customizable parameters like number of trees, maximum depth,
    and feature subsampling ratio.
    
    Attributes:
        n_trees (int): Number of trees in the forest.
        max_features (int or float or str): Number of features to consider for best split:
            - If int, consider max_features features at each split.
            - If float, consider max_features * n_features features at each split.
            - If "sqrt", consider sqrt(n_features) features at each split.
            - If "log2", consider log2(n_features) features at each split.
        bootstrap (bool): Whether to use bootstrap samples when building trees.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum number of samples required to split a node.
        min_samples_leaf (int): Minimum number of samples required at a leaf node.
        trees (list): List of DecisionTree models comprising the forest.
        task (str): The type of task, either 'classification' or 'regression'.
        class_labels (numpy.ndarray): Unique class labels (only for classification).
    """
    
    def __init__(self, n_trees=100, max_features='sqrt', bootstrap=True, 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 random_state=None):
        """
        Initialize the RandomForest.
        
        Args:
            n_trees (int): Number of trees in the forest. Default is 100.
            max_features (int or float or str): Number of features to consider for best split.
                Default is "sqrt".
            bootstrap (bool): Whether to use bootstrap samples when building trees. 
                Default is True.
            max_depth (int, optional): Maximum depth of each tree. None means unlimited.
                Default is None.
            min_samples_split (int): Minimum number of samples required to split a node.
                Default is 2.
            min_samples_leaf (int): Minimum number of samples required at a leaf node.
                Default is 1.
            random_state (int, optional): Seed for random number generation to ensure
                reproducibility. Default is None.
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.task = None
        self.class_labels = None
        self.n_classes = None
        
        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
    
    def _calculate_max_features(self, n_features):
        """
        Calculate the number of features to consider for best split.
        
        Args:
            n_features (int): Total number of features.
            
        Returns:
            int: Number of features to consider.
        """
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features parameter: {self.max_features}")
    
    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the data.
        
        Args:
            X (numpy.ndarray): Feature data.
            y (numpy.ndarray): Target data.
            
        Returns:
            tuple: (X_bootstrap, y_bootstrap) - bootstrap sample of features and targets.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """
        Build the random forest.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_features).
            y (numpy.ndarray): Target values, shape (n_samples,).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        
        # Determine the task (classification or regression)
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10:
            self.task = 'regression'
        else:
            self.task = 'classification'
            self.class_labels = np.unique(y)
            self.n_classes = len(self.class_labels)
        
        # Calculate max_features
        self.max_features_value = self._calculate_max_features(n_features)
        
        # Create forest
        self.trees = []
        for i in range(self.n_trees):
            # Create a new tree
            tree = self._create_tree()
            
            # Bootstrap sample if enabled
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            # Fit the tree with feature subsampling
            self._fit_tree(tree, X_sample, y_sample)
            
            # Add tree to forest
            self.trees.append(tree)
    
    def _create_tree(self):
        """
        Create a new DecisionTree with random feature selection.
        
        Returns:
            RandomTreeEstimator: A new tree estimator.
        """
        return RandomTreeEstimator(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features_value
        )
    
    def _fit_tree(self, tree, X, y):
        """
        Fit a single tree in the forest.
        
        Args:
            tree (RandomTreeEstimator): Tree to fit.
            X (numpy.ndarray): Feature data.
            y (numpy.ndarray): Target data.
        """
        tree.fit(X, y)
    
    def predict(self, X):
        """
        Predict classes or values for samples in X.
        
        Args:
            X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted classes or values.
        """
        X = np.asarray(X)
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Aggregate predictions - voting for classification, averaging for regression
        if self.task == 'classification':
            # Transpose to have shape (n_samples, n_trees)
            predictions = predictions.T
            
            # Majority vote for each sample
            final_predictions = np.zeros(predictions.shape[0], dtype=self.class_labels.dtype)
            for i, sample_preds in enumerate(predictions):
                # Count occurrences of each class
                unique, counts = np.unique(sample_preds, return_counts=True)
                # Select class with highest count
                final_predictions[i] = unique[np.argmax(counts)]
                
            return final_predictions
        else:
            # Average for regression
            return np.mean(predictions, axis=0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X (only for classification).
        
        Args:
            X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Class probabilities, shape (n_samples, n_classes).
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
            
        X = np.asarray(X)
        
        # Get probabilities from all trees
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees])
        
        # Average probabilities across trees
        return np.mean(all_proba, axis=0)
    
    def score(self, X, y):
        """
        Return the accuracy (classification) or R^2 score (regression) on the given test data and labels.
        
        Args:
            X (numpy.ndarray): Test samples, shape (n_samples, n_features).
            y (numpy.ndarray): True labels for X.
            
        Returns:
            float: Accuracy or R^2 score.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            # Classification: return accuracy
            return np.mean(y_pred == y)
        else:
            # Regression: return R^2 score
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            return 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    def feature_importances(self):
        """
        Calculate feature importances.
        
        The importance of a feature is computed as the normalized total reduction of impurity
        brought by that feature across all trees in the forest.
        
        Returns:
            numpy.ndarray: Feature importances, shape (n_features,).
        """
        # Get feature importances from all trees
        feature_importances = np.zeros(self.trees[0].n_features)
        
        for tree in self.trees:
            feature_importances += tree.feature_importances()
            
        # Normalize
        if np.sum(feature_importances) > 0:
            feature_importances /= np.sum(feature_importances)
            
        return feature_importances


class RandomTreeEstimator(DecisionTree):
    """
    A Decision Tree estimator with random feature selection for use in Random Forests.
    
    This class extends the DecisionTree class to consider only a random subset of features
    when looking for the best split at each node.
    
    Attributes:
        max_features (int): Number of features to consider for best split.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        """
        Initialize the RandomTreeEstimator.
        
        Args:
            max_depth (int, optional): Maximum depth of the tree. None means unlimited.
            min_samples_split (int): Minimum number of samples required to split a node.
            min_samples_leaf (int): Minimum number of samples required at a leaf node.
            max_features (int): Number of features to consider for best split.
        """
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        self.max_features = max_features
        self.n_features = None
        self.feature_importances_array = None
    
    def fit(self, X, y):
        """
        Build the decision tree.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_features).
            y (numpy.ndarray): Target values, shape (n_samples,).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_features = X.shape[1]
        self.feature_importances_array = np.zeros(self.n_features)
        
        # Call the parent's fit method
        super().fit(X, y)
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on, considering only a random subset of features.
        
        Args:
            X (numpy.ndarray): Feature data.
            y (numpy.ndarray): Target data.
            
        Returns:
            tuple: (best_feature_idx, best_threshold) or (None, None) if no good split is found.
        """
        n_samples, n_features = X.shape
        
        # If all labels are the same, no need to split further
        if np.all(y == y[0]):
            return None, None
        
        # Get a random subset of features to consider
        feature_indices = np.random.choice(n_features, size=min(self.max_features, n_features), replace=False)
        
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        
        # Calculate parent impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Try each feature in the random subset
        for feature_idx in feature_indices:
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # If there's only one value, we can't split on this feature
            if len(thresholds) == 1:
                continue
            
            # Try each threshold (midpoint between consecutive values)
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2
                
                # Split the data
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                # Check if split results in sufficient samples in each node
                if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
                    continue
                
                # Calculate impurity for each child
                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])
                
                # Calculate the weighted sum of child impurities
                n_left = np.sum(left_indices)
                n_right = np.sum(right_indices)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                
                # Calculate the information gain
                gain = parent_impurity - weighted_impurity
                
                # Update best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    
                    # Record the feature importance (impurity reduction)
                    if best_feature_idx is not None:
                        self.feature_importances_array[best_feature_idx] += gain * n_samples
        
        return best_feature_idx, best_threshold
    
    def feature_importances(self):
        """
        Get feature importances for this tree.
        
        Returns:
            numpy.ndarray: Feature importances.
        """
        return self.feature_importances_array
    

class AdaBoost:
    """
    An AdaBoost implementation for binary classification.
    
    AdaBoost (Adaptive Boosting) is an ensemble method that combines multiple "weak learners"
    (typically decision stumps) into a strong classifier. It works by training each subsequent
    classifier on a weighted version of the dataset, where instances misclassified by previous
    classifiers are given more weight.
    
    This implementation uses decision stumps (one-level decision trees) as weak learners.
    
    Attributes:
        n_estimators (int): The maximum number of estimators to use.
        learning_rate (float): Weight applied to each classifier at each boosting iteration.
        weak_learners (list): List of weak learners (decision stumps).
        alphas (list): List of weights for each weak learner.
    """
    
    class DecisionStump:
        """
        A decision stump implementation used as a weak learner in AdaBoost.
        
        A decision stump is a one-level decision tree that makes predictions based on the value
        of a single feature. It's essentially a decision tree with only one split.
        
        Attributes:
            feature_idx (int): The index of the feature to split on.
            threshold (float): The threshold value for the split.
            polarity (int): Determines which side of the split gets the positive label (1 or -1).
            alpha (float): The weight of this stump in the final ensemble.
        """
        
        def __init__(self):
            self.feature_idx = None
            self.threshold = None
            self.polarity = 1
            self.alpha = None
        
        def predict(self, X):
            """
            Predict class labels (-1 or 1) for samples in X.
            
            Args:
                X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
                
            Returns:
                numpy.ndarray: Predicted class labels (-1 or 1).
            """
            X = np.asarray(X)
            n_samples = X.shape[0]
            predictions = np.ones(n_samples)
            
            # Apply the decision rule
            feature_values = X[:, self.feature_idx]
            if self.polarity == 1:
                predictions[feature_values < self.threshold] = -1
            else:
                predictions[feature_values >= self.threshold] = -1
                
            return predictions
    
    def __init__(self, n_estimators=50, learning_rate=1.0):
        """
        Initialize the AdaBoost classifier.
        
        Args:
            n_estimators (int): The maximum number of estimators to use. Default is 50.
            learning_rate (float): Weight applied to each classifier at each boosting iteration.
                Default is 1.0.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weak_learners = []
        self.alphas = []
    
    def fit(self, X, y):
        """
        Build the AdaBoost classifier.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_features).
            y (numpy.ndarray): Target values, shape (n_samples,).
                Values should be either 0/1 or -1/1 (will be converted to -1/1 internally).
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Convert 0/1 labels to -1/1
        if np.all(np.isin(y, [0, 1])):
            y = 2 * y - 1
            
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("AdaBoost supports only binary classification with labels [-1, 1] or [0, 1]")
        
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples
        
        # Train weak learners
        for _ in range(self.n_estimators):
            # Create and train a weak learner
            weak_learner = self._train_weak_learner(X, y, weights)
            
            # Make predictions with the weak learner
            predictions = weak_learner.predict(X)
            
            # Calculate weighted error
            error = np.sum(weights * (predictions != y)) / np.sum(weights)
            
            # If error is 0 or 0.5, stop training (perfect classifier or no better than random)
            if error == 0 or error >= 0.5:
                if error == 0:  # Perfect classifier
                    self.weak_learners.append(weak_learner)
                    self.alphas.append(self.learning_rate * np.log((1 - error) / max(error, 1e-10)))
                break
            
            # Calculate the weight of the weak learner
            alpha = self.learning_rate * np.log((1 - error) / max(error, 1e-10))
            weak_learner.alpha = alpha
            
            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            
            # Normalize weights
            weights /= np.sum(weights)
            
            # Add the weak learner to the ensemble
            self.weak_learners.append(weak_learner)
            self.alphas.append(alpha)
    
    def _train_weak_learner(self, X, y, weights):
        """
        Train a decision stump on the weighted data.
        
        Args:
            X (numpy.ndarray): Feature data, shape (n_samples, n_features).
            y (numpy.ndarray): Target data, shape (n_samples,).
            weights (numpy.ndarray): Weights for each sample, shape (n_samples,).
            
        Returns:
            DecisionStump: Trained decision stump.
        """
        n_samples, n_features = X.shape
        
        # Initialize with the worst possible error
        best_stump = self.DecisionStump()
        min_error = float('inf')
        
        # Try each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            if len(thresholds) == 1:
                continue
                
            # Try each threshold
            for threshold in thresholds:
                # Try both polarities
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1
                    
                    # Calculate weighted error
                    error = np.sum(weights * (predictions != y))
                    
                    # Update best stump if this is better
                    if error < min_error:
                        min_error = error
                        best_stump.feature_idx = feature_idx
                        best_stump.threshold = threshold
                        best_stump.polarity = polarity
        
        return best_stump
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted class labels (0 or 1).
        """
        X = np.asarray(X)
        
        # Weighted sum of weak learner predictions
        weighted_preds = np.zeros(X.shape[0])
        
        for i, learner in enumerate(self.weak_learners):
            weighted_preds += self.alphas[i] * learner.predict(X)
            
        # Convert to class labels
        predictions = np.sign(weighted_preds)
        
        # Convert -1/1 back to 0/1
        return (predictions + 1) / 2
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        The probabilities are calculated based on the distance from the decision boundary,
        transformed using a sigmoid function.
        
        Args:
            X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Class probabilities, shape (n_samples, 2).
        """
        X = np.asarray(X)
        
        # Weighted sum of weak learner predictions
        weighted_preds = np.zeros(X.shape[0])
        
        for i, learner in enumerate(self.weak_learners):
            weighted_preds += self.alphas[i] * learner.predict(X)
            
        # Convert to probabilities using sigmoid function
        probs_positive = 1 / (1 + np.exp(-2 * weighted_preds))
        probs = np.zeros((X.shape[0], 2))
        probs[:, 0] = 1 - probs_positive
        probs[:, 1] = probs_positive
        
        return probs
    
    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.
        
        Args:
            X (numpy.ndarray): Test samples, shape (n_samples, n_features).
            y (numpy.ndarray): True labels for X, shape (n_samples,).
            
        Returns:
            float: Accuracy score.
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Convert 0/1 labels to -1/1 if needed for comparison
        if np.all(np.isin(y, [0, 1])):
            y_pred = self.predict(X)
        else:  # Already -1/1
            y_pred = 2 * self.predict(X) - 1
            
        return np.mean(y_pred == y)


class XGBoost:
    """
    A simplified XGBoost (eXtreme Gradient Boosting) implementation for both classification and regression.
    
    XGBoost is an ensemble method that builds a series of decision trees, where each tree tries
    to correct the errors of the previous trees. It uses gradient boosting with second-order gradients
    and has built-in regularization to prevent overfitting.
    
    This implementation is a simplified version of the full XGBoost algorithm, focusing on the core
    principles of gradient boosting with decision trees.
    
    Attributes:
        n_estimators (int): Number of boosting stages (trees) to perform.
        learning_rate (float): Shrinks the contribution of each tree.
        max_depth (int): Maximum depth of the individual trees.
        min_samples_split (int): Minimum number of samples required to split a node.
        min_samples_leaf (int): Minimum number of samples required at a leaf node.
        subsample (float): Fraction of samples to be used for training each tree.
        colsample_bytree (float): Fraction of features to be used for training each tree.
        reg_lambda (float): L2 regularization term on weights.
        trees (list): List of trees in the ensemble.
        task (str): The type of task, either 'classification' or 'regression'.
        initial_prediction (float): Initial prediction (baseline) before any trees.
    """
    
    class XGBoostTree:
        """
        A decision tree implementation specialized for XGBoost.
        
        This tree is designed to be used within XGBoost and focuses on optimizing
        the specific objective function used in gradient boosting.
        
        Attributes:
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split a node.
            min_samples_leaf (int): Minimum number of samples required at a leaf node.
            colsample (float): Fraction of features to consider for each split.
            reg_lambda (float): L2 regularization term.
            tree (dict): The tree structure stored as a nested dictionary.
        """
        
        def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1, 
                     colsample=1.0, reg_lambda=1.0):
            """
            Initialize the XGBoostTree.
            
            Args:
                max_depth (int): Maximum depth of the tree. Default is 3.
                min_samples_split (int): Minimum number of samples required to split a node.
                    Default is 2.
                min_samples_leaf (int): Minimum number of samples required at a leaf node.
                    Default is 1.
                colsample (float): Fraction of features to consider for each split.
                    Default is 1.0.
                reg_lambda (float): L2 regularization term. Default is 1.0.
            """
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.colsample = colsample
            self.reg_lambda = reg_lambda
            self.tree = None
        
        def fit(self, X, gradients, hessians):
            """
            Build the tree using the gradients and hessians.
            
            Args:
                X (numpy.ndarray): Feature data, shape (n_samples, n_features).
                gradients (numpy.ndarray): First-order gradients, shape (n_samples,).
                hessians (numpy.ndarray): Second-order gradients, shape (n_samples,).
            """
            X = np.asarray(X)
            gradients = np.asarray(gradients)
            hessians = np.asarray(hessians)
            
            # Build the tree recursively
            self.tree = self._build_tree(X, gradients, hessians, depth=0)
        
        def _build_tree(self, X, gradients, hessians, depth):
            """
            Recursively build the tree.
            
            Args:
                X (numpy.ndarray): Feature data.
                gradients (numpy.ndarray): First-order gradients.
                hessians (numpy.ndarray): Second-order gradients.
                depth (int): Current depth in the tree.
                
            Returns:
                dict: A node in the tree, either a decision node or a leaf node.
            """
            n_samples, n_features = X.shape
            
            # Calculate the leaf value if this is a leaf node
            leaf_value = self._calculate_leaf_value(gradients, hessians)
            
            # Check stopping criteria
            if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_samples < 2 * self.min_samples_leaf):
                return {'leaf_value': leaf_value}
            
            # Find the best split
            best_gain = 0
            best_feature_idx = None
            best_threshold = None
            best_left_indices = None
            best_right_indices = None
            
            # Randomly select features to consider
            n_features_to_consider = max(1, int(self.colsample * n_features))
            feature_indices = np.random.choice(n_features, size=n_features_to_consider, replace=False)
            
            # Calculate gain for the current node (before split)
            G_parent = np.sum(gradients)
            H_parent = np.sum(hessians)
            gain_parent = (G_parent ** 2) / (H_parent + self.reg_lambda)
            
            # Try each feature
            for feature_idx in feature_indices:
                feature_values = X[:, feature_idx]
                unique_values = np.unique(feature_values)
                
                if len(unique_values) < 2:
                    continue
                
                # Try each threshold (midpoint between consecutive values)
                for i in range(len(unique_values) - 1):
                    threshold = (unique_values[i] + unique_values[i + 1]) / 2
                    
                    # Split the data
                    left_indices = feature_values <= threshold
                    right_indices = ~left_indices
                    
                    # Check if split results in sufficient samples in each node
                    if (np.sum(left_indices) < self.min_samples_leaf or
                        np.sum(right_indices) < self.min_samples_leaf):
                        continue
                    
                    # Calculate gain for children
                    G_left = np.sum(gradients[left_indices])
                    H_left = np.sum(hessians[left_indices])
                    G_right = np.sum(gradients[right_indices])
                    H_right = np.sum(hessians[right_indices])
                    
                    gain_left = (G_left ** 2) / (H_left + self.reg_lambda)
                    gain_right = (G_right ** 2) / (H_right + self.reg_lambda)
                    
                    # Calculate the gain
                    gain = gain_left + gain_right - gain_parent
                    
                    # Update best split if this one is better
                    if gain > best_gain:
                        best_gain = gain
                        best_feature_idx = feature_idx
                        best_threshold = threshold
                        best_left_indices = left_indices
                        best_right_indices = right_indices
            
            # If no good split is found, return a leaf node
            if best_feature_idx is None:
                return {'leaf_value': leaf_value}
            
            # Build left and right subtrees
            left_tree = self._build_tree(
                X[best_left_indices],
                gradients[best_left_indices],
                hessians[best_left_indices],
                depth + 1
            )
            
            right_tree = self._build_tree(
                X[best_right_indices],
                gradients[best_right_indices],
                hessians[best_right_indices],
                depth + 1
            )
            
            # Return a decision node
            return {
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left': left_tree,
                'right': right_tree
            }
        
        def _calculate_leaf_value(self, gradients, hessians):
            """
            Calculate the optimal leaf value.
            
            Args:
                gradients (numpy.ndarray): First-order gradients.
                hessians (numpy.ndarray): Second-order gradients.
                
            Returns:
                float: The optimal leaf value.
            """
            sum_gradients = np.sum(gradients)
            sum_hessians = np.sum(hessians)
            
            # Optimal leaf value: -G / (H + lambda)
            return -sum_gradients / (sum_hessians + self.reg_lambda)
        
        def predict(self, X):
            """
            Predict using the tree.
            
            Args:
                X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
                
            Returns:
                numpy.ndarray: Predicted values, shape (n_samples,).
            """
            X = np.asarray(X)
            return np.array([self._predict_single(x, self.tree) for x in X])
        
        def _predict_single(self, x, node):
            """
            Predict for a single sample using the tree.
            
            Args:
                x (numpy.ndarray): A single sample.
                node (dict): Current node in the tree.
                
            Returns:
                float: Predicted value.
            """
            # If leaf node, return the leaf value
            if 'leaf_value' in node:
                return node['leaf_value']
                
            # Otherwise, traverse the tree
            if x[node['feature_idx']] <= node['threshold']:
                return self._predict_single(x, node['left'])
            else:
                return self._predict_single(x, node['right'])
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 colsample_bytree=1.0, reg_lambda=1.0, random_state=None):
        """
        Initialize the XGBoost model.
        
        Args:
            n_estimators (int): Number of boosting stages (trees) to perform. Default is 100.
            learning_rate (float): Shrinks the contribution of each tree. Default is 0.1.
            max_depth (int): Maximum depth of the individual trees. Default is 3.
            min_samples_split (int): Minimum number of samples required to split a node.
                Default is 2.
            min_samples_leaf (int): Minimum number of samples required at a leaf node.
                Default is 1.
            subsample (float): Fraction of samples to be used for training each tree.
                Default is 1.0.
            colsample_bytree (float): Fraction of features to be used for training each tree.
                Default is 1.0.
            reg_lambda (float): L2 regularization term on weights. Default is 1.0.
            random_state (int, optional): Random seed for reproducibility. Default is None.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.trees = []
        self.task = None
        self.initial_prediction = None
        
        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y):
        """
        Build the XGBoost model.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_features).
            y (numpy.ndarray): Target values, shape (n_samples,).
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Determine the task (classification or regression)
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10:
            self.task = 'regression'
        else:
            self.task = 'classification'
            self.class_labels = np.unique(y)
            self.n_classes = len(self.class_labels)
        
        # Initialize predictions with base value
        if self.task == 'regression':
            self.initial_prediction = np.mean(y)
        else:
            # For binary classification, use log-odds of the mean as initial prediction
            p_mean = np.mean(y)
            self.initial_prediction = np.log(p_mean / (1 - p_mean))
        
        F = np.full(y.shape, self.initial_prediction)
        
        # Train trees
        for _ in range(self.n_estimators):
            # Calculate gradients and hessians
            if self.task == 'regression':
                # MSE objective
                gradients = F - y
                hessians = np.ones_like(gradients)
            else:
                # LogLoss objective
                p = 1 / (1 + np.exp(-F))
                gradients = p - y
                hessians = p * (1 - p)
            
            # Create and train a new tree
            tree = self.XGBoostTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                colsample=self.colsample_bytree,
                reg_lambda=self.reg_lambda
            )
            
            # Subsample if enabled
            if self.subsample < 1.0:
                n_samples = X.shape[0]
                n_subsamples = max(1, int(self.subsample * n_samples))
                subsample_indices = np.random.choice(n_samples, size=n_subsamples, replace=False)
                tree.fit(X[subsample_indices], gradients[subsample_indices], hessians[subsample_indices])
            else:
                tree.fit(X, gradients, hessians)
            
            # Update predictions
            update = self.learning_rate * tree.predict(X)
            F += update
            
            # Add the tree to the ensemble
            self.trees.append(tree)
    
    def predict(self, X):
        """
        Predict class labels or values for samples in X.
        
        Args:
            X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted class labels or values.
        """
        X = np.asarray(X)
        
        # Start with initial prediction
        F = np.full(X.shape[0], self.initial_prediction)
        
        # Add contribution from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        if self.task == 'classification':
            # Convert to probabilities and then to class labels
            probs = 1 / (1 + np.exp(-F))
            return (probs >= 0.5).astype(int)
        else:
            return F
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X (only for classification).
        
        Args:
            X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Class probabilities, shape (n_samples, n_classes).
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
            
        X = np.asarray(X)
        
        # Start with initial prediction
        F = np.full(X.shape[0], self.initial_prediction)
        
        # Add contribution from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        # Calculate probabilities
        probs_positive = 1 / (1 + np.exp(-F))
        probs = np.zeros((X.shape[0], len(self.class_labels)))
        
        for i, c in enumerate(self.class_labels):
            probs[:, i] = 1 - probs_positive if c == 0 else probs_positive
        
        return probs
    
    def score(self, X, y):
        """
        Return the accuracy (classification) or R^2 score (regression) on the given test data and labels.
        
        Args:
            X (numpy.ndarray): Test samples, shape (n_samples, n_features).
            y (numpy.ndarray): True labels for X.
            
        Returns:
            float: Accuracy or R^2 score.
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        if self.task == 'classification':
            # Classification: return accuracy
            y_pred = self.predict(X)
            return np.mean(y_pred == y)
        else:
            # Regression: return R^2 score
            y_pred = self.predict(X)
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            return 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    

class RidgeRegression:
    """
    Ridge Regression implementation with L2 regularization.
    
    Ridge regression is a linear regression model with L2 regularization that helps prevent
    overfitting by penalizing large coefficients. It adds a regularization term to the 
    ordinary least squares objective function, which is the sum of squares of the coefficients
    multiplied by a regularization parameter alpha.
    
    This implementation supports both closed-form solution and gradient descent optimization.
    
    Attributes:
        alpha (float): Regularization strength. Higher values increase regularization.
        solver (str): Optimization method, either 'closed_form' or 'gradient_descent'.
        max_iter (int): Maximum number of iterations for gradient descent.
        tol (float): Convergence tolerance for gradient descent.
        learning_rate (float): Learning rate for gradient descent.
        normalize (bool): Whether to normalize input features.
        W (numpy.ndarray): Weights of the model.
        b (float): Bias term.
        min_val (numpy.ndarray): Minimum values for normalization (if normalize=True).
        max_val (numpy.ndarray): Maximum values for normalization (if normalize=True).
    """
    
    def __init__(self, alpha=1.0, solver='closed_form', max_iter=1000, 
                 tol=1e-4, learning_rate=0.01, normalize=True):
        """
        Initialize the Ridge Regression model.
        
        Args:
            alpha (float): Regularization strength. Higher values increase regularization.
                Default is 1.0.
            solver (str): Optimization method, either 'closed_form' or 'gradient_descent'.
                Default is 'closed_form'.
            max_iter (int): Maximum number of iterations for gradient descent.
                Default is 1000.
            tol (float): Convergence tolerance for gradient descent.
                Default is 1e-4.
            learning_rate (float): Learning rate for gradient descent.
                Default is 0.01.
            normalize (bool): Whether to normalize input features.
                Default is True.
        """
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.W = None
        self.b = None
        self.min_val = None
        self.max_val = None
    
    def fit(self, X, y):
        """
        Fit the Ridge Regression model.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_features).
            y (numpy.ndarray): Target values, shape (n_samples,) or (n_samples, 1).
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)  # Ensure y is (n_samples, 1)
        
        # Normalize features if needed
        if self.normalize:
            X_norm, self.min_val, self.max_val = normalize_data(X)
        else:
            X_norm = X
        
        n_samples, n_features = X_norm.shape
        
        # Add intercept column
        X_with_intercept = np.hstack((np.ones((n_samples, 1)), X_norm))
        
        if self.solver == 'closed_form':
            # Closed-form solution: W = (X^T X + alpha I)^-1 X^T y
            I = np.identity(n_features + 1)
            I[0, 0] = 0  # Don't regularize the bias term
            coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept + self.alpha * I) @ X_with_intercept.T @ y
            
            # Extract bias and weights
            self.b = coefficients[0, 0]
            self.W = coefficients[1:, 0].reshape(-1, 1)
            
        elif self.solver == 'gradient_descent':
            # Initialize weights and bias
            self.W = np.zeros((n_features, 1))
            self.b = 0
            
            # Gradient descent
            for _ in range(self.max_iter):
                # Compute predictions
                y_pred = np.dot(X_norm, self.W) + self.b
                
                # Compute gradients
                error = y_pred - y
                dW = (1/n_samples) * (X_norm.T @ error) + (self.alpha/n_samples) * self.W
                db = (1/n_samples) * np.sum(error)
                
                # Update weights and bias
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db
                
                # Check for convergence
                if np.linalg.norm(dW) < self.tol and np.abs(db) < self.tol:
                    break
        else:
            raise ValueError(f"Unknown solver: {self.solver}. Choose either 'closed_form' or 'gradient_descent'.")
    
    def predict(self, X):
        """
        Predict target values for the input data.
        
        Args:
            X (numpy.ndarray): Input features, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted values, shape (n_samples, 1).
        """
        X = np.asarray(X)
        
        # Normalize features if needed
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Model must be fitted with normalize=True to predict with normalization.")
            X_norm = normalize_data(X, min_val=self.min_val, max_val=self.max_val)
        else:
            X_norm = X
        
        # Make predictions
        return np.dot(X_norm, self.W) + self.b
    
    def score(self, X, y):
        """
        Compute the coefficient of determination R^2.
        
        R^2 = 1 - SS_res / SS_tot
        where SS_res is the residual sum of squares and SS_tot is the total sum of squares.
        
        Args:
            X (numpy.ndarray): Test features, shape (n_samples, n_features).
            y (numpy.ndarray): True target values, shape (n_samples,) or (n_samples, 1).
            
        Returns:
            float: R^2 score.
        """
        y = np.asarray(y).reshape(-1, 1)
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - ss_residual / ss_total if ss_total > 0 else 0


class LassoRegression:
    """
    Lasso Regression implementation with L1 regularization.
    
    Lasso regression is a linear regression model with L1 regularization that helps prevent
    overfitting and performs feature selection by promoting sparsity in the coefficients.
    It adds a regularization term to the ordinary least squares objective function, which is
    the sum of absolute values of the coefficients multiplied by a regularization parameter alpha.
    
    This implementation uses coordinate descent to optimize the objective function.
    
    Attributes:
        alpha (float): Regularization strength. Higher values increase regularization.
        max_iter (int): Maximum number of iterations for coordinate descent.
        tol (float): Convergence tolerance.
        normalize (bool): Whether to normalize input features.
        W (numpy.ndarray): Weights of the model.
        b (float): Bias term.
        min_val (numpy.ndarray): Minimum values for normalization (if normalize=True).
        max_val (numpy.ndarray): Maximum values for normalization (if normalize=True).
    """
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, normalize=True):
        """
        Initialize the Lasso Regression model.
        
        Args:
            alpha (float): Regularization strength. Higher values increase regularization.
                Default is 1.0.
            max_iter (int): Maximum number of iterations for coordinate descent.
                Default is 1000.
            tol (float): Convergence tolerance.
                Default is 1e-4.
            normalize (bool): Whether to normalize input features.
                Default is True.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        self.W = None
        self.b = None
        self.min_val = None
        self.max_val = None
    
    def _soft_threshold(self, x, threshold):
        """
        Apply soft thresholding operator.
        
        Args:
            x (float): Input value.
            threshold (float): Threshold value.
            
        Returns:
            float: Soft thresholded value.
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0
    
    def fit(self, X, y):
        """
        Fit the Lasso Regression model using coordinate descent.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_features).
            y (numpy.ndarray): Target values, shape (n_samples,) or (n_samples, 1).
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)  # Ensure y is (n_samples, 1)
        
        # Normalize features if needed
        if self.normalize:
            X_norm, self.min_val, self.max_val = normalize_data(X)
        else:
            X_norm = X
        
        n_samples, n_features = X_norm.shape
        
        # Initialize weights and bias
        self.W = np.zeros((n_features, 1))
        self.b = np.mean(y)
        
        # Precompute X^T X for efficiency
        XTX = X_norm.T @ X_norm
        XTy = X_norm.T @ (y - self.b)
        
        # Coordinate descent
        for _ in range(self.max_iter):
            W_old = self.W.copy()
            
            # Update each weight coordinate
            for j in range(n_features):
                # Remove the contribution of W_j from the prediction
                r = XTy[j] - np.sum(XTX[j, :] * self.W[:, 0]) + XTX[j, j] * self.W[j, 0]
                
                # Apply soft thresholding
                self.W[j, 0] = self._soft_threshold(r, self.alpha) / XTX[j, j]
            
            # Check for convergence
            if np.linalg.norm(self.W - W_old) < self.tol:
                break
        
        # Update bias term
        self.b = np.mean(y - X_norm @ self.W)
    
    def predict(self, X):
        """
        Predict target values for the input data.
        
        Args:
            X (numpy.ndarray): Input features, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted values, shape (n_samples, 1).
        """
        X = np.asarray(X)
        
        # Normalize features if needed
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Model must be fitted with normalize=True to predict with normalization.")
            X_norm = normalize_data(X, min_val=self.min_val, max_val=self.max_val)
        else:
            X_norm = X
        
        # Make predictions
        return X_norm @ self.W + self.b
    
    def score(self, X, y):
        """
        Compute the coefficient of determination R^2.
        
        R^2 = 1 - SS_res / SS_tot
        where SS_res is the residual sum of squares and SS_tot is the total sum of squares.
        
        Args:
            X (numpy.ndarray): Test features, shape (n_samples, n_features).
            y (numpy.ndarray): True target values, shape (n_samples,) or (n_samples, 1).
            
        Returns:
            float: R^2 score.
        """
        y = np.asarray(y).reshape(-1, 1)
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - ss_residual / ss_total if ss_total > 0 else 0


class ElasticNet:
    """
    ElasticNet Regression implementation with combined L1 and L2 regularization.
    
    ElasticNet combines the penalties of Ridge and Lasso regression to overcome some of
    their limitations. It uses a linear combination of L1 and L2 penalties, controlled by
    the l1_ratio parameter. This makes it effective for feature selection while still
    handling correlated features better than Lasso.
    
    This implementation uses coordinate descent to optimize the objective function.
    
    Attributes:
        alpha (float): Regularization strength. Higher values increase regularization.
        l1_ratio (float): The mixing parameter between L1 and L2 penalties (between 0 and 1).
            l1_ratio=0 is equivalent to Ridge, l1_ratio=1 is equivalent to Lasso.
        max_iter (int): Maximum number of iterations for coordinate descent.
        tol (float): Convergence tolerance.
        normalize (bool): Whether to normalize input features.
        W (numpy.ndarray): Weights of the model.
        b (float): Bias term.
        min_val (numpy.ndarray): Minimum values for normalization (if normalize=True).
        max_val (numpy.ndarray): Maximum values for normalization (if normalize=True).
    """
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, normalize=True):
        """
        Initialize the ElasticNet Regression model.
        
        Args:
            alpha (float): Regularization strength. Higher values increase regularization.
                Default is 1.0.
            l1_ratio (float): The mixing parameter between L1 and L2 penalties (between 0 and 1).
                l1_ratio=0 is equivalent to Ridge, l1_ratio=1 is equivalent to Lasso.
                Default is 0.5.
            max_iter (int): Maximum number of iterations for coordinate descent.
                Default is 1000.
            tol (float): Convergence tolerance.
                Default is 1e-4.
            normalize (bool): Whether to normalize input features.
                Default is True.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        self.W = None
        self.b = None
        self.min_val = None
        self.max_val = None
    
    def _soft_threshold(self, x, threshold):
        """
        Apply soft thresholding operator.
        
        Args:
            x (float): Input value.
            threshold (float): Threshold value.
            
        Returns:
            float: Soft thresholded value.
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0
    
    def fit(self, X, y):
        """
        Fit the ElasticNet Regression model using coordinate descent.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_features).
            y (numpy.ndarray): Target values, shape (n_samples,) or (n_samples, 1).
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)  # Ensure y is (n_samples, 1)
        
        # Normalize features if needed
        if self.normalize:
            X_norm, self.min_val, self.max_val = normalize_data(X)
        else:
            X_norm = X
        
        n_samples, n_features = X_norm.shape
        
        # Initialize weights and bias
        self.W = np.zeros((n_features, 1))
        self.b = np.mean(y)
        
        # Precompute X^T X for efficiency
        XTX = X_norm.T @ X_norm
        XTy = X_norm.T @ (y - self.b)
        
        # Calculate alpha for L1 and L2 terms
        alpha_l1 = self.alpha * self.l1_ratio
        alpha_l2 = self.alpha * (1 - self.l1_ratio)
        
        # Coordinate descent
        for _ in range(self.max_iter):
            W_old = self.W.copy()
            
            # Update each weight coordinate
            for j in range(n_features):
                # Remove the contribution of W_j from the prediction
                r = XTy[j] - np.sum(XTX[j, :] * self.W[:, 0]) + XTX[j, j] * self.W[j, 0]
                
                # Apply soft thresholding with both L1 and L2 terms
                if alpha_l2 > 0:
                    r /= (1 + alpha_l2)
                
                self.W[j, 0] = self._soft_threshold(r, alpha_l1) / XTX[j, j]
            
            # Check for convergence
            if np.linalg.norm(self.W - W_old) < self.tol:
                break
        
        # Update bias term
        self.b = np.mean(y - X_norm @ self.W)
    
    def predict(self, X):
        """
        Predict target values for the input data.
        
        Args:
            X (numpy.ndarray): Input features, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted values, shape (n_samples, 1).
        """
        X = np.asarray(X)
        
        # Normalize features if needed
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Model must be fitted with normalize=True to predict with normalization.")
            X_norm = normalize_data(X, min_val=self.min_val, max_val=self.max_val)
        else:
            X_norm = X
        
        # Make predictions
        return X_norm @ self.W + self.b
    
    def score(self, X, y):
        """
        Compute the coefficient of determination R^2.
        
        R^2 = 1 - SS_res / SS_tot
        where SS_res is the residual sum of squares and SS_tot is the total sum of squares.
        
        Args:
            X (numpy.ndarray): Test features, shape (n_samples, n_features).
            y (numpy.ndarray): True target values, shape (n_samples,) or (n_samples, 1).
            
        Returns:
            float: R^2 score.
        """
        y = np.asarray(y).reshape(-1, 1)
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - ss_residual / ss_total if ss_total > 0 else 0
    