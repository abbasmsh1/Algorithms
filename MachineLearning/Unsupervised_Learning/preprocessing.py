import numpy as np

def normalize_data(data, min_val=None, max_val=None):
    """
    Normalize the input data to a range of [0, 1] using min-max scaling.

    This function scales each feature to [0, 1] based on its minimum and maximum values.
    During training, min and max are computed from the data. During prediction, provided
    min and max values (from training) can be used for consistent scaling. Handles cases
    where max equals min to avoid division by zero.

    Args:
        data (numpy.ndarray): Input data to normalize, shape (m, n).
        min_val (numpy.ndarray, optional): Minimum values per feature for scaling. If None, computed from data.
        max_val (numpy.ndarray, optional): Maximum values per feature for scaling. If None, computed from data.

    Returns:
        numpy.ndarray: Normalized data in the range [0, 1].
        numpy.ndarray: Minimum values used for scaling (returned only if min_val is None).
        numpy.ndarray: Maximum values used for scaling (returned only if max_val is None).
    """
    data = np.asarray(data)
    if min_val is None or max_val is None:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        # Handle case where max == min to avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized_data = (data - min_val) / range_val
        return normalized_data, min_val, max_val
    else:
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized_data = (data - min_val) / range_val
        return normalized_data


def standardize(data, mean=None, std=None):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Each feature will have zero mean and unit variance. This is also known as
    Z-score normalization, and is particularly important for many unsupervised 
    learning algorithms that are sensitive to the scale of input features.
    
    Args:
        data (numpy.ndarray): Input data to standardize, shape (m, n).
        mean (numpy.ndarray, optional): Mean values per feature. If None, computed from data.
        std (numpy.ndarray, optional): Standard deviation values per feature. If None, computed from data.
        
    Returns:
        numpy.ndarray: Standardized data with zero mean and unit variance.
        numpy.ndarray: Mean values used for scaling (returned only if mean is None).
        numpy.ndarray: Standard deviation values used for scaling (returned only if std is None).
    """
    data = np.asarray(data)
    
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Handle case where std == 0 to avoid division by zero
        std = np.where(std == 0, 1, std)
        
        standardized_data = (data - mean) / std
        return standardized_data, mean, std
    else:
        # Handle case where std == 0 to avoid division by zero
        std = np.where(std == 0, 1, std)
        
        standardized_data = (data - mean) / std
        return standardized_data


def pca_whitening(data, n_components=None, epsilon=1e-5):
    """
    Apply PCA whitening to the data.
    
    PCA whitening transforms the data to have zero mean and identity covariance matrix.
    It's useful for decorrelating features and can improve performance in many unsupervised
    learning algorithms. The whitening adds a small constant (epsilon) to eigenvalues
    to prevent division by zero.
    
    Args:
        data (numpy.ndarray): Input data, shape (m, n).
        n_components (int, optional): Number of principal components to retain. 
            If None, all components are kept. Default is None.
        epsilon (float, optional): Small constant added to eigenvalues for numerical stability.
            Default is 1e-5.
            
    Returns:
        numpy.ndarray: Whitened data.
        object: PCA object containing the transformation parameters.
    """
    data = np.asarray(data)
    m, n = data.shape
    
    # Center the data
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    
    # Compute covariance matrix
    cov = np.dot(data_centered.T, data_centered) / m
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select n_components
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
    
    # Whiten the data
    whitening = np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues + epsilon)))
    data_whitened = np.dot(data_centered, whitening)
    
    # Create a PCA object to store transformation parameters
    class PCAObject:
        def __init__(self, mean, components, explained_variance, whitening_matrix):
            self.mean = mean
            self.components = components
            self.explained_variance = explained_variance
            self.whitening_matrix = whitening_matrix
            self.n_components = len(explained_variance)
        
        def transform(self, X):
            """Transform data using the PCA whitening parameters."""
            X = np.asarray(X)
            X_centered = X - self.mean
            return np.dot(X_centered, self.whitening_matrix)
        
        def inverse_transform(self, X):
            """Inverse transform data back to original space."""
            X = np.asarray(X)
            inverse_whitening = np.dot(self.whitening_matrix, np.diag(np.sqrt(self.explained_variance + epsilon)))
            return np.dot(X, inverse_whitening.T) + self.mean
    
    pca_obj = PCAObject(
        mean=mean,
        components=eigenvectors,
        explained_variance=eigenvalues,
        whitening_matrix=whitening
    )
    
    return data_whitened, pca_obj


def remove_outliers(data, method='zscore', threshold=3.0):
    """
    Remove outliers from the dataset.
    
    This function identifies and removes outliers using various methods.
    
    Args:
        data (numpy.ndarray): Input data, shape (m, n).
        method (str): Method to use for outlier detection. Options:
            - 'zscore': Remove points with z-score above threshold
            - 'iqr': Remove points outside of Q1-threshold*IQR and Q3+threshold*IQR
        threshold (float): Threshold for outlier detection. Default is 3.0 for zscore
            and 1.5 for IQR.
            
    Returns:
        numpy.ndarray: Data with outliers removed.
        numpy.ndarray: Mask of non-outlier points.
    """
    data = np.asarray(data)
    
    if method == 'zscore':
        # Compute z-scores
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        z_scores = np.abs((data - mean) / std)
        
        # Identify outliers
        mask = np.all(z_scores < threshold, axis=1)
        
    elif method == 'iqr':
        # Use threshold of 1.5 by default for IQR method if not specified
        if threshold == 3.0:
            threshold = 1.5
            
        # Compute quartiles and IQR
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        
        # Identify outliers
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Create mask for non-outlier points
        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        
    else:
        raise ValueError(f"Unknown outlier removal method: {method}. Use 'zscore' or 'iqr'.")
    
    return data[mask], mask


def impute_missing_values(data, strategy='mean'):
    """
    Impute missing values in the dataset.
    
    Args:
        data (numpy.ndarray): Input data, which may contain NaN values.
        strategy (str): Strategy to use for imputation. Options:
            - 'mean': Replace with mean of column
            - 'median': Replace with median of column
            - 'most_frequent': Replace with most frequent value in column
            
    Returns:
        numpy.ndarray: Data with missing values imputed.
        dict: Dictionary containing the imputation values for each column.
    """
    data = np.asarray(data)
    imputed_data = data.copy()
    imputation_values = {}
    
    for col in range(data.shape[1]):
        # Get column data, excluding NaN values
        col_data = data[:, col]
        mask = ~np.isnan(col_data)
        valid_data = col_data[mask]
        
        if len(valid_data) == 0:
            # If all values are NaN, replace with 0
            imputation_values[col] = 0
        else:
            # Choose imputation strategy
            if strategy == 'mean':
                imputation_values[col] = np.mean(valid_data)
            elif strategy == 'median':
                imputation_values[col] = np.median(valid_data)
            elif strategy == 'most_frequent':
                values, counts = np.unique(valid_data, return_counts=True)
                imputation_values[col] = values[np.argmax(counts)]
            else:
                raise ValueError(f"Unknown imputation strategy: {strategy}. Use 'mean', 'median', or 'most_frequent'.")
        
        # Replace NaN values
        imputed_data[~mask, col] = imputation_values[col]
    
    return imputed_data, imputation_values


def train_test_split(X, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Args:
        X (numpy.ndarray): Features data, shape (n_samples, n_features).
        test_size (float): Proportion of the dataset to include in the test split, between 0.0 and 1.0.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
            
    Returns:
        X_train (numpy.ndarray): Training features data.
        X_test (numpy.ndarray): Testing features data.
    """
    X = np.asarray(X)
    
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create random indices for shuffling
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    # Calculate the split point
    test_samples = int(n_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    return X_train, X_test 