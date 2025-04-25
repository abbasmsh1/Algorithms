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


def minmax_scale(data, feature_range=(0, 1), min_val=None, max_val=None):
    """
    Scale features to a specified range using min-max scaling.

    This function is similar to normalize_data but allows for specifying a custom range.
    It scales each feature to the range [min, max] based on its minimum and maximum values.

    Args:
        data (numpy.ndarray): Input data to scale, shape (m, n).
        feature_range (tuple): Desired range of transformed data, default is (0, 1).
        min_val (numpy.ndarray, optional): Minimum values per feature for scaling. If None, computed from data.
        max_val (numpy.ndarray, optional): Maximum values per feature for scaling. If None, computed from data.

    Returns:
        numpy.ndarray: Scaled data in the specified range.
        numpy.ndarray: Minimum values used for scaling (returned only if min_val is None).
        numpy.ndarray: Maximum values used for scaling (returned only if max_val is None).
    """
    data = np.asarray(data)
    min_range, max_range = feature_range
    
    if min_val is None or max_val is None:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        
        # Handle case where max == min to avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        
        # Scale to [0, 1] and then to desired range
        scaled_data = (data - min_val) / range_val
        scaled_data = scaled_data * (max_range - min_range) + min_range
        
        return scaled_data, min_val, max_val
    else:
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        
        # Scale to [0, 1] and then to desired range
        scaled_data = (data - min_val) / range_val
        scaled_data = scaled_data * (max_range - min_range) + min_range
        
        return scaled_data


def standardize(data, mean=None, std=None):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Each feature will have zero mean and unit variance. This is also known as
    Z-score normalization.
    
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


def robust_scale(data, quantile_range=(25.0, 75.0), center=None, scale=None):
    """
    Scale features using statistics that are robust to outliers.
    
    This scaler removes the median and scales the data according to the quantile range
    (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile
    (25th quantile) and the 3rd quartile (75th quantile).
    
    Args:
        data (numpy.ndarray): Input data to scale, shape (m, n).
        quantile_range (tuple): Quantile range used to calculate scale. Default is (25.0, 75.0).
        center (numpy.ndarray, optional): Median values per feature. If None, computed from data.
        scale (numpy.ndarray, optional): Quantile range values per feature. If None, computed from data.
        
    Returns:
        numpy.ndarray: Robustly scaled data.
        numpy.ndarray: Median values used for centering (returned only if center is None).
        numpy.ndarray: Quantile range values used for scaling (returned only if scale is None).
    """
    data = np.asarray(data)
    q_min, q_max = quantile_range
    
    if center is None or scale is None:
        center = np.median(data, axis=0)
        q_min_val = np.percentile(data, q_min, axis=0)
        q_max_val = np.percentile(data, q_max, axis=0)
        scale = q_max_val - q_min_val
        
        # Handle case where scale == 0 to avoid division by zero
        scale = np.where(scale == 0, 1, scale)
        
        scaled_data = (data - center) / scale
        return scaled_data, center, scale
    else:
        # Handle case where scale == 0 to avoid division by zero
        scale = np.where(scale == 0, 1, scale)
        
        scaled_data = (data - center) / scale
        return scaled_data


def one_hot_encode(data, categories=None):
    """
    Convert categorical variable into binary one-hot encoded variables.
    
    Each sample is represented as a 1 in the column for its category and 0
    in all other columns.
    
    Args:
        data (numpy.ndarray): Input categorical data to encode, shape (m,).
        categories (list, optional): List of unique categories. If None, computed from data.
        
    Returns:
        numpy.ndarray: One-hot encoded data, shape (m, n_categories).
        list: List of categories (returned only if categories is None).
    """
    data = np.asarray(data).flatten()
    
    if categories is None:
        categories = sorted(list(set(data)))
        
    n_samples = len(data)
    n_categories = len(categories)
    
    # Create a mapping from category to index
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    # Initialize one-hot encoded array
    one_hot = np.zeros((n_samples, n_categories))
    
    # Fill the array
    for i, cat in enumerate(data):
        if cat in cat_to_idx:
            one_hot[i, cat_to_idx[cat]] = 1
    
    if categories is None:
        return one_hot, categories
    else:
        return one_hot


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    
    This transformer converts categorical labels to numerical labels. It's useful for
    preparing categorical target values for classification algorithms that require
    numerical inputs.
    
    Attributes:
        classes_ (numpy.ndarray): Holds the label for each class after fitting.
        class_to_index_ (dict): Mapping from original class labels to numerical indices.
        index_to_class_ (dict): Mapping from numerical indices back to original class labels.
    """
    
    def __init__(self):
        """Initialize the LabelEncoder."""
        self.classes_ = None
        self.class_to_index_ = None
        self.index_to_class_ = None
    
    def fit(self, y):
        """
        Fit label encoder by storing the unique class labels.
        
        Args:
            y (array-like): Target values to be encoded.
            
        Returns:
            self: Returns self for method chaining.
        """
        y = np.asarray(y).flatten()
        self.classes_ = np.unique(y)
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.index_to_class_ = {idx: cls for idx, cls in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        """
        Transform labels to normalized encoding (integers).
        
        Args:
            y (array-like): Target values to be transformed.
            
        Returns:
            numpy.ndarray: Transformed values as integers from 0 to n_classes-1.
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet. Call fit() first.")
        
        y = np.asarray(y).flatten()
        encoded = np.zeros(len(y), dtype=int)
        
        for i, label in enumerate(y):
            if label in self.class_to_index_:
                encoded[i] = self.class_to_index_[label]
            else:
                raise ValueError(f"Label '{label}' not found in the fitted classes.")
        
        return encoded
    
    def fit_transform(self, y):
        """
        Fit label encoder and return encoded labels.
        
        Args:
            y (array-like): Target values to be encoded.
            
        Returns:
            numpy.ndarray: Transformed values as integers from 0 to n_classes-1.
        """
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        """
        Transform integer labels back to original labels.
        
        Args:
            y (array-like): Target values to be inverse-transformed.
            
        Returns:
            numpy.ndarray: Original labels.
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet. Call fit() first.")
        
        y = np.asarray(y).flatten()
        decoded = np.empty(len(y), dtype=object)
        
        for i, label_idx in enumerate(y):
            if label_idx in self.index_to_class_:
                decoded[i] = self.index_to_class_[label_idx]
            else:
                raise ValueError(f"Index '{label_idx}' not found in the fitted indices.")
        
        return decoded


def label_encode(y, classes=None):
    """
    Encode target labels with value between 0 and n_classes-1.
    
    Functional version of LabelEncoder for quick use.
    
    Args:
        y (array-like): Target values to be encoded.
        classes (array-like, optional): Predefined class order. If None, computed from data.
        
    Returns:
        numpy.ndarray: Transformed values as integers from 0 to n_classes-1.
        numpy.ndarray: Classes in the order they are encoded (returned only if classes is None).
    """
    y = np.asarray(y).flatten()
    
    if classes is None:
        classes = np.unique(y)
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        
        encoded = np.zeros(len(y), dtype=int)
        for i, label in enumerate(y):
            encoded[i] = class_to_index[label]
            
        return encoded, classes
    else:
        classes = np.asarray(classes)
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        
        encoded = np.zeros(len(y), dtype=int)
        for i, label in enumerate(y):
            if label in class_to_index:
                encoded[i] = class_to_index[label]
            else:
                raise ValueError(f"Label '{label}' not found in the provided classes.")
                
        return encoded


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Args:
        X (numpy.ndarray): Features data, shape (n_samples, n_features).
        y (numpy.ndarray): Target data, shape (n_samples,) or (n_samples, n_targets).
        test_size (float): Proportion of the dataset to include in the test split, between 0.0 and 1.0.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
            
    Returns:
        X_train (numpy.ndarray): Training features data.
        X_test (numpy.ndarray): Testing features data.
        y_train (numpy.ndarray): Training target data.
        y_test (numpy.ndarray): Testing target data.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
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
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

