import numpy as np
from preprocessing import standardize, normalize_data

class KMeans:
    """
    K-Means clustering algorithm implementation.
    
    K-Means partitions data into K clusters by minimizing the within-cluster
    sum of squares (WCSS). It iteratively assigns data points to the nearest
    centroid and then updates centroids until convergence.
    
    Attributes:
        n_clusters (int): Number of clusters to form.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for declaring convergence.
        centroids (numpy.ndarray): Cluster centroids.
        labels (numpy.ndarray): Cluster labels for each data point.
        inertia (float): Within-cluster sum of squares.
        n_iter (int): Number of iterations performed.
        normalize (bool): Whether to normalize the input data.
        random_state (int): Random seed for centroid initialization.
    """
    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, normalize=True, random_state=None):
        """
        Initialize the KMeans model.
        
        Args:
            n_clusters (int): Number of clusters to form. Default is 8.
            max_iter (int): Maximum number of iterations for the algorithm. Default is 300.
            tol (float): Tolerance for declaring convergence. Default is 1e-4.
            normalize (bool): Whether to normalize the input data. Default is True.
            random_state (int): Random seed for centroid initialization. Default is None.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iter = 0
        self.normalize = normalize
        self.min_val = None
        self.max_val = None
        self.random_state = random_state
    
    def _initialize_centroids(self, X):
        """
        Initialize cluster centroids using k-means++ algorithm.
        
        This method implements the k-means++ initialization, which selects
        initial centroids that are distant from each other, leading to better
        convergence and results.
        
        Args:
            X (numpy.ndarray): Input data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Initial centroids, shape (n_clusters, n_features).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        # Choose the rest of the centroids using k-means++ initialization
        for i in range(1, self.n_clusters):
            # Calculate distances from points to the existing centroids
            distances = np.array([np.min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) for x in X])
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.random()
            
            for j, p in enumerate(cumulative_probs):
                if r <= p:
                    centroids[i] = X[j]
                    break
        
        return centroids
    
    def fit(self, X):
        """
        Compute k-means clustering.
        
        Args:
            X (numpy.ndarray): Training data, shape (n_samples, n_features).
            
        Returns:
            self: The fitted estimator.
        """
        X = np.asarray(X)
        
        # Normalize data if specified
        if self.normalize:
            X_processed, self.min_val, self.max_val = normalize_data(X)
        else:
            X_processed = X
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(X_processed)
        
        # Main k-means loop
        for i in range(self.max_iter):
            # Assign each point to the nearest centroid
            labels, distances = self._assign_clusters(X_processed)
            
            # Store the previous centroids for convergence check
            previous_centroids = self.centroids.copy()
            
            # Update centroids based on the mean of assigned points
            for j in range(self.n_clusters):
                cluster_points = X_processed[labels == j]
                if len(cluster_points) > 0:
                    self.centroids[j] = np.mean(cluster_points, axis=0)
            
            # Check for convergence
            if np.linalg.norm(self.centroids - previous_centroids) < self.tol:
                break
        
        # Final assignment with updated centroids
        self.labels, distances = self._assign_clusters(X_processed)
        
        # Calculate inertia (sum of squared distances to closest centroid)
        self.inertia = np.sum(distances)
        self.n_iter = i + 1
        
        return self
    
    def _assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid.
        
        Args:
            X (numpy.ndarray): Input data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Cluster labels for each point.
            numpy.ndarray: Squared distances to the nearest centroid.
        """
        # Calculate distances from each point to each centroid
        distances = np.array([np.linalg.norm(X - centroid, axis=1)**2 for centroid in self.centroids])
        
        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=0)
        
        # Get the minimum distance for each point
        min_distances = np.min(distances, axis=0)
        
        return labels, min_distances
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        Args:
            X (numpy.ndarray): New data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Index of the cluster each sample belongs to.
        """
        X = np.asarray(X)
        
        # Normalize data if model was trained with normalization
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Model must be fitted with normalize=True to predict with normalization.")
            X_processed = normalize_data(X, min_val=self.min_val, max_val=self.max_val)
        else:
            X_processed = X
        
        # Assign clusters
        labels, _ = self._assign_clusters(X_processed)
        
        return labels
    
    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Args:
            X (numpy.ndarray): Training data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels
    
    def transform(self, X):
        """
        Transform X to a cluster-distance space.
        
        Args:
            X (numpy.ndarray): New data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Distances to each cluster center, shape (n_samples, n_clusters).
        """
        X = np.asarray(X)
        
        # Normalize data if model was trained with normalization
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Model must be fitted with normalize=True to transform with normalization.")
            X_processed = normalize_data(X, min_val=self.min_val, max_val=self.max_val)
        else:
            X_processed = X
        
        # Calculate distances from each point to each centroid
        distances = np.array([np.linalg.norm(X_processed - centroid, axis=1) for centroid in self.centroids]).T
        
        return distances
    
    def get_cluster_centers(self):
        """
        Get the cluster centers.
        
        Returns:
            numpy.ndarray: Cluster centers, shape (n_clusters, n_features).
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
            
        return self.centroids


class PCA:
    """
    Principal Component Analysis (PCA) implementation.
    
    PCA performs a linear dimensionality reduction by projecting the data onto
    lower-dimensional space while preserving as much variance as possible.
    
    Attributes:
        n_components (int): Number of components to keep.
        standardize (bool): Whether to standardize the input data.
        components (numpy.ndarray): Principal components.
        explained_variance (numpy.ndarray): Variance explained by each component.
        explained_variance_ratio (numpy.ndarray): Ratio of variance explained by each component.
        mean (numpy.ndarray): Mean of training data.
        std (numpy.ndarray): Standard deviation of training data.
    """
    
    def __init__(self, n_components=None, standardize=True):
        """
        Initialize the PCA model.
        
        Args:
            n_components (int, optional): Number of components to keep. If None, all components
                are kept. Default is None.
            standardize (bool): Whether to standardize the input data. Default is True.
        """
        self.n_components = n_components
        self.standardize = standardize
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.mean = None
        self.std = None
    
    def fit(self, X):
        """
        Fit the PCA model.
        
        Args:
            X (numpy.ndarray): Training data, shape (n_samples, n_features).
            
        Returns:
            self: The fitted estimator.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Standardize data if specified
        if self.standardize:
            X_processed, self.mean, self.std = standardize(X)
        else:
            X_processed = X
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.dot(X_processed.T, X_processed) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine number of components to keep
        if self.n_components is None:
            self.n_components = n_features
        
        # Store components and explained variance
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        """
        Transform X to lower-dimensional space.
        
        Args:
            X (numpy.ndarray): New data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Transformed data, shape (n_samples, n_components).
        """
        X = np.asarray(X)
        
        # Standardize data if model was fitted with standardization
        if self.standardize:
            X_processed = (X - self.mean) / self.std
        else:
            X_processed = X - self.mean
        
        # Project data onto principal components
        X_transformed = np.dot(X_processed, self.components)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit the model and apply dimensionality reduction on X.
        
        Args:
            X (numpy.ndarray): Training data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Transformed data, shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Transform data back to its original space.
        
        Args:
            X (numpy.ndarray): Transformed data, shape (n_samples, n_components).
            
        Returns:
            numpy.ndarray: Reconstructed data, shape (n_samples, n_features).
        """
        X = np.asarray(X)
        
        # Project back to original space
        X_reconstructed = np.dot(X, self.components.T)
        
        # Reverse standardization if applied
        if self.standardize:
            X_reconstructed = X_reconstructed * self.std + self.mean
        else:
            X_reconstructed = X_reconstructed + self.mean
        
        return X_reconstructed
    
    def get_components(self):
        """
        Get the principal components.
        
        Returns:
            numpy.ndarray: Principal components, shape (n_features, n_components).
        """
        if self.components is None:
            raise ValueError("Model has not been fitted yet.")
            
        return self.components


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN) implementation.
    
    DBSCAN is a density-based clustering algorithm that groups together points that are
    closely packed together, marking as outliers points that lie in low-density regions.
    
    Attributes:
        eps (float): The maximum distance between two samples for them to be considered neighbors.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
        metric (str): The metric to use for distance computation.
        labels (numpy.ndarray): Cluster labels for each point. Noisy samples are labeled as -1.
        core_sample_indices (numpy.ndarray): Indices of core samples.
        components (numpy.ndarray): Copy of each core sample found by DBSCAN.
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Initialize the DBSCAN model.
        
        Args:
            eps (float): The maximum distance between two samples for them to be considered neighbors.
                Default is 0.5.
            min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
                Default is 5.
            metric (str): The metric to use for distance computation. Currently only 'euclidean' is supported.
                Default is 'euclidean'.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels = None
        self.core_sample_indices = None
        self.components = None
    
    def fit(self, X):
        """
        Perform DBSCAN clustering.
        
        Args:
            X (numpy.ndarray): Training data, shape (n_samples, n_features).
            
        Returns:
            self: The fitted estimator.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Initialize all points as 'unvisited'
        self.labels = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)
        
        # Find neighbors for all points
        neighbors = self._find_neighbors(X)
        
        # Find core samples
        core_samples = np.array([i for i, neighbors_i in enumerate(neighbors) 
                               if len(neighbors_i) >= self.min_samples])
        
        # Initialize cluster label
        current_cluster = 0
        
        # Iterate through all core samples and expand clusters
        for i in core_samples:
            if visited[i]:
                continue
                
            # Mark as visited
            visited[i] = True
            
            # Assign to cluster
            self.labels[i] = current_cluster
            
            # Get neighbors
            seeds = neighbors[i].copy()
            
            # Expand cluster
            while seeds:
                # Get a point from the seeds
                j = seeds.pop()
                
                # If not visited, mark as visited and add its neighbors to seeds
                if not visited[j]:
                    visited[j] = True
                    
                    # If j is a core point, add its neighbors to seeds
                    if len(neighbors[j]) >= self.min_samples:
                        seeds.update(neighbors[j])
                
                # If j is not yet a member of a cluster, add it to the current cluster
                if self.labels[j] == -1:
                    self.labels[j] = current_cluster
            
            # Move to the next cluster
            current_cluster += 1
        
        # Store core sample indices and components
        self.core_sample_indices = np.where(np.array([len(n) for n in neighbors]) >= self.min_samples)[0]
        self.components = X[self.core_sample_indices]
        
        return self
    
    def _find_neighbors(self, X):
        """
        Find all neighbors for each point within eps distance.
        
        Args:
            X (numpy.ndarray): Input data, shape (n_samples, n_features).
            
        Returns:
            list: List of sets, where each set contains the indices of neighbors for a point.
        """
        n_samples = X.shape[0]
        neighbors = [set() for _ in range(n_samples)]
        
        # Compute pairwise distances and find neighbors
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Compute distance between points i and j
                if self.metric == 'euclidean':
                    distance = np.linalg.norm(X[i] - X[j])
                else:
                    raise ValueError(f"Unsupported metric: {self.metric}")
                
                # If distance is less than eps, add to neighbors
                if distance <= self.eps:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        return neighbors
    
    def fit_predict(self, X):
        """
        Perform DBSCAN clustering and return cluster labels.
        
        Args:
            X (numpy.ndarray): Training data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Cluster labels for each point. Noisy samples are labeled as -1.
        """
        self.fit(X)
        return self.labels


class AgglomerativeClustering:
    """
    Hierarchical Clustering using a bottom-up approach.
    
    Agglomerative Clustering is a hierarchical clustering algorithm that builds
    nested clusters by merging them successively. This implementation uses
    Ward's method, which minimizes the variance of the clusters being merged.
    
    Attributes:
        n_clusters (int): The number of clusters to form.
        linkage (str): The linkage criterion to use. Currently supports 'ward', 'single',
            'complete', and 'average'.
        distance_threshold (float): The linkage distance threshold above which clusters
            will not be merged. If None, n_clusters is used.
        affinity (str): Metric used to compute the linkage. Currently only 'euclidean' is supported.
        labels (numpy.ndarray): Cluster labels for each point.
        distances (list): Distances at which each merge occurred.
        n_clusters_ (int): The number of clusters found.
    """
    
    def __init__(self, n_clusters=2, linkage='ward', distance_threshold=None, affinity='euclidean'):
        """
        Initialize the AgglomerativeClustering model.
        
        Args:
            n_clusters (int): The number of clusters to form. Used when distance_threshold is None.
                Default is 2.
            linkage (str): The linkage criterion to use: {'ward', 'single', 'complete', 'average'}.
                Default is 'ward'.
            distance_threshold (float, optional): The linkage distance threshold above which
                clusters will not be merged. If None, n_clusters is used. Default is None.
            affinity (str): Metric used to compute the linkage. Currently only 'euclidean' is supported.
                Default is 'euclidean'.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.affinity = affinity
        self.labels = None
        self.distances = None
        self.n_clusters_ = None
    
    def fit(self, X):
        """
        Fit the hierarchical clustering from features.
        
        Args:
            X (numpy.ndarray): Training data, shape (n_samples, n_features).
            
        Returns:
            self: The fitted estimator.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Initialize each point as its own cluster
        clusters = [{i} for i in range(n_samples)]
        self.labels = np.arange(n_samples)
        self.distances = []
        
        # Compute initial distance matrix
        distance_matrix = self._compute_distance_matrix(X)
        
        # Main clustering loop
        while len(clusters) > 1:
            # Find the two closest clusters
            min_dist = float('inf')
            min_i, min_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Compute the distance between clusters i and j
                    dist = self._cluster_distance(clusters[i], clusters[j], distance_matrix)
                    
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = i, j
            
            # Record the distance at which the merge occurred
            self.distances.append(min_dist)
            
            # Check if we should stop merging based on distance threshold
            if self.distance_threshold is not None and min_dist > self.distance_threshold:
                break
                
            # Merge the two clusters
            new_cluster = clusters[min_i] | clusters[min_j]
            new_cluster_label = len(self.distances) - 1
            
            # Update labels
            for idx in new_cluster:
                self.labels[idx] = new_cluster_label
            
            # Update clusters list
            clusters.pop(max(min_i, min_j))
            clusters.pop(min(min_i, min_j))
            clusters.append(new_cluster)
            
            # Check if we have reached the desired number of clusters
            if self.distance_threshold is None and len(clusters) <= self.n_clusters:
                break
        
        # Relabel clusters from 0 to n_clusters-1
        unique_labels = np.unique(self.labels)
        mapping = {label: i for i, label in enumerate(unique_labels)}
        self.labels = np.array([mapping[label] for label in self.labels])
        self.n_clusters_ = len(unique_labels)
        
        return self
    
    def _compute_distance_matrix(self, X):
        """
        Compute the pairwise distance matrix.
        
        Args:
            X (numpy.ndarray): Input data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Distance matrix, shape (n_samples, n_samples).
        """
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if self.affinity == 'euclidean':
                    distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
                else:
                    raise ValueError(f"Unsupported affinity: {self.affinity}")
                
                # Distance matrix is symmetric
                distance_matrix[j, i] = distance_matrix[i, j]
        
        return distance_matrix
    
    def _cluster_distance(self, cluster1, cluster2, distance_matrix):
        """
        Compute the distance between two clusters.
        
        Args:
            cluster1 (set): Indices of points in the first cluster.
            cluster2 (set): Indices of points in the second cluster.
            distance_matrix (numpy.ndarray): Pairwise distance matrix.
            
        Returns:
            float: Distance between the two clusters.
        """
        if self.linkage == 'single':
            # Single linkage: minimum distance between any points in the clusters
            return min(distance_matrix[i, j] for i in cluster1 for j in cluster2)
            
        elif self.linkage == 'complete':
            # Complete linkage: maximum distance between any points in the clusters
            return max(distance_matrix[i, j] for i in cluster1 for j in cluster2)
            
        elif self.linkage == 'average':
            # Average linkage: average distance between all pairs of points in the clusters
            return sum(distance_matrix[i, j] for i in cluster1 for j in cluster2) / (len(cluster1) * len(cluster2))
            
        elif self.linkage == 'ward':
            # Ward's method: increase in variance when merging the clusters
            # This is a simplified implementation that doesn't compute the true Ward criterion
            distances = [distance_matrix[i, j] for i in cluster1 for j in cluster2]
            return np.mean(distances) * len(cluster1) * len(cluster2) / (len(cluster1) + len(cluster2))
        
        else:
            raise ValueError(f"Unsupported linkage: {self.linkage}")
    
    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.
        
        Args:
            X (numpy.ndarray): Training data, shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Cluster labels for each point.
        """
        self.fit(X)
        return self.labels 