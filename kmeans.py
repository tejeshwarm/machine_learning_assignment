import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100, tolerance=1e-4):
        self.k = k  # number of clusters
        self.max_iters = max_iters  # maximum number of iterations
        self.tolerance = tolerance  # tolerance to declare convergence

    def fit(self, data):
        self.data = data
        n_samples, n_features = data.shape
        
        # Randomly initialize centroids from the data
        np.random.seed(42)
        self.centroids = data[np.random.choice(n_samples, self.k, replace=False)]
        initial_centroids = self.centroids.copy()
        
        for epoch in range(self.max_iters):
            # Assign clusters
            self.labels = self._assign_clusters(self.centroids)

            # Recalculate centroids
            new_centroids = np.array([self.data[self.labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence (centroid changes below tolerance)
            if np.linalg.norm(new_centroids - self.centroids) < self.tolerance:
                print(f"Converged at epoch {epoch}")
                break

            self.centroids = new_centroids

        self.epochs = epoch + 1  # Total number of epochs
        self.error = self._calculate_error()  # Final error rate (SSE)
        
        return initial_centroids, self.centroids, self.epochs, self.error

    def _assign_clusters(self, centroids):
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_error(self):
        sse = 0
        for i in range(self.k):
            cluster_data = self.data[self.labels == i]
            sse += np.sum((cluster_data - self.centroids[i]) ** 2)
        return sse

# Example usage
if __name__ == "__main__":
    # Create a simple dataset
    data = np.random.rand(10, 2)

    # Initialize and fit KMeans
    kmeans = KMeans(k=3)
    initial_centroids, final_centroids, epochs, error = kmeans.fit(data)

    # Print results
    print("Initial centroids:")
    print(initial_centroids)
    print("\nFinal centroids:")
    print(final_centroids)
    print(f"\nTotal epochs: {epochs}")
    print(f"\nFinal error (SSE): {error}")
    
    # Optional: Plot the final clusters
    labels = kmeans._assign_clusters(final_centroids)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(final_centroids[:, 0], final_centroids[:, 1], s=300, c='red')
    plt.title("Final Clusters")
    plt.show()
