import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage="single"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, data):
        self.data = data
        n_samples = data.shape[0]
        
        # Compute the pairwise distance matrix
        self.dist_matrix = squareform(pdist(data, metric='euclidean'))
        np.fill_diagonal(self.dist_matrix, np.inf)  # Ignore self-distances
        
        # Start each point in its own cluster
        self.clusters = {i: [i] for i in range(n_samples)}
        self.epochs = 0
        self.initial_clusters = self.clusters.copy()
        
        # Hierarchical clustering process
        while len(self.clusters) > self.n_clusters:
            self._merge_closest_clusters()
            self.epochs += 1
        
        self.final_clusters = self.clusters
        self.error = self._calculate_sse()  # Final error rate (SSE)

        return self.initial_clusters, self.final_clusters, self.epochs, self.error

    def _merge_closest_clusters(self):
        # Find the closest pair of clusters
        cluster_ids = list(self.clusters.keys())
        closest_pair = None
        min_dist = np.inf
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                dist = self._cluster_distance(cluster_ids[i], cluster_ids[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (cluster_ids[i], cluster_ids[j])
        
        # Merge the closest pair
        c1, c2 = closest_pair
        self.clusters[c1] = self.clusters[c1] + self.clusters[c2]
        del self.clusters[c2]

    def _cluster_distance(self, c1, c2):
        # Linkage method determines how to compute distance between clusters
        points_c1 = self.clusters[c1]
        points_c2 = self.clusters[c2]
        if self.linkage == "single":
            return np.min(self.dist_matrix[np.ix_(points_c1, points_c2)])
        elif self.linkage == "complete":
            return np.max(self.dist_matrix[np.ix_(points_c1, points_c2)])
        elif self.linkage == "average":
            return np.mean(self.dist_matrix[np.ix_(points_c1, points_c2)])
        else:
            raise ValueError(f"Unknown linkage type: {self.linkage}")

    def _calculate_sse(self):
        sse = 0
        for cluster in self.clusters.values():
            if len(cluster) > 1:
                cluster_data = self.data[cluster]
                centroid = np.mean(cluster_data, axis=0)
                sse += np.sum((cluster_data - centroid) ** 2)
        return sse

# Example usage
if __name__ == "__main__":
    # Create a simple dataset
    data = np.random.rand(10, 2)

    # Initialize and fit Hierarchical Clustering
    hc = HierarchicalClustering(n_clusters=3, linkage="single")
    initial_clusters, final_clusters, epochs, error = hc.fit(data)

    # Print results
    print("Initial clusters:")
    print(initial_clusters)
    print("\nFinal clusters:")
    print(final_clusters)
    print(f"\nTotal merges (epochs): {epochs}")
    print(f"\nFinal error (SSE): {error}")
    
    # Optional: Visualize the clusters
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, cluster in enumerate(final_clusters.values()):
        cluster_data = data[cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i % len(colors)], label=f'Cluster {i+1}')
    
    plt.title("Final Clusters after Hierarchical Clustering")
    plt.legend()
    plt.show()
