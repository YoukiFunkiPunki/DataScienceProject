import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score

# Step 1: Load and explore the dataset
data = np.load('data.npy')

data_array = np.array(data)

# Reshape the data to (18 * 3879, 1) for clustering
reshaped_data = data_array.reshape(-1, 1)

# Determine the number of clusters using the Elbow Method
inertia = []
possible_clusters = range(1, 11)  # You can adjust the range based on your data

for n_clusters in possible_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reshaped_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(possible_clusters, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Apply K-means clustering with the chosen number of clusters
chosen_k = 4
kmeans = KMeans(n_clusters=chosen_k, random_state=42)
clusters = kmeans.fit_predict(reshaped_data)

# Visualize the clustered data
plt.scatter(range(len(reshaped_data)), reshaped_data, c=clusters, cmap='viridis', s=5)
plt.title('K-means Clustering')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.show()

# Evaluate the clustering using Davies-Bouldin index
db_index = davies_bouldin_score(reshaped_data, clusters)
print(f'Davies-Bouldin Index: {db_index}')
