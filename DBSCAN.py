import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score

# Step 1: Load and explore the dataset
data = np.load('data.npy')
data_array = np.array(data)

# Reshape the data to (18 * 3879, 1) for clustering
reshaped_data = data_array.reshape(-1, 1)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(reshaped_data)

# Visualize the clustered data
plt.scatter(range(len(reshaped_data)), reshaped_data, c=clusters, cmap='viridis', s=5)
plt.title('DBSCAN Clustering')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.show()

# Evaluate the clustering using Davies-Bouldin index
db_index = davies_bouldin_score(reshaped_data, clusters)
print(f'Davies-Bouldin Index (DBSCAN): {db_index}')
