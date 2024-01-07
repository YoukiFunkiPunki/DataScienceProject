import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score

# Step 1: Load and explore the dataset
data = np.load('data.npy')
data_array = np.array(data)

# Reshape the data to (18 * 3879, 1) for clustering
reshaped_data = data_array.reshape(-1, 1)

# Apply Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=4, random_state=42)
clusters = gmm.fit_predict(reshaped_data)

# Visualize the clustered data
plt.scatter(range(len(reshaped_data)), reshaped_data, c=clusters, cmap='viridis', s=5)
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.show()

# Evaluate the clustering using Davies-Bouldin index
db_index = davies_bouldin_score(reshaped_data, clusters)
print(f'Davies-Bouldin Index (GMM): {db_index}')
