import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Assuming your 'data' is a list of 18 lists, each with 3879 data points
# You can convert it to a NumPy array for better processing
data = np.load('data.npy')
data_array = np.array(data)

# Reshape the data to (18 * 3879, 1) for clustering
reshaped_data = data_array.reshape(-1, 1)

# Standardize the data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(reshaped_data)

# Apply MiniBatchKMeans clustering
minibatch_kmeans = MiniBatchKMeans(n_clusters=3, batch_size=1000, random_state=42)
clusters = minibatch_kmeans.fit_predict(standardized_data)

# Visualize the clustered data
plt.scatter(range(len(standardized_data)), standardized_data, c=clusters, cmap='viridis', s=5)
plt.title('MiniBatchKMeans Clustering')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.show()

# Evaluate the clustering using Davies-Bouldin index
db_index = davies_bouldin_score(standardized_data, clusters)
print(f'Davies-Bouldin Index (MiniBatchKMeans): {db_index}')
