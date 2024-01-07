import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load and explore the dataset
data = np.load('data.npy')

# Data preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Determine the optimal number of clusters using the elbow method and silhouette scores
possible_clusters = range(2, 10)  # Adjust as needed
inertia_values = []
silhouette_scores = []

for n_clusters in possible_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled, cluster_labels))

# Plot elbow method and silhouette scores
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(possible_clusters, inertia_values, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(possible_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

# Print suggested optimal K based on Elbow Method and Silhouette Analysis
elbow_optimal_k = np.argmin(np.diff(inertia_values)) + 2  # Adding 2 because range starts from 2
silhouette_optimal_k = possible_clusters[np.argmax(silhouette_scores)]

print(f'Suggested Optimal K (Elbow Method): {elbow_optimal_k}')
print(f'Suggested Optimal K (Silhouette Analysis): {silhouette_optimal_k}')

plt.show()
