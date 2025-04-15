import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ucimlrepo import fetch_ucirepo


# fetch dataset
breast_cancer_winsconsin_diagnostic = fetch_ucirepo(id=17)


X = breast_cancer_winsconsin_diagnostic.data.features.loc[
    :, "radius1":"texture1"
].values

# Visualização inicial
plt.scatter(X[:, 0], X[:, 1], s=50, c="gray")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dados Brutos")
plt.show()

k = 3  # Número de clusters
# Executa o algoritmo K-Means.
n_samples = X.shape[0]
random_indices = np.random.choice(n_samples, k, replace=False)
centroids = X[random_indices]
max_iters = 100
for i in range(max_iters):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    new_centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])

    # Verifica convergência
    if np.all(np.abs(new_centroids - centroids) < 1e-4):
        break
    centroids = new_centroids

# Visualização dos Clusters
for i in range(k):
    plt.scatter(
        X[cluster_labels == i, 0], X[cluster_labels == i, 1], label=f"Cluster {i + 1}"
    )
plt.scatter(
    centroids[:, 0], centroids[:, 1], s=200, c="red", marker="X", label="Centroides"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clusters Formados")
plt.legend()
plt.show()

kmeans_sklearn = KMeans(n_clusters=k, random_state=42)
kmeans_sklearn.fit(X)

# Visualização dos Clusters com scikit-learn
# plt.scatter(X[:, 0], X[:, 1], c=kmeans_sklearn.labels_, cmap='viridis', s=50)
# plt.scatter(kmeans_sklearn.cluster_centers_[:, 0], kmeans_sklearn.cluster_centers_[:, 1],
#            s=200, c='red', marker='X', label='Centroides')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Clusters com scikit-learn')
# plt.legend()
# plt.show()
