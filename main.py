import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from ucimlrepo import fetch_ucirepo


# Coleta de dados do ucimlrepo
breast_cancer_winsconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_winsconsin_diagnostic.data.features.loc[
    :, "radius1":"texture1"
].values
y = breast_cancer_winsconsin_diagnostic.data.targets.values.flatten()

# Convertendo a coluna diagnosticos para binário
unique_labels = np.unique(y)
mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
y_binary = np.array([mapping[label] for label in y])

k = 2 # Assumindo 2 categorias de diagnśticos e clusters 

kmeans_sklearn = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_sklearn.fit(X)
cluster_labels = kmeans_sklearn.labels_

import itertools

unique_clusters = np.unique(cluster_labels)
num_clusters = len(unique_clusters)

if num_clusters == 2:
    permutations = list(itertools.permutations([0, 1]))
    max_jaccard = 0
    for p in permutations:
        mapped_labels = np.array([p[c] for c in cluster_labels])
        jaccard = jaccard_score(y_binary, mapped_labels)
        if jaccard > max_jaccard:
            max_jaccard = jaccard
    print(f"Jaccard Score: {max_jaccard:.4f}")
else:
    print("O número de clusters é maior que 2")
    print("COnsidere usar apenas duas n_clusters")
