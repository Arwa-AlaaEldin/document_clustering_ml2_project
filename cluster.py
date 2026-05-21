import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def run_kmeans(X, n_clusters, random_state=42):
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return {
        "labels": labels,
        "algorithm": "KMeans",
        "n_clusters": n_clusters,
        "extra": {"inertia": model.inertia_}
    }


def run_hierarchical(X, n_clusters, linkage="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return {
        "labels": labels,
        "algorithm": f"Hierarchical({linkage})",
        "n_clusters": n_clusters,
        "extra": {},
    }


def run_gmm(X, n_clusters, random_state=42):
    X = np.asarray(X, dtype=np.float64)

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        n_init=5,
        max_iter=500,
        reg_covar=1e-3,
        random_state=random_state,
    )

    model.fit(X)
    labels = model.predict(X)

    return {
        "labels": labels,
        "algorithm": "GMM",
        "n_clusters": n_clusters,
        "extra": {
            "bic": model.bic(X),
            "aic": model.aic(X)
        },
    }


def find_best_k(X, k_range=range(4, 16), random_state=42,):

    best_k, best_score = k_range.start, -1.0
    print(f"[K-search] Sweeping k={k_range.start}..{k_range.stop - 1}")
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=5, random_state=random_state).fit_predict(X)
        score = silhouette_score(X, labels, sample_size=3000, random_state=random_state)
        print(f"  k={k:3d}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"[K-search] Best k={best_k}  (silhouette={best_score:.4f})")
    return best_k