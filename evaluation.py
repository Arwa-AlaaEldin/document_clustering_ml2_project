import numpy as np
from sklearn.metrics import silhouette_score


# return dict with silhouette key
def compute_metrics(X, labels, y_true: np.ndarray | None = None, sample_size=5000, random_state=42,):

    n_unique = len(set(labels))
    if n_unique < 2:
        return {}

    # subsample for efficiency on large corpora
    if len(labels) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(labels), size=sample_size, replace=False)
        X_s, l_s = X[idx], labels[idx]
    else:
        X_s, l_s = X, labels

    return {
        "silhouette": silhouette_score(X_s, l_s),
    }


# pick the result with the highest silhouette score
def best_result(all_results: list[dict], metric="silhouette"):

    valid = [r for r in all_results if metric in r.get("metrics", {})]
    if not valid:
        return all_results[0]
    return max(valid, key=lambda r: r["metrics"][metric])


# print a compact comparison table
def print_summary(results: list[dict]):
  
    header = f"{'Pipeline':<30} {'Algo':<22} {'k':>4} {'Silhouette':>12}"
    print("\n" + header)
    print("─" * len(header))
    for r in results:
        m = r.get("metrics", {})
        print(
            f"{r['pipeline']:<30} {r['algorithm']:<22} {r['n_clusters']:>4} "
            f"{m.get('silhouette', float('nan')):>12.4f}"
        )
