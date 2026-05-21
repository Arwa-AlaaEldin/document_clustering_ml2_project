"""
Pipelines, Algorithms:
  1. TF-IDF + LSA  -> KMeans, Hierarchical, GMM
  2. TF-IDF + UMAP -> KMeans, Hierarchical, GMM
  3. SBERT + UMAP -> KMeans, Hierarchical, GMM

Results are saved to results/<dataset>/results.pkl for Streamlit.
The best result is visualised and saved to figures/<dataset>/best_result.png

"""


import numpy as np
import argparse
import os
import pickle
import time
from preprocessing.data_loader import load_newsgroups, load_wikipedia
from preprocessing.feature_extractor import build_tfidf, apply_lsa, apply_umap, build_sbert, top_terms_per_cluster
from cluster import run_kmeans, run_hierarchical, run_gmm, find_best_k
from evaluation import compute_metrics, best_result, print_summary

RANDOM_STATE = 42


# helpers
def _banner(msg):
    print(f"\n{'━' * 60}")
    print(f"  {msg}")
    print(f"{'━' * 60}")


def _elapsed(t0):
    s = time.perf_counter() - t0
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


# core pipeline
# run all 9 combinations and return a list of result dicts
# each dict contains: pipeline, algorithm, n_clusters, labels, metrics, top_terms, X (embedding used), dataset
def run_pipeline(dataset, df, k, results_dir="results", figures_dir="figures", visualise=True):
    
    texts = df["clean_text"].tolist()
    y_true = df["label"].values if "label" in df.columns else None
    all_results = []

    # Step 1: TF-IDF
    _banner("Step 1 · TF-IDF")
    tfidf_vec, X_tfidf = build_tfidf(texts, dataset=dataset)

    # Step 2: LSA
    _banner("Step 2 · TF-IDF + LSA → cluster")
    n_lsa = 100 if dataset == "newsgroups" else 200
    X_lsa = apply_lsa(X_tfidf, n_components=n_lsa, random_state=RANDOM_STATE)

    for algo_fn, algo_name in [
        (lambda X: run_kmeans(X, k, RANDOM_STATE), "KMeans"),
        (lambda X: run_hierarchical(X, k), "Hierarchical"),
        (lambda X: run_gmm(X, k, RANDOM_STATE), "GMM"),
    ]:
        print(f"\n[TF-IDF+LSA] Running {algo_name}…")
        t0 = time.perf_counter()
        res = algo_fn(X_lsa)
        elapsed = time.perf_counter() - t0

        top_terms = top_terms_per_cluster(tfidf_vec, X_tfidf, res["labels"])
        metrics = compute_metrics(X_lsa, res["labels"], y_true, random_state=RANDOM_STATE)
        print(f"  silhouette={metrics.get('silhouette', float('nan')):.4f}  "
              f"nmi={metrics.get('nmi', float('nan')):.4f}  ({elapsed:.1f}s)")

        all_results.append({
            "pipeline": "TF-IDF + LSA",
            "algorithm": res["algorithm"],
            "n_clusters": res["n_clusters"],
            "labels": res["labels"],
            "metrics": metrics,
            "top_terms": top_terms,
            "X": X_lsa,
            "dataset": dataset,
        })

    # Step 3: TF-IDF + UMAP
    _banner("Step 3 · TF-IDF + UMAP → cluster")
    # use 50D UMAP for clustering (preserves structure better than 2D)
    X_tfidf_umap = apply_umap(X_tfidf, n_components=50, metric="cosine", n_neighbors=30, random_state=RANDOM_STATE)

    for algo_fn, algo_name in [
        (lambda X: run_kmeans(X, k, RANDOM_STATE),  "KMeans"),
        (lambda X: run_hierarchical(X, k), "Hierarchical"),
        (lambda X: run_gmm(X, k, RANDOM_STATE), "GMM"),
    ]:
        print(f"\n[TF-IDF+UMAP] Running {algo_name}…")
        t0 = time.perf_counter()
        res = algo_fn(X_tfidf_umap)
        elapsed = time.perf_counter() - t0

        top_terms = top_terms_per_cluster(tfidf_vec, X_tfidf, res["labels"])
        metrics = compute_metrics(X_tfidf_umap, res["labels"], y_true, random_state=RANDOM_STATE)
        print(f"  silhouette={metrics.get('silhouette', float('nan')):.4f}  "
              f"nmi={metrics.get('nmi', float('nan')):.4f}  ({elapsed:.1f}s)")

        all_results.append({
            "pipeline": "TF-IDF + UMAP",
            "algorithm": res["algorithm"],
            "n_clusters": res["n_clusters"],
            "labels": res["labels"],
            "metrics": metrics,
            "top_terms": top_terms,
            "X": X_tfidf_umap,
            "dataset": dataset,
        })

    # Step 4: SBERT + UMAP
    _banner("Step 4 · SBERT + UMAP → cluster")
    sbert_cache = os.path.join("cache", f"{dataset}_sbert.npy")
    X_sbert = build_sbert(texts, cache_path=sbert_cache)

    # UMAP on SBERT embeddings (50D for clustering)
    X_sbert_umap = apply_umap(X_sbert, n_components=50, metric="cosine", n_neighbors=30, random_state=RANDOM_STATE)

    for algo_fn, algo_name in [
        (lambda X: run_kmeans(X, k, RANDOM_STATE), "KMeans"),
        (lambda X: run_hierarchical(X, k), "Hierarchical"),
        (lambda X: run_gmm(X, k, RANDOM_STATE), "GMM"),
    ]:
        print(f"\n[SBERT+UMAP] Running {algo_name}…")
        t0 = time.perf_counter()
        res = algo_fn(X_sbert_umap)
        elapsed = time.perf_counter() - t0

        # top terms: fall back to TF-IDF for interpretability
        top_terms = top_terms_per_cluster(tfidf_vec, X_tfidf, res["labels"])
        metrics = compute_metrics(X_sbert_umap, res["labels"], y_true, random_state=RANDOM_STATE)
        print(f"  silhouette={metrics.get('silhouette', float('nan')):.4f}  "
              f"nmi={metrics.get('nmi', float('nan')):.4f}  ({elapsed:.1f}s)")

        all_results.append({
            "pipeline":   "SBERT + UMAP",
            "algorithm":  res["algorithm"],
            "n_clusters": res["n_clusters"],
            "labels":     res["labels"],
            "metrics":    metrics,
            "top_terms":  top_terms,
            "X":          X_sbert_umap,
            "dataset":    dataset,
        })

    # summary & best result
    _banner("Summary")
    print_summary(all_results)

    best = best_result(all_results, metric="silhouette")
    print(f"\n★ Best: {best['pipeline']} + {best['algorithm']}  "
          f"silhouette={best['metrics'].get('silhouette', float('nan')):.4f}  "
          f"nmi={best['metrics'].get('nmi', float('nan')):.4f}")

    # save results
    out_dir = os.path.join(results_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    # strip large X arrays before saving (keep labels + metrics only)
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "X"}
        save_results.append(sr)

    results_path = os.path.join(out_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(save_results, f)
    print(f"\n[Save] Results → {results_path}")

    # also add doc texts and category info for Streamlit display
    meta_path = os.path.join(out_dir, "meta.pkl")
    meta = {
        "texts": df["clean_text"].tolist(),
        "raw_texts": df["text"].tolist() if "text" in df.columns else df["clean_text"].tolist(),
        "labels": y_true,
        "category": df["category"].tolist() if "category" in df.columns else None,
        "names": df["name"].tolist() if "name" in df.columns else None,
        "dataset": dataset,
        "k": k,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[Save] Meta     → {meta_path}")

    # visualise best
    if visualise:
        _banner("Visualising best result")
        from visualization import plot_best_result
        plot_best_result(best, dataset=dataset, figures_dir=figures_dir, random_state=RANDOM_STATE)

    return all_results


def main():
    p = argparse.ArgumentParser(description="Document clustering pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="newsgroups", choices=["newsgroups", "wikipedia"])
    p.add_argument("--data_path",   default=None, help="Path to Wikipedia CSV/JSON/JSONL (required if dataset=wikipedia)")
    p.add_argument("--k", type=int, default=None, help="Number of clusters. If omitted, auto-selected via silhouette sweep.")
    p.add_argument("--k_min", type=int, default=4)
    p.add_argument("--k_max", type=int, default=15)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--figures_dir", default="figures")
    p.add_argument("--no_visualise", action="store_true")
    args = p.parse_args()

    wall_t0 = time.perf_counter()
    _banner(f"Document Clustering Pipeline — {args.dataset}")

    # Load data
    if args.dataset == "newsgroups":
        df = load_newsgroups()
    else:
        if not args.data_path:
            raise ValueError("--data_path required for wikipedia dataset")
        df = load_wikipedia(args.data_path)

    print(f"\nDataset: {len(df):,} documents")

    # auto-select k if not provided 
    if args.k is None:
        _banner("Auto-selecting k via silhouette sweep (TF-IDF + LSA)")
        from preprocessing.feature_extractor import build_tfidf, apply_lsa
        _, X_tfidf_tmp = build_tfidf(df["clean_text"].tolist(), dataset=args.dataset)
        X_lsa_tmp = apply_lsa(X_tfidf_tmp, n_components=100, random_state=RANDOM_STATE)
        k = find_best_k(X_lsa_tmp, k_range=range(args.k_min, args.k_max + 1), random_state=RANDOM_STATE)
    else:
        k = args.k

    print(f"\nUsing k={k}")

    # Run
    run_pipeline(
        dataset=args.dataset,
        df=df,
        k=k,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        visualise=not args.no_visualise,
    )

    print(f"\n✓ Total time: {_elapsed(wall_t0)}")


if __name__ == "__main__":
    main()