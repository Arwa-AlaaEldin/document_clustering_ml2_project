"""
Pipelines:
  1. TF-IDF -> LSA
  2. TF-IDF -> UMAP
  3. SBERT -> UMAP
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import umap
from sentence_transformers import SentenceTransformer


# TF-IDF
# fit TF-IDF and return (vectorizer, sparse_matrix), hyperparameters are tuned per dataset
def build_tfidf(texts: list[str], dataset="newsgroups"):
    
    kwargs = dict(
        max_features=10_000 if dataset == "newsgroups" else 15_000,
        min_df=3, max_df=0.85,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        sublinear_tf=True,
    )
    if dataset == "wikipedia":
        kwargs["ngram_range"] = (1, 2)

    vec = TfidfVectorizer(**kwargs)
    X = vec.fit_transform(texts)
    print(f"[TF-IDF] Matrix: {X.shape[0]:,} × {X.shape[1]:,}")
    return vec, X


# LSA
# TruncatedSVD(LSA) + L2 normalisation
def apply_lsa(X_tfidf, n_components=100, random_state=42):

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_lsa = svd.fit_transform(X_tfidf)
    X_lsa = Normalizer().fit_transform(X_lsa)
    explained = svd.explained_variance_ratio_.sum()
    print(f"[LSA] {n_components} components, variance explained: {explained:.1%}")
    return X_lsa


# UMAP dimensionality reduction
def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42):
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(X)
    print(f"[UMAP] Input {X.shape} → Output {X_umap.shape}")
    return X_umap


# SBERT
# encode texts with Sentence-BERT, caches embeddings to disk
def build_sbert(texts: list[str], model_name="all-MiniLM-L6-v2", batch_size=64, cache_path: str | None = None):

    if cache_path and os.path.exists(cache_path):
        print(f"[SBERT] Loading embeddings from cache: {cache_path}")
        return np.load(cache_path)

    print(f"[SBERT] Encoding {len(texts):,} docs with '{model_name}'…")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"[SBERT] Saved embeddings to {cache_path}")

    print(f"[SBERT] Embeddings shape: {embeddings.shape}")
    return embeddings


# return top n TF-IDF terms for each cluster
def top_terms_per_cluster(vectorizer, X_tfidf, labels, n=10,):
    feature_names = np.array(vectorizer.get_feature_names_out())
    result = {}
    for cid in sorted(set(labels)):
        mask = labels == cid
        scores = np.asarray(X_tfidf[mask].sum(axis=0)).ravel()
        top_idx = scores.argsort()[::-1][:n]
        result[cid] = feature_names[top_idx].tolist()
    return result