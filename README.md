# 📄 Text Document Clustering — NLP Pipeline Comparison

A systematic study comparing **9 end-to-end pipelines** for unsupervised text clustering across two real-world datasets, evaluating the impact of different vectorization, dimensionality reduction, and clustering strategies.

---

## 📌 Project Overview

This project investigates how different combinations of text representation, dimensionality reduction, and clustering algorithms affect the quality of unsupervised document clustering. The goal is to identify which pipeline components contribute most to meaningful cluster formation — measured by **Silhouette Score**.

Two benchmark datasets were used:
- **Wikipedia** articles (14 true topic clusters)
- **20 Newsgroups** (15 newsgroup categories)

---

## 🔧 Pipeline Components

Each pipeline is composed of three stages:

```
Text → [Vectorizer] → [Dimensionality Reduction] → [Clustering Algorithm]
```

### 1. Vectorizers

| Method | Description |
|--------|-------------|
| **TF-IDF** | Term Frequency–Inverse Document Frequency. Converts raw text into sparse numerical vectors based on word importance across the corpus. Fast and interpretable, but loses semantic meaning and word order. |
| **SBERT** | Sentence-BERT (Sentence Transformers). Uses a pretrained transformer model to produce dense semantic embeddings (768-dim). Captures contextual meaning, synonymy, and sentence-level semantics far beyond bag-of-words approaches. |

### 2. Dimensionality Reduction

| Method | Description |
|--------|-------------|
| **LSA** (Latent Semantic Analysis) | Applies Truncated SVD to TF-IDF vectors. Reduces sparse high-dimensional matrices into a lower-dimensional latent semantic space (~100 dims). Captures co-occurrence patterns but is linear and limited in expressiveness. |
| **UMAP** (Uniform Manifold Approximation and Projection) | A non-linear manifold learning technique. Projects high-dimensional vectors into a compact 2D/nD space while preserving both local and global structure. Far superior for clustering compared to PCA/LSA, especially with dense embeddings. |

### 3. Clustering Algorithms

| Method | Description |
|--------|-------------|
| **KMeans** | Partitions documents into *k* spherical clusters by minimizing within-cluster variance. Fast and scalable, but assumes convex clusters and requires knowing *k* in advance. |
| **Hierarchical (Ward)** | Agglomerative clustering using Ward linkage, which minimizes within-cluster variance at each merge step. Produces a dendrogram; no random initialization; deterministic. |
| **GMM** (Gaussian Mixture Model) | Probabilistic model that assumes data is generated from a mixture of Gaussians. Provides soft cluster assignments and handles elliptical cluster shapes. More flexible than KMeans. |

---

## 🧪 Experimental Pipelines

9 pipelines were evaluated on each dataset:

| # | Vectorizer | Dim. Reduction | Clustering |
|---|-----------|----------------|------------|
| 0 | TF-IDF | LSA | KMeans |
| 1 | TF-IDF | LSA | Hierarchical (Ward) |
| 2 | TF-IDF | LSA | GMM |
| 3 | TF-IDF | UMAP | KMeans |
| 4 | TF-IDF | UMAP | Hierarchical (Ward) |
| 5 | TF-IDF | UMAP | GMM |
| 6 | SBERT | UMAP | KMeans |
| 7 | SBERT | UMAP | Hierarchical (Ward) |
| 8 | SBERT | UMAP | GMM |

---

## 📊 Results

### Metric: Silhouette Score
The **Silhouette Score** ranges from -1 to 1. A higher score indicates that documents are well-matched to their own cluster and clearly separated from neighboring clusters. Scores above **0.4** are generally considered strong for text data.

---

### Wikipedia Dataset (k = 14)

| # | Pipeline | Silhouette Score |
|---|----------|-----------------|
| 0 | TF-IDF + LSA + KMeans | 0.0645 |
| 1 | TF-IDF + LSA + Hierarchical | 0.0454 |
| 2 | TF-IDF + LSA + GMM | 0.0589 |
| 3 | **TF-IDF + UMAP + KMeans** | **0.4502** ✅ |
| 4 | TF-IDF + UMAP + Hierarchical | 0.4379 |
| 5 | TF-IDF + UMAP + GMM | 0.4466 |
| 6 | SBERT + UMAP + KMeans | 0.4335 |
| 7 | SBERT + UMAP + Hierarchical | 0.4444 |
| 8 | SBERT + UMAP + GMM | 0.3832 |

**Best:** TF-IDF + UMAP + KMeans → `0.4502`

---

### 20 Newsgroups Dataset (k = 15)

| # | Pipeline | Silhouette Score |
|---|----------|-----------------|
| 0 | TF-IDF + LSA + KMeans | 0.0532 |
| 1 | TF-IDF + LSA + Hierarchical | 0.0305 |
| 2 | TF-IDF + LSA + GMM | 0.0496 |
| 3 | TF-IDF + UMAP + KMeans | 0.3865 |
| 4 | TF-IDF + UMAP + Hierarchical | 0.3645 |
| 5 | TF-IDF + UMAP + GMM | 0.3683 |
| 6 | **SBERT + UMAP + KMeans** | **0.4380** ✅ |
| 7 | SBERT + UMAP + Hierarchical | 0.4276 |
| 8 | SBERT + UMAP + GMM | 0.3768 |

**Best:** SBERT + UMAP + KMeans → `0.4380`

---

## 📈 Visualizations

Each pipeline was visualized using **2D UMAP projections** of the document embeddings, color-coded by predicted cluster label. These plots reveal:

- **LSA pipelines**: Scattered, overlapping point clouds with no clear visual separation — consistent with near-zero silhouette scores.
- **TF-IDF + UMAP pipelines**: Compact, well-separated island clusters. UMAP successfully compresses the TF-IDF space into a meaningful layout.
- **SBERT + UMAP pipelines**: Tight semantic clusters with smooth intra-cluster density. Semantic similarity drives coherent groupings even for topically adjacent documents.

> Visualization files are located in the `figures/` directory

---

## 💡 Key Findings & Analysis

### Finding 1 — UMAP is the single most impactful component
Switching from LSA to UMAP as the dimensionality reduction step caused a **~6× improvement** in Silhouette Score across all clustering algorithms (e.g., KMeans on Wikipedia: 0.0645 → 0.4502). This is the largest single gain observed in the entire study.

**Why?** LSA performs linear dimensionality reduction and cannot capture the non-linear manifold structure in text embeddings. UMAP, being non-linear, preserves local neighborhood structure and creates compact cluster geometries that clustering algorithms can cleanly partition.

### Finding 2 — TF-IDF + UMAP is surprisingly competitive with SBERT + UMAP
On Wikipedia, TF-IDF + UMAP slightly *outperforms* SBERT + UMAP. On Newsgroups, SBERT + UMAP is the winner. The difference is narrow (~0.01–0.05), suggesting that once UMAP compresses the representation, the gap between TF-IDF and SBERT narrows significantly.

**Why?** UMAP can extract a clean manifold even from TF-IDF's sparse high-dimensional space. For short, topically distinct documents (like Wikipedia articles), term-frequency signals are already highly discriminative.

### Finding 3 — SBERT shines on semantically nuanced datasets
On 20 Newsgroups — a dataset with overlapping topics (e.g., *comp.graphics* vs *comp.os.ms-windows*) — SBERT + UMAP achieves the highest score (0.4380 vs 0.3865 for TF-IDF + UMAP + KMeans). Semantic embeddings better separate documents that share vocabulary but differ in meaning.

### Finding 4 — KMeans is the most reliable clustering algorithm
Across both datasets, KMeans consistently achieves the highest or near-highest score. Its centroid-based approach aligns well with UMAP's compact cluster shapes. Hierarchical (Ward) comes close. GMM slightly underperforms, possibly due to the non-Gaussian shapes produced by UMAP projections.

### Finding 5 — LSA is not suitable for high-quality clustering
All three LSA-based pipelines score below 0.07 on both datasets. This is essentially random clustering. LSA's linear compression cannot produce a geometry that any of the three clustering algorithms can meaningfully exploit.

---

## 🗂️ Project Structure

```text
DOC_CLUSTERING_FINAL/
├── cache/
│   ├── newsgroups_all_True.pkl
│   ├── newsgroups_all.parquet
│   ├── newsgroups_sbert.npy
│   ├── wikipedia_sbert.npy
│   └── wikipedia.parquet
│
├── figures/
│   ├── newsgroups/
│   └── wikipedia/
│
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py
│   └── feature_extractor.py
│
├── results/
│   ├── newsgroups/
│   └── wikipedia/
│
├── app.py                 # Streamlit dashboard
├── cluster.py             # Clustering algorithms
├── evaluation.py          # Evaluation metrics & reports
├── pipeline.py            # End-to-end clustering pipelines
├── visualization.py       # UMAP visualizations & plotting
│
└── README.md
```

---

## 🛠️ Dependencies

```
scikit-learn
sentence-transformers
umap-learn
numpy
pandas
matplotlib
seaborn
```

Install with:
```bash
pip install scikit-learn sentence-transformers umap-learn numpy pandas matplotlib seaborn
```

---

## 🏁 Conclusion

The clearest takeaway from this study is that **dimensionality reduction quality matters more than the choice of clustering algorithm**. UMAP is a critical enabler — without it, even the best vectorizer (SBERT) cannot produce clusters that any standard algorithm can reliably partition.

For production text clustering systems:
- Use **SBERT + UMAP + KMeans** when semantic precision is required (overlapping topics, nuanced language).
- Use **TF-IDF + UMAP + KMeans** when speed and interpretability matter and topics are topically distinct.
- Avoid **LSA** as a dimensionality reduction step for clustering tasks.
