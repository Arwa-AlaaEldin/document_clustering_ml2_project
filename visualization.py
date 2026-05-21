import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

# palette

PALETTE = [
    "#A9907E", "#ABC4AA", "#675D50", "#C4A882", "#7A9E7E",
    "#D4B896", "#8B7355", "#6B8F71", "#BFA882", "#5C7A61",
    "#E8C9A0", "#4A6741", "#9B8270", "#7FB3A3", "#C49A6C",
    "#456B6B", "#B8907A", "#8FAF7F", "#7D6655", "#5B8A8A",
]

BG = "#FDFAF6"
ESPRESSO = "#675D50"
CREAM = "#F3DEBA"
CLAY = "#A9907E"
SAGE = "#ABC4AA"


def _apply_base_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CREAM)
    ax.spines["bottom"].set_color(CREAM)
    ax.tick_params(colors=CLAY, labelsize=8)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=ESPRESSO, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=CLAY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=CLAY)


def _color_map(labels):
    unique = sorted(set(labels))
    return {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(unique)}


# UMAP
def _umap_2d(X: np.ndarray, random_state: int = 42) -> np.ndarray:

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=random_state)
    if len(X) > 15_000:
        reducer2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=random_state)
        return reducer2.fit_transform(X)
    return reducer.fit_transform(X)


def plot_umap(X, labels, random_state=42):

    if hasattr(X, "toarray"):
        X = X.toarray()
        
    X2d = X if X.shape[1] == 2 else _umap_2d(X, random_state)

    cmap = _color_map(labels)
    pt_colors = [cmap[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
    ax.scatter(X2d[:, 0], X2d[:, 1], c=pt_colors, s=6, alpha=0.55, linewidths=0)
    patches = [mpatches.Patch(color=cmap[l], label=f"C{l}") for l in sorted(cmap)]
    ax.legend(handles=patches, fontsize=7, loc="best", ncol=max(1, len(cmap) // 8), markerscale=2, framealpha=0.7, edgecolor=CREAM)
    
    _apply_base_style(ax, title="UMAP 2-D Projection", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig


# PCA
def plot_pca(X, labels, random_state=42):

    if hasattr(X, "toarray"):
        X = X.toarray()
        
    pca = PCA(n_components=2, random_state=random_state)
    X2d = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    cmap = _color_map(labels)
    pt_colors = [cmap[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
    ax.scatter(X2d[:, 0], X2d[:, 1], c=pt_colors, s=6, alpha=0.55, linewidths=0)
    patches = [mpatches.Patch(color=cmap[l], label=f"C{l}") for l in sorted(cmap)]
    ax.legend(handles=patches, fontsize=7, loc="best", ncol=max(1, len(cmap) // 8), markerscale=2, framealpha=0.7, edgecolor=CREAM)
    _apply_base_style(
        ax,
        title=f"PCA 2-D  (PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%)",
        xlabel=f"PC 1 ({ev[0]*100:.1f}%)",
        ylabel=f"PC 2 ({ev[1]*100:.1f}%)",
    )
    plt.tight_layout()
    return fig


# t-SNE
def plot_tsne(X, labels, perplexity=30, random_state=42, max_samples=8_000):

    if hasattr(X, "toarray"):
        X = X.toarray()

    if len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, labels = X[idx], labels[idx]

    print(f"[t-SNE] Fitting on {len(X)} samples")
    X2d = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1).fit_transform(X)

    cmap = _color_map(labels)
    pt_colors = [cmap[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
    ax.scatter(X2d[:, 0], X2d[:, 1], c=pt_colors, s=6, alpha=0.55, linewidths=0)
    patches = [mpatches.Patch(color=cmap[l], label=f"C{l}") for l in sorted(cmap)]
    ax.legend(handles=patches, fontsize=7, loc="best", ncol=max(1, len(cmap) // 8), markerscale=2, framealpha=0.7, edgecolor=CREAM)
    _apply_base_style(ax, title=f"t-SNE 2-D  (perplexity={perplexity})", xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig


# Dendrogram
def plot_dendrogram(X, labels, n_samples=500, linkage_method="ward", title="Hierarchical Clustering Dendrogram", random_state=42):
    
    if hasattr(X, "toarray"):
        X = X.toarray()

    n = min(n_samples, len(X))
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=n, replace=False)
    X_sub = X[idx]
    lbl_sub = labels[idx]

    print(f"[Dendrogram] Computing linkage on {n} samples…")
    Z = scipy_linkage(X_sub, method=linkage_method, metric="euclidean")

    cmap = _color_map(lbl_sub)

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG)
    dendrogram(
        Z,
        ax=ax,
        no_labels=True,
        color_threshold=0.7 * max(Z[:, 2]),
        above_threshold_color=CLAY,
        leaf_rotation=90,
    )
    _apply_base_style(ax, title=title, xlabel="Documents (sampled)", ylabel="Distance")
    ax.set_facecolor(BG)

    patches = [mpatches.Patch(color=cmap[l], label=f"Cluster {l}") for l in sorted(cmap)]
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              ncol=max(1, len(cmap) // 8), markerscale=1.5,
              framealpha=0.7, edgecolor=CREAM,
              title="True cluster", title_fontsize=8)

    plt.tight_layout()
    return fig


# pipeline xxecution saver
def plot_best_result(result, dataset, figures_dir="figures", random_state=42,):
    out_dir = os.path.join(figures_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    X = result["X"]
    labels = result["labels"]

    # 1. Generate & Save UMAP
    print("[Viz] Plotting UMAP Projection...")
    fig_umap = plot_umap(X, labels, random_state)
    umap_path = os.path.join(out_dir, "umap.png")
    fig_umap.savefig(umap_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig_umap)

    # 2. Generate & Save PCA
    print("[Viz] Plotting PCA Projection...")
    fig_pca = plot_pca(X, labels, random_state)
    pca_path = os.path.join(out_dir, "pca.png")
    fig_pca.savefig(pca_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig_pca)

    # 3. Generate & Save t-SNE
    print("[Viz] Plotting t-SNE Projection...")
    fig_tsne = plot_tsne(X, labels, random_state=random_state)
    tsne_path = os.path.join(out_dir, "tsne.png")
    fig_tsne.savefig(tsne_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig_tsne)

    # 4. Generate & Save Dendrogram
    print("[Viz] Plotting Hierarchical Dendrogram...")
    fig_dendro = plot_dendrogram(X, labels, random_state=random_state)
    dendro_path = os.path.join(out_dir, "dendrogram.png")
    fig_dendro.savefig(dendro_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig_dendro)

    print(f"[Viz] All projections successfully cached in: {out_dir}")
    return umap_path