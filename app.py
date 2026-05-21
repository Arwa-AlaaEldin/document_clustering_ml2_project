import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Clustering",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS & Theme ─────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

  :root {
    --clay:    #A9907E;
    --cream:   #F3DEBA;
    --sage:    #ABC4AA;
    --espresso:#675D50;
    --white:   #FDFAF6;
    --text:    #3A342E;
  }

  html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    color: var(--text);
  }

  /* App background */
  .stApp {
    background: linear-gradient(160deg, #FDFAF6 0%, #F3DEBA22 50%, #ABC4AA18 100%);
  }

  /* Hero header */
  .hero-header {
    background: linear-gradient(135deg, var(--espresso) 0%, var(--clay) 60%, #C4A882 100%);
    border-radius: 18px;
    padding: 2.4rem 2.8rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(243,222,186,0.13);
  }
  .hero-header::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 60px;
    width: 150px; height: 150px;
    border-radius: 50%;
    background: rgba(171,196,170,0.10);
  }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: var(--cream) !important;
    margin: 0 0 0.3rem;
    line-height: 1.15;
  }
  .hero-sub {
    color: rgba(243,222,186,0.78);
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.03em;
  }

  /* Dataset pill selector */
  .dataset-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: white;
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 2px 12px rgba(103,93,80,0.09);
    border-left: 5px solid var(--clay);
  }
  .dataset-label {
    font-family: 'Playfair Display', serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--espresso);
    white-space: nowrap;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    padding: 0;
    border-bottom: 2px solid var(--cream);
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.55rem 1.2rem;
    border-radius: 10px 10px 0 0;
    color: var(--clay);
    background: transparent;
    border: none;
  }
  .stTabs [aria-selected="true"] {
    background: var(--cream) !important;
    color: var(--espresso) !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    border-top: 4px solid var(--clay);
    box-shadow: 0 2px 10px rgba(103,93,80,0.08);
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--clay) !important;
    font-weight: 600;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif;
    font-size: 1.55rem;
    color: var(--espresso) !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1.5px solid var(--cream);
  }

  /* Section headers */
  h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--espresso) !important;
  }

  /* Expanders */
  .streamlit-expanderHeader {
    background: var(--cream) !important;
    border-radius: 10px !important;
    font-weight: 600;
    color: var(--espresso);
  }

  /* Selectbox */
  [data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: var(--clay) !important;
    background: white !important;
  }

  /* Info / warning boxes */
  .stAlert {
    border-radius: 12px;
  }

  /* Bar chart */
  [data-testid="stArrowVegaLiteChart"] {
    border-radius: 12px;
    overflow: hidden;
  }

  /* Hide default sidebar toggle decoration when sidebar is off */
  section[data-testid="stSidebar"] { display: none !important; }

  /* Cluster term pills */
  .term-pill {
    display: inline-block;
    background: var(--cream);
    color: var(--espresso);
    border-radius: 20px;
    padding: 2px 11px;
    font-size: 0.82rem;
    margin: 2px 3px;
    font-weight: 500;
    border: 1px solid rgba(169,144,126,0.35);
  }

  /* Viz caption */
  .viz-caption {
    font-size: 0.82rem;
    color: var(--clay);
    text-align: center;
    margin-top: 0.4rem;
    font-style: italic;
  }

  /* Cluster card */
  .cluster-card {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.7rem;
    border-left: 4px solid var(--sage);
    box-shadow: 0 2px 8px rgba(103,93,80,0.07);
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_results(dataset: str):
    results_path = os.path.join(RESULTS_DIR, dataset, "results.pkl")
    meta_path    = os.path.join(RESULTS_DIR, dataset, "meta.pkl")
    if not os.path.exists(results_path):
        return None, None
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    return results, meta


def available_datasets() -> list[str]:
    if not os.path.isdir(RESULTS_DIR):
        return []
    return [d for d in os.listdir(RESULTS_DIR)
            if os.path.isfile(os.path.join(RESULTS_DIR, d, "results.pkl"))]


def build_metrics_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.get("metrics", {})
        rows.append({
            "Pipeline":      r["pipeline"],
            "Algorithm":     r["algorithm"],
            "k":             r["n_clusters"],
            "Silhouette":  round(m.get("silhouette", float("nan")), 4),
        })
    return pd.DataFrame(rows)


def highlight_best(df: pd.DataFrame):
    styled = df.style

    def _highlight(series, higher_better=True):
        best = series.max() if higher_better else series.min()
        return [
            f"background-color: #ABC4AA44; font-weight: bold; color: #675D50;"
            if v == best else ""
            for v in series
        ]

    styled = styled.apply(_highlight, higher_better=True, subset=["Silhouette"])
    return styled


# ── Hero Header ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <div class="hero-title">Document Clustering</div>
  <div class="hero-sub">
    Unsupervised Learning Project Doc Clustering using KMeans · Hierarchical · GMM
  </div>
</div>
""", unsafe_allow_html=True)

# ── Dataset selector (top, not sidebar) ───────────────────────────────────────

datasets = available_datasets()

if not datasets:
    st.warning(
        "No results found. Run the pipeline first:\n\n"
        "```bash\n"
        "python pipeline.py --dataset newsgroups\n"
        "python pipeline.py --dataset wikipedia --data_path people_wiki.csv\n"
        "```"
    )
    st.stop()

col_ds, col_rest = st.columns([1, 3])
with col_ds:
    st.subheader("Choose Dataset")
    dataset = st.selectbox("", datasets, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

results, meta = load_results(dataset)

if results is None:
    st.error(f"Could not load results for '{dataset}'.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_overview, tab_detail, tab_viz, tab_explore = st.tabs(
    ["Overview", "Detail", "Visualisations", "Explore Clusters"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.subheader(f"All results — {dataset}")

    df_metrics = build_metrics_df(results)
    st.dataframe(highlight_best(df_metrics), use_container_width=True, height=370)

    st.markdown("---")
    st.subheader("Best per pipeline (by Silhouette)")
    best_rows = []
    for pipeline in df_metrics["Pipeline"].unique():
        sub = df_metrics[df_metrics["Pipeline"] == pipeline]
        best_rows.append(sub.loc[sub["Silhouette"].idxmax()])
    st.dataframe(pd.DataFrame(best_rows).reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.subheader("Silhouette score comparison")
    chart_df = df_metrics.copy()
    chart_df["Config"] = chart_df["Pipeline"] + " + " + chart_df["Algorithm"]
    st.bar_chart(chart_df.set_index("Config")[["Silhouette"]], color="#A9907E")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Detail
# ══════════════════════════════════════════════════════════════════════════════

with tab_detail:
    st.subheader("Inspect a single result")
    options = [f"{r['pipeline']} + {r['algorithm']}" for r in results]
    choice  = st.selectbox("Select configuration", options)
    idx     = options.index(choice)
    r       = results[idx]

    m = r.get("metrics", {})
    sil = m.get("silhouette", float("nan"))

    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette",  f"{sil:.4f}")
    col2.metric("Algorithm",     r["algorithm"])
    col3.metric("k (clusters)",  r["n_clusters"])

    st.markdown("---")
    st.subheader("Cluster sizes")
    labels = r["labels"]
    unique, counts = np.unique(labels, return_counts=True)
    size_df = pd.DataFrame({"Cluster": unique, "Documents": counts})
    st.bar_chart(size_df.set_index("Cluster"), color="#ABC4AA")

    st.markdown("---")
    st.subheader("Top terms per cluster")
    top_terms = r.get("top_terms", {})
    if top_terms:
        cluster_ids = sorted(top_terms.keys())
        n_cols = 3
        for i in range(0, len(cluster_ids), n_cols):
            cols = st.columns(n_cols)
            for j, cid in enumerate(cluster_ids[i:i + n_cols]):
                with cols[j]:
                    st.markdown(
                        f'<div class="cluster-card">'
                        f'<strong style="color:#675D50;font-family:Playfair Display,serif">Cluster {cid}</strong><br>'
                        + "".join(f'<span class="term-pill">{t}</span>' for t in top_terms[cid][:10])
                        + "</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("No top-term data available.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Visualisations (PCA, t-SNE, Dendrogram)
# ══════════════════════════════════════════════════════════════════════════════

with tab_viz:
    st.subheader("Visualisations")

    st.markdown("#### Best Configuration Projections (Pre-saved Assets)")
    fig_dir = os.path.join(FIGURES_DIR, dataset)
    if os.path.isdir(fig_dir):
        png_files = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
        if png_files:
            img_cols = st.columns(min(len(png_files), 3))
            for i, fname in enumerate(sorted(png_files)):
                with img_cols[i % 3]:
                    st.image(os.path.join(fig_dir, fname),
                             caption=fname.replace("_", " ").replace(".png", "").upper(),
                             use_container_width=True)
        else:
            st.info("No explicit custom asset output files located inside figures/" + dataset)
    else:
        st.info("Target dataset filesystem paths unavailable.")

    st.markdown("---")

    # ── 3. On-the-Fly Projection Explorer
    st.markdown("#### Projection Explorer")

    viz_options = [f"{r['pipeline']} + {r['algorithm']}" for r in results]
    viz_choice  = st.selectbox("Select configuration to explore interactively", viz_options, key="viz_config")
    viz_idx     = viz_options.index(viz_choice)
    viz_r       = results[viz_idx]
    viz_labels  = np.array(viz_r["labels"])

    has_embeddings = meta is not None and meta.get("texts") is not None

    if not has_embeddings:
        st.info("Metadata vectors are missing — run pipeline with metadata flags to enable the interactive explorer.")
    else:
        viz_type = st.radio(
            "Projection Algorithm",
            ["PCA", "t-SNE", "Dendrogram"],
            horizontal=True,
            key="viz_type",
        )

        if st.button("compute", type="primary"):
            with st.spinner("Calculating projection dimensions…"):
                try:
                    from visualization import plot_pca, plot_tsne, plot_dendrogram

                    import sys, os as _os
                    sys.path.insert(0, _os.path.dirname(__file__))
                    from preprocessing.feature_extractor import build_tfidf, apply_lsa

                    texts_clean = meta["texts"]
                    _, X_tfidf  = build_tfidf(texts_clean, dataset=dataset)
                    X_lsa       = apply_lsa(X_tfidf, n_components=50, random_state=42)

                    if viz_type == "PCA":
                        fig = plot_pca(X_lsa, viz_labels)
                    elif viz_type == "t-SNE":
                        fig = plot_tsne(X_lsa, viz_labels)
                    else:
                        fig = plot_dendrogram(X_lsa, viz_labels, n_samples=500, title=f"Dendrogram — {viz_choice}")

                    _, dynamic_plot_col, _ = st.columns([1, 2, 1])
                    with dynamic_plot_col:
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(f'<p class="viz-caption">{viz_type} On-The-Fly View · {viz_choice}</p>', unsafe_allow_html=True)

                except ImportError as e:
                    st.error(f"Missing dependency component: {e}")
                except Exception as e:
                    st.error(f"Visualisation engine execution failed: {e}")
                    st.exception(e)
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Explore Clusters
# ══════════════════════════════════════════════════════════════════════════════

with tab_explore:
    st.subheader("Browse documents by cluster")

    if meta is None:
        st.info("Meta data not available. Re-run the pipeline.")
    else:
        options2 = [f"{r['pipeline']} + {r['algorithm']}" for r in results]
        choice2  = st.selectbox("Configuration", options2, key="explore_config")
        idx2     = options2.index(choice2)
        r2       = results[idx2]
        labels2  = r2["labels"]

        cluster_id = st.selectbox("Cluster", sorted(set(labels2)))

        raw_texts  = meta.get("raw_texts") or meta.get("texts", [])
        categories = meta.get("category") or [None] * len(raw_texts)
        names      = meta.get("names")

        mask = np.where(labels2 == cluster_id)[0]

        c1, c2 = st.columns([2, 5])
        c1.metric("Documents in cluster", len(mask))
        if r2.get("top_terms") and cluster_id in r2["top_terms"]:
            with c2:
                st.markdown(
                    "**Top terms:** "
                    + "".join(
                        f'<span class="term-pill">{t}</span>'
                        for t in r2["top_terms"][cluster_id][:8]
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        n_show = st.slider("Show N documents", 1, min(50, len(mask)), 5)
        for i in mask[:n_show]:
            label_str = f" · `{categories[i]}`" if categories and categories[i] else ""
            name_str  = f"**{names[i]}** — " if names and names[i] else ""
            with st.expander(f"Doc {i}{label_str}"):
                st.write(name_str + (raw_texts[i][:800] + "…"
                                     if len(raw_texts[i]) > 800 else raw_texts[i]))
