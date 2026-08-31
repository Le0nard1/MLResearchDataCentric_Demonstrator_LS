"""Real-Data Example — the iterative data-selective loop on iris_extended.

The synthetic studies (pages 07–10) control everything: the target function,
the induced gap, the candidate pool. This page runs the same loop on a real,
fixed tabular dataset (data/raw/iris_extended.csv, 1200 rows, two categorical
columns), where the dataset itself is the pool, consumed without replacement.

Three encodings answer the categorical question three ways: drop the
categorical columns (numeric), one-hot them into the distance so "closest"
spans categories (onehot), or match the guided draw to the weakspot's own
categorical distribution (stratified). Policies mirror the paper: static,
dynamic (×0.1/round), adaptive (severity-gated).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.realdata import loop_real as R

st.set_page_config(page_title="Real-Data Example", layout="wide")
st.title("🌱 Real-Data Example — iris_extended")
st.markdown(
    "The iterative guided-vs-random loop on a **real fixed dataset**: no ground "
    "truth, no induced gap, the dataset is the candidate pool and is consumed "
    "without replacement. **Guided** selects around the current weakspot "
    "(kNN-smoothed error peak in feature space); **random** draws the same "
    "counts from the same pool."
)

df_raw, cat_cols = R.load_dataset(
    Path(__file__).resolve().parents[2] / "data/raw/iris_extended.csv")
num_targets = [c for c in df_raw.columns if c not in cat_cols]

with st.sidebar:
    st.header("Setup")
    target = st.selectbox("Target column", num_targets,
                          index=num_targets.index(R.DEFAULTS["target"]))
    encoding = st.selectbox("Categorical handling", list(R.ENCODINGS), index=1,
                            help="numeric: drop categorical columns everywhere. "
                                 "onehot: one-hot into model + distance. "
                                 "stratified: quotas from the weakspot's own "
                                 "categorical distribution.")
    policy = st.selectbox("Mix policy", list(R.POLICIES), index=2)
    st.header("Loop")
    n_iterations = st.slider("Iterations K", 2, 20, R.DEFAULTS["n_iterations"])
    n_train = st.slider("Initial training rows", 20, 200, R.DEFAULTS["n_train"], 10)
    n_select = st.slider("Rows added per round", 10, 100, R.DEFAULTS["n_select"], 5)
    n_eval = st.slider("Evaluation rows", 100, 500, R.DEFAULTS["n_eval"], 50)
    st.header("Selection")
    mix_ratio = st.slider("Starting mix α₀", 0.0, 1.0, R.DEFAULTS["mix_ratio"], 0.05)
    sel_sigma = st.slider("Kernel width σ (× median distance)", 0.1, 1.5,
                          R.DEFAULTS["sel_sigma"], 0.05)
    seed = st.number_input("Seed", 0, 9999, 42)
    seeds_avg = st.slider("Seeds to average", 1, 15, 1,
                          help="Runs the loop repeatedly from consecutive seeds "
                               "and averages the curves.")
    st.header("Induced deficit")
    st.caption("Real-data analogue of the papers' scarce pool: withhold one "
               "category from the initial training set and thin it from the "
               "candidate pool. The evaluation set is untouched.")
    deficit_col = st.selectbox("Deficit column", ["(none)"] + cat_cols)
    deficit_val, deficit_frac = None, 0.0
    if deficit_col != "(none)":
        deficit_val = st.selectbox(
            "Withheld category",
            sorted(df_raw[deficit_col].astype(str).unique()))
        deficit_frac = st.slider("Pool thinning", 0.0, 1.0, 0.9, 0.05)

cfg = dict(target=target, encoding=encoding, policy=policy,
           n_iterations=int(n_iterations), n_train=int(n_train),
           n_select=int(n_select), n_eval=int(n_eval),
           mix_ratio=float(mix_ratio), sel_sigma=float(sel_sigma),
           deficit_col=None if deficit_col == "(none)" else deficit_col,
           deficit_val=deficit_val, deficit_frac=float(deficit_frac))

if st.button("Run", type="primary"):
    prog = st.progress(0.0)
    runs = []
    for i in range(int(seeds_avg)):
        runs.append(R.run_real({**cfg, "seed": int(seed) + i}))
        prog.progress((i + 1) / int(seeds_avg))
    prog.empty()
    st.session_state["real_runs"] = runs

runs = st.session_state.get("real_runs")
if not runs:
    st.info("Configure in the sidebar and press **Run**.")
    st.stop()

K = len(runs[0]["mae_g"]) - 1
its = list(range(K + 1))
mg = np.mean([r["mae_g"] for r in runs], axis=0)
mr = np.mean([r["mae_r"] for r in runs], axis=0)
gap = mg - mr

c1, c2, c3 = st.columns(3)
c1.metric("Across-loop gap (guided − random)",
          f"{np.mean(gap[1:]):+.3f}",
          help="Negative = guided ahead on eval MAE, averaged over the loop.")
c2.metric("Final MAE guided / random", f"{mg[-1]:.2f} / {mr[-1]:.2f}")
c3.metric("Initial MAE", f"{mg[0]:.2f}")

left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_scatter(x=its, y=mg, mode="lines+markers", name="guided",
                    line=dict(color="#2ca02c"))
    fig.add_scatter(x=its, y=mr, mode="lines+markers", name="random",
                    line=dict(color="#d62728"))
    fig.update_layout(title="Evaluation MAE over the loop",
                      xaxis_title="iteration", yaxis_title="MAE", height=380)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = go.Figure()
    fig.add_scatter(x=its[1:], y=gap[1:], mode="lines+markers", name="gap",
                    line=dict(color="#1f77b4"))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title="Gap (guided − random; below 0 = guided ahead)",
                      xaxis_title="iteration", yaxis_title="MAE gap", height=380)
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_scatter(x=its[1:], y=np.mean([r["sev"] for r in runs], axis=0),
                    mode="lines+markers", name="severity",
                    line=dict(color="#9467bd"))
    fig.add_scatter(x=its[1:], y=np.mean([r["mix"] for r in runs], axis=0),
                    mode="lines+markers", name="realised mix α",
                    line=dict(color="#ff7f0e"))
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    fig.update_layout(title="Detected severity and realised mix per round",
                      xaxis_title="iteration", height=380)
    st.plotly_chart(fig, use_container_width=True)
with right:
    # Categorical distributions: weakspot members vs guided picks (first run).
    r0 = runs[0]
    rows = []
    for i, (cw, cp) in enumerate(zip(r0["cat_weak"], r0["cat_pick"]), start=1):
        for cat, v in cw.items():
            rows.append(dict(iteration=i, category=cat, share=v, what="weakspot"))
        for cat, v in cp.items():
            rows.append(dict(iteration=i, category=cat, share=v, what="picked"))
    dd = pd.DataFrame(rows)
    fig = px.bar(dd, x="iteration", y="share", color="category",
                 facet_row="what", height=380,
                 title="Categorical distribution: weakspot members vs guided picks "
                       f"(seed {r0['cfg']['seed']})")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Per-run details"):
    for r in runs:
        s = R.summarise(r)
        st.write(f"seed {r['cfg']['seed']}: auc gap {s['auc_gap']:+.3f}, "
                 f"final {s['final_mae_g']:.2f} vs {s['final_mae_r']:.2f}")
