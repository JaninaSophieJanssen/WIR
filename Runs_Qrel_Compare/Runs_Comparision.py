# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.14",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pyterrier as pt
    import numpy as np
    import gzip
    import pandas as pd
    from ranx import Qrels, Run, evaluate, compare, plot
    import os
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return Qrels, Run, compare, os, pd, plt, pt, sys


@app.cell
def _(pt):
    # Initialize PyTerrier
    if not pt.java.started():
        pt.init()

    # Load the indexe for baseline_dense and baseline_bm25
    index = pt.Artifact.from_hf('jueri/longeval-2026-snapshot-1-index')
    return


@app.cell
def _(Qrels):
    #Qrels through ranx
    qrel_human = Qrels.from_file("qrel_human.txt", kind='trec')
    qrel_llm = Qrels.from_file("qrel_llm.txt", kind='trec')
    qrel = Qrels.from_file(r'qrels.txt', kind='trec')
    return qrel, qrel_human, qrel_llm


@app.cell
def _(Run):
    bm25_run = Run.from_file("./bm25-snapshot-3-run/run.txt")   # TREC-Style file with txt extension
    dense_run = Run.from_file("./qwen3-snapshot-3-run/run.txt")   # TREC-Style file with txt extension
    return bm25_run, dense_run


@app.cell
def _(bm25_run, compare, dense_run, qrel, qrel_human, qrel_llm):
    report_human = compare(
        qrels=qrel_human,
        runs=[bm25_run, dense_run],
        metrics=["precision@5", "precision@10", "precision@20", "precision@50", "precision@100",
                 "recall@5", "recall@10", "recall@20", "recall@50", "recall@100",
                 "f1@5", "f1@10", "f1@20", "f1@50", "f1@100",
                 "r-precision",
                 "bpref",
                 "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50", "ndcg@100",
                 "map@5", "map@10", "map@20", "map@50", "map@100"],
        max_p=1.0,  # P-value threshold
        make_comparable=True
    )

    report_llm = compare(
        qrels=qrel_llm,
        runs=[bm25_run, dense_run],
        metrics=["precision@5", "precision@10", "precision@20", "precision@50", "precision@100",
                 "recall@5", "recall@10", "recall@20", "recall@50", "recall@100",
                 "f1@5", "f1@10", "f1@20", "f1@50", "f1@100",
                 "r-precision",
                 "bpref",
                 "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50", "ndcg@100",
                 "map@5", "map@10", "map@20", "map@50", "map@100"],
        max_p=1.0,  # P-value threshold
        make_comparable=True
    )

    report = compare(
        qrels=qrel,
        runs=[bm25_run, dense_run],
        metrics=["precision@5", "precision@10", "precision@20", "precision@50", "precision@100",
                 "recall@5", "recall@10", "recall@20", "recall@50", "recall@100",
                 "f1@5", "f1@10", "f1@20", "f1@50", "f1@100",
                 "r-precision",
                 "bpref",
                 "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50", "ndcg@100",
                 "map@5", "map@10", "map@20", "map@50", "map@100"],
        max_p=1.0,  # P-value threshold
        make_comparable=True
    )
    return report, report_human, report_llm


@app.cell
def _(pd, report, report_human, report_llm):
    results_human = pd.DataFrame.from_dict(report_human.results, orient='index')
    results_llm = pd.DataFrame.from_dict(report_llm.results, orient='index')
    results = pd.DataFrame.from_dict(report.results, orient='index')

    results_human['qrel'] = 'human'
    results_llm['qrel'] = 'llm'
    results['qrel'] = 'baseline'

    df = pd.concat([results_human, results_llm, results], axis=0)
    df = df.reset_index()
    df.rename(columns={'index': 'model'}, inplace=True)
    return (df,)


@app.cell
def _(df, os, pd, plt, sys):
    """
    plot_comparisons.py

    Reads a CSV of retrieval-evaluation metrics and produces comparison plots.

    Rules encoded from the request:
      * Any column whose name contains "@" (e.g. "precision@5", "precision@10", ...)
        belongs to a group (the part before the "@"). All points in a group are
        plotted together in ONE plot, as different points along the x-axis (k).
      * Every plot compares the different values of the 'qrel' column
        (e.g. human / llm / baseline), and is always produced separately for
        'bm25' and 'dense' (both are always shown, side by side).
      * qrel values get fixed colors: purple, orange, red (in that order, cycling
        if there happen to be more than three).
      * All plot backgrounds (figure + axes) are white.

    Usage:
        python plot_comparisons.py path/to/file.csv [output_dir]
    """

    QREL_COL = "qrel"
    modelS = ["bm25", "dense"]                        # fixed order, both always plotted

    # Fixed color per qrel value. If more qrel values than colors exist, the
    # palette cycles.
    QREL_COLOR_ORDER = ["#3eaee0", "#84ba5a", "#ea94a8"]

    def get_metric_groups(df: pd.DataFrame):
        """
        Split all metric columns into:
          - grouped: dict {prefix: [(k, colname), ...]} for columns containing "@"
          - singles: list of column names with no "@" (plain metrics)
        """
        grouped = {}
        singles = []

        for col in df.columns:
            if col in ("model", QREL_COL):
                continue
            if "@" in col:
                prefix, suffix = col.split("@", 1)
                try:
                    k = float(suffix)
                except ValueError:
                    k = suffix
                grouped.setdefault(prefix, []).append((k, col))
            else:
                singles.append(col)

        # sort each group's points by their k value
        for prefix in grouped:
            grouped[prefix].sort(key=lambda item: item[0])

        return grouped, singles


    def build_qrel_colors(qrels):
        """Assign a fixed color to each qrel value, cycling the palette if needed."""
        return {
            qrel: QREL_COLOR_ORDER[i % len(QREL_COLOR_ORDER)]
            for i, qrel in enumerate(qrels)
        }


    # ---------------------------------------------------------------------------
    # Plot styling helpers
    # ---------------------------------------------------------------------------
    def style_axis(ax):
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)


    # ---------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------
    LINE_STYLES = ["-", "--", ":", "-."]
    MARKERS = ["o", "s", "^", "D"]


    def plot_grouped_metric(df, prefix, points, qrel_colors, out_dir):
        """One figure per metric group (e.g. 'precision'), with a subplot for
        bm25 and a subplot for dense. Each qrel is a colored line across k."""
        plt.style.use('seaborn-v0_8-whitegrid')  # Apply light style

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        qrels = list(qrel_colors.keys())
        xs = [k for k, _ in points]
        xlabels = [str(k if k != int(k) else int(k)) for k in xs] if all(
            isinstance(k, float) for k, _ in points
        ) else [str(k) for k in xs]

        for ax, model in zip(axes, modelS):
            sub = df[df["model"] == model]
            n = len(qrels)
            for idx, qrel in enumerate(qrels):
                row = sub[sub[QREL_COL] == qrel]
                if row.empty:
                    continue
                ys = [row.iloc[0][col] for _, col in points]
                ax.plot(
                    range(len(xs)), ys,
                    marker=MARKERS[idx % len(MARKERS)],
                    linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
                    linewidth=2 + (n - idx),
                    markersize=6 + 2 * (n - idx),
                    alpha=0.85,
                    label=qrel, color=qrel_colors[qrel],
                    zorder=10 + idx,
                )
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xlabels)
            ax.set_title(model)
            ax.set_xlabel("k")
            ax.set_ylabel(prefix)
            style_axis(ax)
            ax.legend(title="qrel", facecolor='white', edgecolor='black')

        fig.suptitle(f"{prefix}@k — comparison across qrels", fontsize=14, fontweight="bold")
        fig.tight_layout()

        return fig



    def plot_single_metric(df, metric, qrel_colors, out_dir):
        """One figure per plain (non-grouped) metric, with a subplot for bm25
        and a subplot for dense. Each qrel is a colored bar."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        qrels = list(qrel_colors.keys())

        for ax, model in zip(axes, modelS):
            sub = df[df["model"] == model]
            labels, vals, colors = [], [], []
            for qrel in qrels:
                row = sub[sub[QREL_COL] == qrel]
                if row.empty:
                    continue
                labels.append(qrel)
                vals.append(row.iloc[0][metric])
                colors.append(qrel_colors[qrel])
            ax.bar(labels, vals, color=colors)
            ax.set_title(model)
            ax.set_ylabel(metric)
            style_axis(ax)

        fig.suptitle(f"{metric} — comparison across qrels", fontsize=14, fontweight="bold")
        fig.tight_layout()

        return fig


    # ---------------------------------------------------------------------------
    # Main
    # ---------------------------------------------------------------------------


    out_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"
    os.makedirs(out_dir, exist_ok=True)

    df 
    grouped, singles = get_metric_groups(df)

    qrels = df[QREL_COL].unique().tolist()
    qrel_colors = build_qrel_colors(qrels)

    saved = []
    for prefix, points in grouped.items():
        saved.append(plot_grouped_metric(df, prefix, points, qrel_colors, out_dir))
    for metric in singles:
        saved.append(plot_single_metric(df, metric, qrel_colors, out_dir))

    print(f"Saved {len(saved)} plot(s) to '{out_dir}/':")
    for p in saved:
        print(" -", p)
    return (saved,)


@app.cell
def _(saved):
    saved
    return


@app.cell
def _(pd):
    # Load the runs
    bm25 = pd.read_csv("./bm25-snapshot-3-run/run.txt", sep=" ", header=None, names=['qid', 'Q', 'docid', 'rank', 'score', 'bm25'])
    dense = pd.read_csv("./qwen3-snapshot-3-run/run.txt", sep=" ", header=None, names=['qid', 'Q', 'docid', 'rank', 'score', 'dense'])

    # Load the qrels
    llms = pd.read_csv("qrel_llm.txt", sep=" ", header=None, names=['qid', 'zero', 'docid', 'relevance'])
    humans = pd.read_csv("qrel_human.txt", sep=" ", header=None, names=['qid', 'zero', 'docid', 'relevance'])
    qre = pd.read_csv("qrels.txt", sep=" ", header=None, names=['qid', 'zero', 'docid', 'relevance'])

    # Rename relevance columns for clarity
    llms = llms.rename(columns={'relevance': 'llm_relevance'})
    humans = humans.rename(columns={'relevance': 'human_relevance'})
    qre = qre.rename(columns={'relevance': 'qre_relevance'})

    # Find the intersection of qid and docid across all three qrels
    common_qid_docid = set(qre[['qid', 'docid']].apply(tuple, axis=1)) & \
                      set(llms[['qid', 'docid']].apply(tuple, axis=1)) & \
                      set(humans[['qid', 'docid']].apply(tuple, axis=1))

    # Filter the runs to only include rows where (qid, docid) is in the intersection
    dense_filtered = dense[dense.apply(lambda row: (row['qid'], row['docid']) in common_qid_docid, axis=1)]
    bm25_filtered = bm25[bm25.apply(lambda row: (row['qid'], row['docid']) in common_qid_docid, axis=1)]

    # Merge relevance scores with the filtered runs
    dense_merged = pd.merge(dense_filtered, qre, on=['qid', 'docid'], how='left')
    dense_merged = pd.merge(dense_merged, llms, on=['qid', 'docid'], how='left')
    dense_merged = pd.merge(dense_merged, humans, on=['qid', 'docid'], how='left')

    bm25_merged = pd.merge(bm25_filtered, qre, on=['qid', 'docid'], how='left')
    bm25_merged = pd.merge(bm25_merged, llms, on=['qid', 'docid'], how='left')
    bm25_merged = pd.merge(bm25_merged, humans, on=['qid', 'docid'], how='left')

    # Compare relevance scores for dense and bm25
    dense_comparison = dense_merged[['qid', 'docid', 'rank','qre_relevance', 'llm_relevance', 'human_relevance']]
    bm25_comparison = bm25_merged[['qid', 'docid', 'rank','qre_relevance', 'llm_relevance', 'human_relevance']]
    return bm25_comparison, dense_comparison, humans, llms


@app.cell
def _(bm25_comparison, dense_comparison):
    # Filter rows where llm_relevance and human_relevance are different for dense_comparison
    dense_different_relevance = dense_comparison[dense_comparison['llm_relevance'] != dense_comparison['human_relevance']]

    # Filter rows where llm_relevance and human_relevance are different for bm25_comparison
    bm25_different_relevance = bm25_comparison[bm25_comparison['llm_relevance'] != bm25_comparison['human_relevance']]
    return (bm25_different_relevance,)


@app.cell
def _(bm25_different_relevance):
    bm25_different_relevance
    return


@app.cell
def _(humans, llms, pd):
    # Merge llms and humans on 'qid' and 'docid'
    merged_llms_humans = pd.merge(
        llms,
        humans,
        on=['qid', 'docid'],
        how='inner'  # Only keep rows where both 'qid' and 'docid' match
    )

    # Filter rows where llm_relevance and human_relevance are different
    different_relevance_df = merged_llms_humans[
        merged_llms_humans['llm_relevance'] != merged_llms_humans['human_relevance']
    ]

    # Display the resulting DataFrame
    print("Rows where LLM and Human relevance scores differ for matching (qid, docid):")
    print(different_relevance_df)
    return (different_relevance_df,)


@app.cell
def _(different_relevance_df):
    different_relevance_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
