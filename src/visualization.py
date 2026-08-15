"""Figures: per-task comparison grids and aggregate summary plots.

Two rules from the brief drive the design:

  * every requested reconstruction is shown -- if a task has more configurations than fit in
    one readable figure, the grid PAGINATES rather than silently dropping the tail;
  * every panel is labelled with the model, the method, t0, K, the MPC step count and
    lambda, so a figure can never be misattributed.

Nothing here is a dashboard; they are matplotlib figures written to disk and, in a notebook,
displayed inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .utils import in_ipython, to_uint8

MAX_COLUMNS_PER_PAGE = 6          # reconstruction columns, on top of the 2 reference columns
MAX_ROWS = 4                      # images per page


def _setup():
    import matplotlib
    if not in_ipython():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["figure.max_open_warning"] = 0
    return plt


def _finish(fig, path: Path, show: bool) -> Path:
    plt = _setup()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    if show and in_ipython():
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_source_images(images: np.ndarray, names: Sequence[str], labels: Sequence[int],
                       path: Path, show: bool = True, limit: int = 8) -> Path:
    plt = _setup()
    k = min(limit, len(images))
    fig, axes = plt.subplots(1, k, figsize=(2.0 * k, 2.5), squeeze=False)
    for i in range(k):
        axes[0][i].imshow(to_uint8(images[i]))
        axes[0][i].axis("off")
        axes[0][i].set_title("%s\n(class %d)" % (names[i], labels[i]), fontsize=8)
    fig.suptitle("Ground truth -- one shared pool for every experiment, model and method",
                 fontweight="bold")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_problem_overview(problems: Dict[str, Any], path: Path, show: bool = True,
                          n_show: int = 3) -> Path:
    """One row per problem: ground truth, degraded observation, initialisation guide."""
    plt = _setup()
    rows = len(problems)
    fig, axes = plt.subplots(rows, 3 * n_show, figsize=(2.0 * 3 * n_show, 2.4 * rows),
                             squeeze=False)
    for r, (_key, p) in enumerate(problems.items()):
        for i in range(min(n_show, len(p.measurement))):
            panels = [("ground truth", p.ground_truth[i]),
                      ("degraded y", p.display_measurement[i]),
                      ("guide g(y)", p.initialization_guide[i])]
            for c, (title, img) in enumerate(panels):
                ax = axes[r][3 * i + c]
                ax.imshow(to_uint8(img))
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if r == 0:
                    ax.set_title(title, fontsize=8)
                if 3 * i + c == 0:
                    ax.set_ylabel(p.name, fontsize=8)
    fig.suptitle("Inverse-problem instances: y = A(x*) + eta, built once and shared by "
                 "SDEdit, MPC-RHC and MPC-delta_t", fontweight="bold")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_comparison_grid(experiment: str, model: str, records: Sequence[Dict[str, Any]],
                         problem, figures_dir: Path, show: bool = True,
                         max_rows: int = MAX_ROWS,
                         max_columns: int = MAX_COLUMNS_PER_PAGE) -> List[Path]:
    """Ground truth | degraded | every requested reconstruction, paginated.

    `records` are dicts with keys 'title' and 'images' (an (N,256,256,3) array).  Sorted so
    that SDEdit appears first, which makes the MPC columns read as deltas against it.
    """
    plt = _setup()
    if not records:
        return []
    order = {"sdedit": 0, "mpc_rhc": 1, "mpc_delta_t": 2}
    records = sorted(records, key=lambda r: (order.get(r.get("method", ""), 9),
                                             r.get("t0", 0.0), r.get("sort_key", "")))
    n_images = min(max_rows, len(problem.ground_truth))
    pages = [records[i:i + max_columns] for i in range(0, len(records), max_columns)]
    paths: List[Path] = []

    for page_idx, page in enumerate(pages, 1):
        n_cols = 2 + len(page)
        fig, axes = plt.subplots(n_images, n_cols,
                                 figsize=(2.05 * n_cols, 2.35 * n_images), squeeze=False)
        for row in range(n_images):
            panels = [("original", problem.ground_truth[row]),
                      ("degraded /\nguide", problem.display_measurement[row])]
            panels += [(rec["title"], rec["images"][row]) for rec in page]
            for col, (title, img) in enumerate(panels):
                ax = axes[row][col]
                ax.imshow(to_uint8(img))
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if row == 0:
                    ax.set_title(title, fontsize=7.5, linespacing=1.25)
            axes[row][0].set_ylabel(problem.image_ids[row], fontsize=6.5)
        suffix = ("  (page %d/%d)" % (page_idx, len(pages))) if len(pages) > 1 else ""
        fig.suptitle("%s | %s  --  same image, same y, same t0, same epsilon%s"
                     % (experiment, model.upper(), suffix), fontweight="bold", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        paths.append(_finish(fig, figures_dir / ("%s_%s_page_%02d.png"
                                                 % (experiment, model, page_idx)), show))
    return paths


def plot_metric_summary(rows: Sequence[Dict[str, Any]], path: Path,
                        show: bool = True) -> Optional[Path]:
    """Bar charts of PSNR / SSIM / LPIPS / runtime-per-image, grouped by method."""
    plt = _setup()
    rows = [r for r in rows if r.get("status") == "ok"]
    if not rows:
        return None
    tasks = sorted({r["task"] for r in rows})
    methods = [m for m in ("sdedit", "mpc_rhc", "mpc_delta_t")
               if any(r["method"] == m for r in rows)]
    metrics = [("PSNR (dB)", "psnr", True), ("SSIM", "ssim", True),
               ("LPIPS (lower is better)", "lpips", False),
               ("runtime / image (s)", "runtime_per_image", False)]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(1.6 * max(4, len(tasks)) + 3,
                                                       3.1 * len(metrics)), squeeze=False)
    width = 0.8 / max(1, len(methods))
    for ax_idx, (label, key, _higher) in enumerate(metrics):
        ax = axes[ax_idx][0]
        base = np.arange(len(tasks))
        for m_idx, method in enumerate(methods):
            values = []
            for task in tasks:
                vals = [r[key] for r in rows
                        if r["task"] == task and r["method"] == method
                        and r.get(key) is not None]
                values.append(float(np.mean(vals)) if vals else np.nan)
            ax.bar(base + m_idx * width, values, width, label=method)
        ax.set_xticks(base + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(tasks, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        if ax_idx == 0:
            ax.legend(fontsize=8, ncol=len(methods))
        if key == "runtime_per_image":
            ax.set_yscale("log")
    fig.suptitle("Reconstruction quality and cost by method (averaged over configurations)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, path, show)


def plot_quality_vs_cost(rows: Sequence[Dict[str, Any]], path: Path,
                         show: bool = True) -> Optional[Path]:
    """LPIPS versus runtime, and measurement consistency versus perceptual quality."""
    plt = _setup()
    rows = [r for r in rows if r.get("status") == "ok"]
    usable = [r for r in rows if r.get("lpips") is not None]
    if not usable:
        usable = rows
    if not usable:
        return None
    markers = {"sdedit": "o", "mpc_rhc": "s", "mpc_delta_t": "^"}
    colors = {"jit": "tab:blue", "pmf": "tab:orange", "sit": "tab:green", "imf": "tab:red"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r in usable:
        style = dict(marker=markers.get(r["method"], "x"),
                     color=colors.get(r["model"], "grey"), s=48, alpha=0.85)
        y = r.get("lpips")
        if y is not None and r.get("runtime_per_image"):
            axes[0].scatter(r["runtime_per_image"], y, **style)
        if y is not None and r.get("measurement_rmse") is not None:
            axes[1].scatter(r["measurement_rmse"], y, **style)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("runtime per image (s, log scale)")
    axes[0].set_ylabel("LPIPS (lower is better)")
    axes[0].set_title("Perceptual quality vs compute")
    axes[1].set_xlabel("measurement RMSE  ||A(x_hat) - y||")
    axes[1].set_ylabel("LPIPS (lower is better)")
    axes[1].set_title("Measurement consistency vs perceptual quality")
    for ax in axes:
        ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker=markers[m], linestyle="", color="black", label=m)
               for m in markers if any(r["method"] == m for r in usable)]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=colors[m], label=m.upper())
                for m in colors if any(r["model"] == m for r in usable)]
    axes[0].legend(handles=handles, fontsize=8)
    fig.suptitle("MPC buys measurement consistency; the question is what it costs and whether "
                 "perceptual quality follows", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, path, show)


def plot_stroke_parity(reference: np.ndarray, rendered: np.ndarray, truth: np.ndarray,
                       path: Path, show: bool = True, limit: int = 3) -> Path:
    """Original PIL stroke transform vs the frozen-geometry differentiable renderer."""
    plt = _setup()
    k = min(limit, len(truth))
    fig, axes = plt.subplots(k, 3, figsize=(7.5, 2.5 * k), squeeze=False)
    for i in range(k):
        for c, (title, img) in enumerate((("ground truth", truth[i]),
                                          ("original SDEdit stroke\n(PIL, non-differentiable)",
                                           reference[i]),
                                          ("frozen-geometry renderer\n(differentiable, = y)",
                                           rendered[i]))):
            axes[i][c].imshow(to_uint8(img))
            axes[i][c].axis("off")
            if i == 0:
                axes[i][c].set_title(title, fontsize=8)
    fig.suptitle("Stroke operator parity", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, path, show)
