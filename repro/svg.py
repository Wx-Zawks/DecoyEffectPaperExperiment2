from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


FONT_FAMILY = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
SERIES_COLORS = ["#6C757D", "#E57373", "#64B5F6", "#81C784", "#BA68C8", "#FFD54F", "#26A69A", "#5C6BC0", "#A1887F", "#FF8A65", "#90CAF9"]
MARKERS = ["s", "o", "x", "*", "D", "^", "v", ">", "<", "p", "h"]
LINESTYLES = ["--", "--", "--", "--", "-", "-", "-", "-", "-.", "-.", "-."]
SURFACE_CMAP = plt.cm.jet
GRID_COLOR = "#B0BEC5"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "axes.unicode_minus": False,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "grid.color": GRID_COLOR,
            "grid.linestyle": ":",
            "grid.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "savefig.dpi": 220,
        }
    )


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=13)


def paper_surface_pair(
    x_values: list[float],
    y_values: list[float],
    z_top: list[list[float]],
    z_bottom: list[list[float]],
    zlabels: tuple[str, str],
    subtitles: tuple[str, str],
    caption: str,
    output_path: Path,
) -> None:
    _apply_style()
    x_grid, y_grid = np.meshgrid(np.array(x_values), np.array(y_values))
    fig = plt.figure(figsize=(7.0, 10.5), dpi=220)

    for index, (z_matrix, z_label, subtitle) in enumerate(
        ((np.array(z_top), zlabels[0], subtitles[0]), (np.array(z_bottom), zlabels[1], subtitles[1])),
        start=1,
    ):
        ax = fig.add_subplot(2, 1, index, projection="3d")
        surf = ax.plot_surface(
            x_grid,
            y_grid,
            z_matrix,
            cmap=SURFACE_CMAP,
            linewidth=0.35,
            edgecolor="#888888",
            antialiased=True,
        )
        ax.view_init(elev=22, azim=-122)
        ax.set_xlabel(r"$\chi$", labelpad=8)
        ax.set_ylabel(r"$\gamma$", labelpad=8)
        ax.set_zlabel(z_label, labelpad=8)
        ax.set_title(subtitle, pad=8)
        ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        ax.xaxis._axinfo["grid"]["linestyle"] = ":"
        ax.yaxis._axinfo["grid"]["linestyle"] = ":"
        ax.zaxis._axinfo["grid"]["linestyle"] = ":"
        fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.06, aspect=20)

    _caption(fig, caption)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.99))
    fig.savefig(output_path)
    plt.close(fig)


def paper_line_chart(
    x_values: list[float | int],
    series: list[tuple[str, list[float]]],
    x_label: str,
    y_label: str,
    caption: str,
    output_path: Path,
    x_as_labels: bool = False,
    figsize: tuple[float, float] = (6.2, 4.8),
    y_lim: tuple[float, float] | None = None,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    ax.grid(True)
    for index, (label, values) in enumerate(series):
        plot_values = [np.nan if value is None else value for value in values]
        ax.plot(
            x_values if not x_as_labels else range(len(x_values)),
            plot_values,
            label=label,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            linewidth=1.2,
            markersize=4.5,
            markerfacecolor="none",
        )
    if x_as_labels:
        ax.set_xticks(range(len(x_values)), [str(value) for value in x_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    legend = ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#666666")
    legend.get_frame().set_linewidth(0.8)
    _caption(fig, caption)
    fig.tight_layout(rect=(0.04, 0.08, 0.99, 0.99))
    fig.savefig(output_path)
    plt.close(fig)


def paper_scatter_compare(
    panels: list[tuple[str, list[float], list[float]]],
    x_label: str,
    y_label: str,
    caption: str,
    output_path: Path,
) -> None:
    _apply_style()
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.0, 8.8), dpi=220, sharex=False)
    if len(panels) == 1:
        axes = [axes]
    for axis, (subtitle, x_values, y_values) in zip(axes, panels):
        axis.grid(True)
        axis.scatter(x_values, y_values, s=22, marker="s", c="#111111", edgecolors="#111111", linewidths=0.2)
        axis.set_title(subtitle, pad=6)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_xlim(-0.5, max(x_values) + 1.0 if x_values else 1.0)
    _caption(fig, caption)
    fig.tight_layout(rect=(0.05, 0.06, 0.99, 0.99))
    fig.savefig(output_path)
    plt.close(fig)


def paper_task_bar_chart(
    x_labels: list[int],
    series: list[tuple[str, list[float | None]]],
    x_label: str,
    y_label: str,
    caption: str,
    output_path: Path,
    figsize: tuple[float, float] = (7.0, 4.9),
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    ax.grid(True, axis="y")
    positions = np.arange(len(x_labels))
    width = 0.38 if len(series) == 2 else 0.8 / max(1, len(series))
    has_missing = False
    for index, (label, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2.0) * width
        bar_positions = []
        bar_values = []
        missing_positions = []
        for position, value in zip(positions, values):
            if value is None or not np.isfinite(value):
                missing_positions.append(position + offset)
                continue
            bar_positions.append(position + offset)
            bar_values.append(value)
        ax.bar(
            bar_positions,
            bar_values,
            width=width,
            label=label,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            edgecolor="#444444",
            linewidth=0.6,
            alpha=0.65,
        )
        if missing_positions:
            has_missing = True
            ax.scatter(
                missing_positions,
                [0.0 for _ in missing_positions],
                marker="x",
                s=22,
                color="#6C757D",
                linewidths=0.9,
                zorder=4,
                label="_nolegend_",
            )
    if has_missing:
        ax.scatter([], [], marker="x", s=22, color="#6C757D", linewidths=0.9, label="未成交")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(positions[::5], [str(x_labels[index]) for index in range(0, len(x_labels), 5)])
    legend = ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#666666")
    legend.get_frame().set_linewidth(0.8)
    _caption(fig, caption)
    fig.tight_layout(rect=(0.05, 0.08, 0.99, 0.99))
    fig.savefig(output_path)
    plt.close(fig)


def paper_mechanism_summary_chart(
    mechanisms: list[str],
    completion_rates: dict[str, list[float]],
    completed_counts: dict[str, list[int]],
    total_tasks: int,
    bid_samples: dict[str, list[float]],
    price_samples: dict[str, list[float]],
    caption: str,
    output_path: Path,
) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), dpi=220)

    x_positions = np.arange(len(mechanisms))
    mean_rates = [float(np.mean(completion_rates.get(mechanism, [0.0]))) for mechanism in mechanisms]
    axes[0].bar(
        x_positions,
        mean_rates,
        color=[SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(len(mechanisms))],
        edgecolor="#444444",
        linewidth=0.7,
        alpha=0.72,
    )
    axes[0].set_xticks(x_positions, mechanisms)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("任务完成率")
    axes[0].set_title("(a) 任务完成率", pad=6)
    axes[0].grid(True, axis="y")
    for position, mechanism, rate in zip(x_positions, mechanisms, mean_rates):
        counts = completed_counts.get(mechanism, [])
        mean_count = float(np.mean(counts)) if counts else 0.0
        axes[0].text(position, rate + 0.03, f"{mean_count:.1f}/{total_tasks}", ha="center", va="bottom", fontsize=8)

    _boxplot_panel(axes[1], mechanisms, bid_samples, "(b) 中标雾节点出价", "雾节点出价")
    _boxplot_panel(axes[2], mechanisms, price_samples, "(c) 任务成交报酬", "任务成交报酬")

    _caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.1, 0.99, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def _boxplot_panel(axis: plt.Axes, mechanisms: list[str], samples: dict[str, list[float]], title: str, y_label: str) -> None:
    data = [samples.get(mechanism, []) for mechanism in mechanisms]
    positions = np.arange(1, len(mechanisms) + 1)
    box = axis.boxplot(
        data,
        positions=positions,
        labels=mechanisms,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        boxprops={"edgecolor": "#444444", "linewidth": 0.8},
        whiskerprops={"color": "#444444", "linewidth": 0.8},
        capprops={"color": "#444444", "linewidth": 0.8},
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(SERIES_COLORS[index % len(SERIES_COLORS)])
        patch.set_alpha(0.58)
    axis.set_title(title, pad=6)
    axis.set_ylabel(y_label)
    axis.grid(True, axis="y")
    for position, values in zip(positions, data):
        axis.text(
            position,
            0.96,
            f"n={len(values)}",
            ha="center",
            va="top",
            fontsize=8,
            transform=axis.get_xaxis_transform(),
        )


def paper_task_heatmap_pair(
    task_indexes: list[int],
    bid_values: dict[str, list[float | None]],
    price_values: dict[str, list[float | None]],
    mechanisms: list[str],
    caption: str,
    output_path: Path,
) -> None:
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 4.8), dpi=220, sharex=True)
    _task_heatmap_panel(axes[0], task_indexes, bid_values, mechanisms, "(a) 中标雾节点出价")
    _task_heatmap_panel(axes[1], task_indexes, price_values, mechanisms, "(b) 任务成交报酬")
    axes[1].set_xlabel("按时间成本升序排列的任务")
    _caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.09, 0.99, 0.98))
    fig.savefig(output_path)
    plt.close(fig)


def _task_heatmap_panel(
    axis: plt.Axes,
    task_indexes: list[int],
    values_by_mechanism: dict[str, list[float | None]],
    mechanisms: list[str],
    title: str,
) -> None:
    matrix = []
    mask = []
    for mechanism in mechanisms:
        values = values_by_mechanism.get(mechanism, [])
        row = []
        row_mask = []
        for task_index in task_indexes:
            value = values[task_index - 1] if task_index - 1 < len(values) else None
            row.append(np.nan if value is None else value)
            row_mask.append(value is None)
        matrix.append(row)
        mask.append(row_mask)
    data = np.ma.masked_invalid(np.array(matrix, dtype=float))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#F7F7F7")
    image = axis.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap)
    axis.set_yticks(range(len(mechanisms)), mechanisms)
    tick_positions = list(range(0, len(task_indexes), 5))
    axis.set_xticks(tick_positions, [str(task_indexes[index]) for index in tick_positions])
    axis.set_title(title, pad=6)
    axis.grid(False)
    for row_index, row_mask in enumerate(mask):
        for column_index, is_missing in enumerate(row_mask):
            if is_missing:
                axis.text(column_index, row_index, "×", ha="center", va="center", color="#6C757D", fontsize=7)
    colorbar = axis.figure.colorbar(image, ax=axis, shrink=0.8, pad=0.01)
    colorbar.ax.tick_params(labelsize=7)


def paper_grouped_bar_chart(
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    x_label: str,
    y_label: str,
    caption: str,
    output_path: Path,
    figsize: tuple[float, float] = (7.2, 4.9),
    y_lim: tuple[float, float] | None = None,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    ax.grid(True, axis="y")
    positions = np.arange(len(x_labels))
    width = 0.8 / max(1, len(series))
    for index, (label, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2.0) * width
        plot_values = [np.nan if value is None else value for value in values]
        ax.bar(
            positions + offset,
            plot_values,
            width=width,
            label=label,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            edgecolor="#444444",
            linewidth=0.7,
            alpha=0.72,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(positions, x_labels, rotation=0)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    legend = ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#666666")
    legend.get_frame().set_linewidth(0.8)
    _caption(fig, caption)
    fig.tight_layout(rect=(0.05, 0.08, 0.99, 0.99))
    fig.savefig(output_path)
    plt.close(fig)


def write_dashboard(title: str, figures: list[tuple[str, str]], output_path: Path) -> None:
    cards = []
    for label, path in figures:
        cards.append(f'<section class="card"><h2>{label}</h2><img src="{path}" alt="{label}"></section>')
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      background: #f4f5f7;
      color: #111827;
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    p {{
      margin: 0 0 22px;
      color: #4b5563;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d1d5db;
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }}
    .card h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    img {{
      width: 100%;
      display: block;
      background: #fff;
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>{title}</h1>
    <p>以下结果已经切换为 PNG，并尽量靠近论文中的绘制方式与版式风格。</p>
    {"".join(cards)}
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
