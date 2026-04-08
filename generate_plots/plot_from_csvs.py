#!/usr/bin/env python3
"""Plot perf vs train_time and perf vs memory_usage from CSV files.

Input: a folder path. The script recursively finds all .csv files, loads rows,
and generates plots grouped by task and dataset.

For each (task, dataset), it creates figures with 3 subplots for num_layers = 2, 3, 4.
- x-axis: train_time (figure 1) / memory_usage (figure 2)
- y-axis: perf
- each point: one row (experiment run)
- marker shape: model family (GCN/SAGE/GAT)
- color: mode (bp/sf/ff), where exp_setting second token defines mode,
         and fl is mapped to sf.

Output folder structure:
<output>/plots/<task>/<dataset>/perf_vs_time_layers_2_3_4.png
<output>/plots/<task>/<dataset>/perf_vs_memory_usage_layers_2_3_4.png
<output>/plots/<task>/<dataset>/perf_vs_time_cache_vs_noncache_layers_2_3_4.png
<output>/plots/<task>/<dataset>/perf_vs_memory_usage_cache_vs_noncache_layers_2_3_4.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MODEL_MARKER = {
    "GCN": "o",
    "SAGE": "s",
    "GAT": "^",
}

MODE_COLOR = {
    "bp": "tab:blue",
    "sf": "tab:orange",
    "ff": "tab:green",
}

CACHE_VARIANT_COLOR = {
    # Non-cached: blue family
    "sf": "#1f77b4",
    "ff": "#6baed6",
    # Cached: red family
    "sf-cached": "#d62728",
    "ff-cached": "#ff9896",
}

TOPDOWN_NODE_COLOR = {
    "sf": "#1f77b4",          # blue
    "sf-top2loss": "#d62728", # red
    "sf-top2input": "#ffdf00",# yellow
}

TOPDOWN_LINK_COLOR = {
    "sf": "#1f77b4",          # blue
    "sf-topdown": "#d62728",  # red
}


def parse_float(v: str) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_int(v: str) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_mode(exp_setting: str) -> Optional[str]:
    if not exp_setting:
        return None
    parts = [p for p in str(exp_setting).strip().lower().replace("_", "-").split("-") if p]

    # Accept token anywhere in exp_setting (e.g., node-sf-top2input-cached, link-fl-topdown-cached, etc.)
    for token in parts:
        if token == "fl":
            return "sf"
        if token in {"bp", "sf", "ff"}:
            return token

    # Fallback aliases
    if "backprop" in parts:
        return "bp"
    return None


def parse_model_family(model: str) -> Optional[str]:
    if not model:
        return None
    s = str(model).upper()
    for name in ("GAT", "SAGE", "GCN"):
        if name in s:
            return name
    return None


def parse_task(task: str) -> Optional[str]:
    if not task:
        return None
    t = str(task).strip().lower().replace("_", "-")
    if t == "link-pred":
        return "link_pred"
    if t == "node-class":
        return "node_class"
    return None


def parse_cache_variant(row: Dict[str, str], mode: str) -> Optional[str]:
    """Return one of {sf, sf-cached, ff, ff-cached} for cache-comparison plots.

    Excludes:
    - bp
    - topdown/top2input/top2loss variants
    """
    if mode not in {"sf", "ff"}:
        return None

    model = (row.get("model") or "").strip().lower().replace("_", "-")
    exp_setting = (row.get("exp_setting") or "").strip().lower().replace("_", "-")
    text = f"{model} {exp_setting}"
    if any(tok in text for tok in ("topdown", "top2input", "top2loss")):
        return None

    is_cached = "cached" in text
    return f"{mode}-cached" if is_cached else mode


def parse_topdown_variant(task: str, row: Dict[str, str], mode: str) -> Optional[str]:
    """Return topdown comparison variant for SF-only settings.

    node_class: sf / sf-top2loss / sf-top2input
    link_pred: sf / sf-topdown
    """
    if mode != "sf":
        return None

    model = (row.get("model") or "").strip().lower().replace("_", "-")
    exp_setting = (row.get("exp_setting") or "").strip().lower().replace("_", "-")
    topdown_model = (row.get("topdown_model") or "").strip().lower().replace("_", "-")
    text = f"{model} {exp_setting} {topdown_model}"

    if task == "node_class":
        if "top2input" in text:
            return "sf-top2input"
        if "top2loss" in text:
            return "sf-top2loss"
        return "sf"

    if task == "link_pred":
        if "topdown" in text:
            return "sf-topdown"
        return "sf"

    return None


def load_rows_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return []
    return rows


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    m = n // 2
    if n % 2 == 1:
        return vals[m]
    return 0.5 * (vals[m - 1] + vals[m])


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def build_setup_signature(row: Dict[str, str], *, exclude_fields: set[str]) -> Tuple[Tuple[str, str], ...]:
    items: List[Tuple[str, str]] = []
    for k, v in row.items():
        if k in exclude_fields:
            continue
        items.append((k, (v or "").strip()))
    items.sort(key=lambda x: x[0])
    return tuple(items)


def collect_points(
    csv_files: List[Path],
) -> Tuple[Dict[str, Dict[str, Dict[int, List[dict]]]], Dict[str, Dict[int, List[dict]]]]:
    # task -> dataset -> num_layers -> [points]
    grouped_by_dataset: Dict[str, Dict[str, Dict[int, List[dict]]]] = {}
    # task -> num_layers -> [points] (all datasets together)
    grouped_by_task: Dict[str, Dict[int, List[dict]]] = {}

    exclude_run_fields = {
        "perf",
        "train_time",
        "memory_usage",
        "run_i",
        "run_seed",
        "best_val_epoch",
        "train_epochs",
        "exp_datetime",
    }

    # setup_key -> accumulator
    setup_aggr: Dict[tuple, dict] = {}

    for csv_file in sorted(csv_files):
        for row in load_rows_from_csv(csv_file):
            dataset = (row.get("dataset") or "").strip()
            if not dataset:
                continue

            task = parse_task(row.get("task", ""))
            if task is None:
                continue

            num_layers = parse_int(row.get("num_layers", ""))
            if num_layers not in {1, 2, 3, 4}:
                continue

            perf = parse_float(row.get("perf", ""))
            train_time = parse_float(row.get("train_time", ""))
            memory_usage = parse_float(row.get("memory_usage", ""))
            if perf is None or train_time is None:
                continue
            if perf > 1:
                perf = perf / 100.0

            mode = parse_mode(row.get("exp_setting", ""))
            model_family = parse_model_family(row.get("model", ""))
            if mode is None or model_family is None:
                continue

            cache_variant = parse_cache_variant(row, mode)
            topdown_variant = parse_topdown_variant(task, row, mode)

            setup_key = build_setup_signature(row, exclude_fields=exclude_run_fields)
            base_setup_key = build_setup_signature(row, exclude_fields=exclude_run_fields | {"num_layers"})

            entry = setup_aggr.setdefault(
                setup_key,
                {
                    "task": task,
                    "dataset": dataset,
                    "num_layers": num_layers,
                    "mode": mode,
                    "cache_variant": cache_variant,
                    "topdown_variant": topdown_variant,
                    "model_family": model_family,
                    "base_setup_key": base_setup_key,
                    "perf_vals": [],
                    "train_time_vals": [],
                    "memory_vals": [],
                },
            )
            entry["perf_vals"].append(perf)
            entry["train_time_vals"].append(train_time)
            if memory_usage is not None and memory_usage > 0:
                entry["memory_vals"].append(memory_usage)

    aggregated: List[dict] = []
    for entry in setup_aggr.values():
        perf_mean = mean(entry["perf_vals"])
        train_time_mean = mean(entry["train_time_vals"])
        memory_mean = mean(entry["memory_vals"])
        if perf_mean is None or train_time_mean is None:
            continue
        aggregated.append(
            {
                "task": entry["task"],
                "dataset": entry["dataset"],
                "num_layers": entry["num_layers"],
                "mode": entry["mode"],
                "cache_variant": entry["cache_variant"],
                "topdown_variant": entry["topdown_variant"],
                "model_family": entry["model_family"],
                "base_setup_key": entry["base_setup_key"],
                "perf": perf_mean,
                "train_time": train_time_mean,
                "memory_usage": memory_mean,
            }
        )

    baseline_candidates: Dict[tuple, List[float]] = {}
    for item in aggregated:
        if item["num_layers"] == 1 and item["memory_usage"] is not None:
            baseline_candidates.setdefault(item["base_setup_key"], []).append(item["memory_usage"])

    baseline_memory: Dict[tuple, float] = {}
    for key, values in baseline_candidates.items():
        m = median(values)
        if m is not None and m > 0:
            baseline_memory[key] = m

    for item in aggregated:
        num_layers = item["num_layers"]
        if num_layers not in {2, 3, 4}:
            continue

        baseline = baseline_memory.get(item["base_setup_key"])
        memory_usage = item["memory_usage"]
        memory_usage_rel = None
        if baseline is not None and memory_usage is not None and baseline > 0:
            memory_usage_rel = memory_usage / baseline

        point = {
            "y": item["perf"],
            "train_time": item["train_time"],
            "memory_usage_rel": memory_usage_rel,
            "mode": item["mode"],
            "cache_variant": item["cache_variant"],
            "topdown_variant": item["topdown_variant"],
            "model_family": item["model_family"],
        }
        grouped_by_dataset.setdefault(item["task"], {}).setdefault(item["dataset"], {}).setdefault(num_layers, []).append(point)
        grouped_by_task.setdefault(item["task"], {}).setdefault(num_layers, []).append(point)

    return grouped_by_dataset, grouped_by_task


def plot_dataset(
    task: str,
    dataset: str,
    layer_points: Dict[int, List[dict]],
    output_dir: Path,
    *,
    x_key: str,
    x_label: str,
    file_name: str,
    title_suffix: str,
    group_key: str,
    group_color: Dict[str, str],
    group_order: List[str],
    point_filter: Optional[Callable[[dict], bool]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layer_order = [2, 3, 4]

    for ax, nl in zip(axes, layer_order):
        points = layer_points.get(nl, [])
        plotted = 0
        for p in points:
            if point_filter is not None and not point_filter(p):
                continue
            x = p.get(x_key)
            if x is None:
                continue
            group = p.get(group_key)
            if group not in group_color:
                continue
            ax.scatter(
                x,
                p["y"],
                marker=MODEL_MARKER[p["model_family"]],
                c=group_color[group],
                s=40,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.3,
            )
            plotted += 1

        ax.set_title(f"{dataset} | num_layers={nl}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("perf")
        ax.grid(True, alpha=0.25)
        if plotted == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Legend: combined model-group entries
    legend_order = [(m, g) for m in ("GAT", "GCN", "SAGE") for g in group_order]
    combo_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKER[model_name],
            linestyle="",
            color=group_color[group_name],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=8,
            label=f"{model_name}-{group_name}",
        )
        for model_name, group_name in legend_order
    ]

    fig.legend(
        handles=combo_handles,
        loc="center left",
        bbox_to_anchor=(0.90, 0.5),
        ncol=1,
        frameon=True,
        title="Legend",
    )
    fig.suptitle(f"Perf vs {title_suffix} by Layers | task={task} | dataset={dataset} | mean over run_i")
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.94))

    out_file = output_dir / file_name
    fig.savefig(str(out_file), dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_plot_set(task: str, dataset: str, layer_points: Dict[int, List[dict]], out_dir: Path) -> None:
    plot_dataset(
        task,
        dataset,
        layer_points,
        out_dir,
        x_key="train_time",
        x_label="train_time",
        file_name="perf_vs_time_layers_2_3_4.png",
        title_suffix="Train Time",
        group_key="mode",
        group_color=MODE_COLOR,
        group_order=["bp", "sf", "ff"],
    )
    plot_dataset(
        task,
        dataset,
        layer_points,
        out_dir,
        x_key="memory_usage_rel",
        x_label="memory_usage (x vs num_layers=1)",
        file_name="perf_vs_memory_usage_layers_2_3_4.png",
        title_suffix="Memory Usage",
        group_key="mode",
        group_color=MODE_COLOR,
        group_order=["bp", "sf", "ff"],
    )
    plot_dataset(
        task,
        dataset,
        layer_points,
        out_dir,
        x_key="train_time",
        x_label="train_time",
        file_name="perf_vs_time_cache_vs_noncache_layers_2_3_4.png",
        title_suffix="Train Time (Cache vs Non-Cache)",
        group_key="cache_variant",
        group_color=CACHE_VARIANT_COLOR,
        group_order=["ff", "ff-cached", "sf", "sf-cached"],
        point_filter=lambda p: p.get("cache_variant") in CACHE_VARIANT_COLOR and p.get("model_family") != "GAT",
    )
    plot_dataset(
        task,
        dataset,
        layer_points,
        out_dir,
        x_key="memory_usage_rel",
        x_label="memory_usage (x vs num_layers=1)",
        file_name="perf_vs_memory_usage_cache_vs_noncache_layers_2_3_4.png",
        title_suffix="Memory Usage (Cache vs Non-Cache)",
        group_key="cache_variant",
        group_color=CACHE_VARIANT_COLOR,
        group_order=["ff", "ff-cached", "sf", "sf-cached"],
        point_filter=lambda p: p.get("cache_variant") in CACHE_VARIANT_COLOR and p.get("model_family") != "GAT",
    )

    topdown_colors = TOPDOWN_NODE_COLOR if task == "node_class" else TOPDOWN_LINK_COLOR
    topdown_order = ["sf", "sf-top2loss", "sf-top2input"] if task == "node_class" else ["sf", "sf-topdown"]
    plot_dataset(
        task,
        dataset,
        layer_points,
        out_dir,
        x_key="train_time",
        x_label="train_time",
        file_name="perf_vs_time_topdown_vs_nontopdown_layers_2_3_4.png",
        title_suffix="Train Time (Topdown vs Non-Topdown)",
        group_key="topdown_variant",
        group_color=topdown_colors,
        group_order=topdown_order,
        point_filter=lambda p, c=topdown_colors: p.get("topdown_variant") in c,
    )
    plot_dataset(
        task,
        dataset,
        layer_points,
        out_dir,
        x_key="memory_usage_rel",
        x_label="memory_usage (x vs num_layers=1)",
        file_name="perf_vs_memory_usage_topdown_vs_nontopdown_layers_2_3_4.png",
        title_suffix="Memory Usage (Topdown vs Non-Topdown)",
        group_key="topdown_variant",
        group_color=topdown_colors,
        group_order=topdown_order,
        point_filter=lambda p, c=topdown_colors: p.get("topdown_variant") in c,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot perf vs train_time and perf vs memory_usage from CSV files.")
    parser.add_argument("input_folder", type=Path, help="CSV file or folder to recursively scan for CSV files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots"),
        help="Output root folder for plots (default: ./plots)",
    )
    args = parser.parse_args()

    root = args.input_folder.resolve()
    if not root.exists():
        raise SystemExit(f"Input path does not exist: {root}")

    if root.is_file():
        csv_files = [root] if root.suffix.lower() == ".csv" else []
        input_base = root.parent
    else:
        csv_files = sorted(root.rglob("*.csv"))
        input_base = root

    if not csv_files:
        raise SystemExit("No CSV files found.")

    output_root = args.output.resolve() / "plots"
    generated = 0

    for csv_file in csv_files:
        grouped_by_dataset, grouped_by_task = collect_points([csv_file])
        if not grouped_by_dataset and not grouped_by_task:
            continue

        rel_no_suffix = csv_file.relative_to(input_base).with_suffix("")
        csv_output_root = output_root / rel_no_suffix

        for task, task_group in grouped_by_dataset.items():
            for dataset, layer_points in task_group.items():
                generate_plot_set(task, dataset, layer_points, csv_output_root / task / dataset)

        for task, layer_points in grouped_by_task.items():
            generate_plot_set(task, "all_datasets", layer_points, csv_output_root / task / "all_datasets")

        generated += 1

    if generated == 0:
        raise SystemExit("No plottable rows found in CSV files.")

    print(f"Saved plots to: {output_root}")


if __name__ == "__main__":
    main()
