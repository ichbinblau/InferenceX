#!/usr/bin/env python3
"""Generate Pareto frontier + TTFT bar charts from InferenceX agg JSON results.

Reads per-concurrency JSON files produced by process_agentic_result.py /
process_result.py and emits a comparison figure matching the InferenceX
Pareto dashboard layout:

  Left : Pareto frontier scatter (Median Interactivity vs TTFT P95), one series per offloading mode
  Right: Grouped TTFT bar chart by concurrency

Usage:
    python3 generate_pareto_charts.py /it-share/thshan/InferenceX/json_result/vllm_l1
    python3 generate_pareto_charts.py /path/to/json_dir --output pareto.png
    python3 generate_pareto_charts.py /path/to/parent_dir --recursive
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy are required", file=sys.stderr)
    sys.exit(1)


SERIES_STYLES = [
    {"color": "#E8915C", "marker": "s", "label": "cached"},
    {"color": "#6A85AD", "marker": "o", "label": "uncached"},
    {"color": "#59A14F", "marker": "^", "label": "series-3"},
    {"color": "#AF7AA1", "marker": "D", "label": "series-4"},
]


@dataclass(frozen=True)
class ConfigGroup:
    hw: str
    model: str
    model_prefix: str
    framework: str
    precision: str
    disagg: bool

    @property
    def title_suffix(self) -> str:
        hw = self.hw.upper() if self.hw else "mi355x"
        is_disagg = self.disagg or "disagg" in self.framework.lower()
        disagg_tag = "disagg" if is_disagg else "agg"
        return f"{hw} {self.model} {self.framework} {self.precision} ({disagg_tag})"

    @classmethod
    def from_record(cls, record: dict) -> ConfigGroup:
        raw_model = str(record.get("model") or "unknown")
        # Model can be a filesystem path (e.g. /models/DeepSeek-R1-...); use basename.
        model = raw_model.rstrip("/").split("/")[-1] if "/" in raw_model else raw_model
        return cls(
            hw=str(record.get("hw") or "").strip(),
            model=model,
            model_prefix=str(record.get("infmax_model_prefix") or "unknown"),
            framework=str(record.get("framework") or "unknown"),
            precision=str(record.get("precision") or "unknown"),
            disagg=bool(record.get("disagg", False)),
        )


def load_records(results_dir: Path, recursive: bool) -> list[dict]:
    pattern = "**/*.json" if recursive else "*.json"
    records: list[dict] = []
    for path in sorted(results_dir.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)
    return records


def median_intvty(record: dict) -> float | None:
    """Return mean interactivity (tok/s/user); fall back to median when absent."""
    val = record.get("mean_intvty")
    if val is not None:
        return float(val)
    for key in ("median_intvty", "p50_intvty"):
        val = record.get(key)
        if val is not None:
            return float(val)
    return None


def enrich_median_intvty_from_aiperf(
    records: list[dict],
    artifacts_root: Path | None,
) -> None:
    """Fill median_intvty from aiperf aggregate p50 when JSON lacks it."""
    if artifacts_root is None:
        return
    for record in records:
        if record.get("median_intvty") is not None:
            continue
        conc = record.get("conc")
        if conc is None:
            continue
        agg_path = artifacts_root / f"conc{conc}" / "aiperf_artifacts" / "profile_export_aiperf.json"
        if not agg_path.exists():
            continue
        with open(agg_path) as f:
            agg = json.load(f)
        intvty = agg.get("output_token_throughput_per_user") or agg.get(
            "active_decode_throughput_per_user"
        )
        if isinstance(intvty, dict) and intvty.get("p50") is not None:
            p50 = float(intvty["p50"])
            if p50 > 0:
                record["median_intvty"] = p50


def is_valid_record(record: dict) -> bool:
    if record.get("num_requests_successful", 0) <= 0:
        return False
    if median_intvty(record) is None:
        return False
    if record.get("mean_ttft") is None:
        return False
    if record.get("tput_per_gpu") is None:
        return False
    return True


def offloading_label(record: dict) -> str:
    offloading = str(record.get("offloading") or "none").strip().lower()
    if offloading in ("", "false", "0"):
        offloading = "none"
    return f"offloading:{offloading}"


def compute_pareto_frontier(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    maximize_x: bool = False,
    maximize_y: bool = False,
) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points."""
    n = len(xs)
    if n == 0:
        return np.array([], dtype=bool)

    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_optimal[i]:
            continue
        for j in range(n):
            if i == j or not is_optimal[j]:
                continue
            x_better = xs[j] > xs[i] if maximize_x else xs[j] < xs[i]
            y_better = ys[j] > ys[i] if maximize_y else ys[j] < ys[i]
            x_equal = xs[j] == xs[i]
            y_equal = ys[j] == ys[i]
            if (x_better and (y_better or y_equal)) or (y_better and (x_better or x_equal)):
                is_optimal[i] = False
                break
    return is_optimal


def plot_pareto_panel(ax, series_data: dict[str, list[dict]], title: str) -> None:
    for idx, (series_name, points) in enumerate(sorted(series_data.items())):
        style = SERIES_STYLES[idx % len(SERIES_STYLES)]
        points = sorted(points, key=lambda r: r["conc"])
        xs = np.array([median_intvty(r) for r in points], dtype=float)
        ys = np.array([float(r["tput_per_gpu"]) for r in points], dtype=float)
        concs = [r["conc"] for r in points]

        ax.scatter(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            s=55,
            alpha=0.85,
            label=series_name,
            zorder=3,
        )

        order = np.argsort(xs)
        ax.plot(
            xs[order],
            ys[order],
            color=style["color"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            zorder=2,
        )

        for x, y, conc in zip(xs, ys, concs, strict=True):
            ax.annotate(
                str(conc),
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="#333333",
            )

    ax.set_xlabel("Mean Interactivity (tok/s/user)")
    ax.set_ylabel("Total Throughput per GPU (tok/s)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8)


def plot_ttft_bar_panel(ax, series_data: dict[str, list[dict]], title: str) -> None:
    all_concs = sorted({r["conc"] for pts in series_data.values() for r in pts})
    if not all_concs:
        ax.set_title(title, fontsize=10)
        return

    x = np.arange(len(all_concs))
    n_series = len(series_data)
    width = 0.8 / max(n_series, 1)

    for idx, (series_name, points) in enumerate(sorted(series_data.items())):
        style = SERIES_STYLES[idx % len(SERIES_STYLES)]
        by_conc = {r["conc"]: r["mean_ttft"] * 1000.0 for r in points}
        heights = [by_conc.get(conc, np.nan) for conc in all_concs]
        offset = (idx - (n_series - 1) / 2) * width
        ax.bar(
            x + offset,
            heights,
            width=width,
            color=style["color"],
            label=series_name,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in all_concs])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TTFT Mean (msec)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)


def build_groups(records: list[dict]) -> dict[ConfigGroup, dict[str, list[dict]]]:
    grouped: dict[ConfigGroup, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if not is_valid_record(record):
            continue
        group = ConfigGroup.from_record(record)
        series = offloading_label(record)
        grouped[group][series].append(record)
    for group in grouped:
        for series in grouped[group]:
            grouped[group][series].sort(key=lambda r: int(r["conc"]))
    return grouped


def generate_figure(
    grouped: dict[ConfigGroup, dict[str, list[dict]]],
    output: Path,
    *,
    dpi: int = 150,
) -> None:
    if not grouped:
        raise SystemExit(
            "No valid benchmark records found "
            "(need median/mean interactivity + p95_ttft + successful requests)."
        )

    groups = sorted(grouped.keys(), key=lambda g: (g.model_prefix, g.framework, g.precision))
    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows), squeeze=False)

    for row, group in enumerate(groups):
        series_data = grouped[group]
        suffix = group.title_suffix
        plot_pareto_panel(axes[row, 0], series_data, f"Pareto Frontier - {suffix}")
        plot_ttft_bar_panel(axes[row, 1], series_data, f"TTFT - {suffix}")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Directory containing agg JSON files")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG path (default: <results_dir>/pareto_frontier.png)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for *.json under results_dir",
    )
    parser.add_argument(
        "--aiperf-artifacts",
        type=Path,
        default=None,
        help=(
            "Optional path to aiperf run dir (e.g. .../vllm-disagg_isl_1024_osl_1024) "
            "to load median interactivity from profile_export_aiperf.json p50"
        ),
    )
    args = parser.parse_args()

    records = load_records(args.results_dir, args.recursive)
    enrich_median_intvty_from_aiperf(records, args.aiperf_artifacts)
    grouped = build_groups(records)
    output = args.output or (args.results_dir / "pareto_frontier.png")
    generate_figure(grouped, output)

    n_points = sum(len(v) for series in grouped.values() for v in series.values())
    print(f"Saved {output}")
    print(f"  Config groups: {len(grouped)} | Data points: {n_points}")
    for group, series_data in sorted(grouped.items(), key=lambda kv: kv[0].title_suffix):
        concs = sorted({r['conc'] for pts in series_data.values() for r in pts})
        series_names = ", ".join(sorted(series_data))
        print(f"  - {group.title_suffix}: series=[{series_names}] conc={concs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
