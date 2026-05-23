"""Comparison and statistical analysis module for benchmark results."""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import stdev
from typing import Optional

from scipy.stats import t as t_dist
from scipy.stats import ttest_ind

from .config import CONDUCTRESS_OUTPUT

logger = logging.getLogger(__name__)

# GroupKey: (method, val_size, key_size, io_threads, pipelining, make_args)
GroupKey = tuple[str, int, int, int, int, str]


@dataclass
class ComparisonRow:
    """A single row in the comparison table."""

    method: str
    val_size: int
    key_size: int
    io_threads: int
    pipelining: int
    make_args: str
    mean_a: float
    mean_b: float
    delta_pct: float
    p_value: Optional[float]  # None if insufficient data
    n_a: int  # sample count for specifier A
    n_b: int  # sample count for specifier B
    ci_a_pct: Optional[float] = None  # 95% CI as % of mean for A
    ci_b_pct: Optional[float] = None  # 95% CI as % of mean for B


class AnalysisModule:
    """Statistical comparison module for benchmark results."""

    def __init__(self, results_path: Path = CONDUCTRESS_OUTPUT) -> None:
        self.results_path = results_path

    def load_results(
        self,
        source_filter: Optional[str] = None,
        method_filter: Optional[str] = None,
    ) -> list[dict]:
        """Load and filter results from output.jsonl.

        Args:
            source_filter: If set, only include results matching this source.
            method_filter: If set, only include results matching this method.

        Returns:
            List of result dicts parsed from each JSONL line.
        """
        if not self.results_path.exists():
            logger.warning("No results file found at %s", self.results_path)
            return []

        results: list[dict] = []
        with open(self.results_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON on line %d", line_num)
                    continue

                if source_filter and record.get("source") != source_filter:
                    continue
                if method_filter and record.get("method") != method_filter:
                    continue

                results.append(record)

        return results

    def _extract_group_key(self, record: dict) -> GroupKey:
        """Extract the group key tuple from a result record."""
        data = record.get("data", {})
        return (
            record.get("method", ""),
            data.get("size", 0),
            data.get("key_size", 0),
            data.get("io-threads", 0),
            data.get("pipeline", 0),
            record.get("make_args", ""),
        )

    def _extract_samples(self, record: dict) -> list[float]:
        """Extract RPS samples from a result record.

        For aggregated results (repetitions > 1), use stored per_run_rps.
        For single-run results, use score as a single sample.
        """
        data = record.get("data", {})
        repetitions = data.get("repetitions", 1)

        if repetitions > 1 and "per_run_rps" in data:
            return list(data["per_run_rps"])
        else:
            score = record.get("score")
            if score is not None:
                return [float(score)]
            return []

    def group_results(
        self,
        results: list[dict],
        specifier_a: str,
        specifier_b: str,
    ) -> dict[GroupKey, dict[str, list[float]]]:
        """Group results by parameter combination and collect samples per specifier.

        Args:
            results: List of result dicts from load_results().
            specifier_a: First specifier to compare.
            specifier_b: Second specifier to compare.

        Returns:
            Dict mapping GroupKey to {"a": [samples], "b": [samples]}.
        """
        groups: dict[GroupKey, dict[str, list[float]]] = {}

        for record in results:
            specifier = record.get("specifier", "")
            if specifier not in (specifier_a, specifier_b):
                continue

            key = self._extract_group_key(record)
            if key not in groups:
                groups[key] = {"a": [], "b": []}

            samples = self._extract_samples(record)
            side = "a" if specifier == specifier_a else "b"
            groups[key][side].extend(samples)

        return groups

    def compare(
        self,
        specifier_a: str,
        specifier_b: str,
        source_filter: Optional[str] = None,
        method_filter: Optional[str] = None,
    ) -> list[ComparisonRow]:
        """Compare two specifiers across all matching parameter groups.

        Args:
            specifier_a: First specifier (branch/tag/hash).
            specifier_b: Second specifier (branch/tag/hash).
            source_filter: Optional source filter.
            method_filter: Optional method filter.

        Returns:
            List of ComparisonRow with statistics for each parameter group.
        """
        results = self.load_results(
            source_filter=source_filter,
            method_filter=method_filter,
        )

        if not results:
            return []

        groups = self.group_results(results, specifier_a, specifier_b)
        rows: list[ComparisonRow] = []

        for group_key, samples_dict in sorted(groups.items()):
            samples_a = samples_dict["a"]
            samples_b = samples_dict["b"]

            n_a = len(samples_a)
            n_b = len(samples_b)

            if n_a == 0 or n_b == 0:
                continue

            mean_a = sum(samples_a) / n_a
            mean_b = sum(samples_b) / n_b

            if mean_a != 0:
                delta_pct = ((mean_b - mean_a) / mean_a) * 100.0
            else:
                delta_pct = 0.0

            # Perform Welch's t-test if both sides have >= 2 samples
            p_value: Optional[float] = None
            if n_a >= 2 and n_b >= 2:
                _, p_value = ttest_ind(samples_a, samples_b, equal_var=False)

            # Compute 95% CI as percentage of mean for each side
            ci_a_pct: Optional[float] = None
            ci_b_pct: Optional[float] = None
            if n_a >= 2 and mean_a > 0:
                ci_a = t_dist.ppf(0.975, n_a - 1) * (stdev(samples_a) / sqrt(n_a))
                ci_a_pct = (ci_a / mean_a) * 100.0
            if n_b >= 2 and mean_b > 0:
                ci_b = t_dist.ppf(0.975, n_b - 1) * (stdev(samples_b) / sqrt(n_b))
                ci_b_pct = (ci_b / mean_b) * 100.0

            method, val_size, key_size, io_threads, pipelining, make_args = group_key

            rows.append(
                ComparisonRow(
                    method=method,
                    val_size=val_size,
                    key_size=key_size,
                    io_threads=io_threads,
                    pipelining=pipelining,
                    make_args=make_args,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    delta_pct=delta_pct,
                    p_value=p_value,
                    n_a=n_a,
                    n_b=n_b,
                    ci_a_pct=ci_a_pct,
                    ci_b_pct=ci_b_pct,
                )
            )

        return rows

    def format_table(self, rows: list[ComparisonRow]) -> str:
        """Format comparison results as a readable table.

        Args:
            rows: List of ComparisonRow from compare().

        Returns:
            Formatted table string with CI columns and a measurement quality summary.
        """
        if not rows:
            return "No comparison data available."

        # Header
        header = (
            f"{'Test':<12}| {'Size':>6} | {'Key':>5} | {'IO':>3} | {'Pipe':>4} "
            f"| {'Mean A':>14} | {'±CI%':>5} | {'Mean B':>14} | {'±CI%':>5} "
            f"| {'Delta':>7} | {'p-value':>8} | {'n':>3}"
        )
        separator = (
            f"{'-' * 12}+{'-' * 8}+{'-' * 7}+{'-' * 5}+{'-' * 6}"
            f"+{'-' * 16}+{'-' * 7}+{'-' * 16}+{'-' * 7}"
            f"+{'-' * 9}+{'-' * 10}+{'-' * 5}"
        )

        lines = [header, separator]

        ci_pcts = []
        significant_count = 0

        for row in rows:
            mean_a_str = f"{row.mean_a:,.0f} rps"
            mean_b_str = f"{row.mean_b:,.0f} rps"
            delta_str = f"{row.delta_pct:+.2f}%"

            ci_a_str = f"{row.ci_a_pct:.1f}%" if row.ci_a_pct is not None else "N/A"
            ci_b_str = f"{row.ci_b_pct:.1f}%" if row.ci_b_pct is not None else "N/A"

            if row.ci_a_pct is not None:
                ci_pcts.append(row.ci_a_pct)
            if row.ci_b_pct is not None:
                ci_pcts.append(row.ci_b_pct)

            if row.p_value is not None:
                p_str = f"{row.p_value:.4f}"
                if row.p_value < 0.05:
                    significant_count += 1
            else:
                p_str = "N/A"

            n_str = f"{row.n_a}/{row.n_b}"
            size_str = _format_size(row.val_size)

            line = (
                f"{row.method:<12}| {size_str:>6} | {row.key_size:>5} | {row.io_threads:>3} "
                f"| {row.pipelining:>4} | {mean_a_str:>14} | {ci_a_str:>5} "
                f"| {mean_b_str:>14} | {ci_b_str:>5} "
                f"| {delta_str:>7} | {p_str:>8} | {n_str:>3}"
            )
            lines.append(line)

        # Summary
        lines.append("")
        lines.append(f"Comparisons: {len(rows)}")
        lines.append(f"Significant (p < 0.05): {significant_count}/{len(rows)}")

        if ci_pcts:
            avg_ci = sum(ci_pcts) / len(ci_pcts)
            max_ci = max(ci_pcts)
            lines.append(
                f"Measurement precision: avg ±{avg_ci:.2f}%, max ±{max_ci:.2f}% (95% CI as % of mean)"
            )

            # Guidance on whether more testing would help
            if avg_ci > 2.0:
                lines.append(
                    "⚠ High variance — consider more repetitions to improve precision."
                )
            elif avg_ci > 1.0:
                lines.append(
                    "Moderate precision — more repetitions would help detect sub-1% effects."
                )
            else:
                lines.append("Good precision — sufficient to detect effects ≥1%.")

            # Detectable effect size estimate (rough: need delta > 2x pooled CI to be significant)
            lines.append(
                f"Minimum detectable effect: ~±{avg_ci * 2:.1f}% (approximate)"
            )

        return "\n".join(lines)


def _format_size(size_bytes: int) -> str:
    """Format a byte size into a human-readable string."""
    if size_bytes == 0:
        return "0"
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        kb = size_bytes / 1024
        if kb == int(kb):
            return f"{int(kb)}KB"
        return f"{kb:.1f}KB"
    else:
        mb = size_bytes / (1024 * 1024)
        if mb == int(mb):
            return f"{int(mb)}MB"
        return f"{mb:.1f}MB"


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the analysis CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.analysis",
        description="Compare benchmark results between two specifiers.",
    )
    parser.add_argument(
        "specifier_a",
        help="First specifier (branch, tag, or commit hash)",
    )
    parser.add_argument(
        "specifier_b",
        help="Second specifier (branch, tag, or commit hash)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Filter results by source repository name",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Filter results by test method (e.g., perf-get)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the analysis module.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    module = AnalysisModule()

    if not module.results_path.exists():
        print(f"No results file found at {module.results_path}", file=sys.stderr)
        return 1

    rows = module.compare(
        specifier_a=args.specifier_a,
        specifier_b=args.specifier_b,
        source_filter=args.source,
        method_filter=args.method,
    )

    if not rows:
        print(
            f"No matching results found for {args.specifier_a} and {args.specifier_b}",
            file=sys.stderr,
        )
        return 0

    table = module.format_table(rows)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
