"""
Data Quality Validation Framework
Week 3 — Operationalizing AI | Mrinal Goel

Issues identified in demand_enriched_corrupted.parquet (post Jan 16, 2026):
  1. Negative trip_count values (353 rows) — physically impossible
  2. Extreme outliers in trip_count (311 rows > 1000, max = 99,999)
  3. Duplicate rows (8,134 rows) — inflates demand signals
  4. Distribution shift — std ratio 112x baseline (21 -> 2433)
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List

CUTOFF = pd.Timestamp("2026-01-16")
DATA_PATH = "week3/data/demand_enriched_corrupted.parquet"

REQUIRED_COLUMNS = [
    "PULocationID", "time_bucket", "trip_count", "hour", "minute",
    "dayofweek", "is_weekend", "month", "dayofyear", "weekofyear", "year",
    "slot_of_day", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_holiday", "cbd_pricing_active",
    "borough_id", "service_zone_id", "is_airport_zone", "zone_slot_baseline",
    "lag_15min", "lag_1h", "lag_2h", "lag_1day", "lag_1week",
    "roll_mean_1h", "roll_mean_2h", "roll_mean_1day",
]


class DataQualityValidator:
    """Validates incoming data against quality expectations derived from baseline."""

    def __init__(self, baseline_df: pd.DataFrame = None):
        self.baseline = baseline_df
        self.issues = []
        if baseline_df is not None:
            self._baseline_stats = {
                "trip_count_mean": baseline_df["trip_count"].mean(),
                "trip_count_std": baseline_df["trip_count"].std(),
                "trip_count_max": baseline_df["trip_count"].max(),
                "trip_count_median": baseline_df["trip_count"].median(),
                "trip_count_mad": (baseline_df["trip_count"] - baseline_df["trip_count"].median()).abs().median(),
            }
        else:
            self._baseline_stats = {
                "trip_count_mean": 17.0,
                "trip_count_std": 21.6,
                "trip_count_max": 310.0,
                "trip_count_median": 9.0,
                "trip_count_mad": 8.0,
            }

    def validate(self, df: pd.DataFrame) -> Dict:
        """Run all validation checks. Returns structured result dict."""
        self.issues = []
        self.check_schema(df)
        self.check_value_ranges(df)
        self.check_duplicates(df)
        self.check_distributions(df)
        return {
            "is_valid": len(self.issues) == 0,
            "num_issues": len(self.issues),
            "issues": self.issues,
        }

    def check_schema(self, df: pd.DataFrame):
        """Check required columns exist."""
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            self._add_issue(
                issue_type="schema_violation",
                severity="critical",
                description=f"Missing required columns: {missing}",
                count=len(missing),
                missing_columns=missing,
            )

    def check_value_ranges(self, df: pd.DataFrame):
        """Issue 1: Negative trip counts. Issue 2: Extreme outliers."""
        if df.empty:
            return

        # Issue 1 — negative trip_count
        neg_mask = df["trip_count"] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            self._add_issue(
                issue_type="negative_trip_count",
                severity="critical",
                description=(
                    f"{neg_count} rows have negative trip_count values "
                    f"(min={df.loc[neg_mask, 'trip_count'].min()}). "
                    "Trip counts are physically impossible below 0."
                ),
                count=int(neg_count),
                min_value=float(df.loc[neg_mask, "trip_count"].min()),
            )

        # Issue 2 — extreme outliers (>10x baseline max)
        outlier_threshold = self._baseline_stats["trip_count_max"] * 10
        outlier_mask = df["trip_count"] > outlier_threshold
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            self._add_issue(
                issue_type="outlier_trip_count",
                severity="high",
                description=(
                    f"{outlier_count} rows have trip_count > {outlier_threshold:.0f} "
                    f"(baseline max={self._baseline_stats['trip_count_max']:.0f}, "
                    f"corrupted max={df['trip_count'].max():.0f}). "
                    "Extreme outliers will distort model predictions."
                ),
                count=int(outlier_count),
                threshold=float(outlier_threshold),
                max_value=float(df["trip_count"].max()),
            )

    def check_duplicates(self, df: pd.DataFrame):
        """Issue 3: Duplicate rows inflate demand signals."""
        if df.empty:
            return
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            dup_pct = dup_count / len(df) * 100
            self._add_issue(
                issue_type="duplicate_rows",
                severity="high",
                description=(
                    f"{dup_count} duplicate rows detected ({dup_pct:.1f}% of data). "
                    "Duplicates inflate zone demand counts and corrupt rolling mean features."
                ),
                count=int(dup_count),
                percentage=round(dup_pct, 2),
            )

    def check_distributions(self, df: pd.DataFrame):
        """Issue 4: Distribution shift via std ratio."""
        if df.empty:
            return
        current_std = df["trip_count"].std()
        baseline_std = self._baseline_stats["trip_count_std"]
        std_ratio = current_std / (baseline_std + 1e-9)

        if std_ratio > 5.0:
            self._add_issue(
                issue_type="distribution_shift",
                severity="high",
                description=(
                    f"trip_count std is {std_ratio:.1f}x baseline "
                    f"(baseline std={baseline_std:.1f}, current std={current_std:.1f}). "
                    "Model trained on baseline distribution will produce unreliable predictions."
                ),
                count=len(df),
                baseline_std=round(baseline_std, 2),
                current_std=round(current_std, 2),
                std_ratio=round(std_ratio, 2),
            )

    def _add_issue(self, issue_type, severity, description, count=None, **details):
        self.issues.append({
            "type": issue_type,
            "severity": severity,
            "description": description,
            "count": count,
            **details,
        })


# ── Utility functions ──────────────────────────────────────────────────────────

def compare_distributions(baseline: pd.Series, current: pd.Series, threshold: float = 3.0) -> bool:
    """Returns True if std ratio exceeds threshold."""
    baseline_std = baseline.std()
    current_std = current.std()
    std_ratio = current_std / (baseline_std + 1e-9)
    return bool(std_ratio > threshold)


def detect_outliers(series: pd.Series, baseline_series: pd.Series = None, sigma: float = 3.0) -> pd.Series:
    """Returns boolean Series: True where values are outliers."""
    if baseline_series is not None:
        mean = baseline_series.mean()
        std = baseline_series.std()
    else:
        mean = series.mean()
        std = series.std()
    return (series - mean).abs() > sigma * std


# ── CLI entrypoint (called by GitHub Actions) ──────────────────────────────────

def main():
    print("=" * 60)
    print("Data Quality Validation — Week 3")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    baseline = df[df["time_bucket"] < CUTOFF]
    corrupted = df[df["time_bucket"] >= CUTOFF]

    print(f"Baseline rows : {len(baseline):,}")
    print(f"Corrupted rows: {len(corrupted):,}")
    print()

    validator = DataQualityValidator(baseline_df=baseline)
    result = validator.validate(corrupted)

    if result["is_valid"]:
        print("All checks passed — data quality OK")
        sys.exit(0)
    else:
        print(f"{result['num_issues']} issue(s) found:\n")
        for issue in result["issues"]:
            print(f"  [{issue['severity'].upper()}] {issue['type']}")
            print(f"    {issue['description']}")
            print()
        sys.exit(1)


if __name__ == "__main__":
    main()