"""
Data Quality Validation Tests
Week 3 — Operationalizing AI | Mrinal Goel

Tests:
  - Baseline data passes all checks
  - Corrupted data fails on all 4 identified issues
  - Each issue detected independently
  - API does not crash with bad data (graceful degradation)
"""

import logging
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validation.check_data_quality import DataQualityValidator, detect_outliers, compare_distributions

DATA_PATH = "week3/data/demand_enriched_corrupted.parquet"
CUTOFF = pd.Timestamp("2026-01-16")


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def full_df():
    return pd.read_parquet(DATA_PATH)


@pytest.fixture(scope="session")
def baseline_data(full_df):
    return full_df[full_df["time_bucket"] < CUTOFF].copy()


@pytest.fixture(scope="session")
def corrupted_data(full_df):
    return full_df[full_df["time_bucket"] >= CUTOFF].copy()


@pytest.fixture(scope="session")
def validator(baseline_data):
    return DataQualityValidator(baseline_df=baseline_data)


# ── Baseline passes ────────────────────────────────────────────────────────────

class TestBaselineData:
    """Baseline (pre Jan 16) data should pass all validation checks."""

    def test_baseline_passes_validation(self, baseline_data, validator):
        result = validator.validate(baseline_data)
        assert result["is_valid"], f"Baseline failed validation: {result['issues']}"

    def test_baseline_no_negatives(self, baseline_data):
        assert (baseline_data["trip_count"] >= 0).all()

    def test_baseline_no_duplicates(self, baseline_data):
        assert baseline_data.duplicated().sum() == 0

    def test_baseline_trip_count_reasonable(self, baseline_data):
        assert baseline_data["trip_count"].max() <= 500


# ── Issue 1: Negative trip counts ─────────────────────────────────────────────

class TestIssue1NegativeTripCount:
    """Issue 1: 353 rows with negative trip_count values."""

    def test_corrupted_has_negative_values(self, corrupted_data):
        assert (corrupted_data["trip_count"] < 0).sum() > 0

    def test_validator_detects_negative_trip_count(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        assert not result["is_valid"]
        types = [i["type"] for i in result["issues"]]
        assert "negative_trip_count" in types

    def test_synthetic_negative_data_detected(self, baseline_data):
        """Inject negatives into clean data and verify detection."""
        bad_df = baseline_data.copy()
        bad_df.loc[bad_df.index[:10], "trip_count"] = -5
        v = DataQualityValidator(baseline_df=baseline_data)
        result = v.validate(bad_df)
        assert any(i["type"] == "negative_trip_count" for i in result["issues"])


# ── Issue 2: Extreme outliers ──────────────────────────────────────────────────

class TestIssue2OutlierTripCount:
    """Issue 2: 311 rows with trip_count > 1000; max = 99,999."""

    def test_corrupted_has_outliers(self, corrupted_data):
        assert (corrupted_data["trip_count"] > 1000).sum() > 0

    def test_validator_detects_outliers(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        assert not result["is_valid"]
        types = [i["type"] for i in result["issues"]]
        assert "outlier_trip_count" in types

    def test_detect_outliers_utility(self, baseline_data, corrupted_data):
        outliers = detect_outliers(corrupted_data["trip_count"], baseline_data["trip_count"], sigma=5.0)
        assert outliers.sum() > 0

    def test_synthetic_outlier_detected(self, baseline_data):
        bad_df = baseline_data.copy()
        bad_df.loc[bad_df.index[0], "trip_count"] = 99999
        v = DataQualityValidator(baseline_df=baseline_data)
        result = v.validate(bad_df)
        assert any(i["type"] == "outlier_trip_count" for i in result["issues"])


# ── Issue 3: Duplicate rows ────────────────────────────────────────────────────

class TestIssue3DuplicateRows:
    """Issue 3: 8,134 duplicate rows in corrupted data."""

    def test_corrupted_has_duplicates(self, corrupted_data):
        assert corrupted_data.duplicated().sum() > 0

    def test_validator_detects_duplicates(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        assert not result["is_valid"]
        types = [i["type"] for i in result["issues"]]
        assert "duplicate_rows" in types

    def test_synthetic_duplicate_detected(self, baseline_data):
        bad_df = pd.concat([baseline_data.head(100), baseline_data.head(10)], ignore_index=True)
        v = DataQualityValidator(baseline_df=baseline_data)
        result = v.validate(bad_df)
        assert any(i["type"] == "duplicate_rows" for i in result["issues"])


# ── Issue 4: Distribution shift ────────────────────────────────────────────────

class TestIssue4DistributionShift:
    """Issue 4: Mean shifted from 17 to 79; std from 21 to 2433."""

    def test_corrupted_mean_shifted(self, baseline_data, corrupted_data):
        baseline_mean = baseline_data["trip_count"].mean()
        corrupted_mean = corrupted_data["trip_count"].mean()
        assert corrupted_mean > baseline_mean * 3

    def test_validator_detects_distribution_shift(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        types = [i["type"] for i in result["issues"]]
        assert "distribution_shift" in types

    def test_compare_distributions_utility(self, baseline_data, corrupted_data):
        shifted = compare_distributions(
            baseline_data["trip_count"],
            corrupted_data["trip_count"],
            threshold=3.0,
        )
        assert shifted == True

    def test_clean_data_not_shifted(self, baseline_data):
        sample = baseline_data.sample(1000, random_state=42)
        shifted = compare_distributions(
            baseline_data["trip_count"],
            sample["trip_count"],
            threshold=3.0,
        )
        assert shifted == False


# ── Graceful degradation ───────────────────────────────────────────────────────

class TestGracefulDegradation:
    """Validation issues must be logged; API must not crash."""

    def test_validator_never_raises(self, corrupted_data, validator):
        try:
            result = validator.validate(corrupted_data)
            assert isinstance(result, dict)
            assert "is_valid" in result
        except Exception as e:
            pytest.fail(f"validator.validate() raised an exception: {e}")

    def test_validator_handles_empty_dataframe(self, baseline_data):
        empty_df = pd.DataFrame(columns=baseline_data.columns)
        v = DataQualityValidator(baseline_df=baseline_data)
        try:
            result = v.validate(empty_df)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"Validator crashed on empty DataFrame: {e}")

    def test_issues_are_logged(self, corrupted_data, validator, caplog):
        try:
            from backend.data import check_and_log_data_quality
            with caplog.at_level(logging.WARNING):
                check_and_log_data_quality()
            assert len(caplog.records) > 0
        except ImportError:
            pytest.skip("check_and_log_data_quality not yet integrated into data.py")

    def test_result_structure(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        assert "is_valid" in result
        assert "num_issues" in result
        assert "issues" in result
        assert isinstance(result["issues"], list)
        for issue in result["issues"]:
            assert "type" in issue
            assert "severity" in issue
            assert "description" in issue