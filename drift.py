import numpy as np
import pandas as pd


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two distributions."""
    def _break_into_bins(series, buckets):
        return np.linspace(series.min(), series.max(), buckets + 1)

    bins = _break_into_bins(expected.append(actual), buckets)
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    psi = np.sum((expected_percents - actual_percents) * np.log((expected_percents + 1e-8)/(actual_percents + 1e-8)))
    return psi


def detect_drift(psi: float, threshold: float = 0.2) -> bool:
    """Return True if drift is detected based on PSI."""
    return psi > threshold