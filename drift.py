import numpy as np
import pandas as pd

def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two distributions."""
    def _break_into_bins(series, buckets):
        return np.linspace(series.min(), series.max(), buckets + 1)

    # FIXED: use pd.concat instead of .append
    combined = pd.concat([expected, actual])
    bins = _break_into_bins(combined, buckets)

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # Add epsilon to avoid division by zero or log(0)
    psi = np.sum(
        (expected_percents - actual_percents) * 
        np.log((expected_percents + 1e-8) / (actual_percents + 1e-8))
    )
    return psi

def detect_drift(psi: float, threshold: float = 0.2) -> bool:
    """Return True if drift is detected based on PSI threshold."""
    return psi > threshold