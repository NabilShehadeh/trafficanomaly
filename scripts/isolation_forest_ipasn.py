#!/usr/bin/env python3
"""
Isolation Forest model for cs448b_ipasn.csv dataset.

Dataset columns: date, l_ipn (left IP network), r_asn (right ASN), f (flow count).
Detects anomalous (l_ipn, r_asn, f) patterns.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Default path to dataset (override with --data)
DEFAULT_DATA_PATH = Path.home() / "Downloads" / "cs448b_ipasn.csv"


def load_data(path: Path) -> pd.DataFrame:
    """Load and basic clean of the IP-ASN CSV."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Drop rows with nulls from date parsing
    df = df.dropna(subset=["date"])
    return df


def prepare_features(df: pd.DataFrame):
    """
    Build numeric feature matrix for Isolation Forest.
    Uses l_ipn, r_asn, f and optional date-derived features.
    """
    X = df[["l_ipn", "r_asn", "f"]].copy().astype(np.float64)
    # Optional: add date-based features (day of week, numeric date)
    X["day_of_week"] = df["date"].dt.dayofweek
    X["day_of_year"] = df["date"].dt.dayofyear
    # Log-scale flow count (often heavy-tailed)
    X["log_f"] = np.log1p(df["f"].values)
    return X, X.values


def main():
    parser = argparse.ArgumentParser(description="Isolation Forest on cs448b_ipasn.csv")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to cs448b_ipasn.csv",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected fraction of anomalies (default 0.05)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to save predictions CSV (optional)",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Path to save fitted model (joblib) and scaler (optional)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    print("Loading data...")
    df = load_data(args.data)
    print(f"Rows: {len(df):,}")

    print("Preparing features...")
    X_df, X = prepare_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Isolation Forest...")
    clf = IsolationForest(
        contamination=args.contamination,
        random_state=42,
        n_estimators=100,
        max_samples="auto",
        max_features=1.0,
    )
    clf.fit(X_scaled)

    preds = clf.predict(X_scaled)  # 1 = normal, -1 = anomaly
    scores = clf.decision_function(X_scaled)
    is_anomaly = (preds == -1).astype(int)

    n_anomalies = is_anomaly.sum()
    print(f"Anomalies detected: {n_anomalies:,} ({100 * n_anomalies / len(df):.2f}%)")
    print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")

    out_df = df.copy()
    out_df["anomaly_score"] = scores
    out_df["is_anomaly"] = is_anomaly

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Predictions saved to {args.out}")

    if args.save_model:
        import joblib

        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": clf, "scaler": scaler, "feature_names": list(X_df.columns)}, args.save_model)
        print(f"Model and scaler saved to {args.save_model}")

    # Show a few anomaly examples
    anomaly_rows = out_df[out_df["is_anomaly"] == 1].head(10)
    if not anomaly_rows.empty:
        print("\nSample anomalies (first 10):")
        print(anomaly_rows.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
