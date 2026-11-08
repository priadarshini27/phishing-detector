"""Module A: load PhiUSIIL, clean, recode the label and extract hostname features.

Run:  python -m src.data_prep
Out:  data/features.csv  (FEATURE_COLUMNS + is_phishing + host + reg_domain)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.features import FEATURE_COLUMNS, extract_host_features, normalize_url, registered_domain

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
OUT_CSV = ROOT / "data" / "features.csv"


def load_raw(path: Path = RAW_CSV, columns=("URL", "Domain", "label")) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download PhiUSIIL from UCI (id=967) into data/."
        )
    return pd.read_csv(path, usecols=list(columns) if columns else None, encoding_errors="ignore")


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """The ONLY place the label is recoded. Raw: 1 = legit, 0 = phishing."""
    df = df.copy()
    df["is_phishing"] = (1 - df["label"]).astype(int)
    return df.drop(columns=["label"])


def _safe_host(url: str) -> str | None:
    try:
        return normalize_url(url).host
    except ValueError:
        return None


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["host"] = df["URL"].map(_safe_host)
    bad = df["host"].isna().sum()
    if bad:
        print(f"  dropped {bad} unparseable URLs")
    df = df.dropna(subset=["host"])
    feats = pd.DataFrame([extract_host_features(h) for h in df["host"]], index=df.index)
    feats = feats[FEATURE_COLUMNS]
    feats["is_phishing"] = df["is_phishing"].values
    feats["host"] = df["host"].values
    feats["reg_domain"] = df["host"].map(registered_domain).values
    return feats


def main() -> None:
    t0 = time.time()
    df = load_raw()
    print(f"raw rows: {len(df):,}")
    df = df.drop_duplicates(subset="URL")
    print(f"after dropping duplicate URLs: {len(df):,}")
    df = add_label(df)

    feats = build_features(df)
    # Different raw URLs can collapse to the same host (e.g. with and without a path).
    # Keep one row per (host, label) so test hosts never duplicate training hosts.
    before = len(feats)
    feats = feats.drop_duplicates(subset=["host", "is_phishing"])
    print(f"after collapsing duplicate (host, label) rows: {len(feats):,} (-{before - len(feats):,})")
    conflicted = feats["host"].duplicated(keep=False)
    feats = feats[~conflicted]
    print(f"dropped {conflicted.sum() // 2} hosts that appear under both labels")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_CSV, index=False)
    print("class balance (is_phishing):", feats["is_phishing"].value_counts().to_dict())
    print(f"unique registered domains: {feats['reg_domain'].nunique():,}")
    print(f"wrote {OUT_CSV}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
