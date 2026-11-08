"""Light EDA, including evidence of the dataset's URL-shape shortcut.

Run:  python -m src.eda
Out:  results/eda/*.png, results/eda/eda_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_prep import OUT_CSV, add_label, load_raw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "eda"
COLORS = {"Legitimate": "#2a9d8f", "Phishing": "#e76f51"}


def shortcut_table(raw: pd.DataFrame) -> dict:
    """How the raw URL shape differs by class (why the model is hostname-only)."""
    rows = {}
    for name, grp in raw.groupby(raw["is_phishing"].map({0: "Legitimate", 1: "Phishing"})):
        u = grp["URL"]
        path = u.map(lambda s: urlsplit(s).path)
        rows[name] = {
            "n": int(len(grp)),
            "starts_https": round(float(u.str.startswith("https://").mean()), 3),
            "starts_https_www": round(float(u.str.startswith("https://www.").mean()), 3),
            "has_path": round(float(((path != "") & (path != "/")).mean()), 3),
            "has_query": round(float(u.str.contains(r"\?").mean()), 3),
            "trailing_slash": round(float(u.str.endswith("/").mean()), 3),
            "median_url_length": float(u.str.len().median()),
        }
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = add_label(load_raw().drop_duplicates("URL"))
    feats = pd.read_csv(OUT_CSV)
    feats["cls"] = feats["is_phishing"].map({0: "Legitimate", 1: "Phishing"})

    summary = {
        "raw_rows_after_url_dedupe": int(len(raw)),
        "feature_rows": int(len(feats)),
        "class_balance": feats["cls"].value_counts().to_dict(),
        "unique_registered_domains": int(feats["reg_domain"].nunique()),
        "url_shape_by_class": shortcut_table(raw),
        "top_phishing_registered_domains": feats.loc[feats.is_phishing == 1, "reg_domain"]
        .value_counts().head(15).to_dict(),
        "top_tlds_by_class": {
            c: g["tld_group"].value_counts().head(10).to_dict() for c, g in feats.groupby("cls")
        },
        "feature_means_by_class": feats.groupby("cls")[
            ["host_length", "num_subdomains", "num_hyphens", "num_digits", "host_entropy",
             "is_free_hosting", "num_suspicious_keywords", "brand_in_subdomain"]
        ].mean().round(3).to_dict(orient="index"),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # 1. class balance
    fig, ax = plt.subplots(figsize=(4, 3))
    vc = feats["cls"].value_counts()
    ax.bar(vc.index, vc.values, color=[COLORS[k] for k in vc.index])
    ax.set_title("Class balance (unique hosts)")
    fig.tight_layout(); fig.savefig(OUT / "class_balance.png", dpi=120); plt.close(fig)

    # 2. host length by class
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for c, g in feats.groupby("cls"):
        ax.hist(g["host_length"].clip(upper=80), bins=60, alpha=0.6, label=c, color=COLORS[c], density=True)
    ax.set_xlabel("hostname length (clipped at 80)"); ax.legend(); ax.set_title("Hostname length by class")
    fig.tight_layout(); fig.savefig(OUT / "host_length.png", dpi=120); plt.close(fig)

    # 3. top TLDs by class
    top = feats["tld_group"].value_counts().head(15).index
    ct = pd.crosstab(feats["tld_group"], feats["cls"]).loc[top]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ct.plot.bar(ax=ax, color=[COLORS[c] for c in ct.columns])
    ax.set_title("Top TLDs by class"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(OUT / "top_tlds.png", dpi=120); plt.close(fig)

    # 4. URL-shape shortcut
    st = pd.DataFrame(summary["url_shape_by_class"]).T[
        ["starts_https", "starts_https_www", "has_path", "has_query", "trailing_slash"]]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    st.T.plot.bar(ax=ax, color=[COLORS[c] for c in st.index])
    ax.set_ylim(0, 1.05); ax.set_title("Raw URL shape by class: legit URLs are all https://www.<domain>")
    fig.tight_layout(); fig.savefig(OUT / "url_shape_shortcut.png", dpi=120); plt.close(fig)

    print(json.dumps(summary["url_shape_by_class"], indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
