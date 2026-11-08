"""Ablation: why the production model uses hostname features only.

Same rows, same grouped split (by registered domain), same Random Forest, three
feature sets:
  (a) hostname   - the production features (src/features.py)
  (b) full_url   - the project doc's full-URL features (path, query, https included)
  (c) dataset    - PhiUSIIL's own pre-extracted URL columns (URLSimilarityIndex ...)

(b) and (c) score close to 100% because of the dataset's shortcut (legit = bare
https://www.<domain>). The probe shows the cost: model (b) flags real deep links.

Run:  python -m src.ablation
Out:  results/ablation.json
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.data_prep import add_label, load_raw
from src.features import (
    FEATURE_COLUMNS, SUSPICIOUS_KEYWORDS, TOP_TLDS, extract_host_features, normalize_url,
    registered_domain,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ablation.json"
SEED = 42

DATASET_COLS = [
    "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain", "HasObfuscation",
    "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL",
    "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL",
    "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

PROBE_URLS = [
    "https://github.com/scikit-learn/scikit-learn/issues",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.amazon.com/dp/B08N5WRWNW?ref=nav",
    "https://docs.python.org/3/library/urllib.parse.html",
    "http://www.example.org/",
    "https://www.bbc.co.uk/news/technology",
]


def _entropy(s: str) -> float:
    n = len(s) or 1
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values()) if s else 0.0


def full_url_features(url: str) -> dict:
    """The project doc's original full-URL feature list (not used in production)."""
    u = url.strip()
    p = urlsplit(u if "://" in u else "http://" + u)
    host = (p.hostname or "").lower()
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    n = max(len(u), 1)
    letters, digits = sum(c.isalpha() for c in u), sum(c.isdigit() for c in u)
    specials = sum(not c.isalnum() for c in u)
    return {
        "url_length": len(u), "domain_length": len(host), "path_length": len(p.path),
        "query_length": len(p.query), "num_dots": u.count("."), "num_hyphens": u.count("-"),
        "num_digits": digits, "num_letters": letters,
        "num_subdomains": max(host.count(".") - 1, 0), "num_equals": u.count("="),
        "num_qmarks": u.count("?"), "num_ampersands": u.count("&"), "num_at": u.count("@"),
        "num_percent": u.count("%"),
        "num_other_special": specials - sum(u.count(c) for c in ".-=?&@%"),
        "digit_ratio": digits / n, "letter_ratio": letters / n, "special_ratio": specials / n,
        "is_https": int(u.lower().startswith("https://")),
        "is_ip_domain": int(bool(re.fullmatch(r"[\d.]+", host))),
        "has_obfuscation": int("%" in u),
        "has_suspicious_keyword": int(any(k in u.lower() for k in SUSPICIOUS_KEYWORDS)),
        "tld_length": len(tld), "tld_idx": TOP_TLDS.index(tld) if tld in TOP_TLDS else -1,
        "url_entropy": _entropy(u), "domain_entropy": _entropy(host),
    }


def hostname_frame(urls: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame([extract_host_features(normalize_url(u).host) for u in urls])[FEATURE_COLUMNS]
    f["tld_group"] = f["tld_group"].map({t: i for i, t in enumerate(TOP_TLDS)}).fillna(-1)
    return f.astype(float)


def evaluate(X, y, tr, te) -> tuple[dict, RandomForestClassifier]:
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced",
                                n_jobs=-1, random_state=SEED).fit(X.iloc[tr], y.iloc[tr])
    proba = rf.predict_proba(X.iloc[te])[:, 1]
    pred = (proba >= 0.5).astype(int)
    yt = y.iloc[te]
    return {
        "accuracy": round(accuracy_score(yt, pred), 4),
        "precision_phishing": round(precision_score(yt, pred), 4),
        "recall_phishing": round(recall_score(yt, pred), 4),
        "f1_phishing": round(f1_score(yt, pred), 4),
        "roc_auc": round(roc_auc_score(yt, proba), 4),
        "n_features": X.shape[1],
    }, rf


def main() -> None:
    raw = add_label(load_raw(columns=["URL", "label", *DATASET_COLS]).drop_duplicates("URL"))
    ok = raw["URL"].map(lambda u: _try_host(u) is not None)
    raw = raw[ok].reset_index(drop=True)
    y = raw["is_phishing"]
    groups = raw["URL"].map(lambda u: registered_domain(normalize_url(u).host))
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(raw, y, groups))
    print(f"rows {len(raw):,}  train {len(tr):,}  test {len(te):,}")

    sets = {
        "hostname (production)": hostname_frame(raw["URL"]),
        "full_url (doc features)": pd.DataFrame([full_url_features(u) for u in raw["URL"]]),
        "dataset pre-extracted columns": raw[DATASET_COLS].astype(float),
    }
    results, models = {}, {}
    for name, X in sets.items():
        results[name], models[name] = evaluate(X, y, tr, te)
        print(f"{name:32s} {results[name]}")

    host_rf, url_rf = models["hostname (production)"], models["full_url (doc features)"]
    probe = []
    for u in PROBE_URLS:
        p_url = url_rf.predict_proba(pd.DataFrame([full_url_features(u)]))[0, 1]
        p_host = host_rf.predict_proba(hostname_frame(pd.Series([u])))[0, 1]
        probe.append({"url": u, "p_phishing_full_url_model": round(float(p_url), 3),
                      "p_phishing_hostname_model": round(float(p_host), 3)})
        print(f"  probe {u:55s} full-url={p_url:.2f}  hostname={p_host:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": "Same rows, same grouped split (registered domain), RF 150 trees, threshold 0.5. "
                "Row-level (one row per unique URL), so numbers differ slightly from metrics.json.",
        "results": results, "legit_deep_link_probe": probe}, indent=2))
    print(f"wrote {OUT}")


def _try_host(u):
    try:
        return normalize_url(u).host
    except ValueError:
        return None


if __name__ == "__main__":
    main()
