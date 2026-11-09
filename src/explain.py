"""Per-URL explanations for the Random Forest, plus model loading helpers.

Contributions use the tree-path decomposition (Saabas / "treeinterpreter"):
for every tree, the change in P(phishing) at each split along the sample's decision
path is credited to the feature used at that split. Averaged over trees:

    P(phishing | x) = bias + sum(feature contributions)

No extra dependency is needed; it only uses each fitted tree's ``tree_`` arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.allowlist import KNOWN_DOMAINS
from src.features import (
    FEATURE_COLUMNS, NUMERIC, extract_host_features, found_brand_in_subdomain,
    found_keywords, matched_free_host, normalize_url, registered_domain, url_warnings,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "rf_pipeline.joblib"
META_PATH = ROOT / "models" / "model_meta.json"


def load_model(model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
    pipe = joblib.load(model_path)
    pipe.named_steps["rf"].n_jobs = 1  # single-URL inference: threads only add overhead
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"threshold": 0.5}
    return pipe, meta


def to_frame(feats: dict) -> pd.DataFrame:
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]
    X[NUMERIC] = X[NUMERIC].astype(float)
    return X


def contributions(pipe, X: pd.DataFrame) -> tuple[float, pd.Series]:
    """Return (bias, per-original-feature contribution to P(phishing)) for one row."""
    pre, rf = pipe.named_steps["pre"], pipe.named_steps["rf"]
    Xt = np.asarray(pre.transform(X), dtype=np.float32)
    names = pre.get_feature_names_out()
    phish_idx = list(rf.classes_).index(1)
    contrib = np.zeros(Xt.shape[1])
    bias = 0.0
    indicator, ptr = rf.decision_path(Xt[:1])  # one call for all trees
    path_all = indicator.indices
    for i, est in enumerate(rf.estimators_):
        t = est.tree_
        # nodes of tree i on the path; sklearn numbers children after parents
        nodes = np.sort(path_all[(path_all >= ptr[i]) & (path_all < ptr[i + 1])] - ptr[i])
        val = t.value[nodes, 0, :]
        p = val[:, phish_idx] / val.sum(axis=1)
        bias += p[0]
        np.add.at(contrib, t.feature[nodes[:-1]], p[1:] - p[:-1])
    n = len(rf.estimators_)
    s = pd.Series(contrib / n, index=names)
    s = s.groupby(lambda c: "tld_group" if c.startswith("tld_group_") else c).sum()
    return bias / n, s


def describe(feature: str, feats: dict, host: str, contribution: float, meta: dict) -> str:
    """Plain-language sentence for one feature of one URL."""
    up = contribution > 0
    v = feats[feature]
    med = meta.get("legit_median", {}).get(feature)
    cmp = ""
    if med is not None and feature not in ("tld_group",):
        cmp = f" (typical legit site: {round(med, 2):g})"
    if feature == "is_free_hosting":
        return (f"Hosted on a free hosting / site-builder platform ({matched_free_host(host)})"
                if v else "Not on a free hosting or site-builder platform")
    if feature == "brand_in_subdomain":
        return (f"Brand name '{found_brand_in_subdomain(host)}' appears outside its own domain"
                if v else "No brand name used outside its own domain")
    if feature == "num_suspicious_keywords":
        return (f"Hostname contains sensitive words: {', '.join(found_keywords(host)[:4])}"
                if v else "No sensitive words (login, verify, secure ...) in the hostname")
    if feature == "is_ip":
        return "Hostname is a raw IP address" if v else "Hostname is a domain name, not an IP"
    if feature == "is_punycode":
        return "Hostname uses punycode (look-alike characters possible)" if v else "No punycode"
    if feature == "is_multi_part_suffix":
        return "Country second-level suffix (e.g. .co.uk)" if v else "Single-part domain suffix"
    if feature == "tld_group":
        tld = host.rsplit(".", 1)[-1]
        return f"Top-level domain .{tld} " + ("is common among phishing hosts" if up else "lowers the risk")
    labels = {
        "host_length": f"Hostname length {v:g} characters{cmp}",
        "num_labels": f"{v:g} dot-separated parts in the hostname{cmp}",
        "num_subdomains": f"{v:g} subdomain level(s){cmp}",
        "sld_length": f"Main domain name is {v:g} characters long{cmp}",
        "longest_label_len": f"Longest hostname part is {v:g} characters{cmp}",
        "mean_label_len": f"Average hostname part length {v:g}{cmp}",
        "num_hyphens": f"{v:g} hyphen(s) in the hostname{cmp}",
        "num_digits": f"{v:g} digit(s) in the hostname{cmp}",
        "digit_ratio": f"Digits make up {v:.0%} of the hostname",
        "vowel_ratio": f"Vowel ratio {v:.2f}{cmp}",
        "max_consonant_run": f"Longest run of consonants: {v:g}{cmp}",
        "max_digit_run": f"Longest run of digits: {v:g}{cmp}",
        "letter_digit_switches": f"Letters and digits alternate {v:g} time(s){cmp}",
        "host_entropy": f"Hostname randomness (entropy) {v:.2f}{cmp}",
        "sld_entropy": f"Main domain randomness (entropy) {v:.2f}{cmp}",
        "tld_length": f"TLD length {v:g}{cmp}",
    }
    return labels.get(feature, f"{feature} = {v}")


def analyze_url(url: str, pipe, meta: dict, top_k: int = 5) -> dict:
    """Full analysis used by the app. Raises ValueError for invalid input."""
    parts = normalize_url(url)
    feats = extract_host_features(parts.host)
    X = to_frame(feats)
    proba = float(pipe.predict_proba(X)[0, list(pipe.classes_).index(1)])
    thr = float(meta.get("threshold", 0.5))
    bias, contrib = contributions(pipe, X)
    order = contrib.abs().sort_values(ascending=False).index[:top_k]
    reasons = [
        {"feature": f, "contribution": round(float(contrib[f]), 4),
         "direction": "phishing" if contrib[f] > 0 else "legitimate",
         "text": describe(f, feats, parts.host, float(contrib[f]), meta)}
        for f in order if abs(contrib[f]) >= 0.005
    ]
    reg = registered_domain(parts.host)
    known = reg in KNOWN_DOMAINS and matched_free_host(parts.host) is None and not parts.has_userinfo
    model_says_phishing = proba >= thr
    # '@' before the host is never needed in a normal link and only hides the real
    # destination, so the app treats it as suspicious whatever the model says.
    red_flag_override = parts.has_userinfo and not model_says_phishing
    return {
        "url": parts.raw, "host": parts.host, "registered_domain": reg,
        "proba_phishing": proba, "threshold": thr,
        "model_is_phishing": model_says_phishing,
        "known_domain": known,
        "red_flag_override": red_flag_override,
        "is_phishing": (model_says_phishing and not known) or red_flag_override,
        "base_rate": float(bias), "reasons": reasons,
        "warnings": url_warnings(parts), "features": feats,
    }
