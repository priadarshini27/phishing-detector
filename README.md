# Phishing URL Detector (Random Forest + Streamlit)

Paste a URL, get a **Legitimate / Phishing** verdict, a phishing probability and the
reasons behind it. The app only parses the URL text. **It never opens or fetches the link.**

Dataset: [PhiUSIIL Phishing URL](https://archive.ics.uci.edu/dataset/967) (UCI, 2024, CC BY 4.0)
· Model: scikit-learn Random Forest · App: Streamlit


## Quick start

```bash
pip install -r requirements.txt
# put PhiUSIIL_Phishing_URL_Dataset.csv in data/
python -m src.data_prep        # -> data/features.csv          (~1 min)
python -m src.train --fast     # -> models/, results/          (~2 min; drop --fast to re-run the search, ~8 min)
python -m src.eda              # -> results/eda/
python -m src.ablation         # -> results/ablation.json      (~3 min)
python -m src.realworld_check  # -> results/realworld_check.csv
python -m pytest -q
streamlit run app/app.py
```

`models/rf_pipeline.joblib` (≈46 MB) is a complete scikit-learn `Pipeline` (one-hot TLD, then RF),
so the app runs without the dataset.

## The key finding: why the model only looks at the hostname

Before any modelling, the raw URLs show a strong shortcut (`results/eda/url_shape_shortcut.png`):

| | Legitimate (134,850) | Phishing (100,520) |
|---|---|---|
| starts with `https://` | **100%** | 48.6% |
| starts with `https://www.` | **100%** | 2.4% |
| has a path | **0%** | 27.1% |
| has a query string | **0%** | 6.1% |
| ends with `/` | **0%** | 36.3% |

Every legitimate URL is a bare `https://www.<domain>` homepage. A model fed the full URL
learns "has a path or uses http, so phishing". It scores about 99.6%, but it is useless in
practice. The ablation (`results/ablation.json`; same rows, same split, same RF) shows this:

| Feature set | Accuracy | Phishing precision | Phishing recall | ROC-AUC |
|---|---|---|---|---|
| **Hostname only (production)** | 0.826 | 0.848 | 0.633 | 0.847 |
| Full-URL features (the original plan's list) | 0.996 | 0.999 | 0.990 | 0.998 |
| Dataset's pre-extracted columns | 1.000 | 1.000 | 1.000 | 1.000 |

| Real legitimate deep link | P(phishing), full-URL model | P(phishing), hostname model |
|---|---|---|
| github.com/scikit-learn/scikit-learn/issues | **1.00** | 0.23 |
| www.amazon.com/dp/B08N5WRWNW?ref=nav | **1.00** | 0.24 |
| www.bbc.co.uk/news/technology | **0.99** | 0.00 |
| http://www.example.org/ | **1.00** | 0.04 |

So the production model sees **only the hostname** (scheme, `www.`, port, path and query
are removed). Path, query and scheme are still checked, as separate **rule-based red flags**
in the app.

## Pipeline

```
PhiUSIIL CSV ─► src/data_prep.py ─► data/features.csv ─► src/train.py ─► models/rf_pipeline.joblib
  (URL, label)    dedupe URLs, is_phishing = 1 - label,     grouped split,      + models/model_meta.json
                  normalise → host, one row per host        RF, threshold       + results/*
                                                                   │
            app/app.py ◄── src/explain.py (analyze_url, per-URL contributions) ◄┘
                       ◄── src/features.py (the ONLY feature code: shared by training and app)
```

- **Label:** `is_phishing` = 1 for phishing, 0 for legitimate (the raw dataset uses the reverse). It is recoded only in `data_prep.add_label`.
- **Cleaning:** 425 duplicate URLs and 8 unparseable URLs dropped. URLs collapsing to the same host are kept once per (host, label). 58 hosts that appear under both labels are dropped. This leaves 219,033 hosts (134,792 legitimate, 84,241 phishing).
- **Split:** `GroupShuffleSplit` 80/20 **grouped by registered domain** (e.g. all `*.web.app` hosts land on one side), seed 42. No registered domain appears in both train and test.
- **Model:** `RandomForestClassifier(n_estimators=300, min_samples_leaf=8, max_features="sqrt", class_weight="balanced")`. The hyper-parameters come from `RandomizedSearchCV` (10 candidates, `GroupKFold(3)`, PR-AUC). The decision threshold (0.504) is the max-F1 point on out-of-fold predictions from the training split.

## Features (`src/features.py`, 23 columns, hostname only)

| Group | Features |
|---|---|
| Size | `host_length`, `num_labels`, `num_subdomains`, `sld_length`, `longest_label_len`, `mean_label_len` |
| Characters | `num_hyphens`, `num_digits`, `digit_ratio`, `vowel_ratio`, `max_consonant_run`, `max_digit_run`, `letter_digit_switches` |
| Randomness | `host_entropy`, `sld_entropy` |
| Flags | `is_ip`, `is_punycode`, `is_multi_part_suffix` (e.g. co.uk), `is_free_hosting` (fixed list of hosting and site-builder platforms) |
| Words | `num_suspicious_keywords` (login, verify, secure …), `brand_in_subdomain` (paypal, outlook … used outside the brand's own domain, including digit look-alikes such as `0utlook`) |
| TLD | `tld_length`, `tld_group` (the top 30 TLDs, frozen in code; everything else is "other") |

There is no leakage by construction. Every value comes from the single URL string. Nothing uses
dataset-wide statistics or the label, unlike the dataset's `URLSimilarityIndex`,
`TLDLegitimateProb` and `URLCharProb`.

## Results (held-out test set, 39,858 hosts, grouped by registered domain)

| Metric | Value |
|---|---|
| Accuracy | 0.846 |
| Phishing precision | 0.864 |
| Phishing recall | 0.620 |
| Phishing F1 | 0.722 |
| ROC-AUC | 0.859 |
| PR-AUC | 0.827 |

Confusion matrix at threshold 0.504 (rows = true class): legit `[25,722 correct, 1,256 false alarms]`,
phishing `[4,891 missed, 7,989 caught]`. See `results/confusion_matrix.png`, `results/roc_pr.png`
and `results/feature_importance.png`. The most important features are `num_subdomains`, `tld_group`,
`is_free_hosting`, `num_labels` and `digit_ratio`.

**Where it misses (`phishing_recall_by_segment` in `metrics.json`):** phishing on free hosting
platforms is caught **100%** of the time (1,383/1,383), but phishing on its own ordinary-looking domain
only **57%** (11,497 in test). Most misses look like normal small-business sites
(`mestand.com`, `karlast.com`, `lrservice.com.ua`), which are probably compromised websites.
No hostname-only model can catch them.

### Real-world sheet (`tests/sample_urls.csv`, `results/realworld_check.csv`)

23 hand-written URLs: 12 legitimate (including deep links and subdomains) and 11 made-up phishing-style URLs.
The model alone gets **19/23**. Its misses are three big-site subdomains (en.wikipedia.org 0.75,
accounts.google.com 0.90, docs.python.org 0.81) and the `paypal.com@evil-site.ru` trick.
The app gets **23/23** with two clearly labelled post-model rules:

1. **Known-domain allowlist** (`src/allowlist.py`, about 90 major domains). Because the training data's legit
   hosts are all bare homepages, the model over-flags legitimate subdomains. The allowlist never applies
   to user-content platforms (sites.google.com, *.github.io …) or to URLs with `@`.
2. **`@` userinfo override.** `http://paypal.com@evil-site.ru` really goes to `evil-site.ru`. It is
   shown as "Suspicious" whatever the model's score.

The sheet was written by hand, so treat 23/23 as a demo check, not a metric.

## The app

- Example buttons (legit, phishing-looking, free-hosting lure, `@` trick, big-site subdomain) and input validation (empty, non-http schemes, spaces, no TLD, over 2048 characters).
- A verdict card with the phishing probability against the threshold.
- **"Why":** the top 5 per-URL contributions. These come from the tree-path (Saabas) decomposition in `src/explain.py`, where base rate + contributions = predicted probability, exactly. The test suite checks this.
- **Rule-based red flags** from the full URL: `@`, IP host, non-standard port, plain http, percent-encoding, very long URL, sensitive words in the path or query, `//` in the path, deep subdomains, punycode.

## Limitations

- **Dataset bias:** legitimate examples are homepages only, so the model has seen few legitimate subdomains (this is why the allowlist exists).
- **Hostname-only view:** compromised legitimate-looking domains are mostly missed (57% recall on own-domain phishing). There are no page-content, WHOIS/domain-age or certificate features.
- **`is_free_hosting` cuts both ways:** genuine sites on github.io or netlify.app will look riskier.
- **Time drift:** the data is a 2023-24 snapshot, and phishing infrastructure changes quickly.
- **Safety:** this is a learning project and not a replacement for real security tools.

## Repo layout

```
src/features.py        shared feature extractor + URL red-flag rules
src/data_prep.py       cleaning, label recode, feature table
src/eda.py             EDA plots and summary (results/eda/)
src/train.py           grouped split, RF, threshold, metrics, plots, model export
src/explain.py         model loading, per-URL contributions, analyze_url() used by the app
src/allowlist.py       well-known domains (app-level rule)
src/ablation.py        hostname vs full-URL vs dataset-column comparison
src/realworld_check.py runs tests/sample_urls.csv through the app logic
app/app.py             Streamlit app
tests/                 feature unit tests, model tests, sample URLs
```
