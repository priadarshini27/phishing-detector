# Phishing Website Detection from URLs: Project Plan

*Mini project (ML) · Team of 3 · Dataset: PhiUSIIL Phishing URL (UCI, 2024) · Models: Random Forest and SVM · Demo: live URL checker*

## 1. Project description

**Title:** URL-Based Phishing Website Detection using Random Forest and SVM, with a Live URL Checker.

**Problem.** Phishing sites impersonate legitimate websites to steal credentials. Many of them can be flagged from the URL string alone, before anyone opens the page. This project trains classical ML models to label a URL as **Legitimate** or **Phishing**, and wraps the best model in a small web app where a user pastes a URL and gets a verdict, a confidence score and the reasons behind it.

**Objectives**

- Build one shared, reproducible URL feature extractor that works identically for training data and for new URLs typed into the app.
- Train and compare Random Forest, SVM and a simple baseline (Logistic Regression).
- Evaluate honestly: grouped train/test split by domain, phishing-focused metrics, and a check for data leakage.
- Deliver a working Streamlit demo, a short report and slides.

**Scope.** URL-only prediction. The app never fetches or visits the URL; it only parses the text. No WHOIS or page-content features.

**Deliverables:** GitHub repo, trained model file, Streamlit app, evaluation results (tables and plots), report, demo.

## 2. Dataset: what was checked and what wasn't

**Confirmed from the UCI page**

- PhiUSIIL Phishing URL (Website), donated March 2024, from Prasad and Chandra (Computers & Security).
- 235,795 rows: 134,850 legitimate and 100,945 phishing URLs. No missing values. CC BY 4.0 licence.
- One CSV, `PhiUSIIL_Phishing_URL_Dataset.csv` (about 54 MB). Download from UCI, or use `fetch_ucirepo(id=967)`.
- **Label 1 = legitimate, 0 = phishing.** Ignore the `FILENAME` column.
- First columns: `URL, URLLength, Domain, DomainLength, IsDomainIP, TLD, URLSimilarityIndex, CharContinuationRate, TLDLegitimateProb` and more (56 columns in total including the label).

**Not verified:** the UCI page only lists the first 10 columns, and the file could not be downloaded in the environment used to prepare this plan. The remaining column names (for example `IsHTTPS`, `NoOfSubDomain`, `HasObfuscation`, page-content columns such as `HasPasswordField`, `NoOfJS`) come from the published dataset description and should be confirmed on Day 1 with the sanity-check script in section 9.

**Key design decision: recompute features from the raw `URL` column instead of using the dataset's pre-extracted columns.**

- A live checker only has a URL string. Many dataset columns need the page's HTML (forms, JavaScript, images), so they cannot be produced for a new URL.
- Some columns (`URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`) are derived using reference data or dataset-wide statistics, so they can leak label information and cannot be reproduced exactly for a single new URL.
- One shared extractor solves the "all three members must use the same feature definitions" problem.

## 3. Shared feature contract (URL-only)

About 25 features, all computed from the URL string. The list and its order are frozen once all three members agree. Nobody adds or renames a feature without telling the others.

- **Lengths:** `url_length`, `domain_length`, `path_length`, `query_length`
- **Counts:** `num_dots`, `num_hyphens`, `num_digits`, `num_letters`, `num_subdomains`, `num_equals`, `num_qmarks`, `num_ampersands`, `num_at`, `num_percent`, `num_other_special`
- **Ratios:** `digit_ratio`, `letter_ratio`, `special_ratio`
- **Flags (0/1):** `is_https`, `is_ip_domain`, `has_at_symbol`, `has_obfuscation` (percent-encoded characters), `has_suspicious_keyword` (login, verify, secure, account, update, bank, signin and similar)
- **Domain:** `tld_length`, `tld_group` (the top 15 TLDs, everything else grouped as "other")
- **Randomness:** `url_entropy`, `domain_entropy`

**Label convention (decided here):** the project target is `is_phishing`, where **1 = phishing and 0 = legitimate**. This is the reverse of the raw dataset, so it is done once in Module A and nowhere else. Phishing is then the positive class, and precision and recall mean what you expect them to mean.

## 4. Architecture

```
PhiUSIIL CSV  (URL, Domain, label)
      |
      v
[Module A]  Data prep + feature extraction
            features.py -> extract_features(url)
            output: data/features.csv  (X columns, is_phishing, domain)
      |
      v
[Module B]  Model training + evaluation
            Logistic Regression, Random Forest, SVM
            output: models/best_model.joblib, results/metrics.json, plots
      |
      v
[Module C]  Inference + Streamlit app + docs
            URL -> extract_features -> model -> verdict, confidence, top reasons
```

**Suggested repo layout**

```
phishing-detector/
  data/            raw CSV and features.csv (do not commit the raw CSV)
  src/features.py  shared extractor (owned by A)
  src/train.py     training and evaluation (owned by B)
  app/app.py       Streamlit app (owned by C)
  models/          best_model.joblib
  results/         metrics.json, confusion matrices, plots
  notebooks/       EDA
  tests/           feature unit tests, sample URLs
  README.md, requirements.txt
```

**Interfaces between modules (agree on these in hour 1)**

- `features.py` exposes `FEATURE_COLUMNS` (fixed list and order) and `extract_features(url: str) -> dict` returning a value for every column.
- `data/features.csv` has exactly `FEATURE_COLUMNS` plus `is_phishing` and `domain`.
- `models/best_model.joblib` is a single scikit-learn `Pipeline` (encoding, scaling and classifier together) that accepts a DataFrame with `FEATURE_COLUMNS` and supports `predict_proba`.
- The app imports `extract_features` and loads the model file. It never re-implements any feature logic.

## 5. The three modules

### Module A: Data and Feature Engineering (Member A)

**Responsibilities**

- Load the CSV, run the sanity checks, remove exact duplicate URLs, and recode the label to `is_phishing`.
- Light EDA: class balance, URL length distributions by class, top TLDs by class, a few example URLs of each class.
- Write and unit-test `extract_features(url)`. Handle edge cases: missing scheme (`example.com/login`), IP-address hosts, ports, uppercase, very long URLs, trailing spaces.
- Run the extractor over all rows and save `data/features.csv`, keeping the `domain` column for grouped splitting.
- Leakage report: show that none of the chosen features use dataset-wide statistics or the label.

**Done when:** `features.csv` exists, `extract_features` passes tests on 20+ hand-picked URLs, and the feature list is documented in the README.

### Module B: Modelling and Evaluation (Member B)

**Responsibilities**

- Build preprocessing (one-hot for `tld_group`, scaling for SVM) inside a scikit-learn `Pipeline`.
- Split with `GroupShuffleSplit` on `domain` (80/20) so the same domain never appears in both train and test.
- Train three models: Logistic Regression (baseline), Random Forest, SVM. Use `LinearSVC` (wrapped in `CalibratedClassifierCV` for probabilities) on the full data, or RBF `SVC` on a 30-50k subsample, because RBF SVC is slow on 235k rows.
- Tune lightly with `RandomizedSearchCV` using grouped cross-validation.
- Report accuracy, precision, recall, F1 and ROC-AUC with a confusion matrix for each model, with extra attention to phishing recall.
- Run the ablation: URL-only features versus the dataset's own pre-extracted columns, to show how much the "easy" columns inflate scores.
- Export feature importances (Random Forest) and save the best pipeline.

**Done when:** `best_model.joblib` and `metrics.json` are saved, a comparison table and plots exist, and the best model choice is justified.

### Module C: Application, Demo and Documentation (Member C)

**Responsibilities**

- Build the Streamlit app: text box, "Check URL" button, verdict (Legitimate / Phishing), phishing probability, and the top 3-5 contributing features in plain language (for example "very long URL", "contains @ symbol", "no HTTPS").
- Add input validation and friendly errors. The app only parses text and never opens the link.
- Until the real model arrives, develop against a dummy model that follows the agreed interface, then swap in `best_model.joblib`.
- Build a small real-world test sheet (about 20 URLs: popular legitimate sites including deep links, plus obviously suspicious made-up examples) and record where the model fails.
- Own the README, report, slides and demo script; collect results from A and B.

**Done when:** the app runs from a clean clone with `pip install -r requirements.txt` and `streamlit run app/app.py`, and the report and slides are complete.

## 6. Step-by-step guide (about 2 days)

### Day 1

**Hour 0-1: all together**

1. Create the GitHub repo, a shared `requirements.txt` (pandas, numpy, scikit-learn, joblib, streamlit, matplotlib, tldextract) and the folder structure.
2. Download the dataset and run the sanity-check script (section 9). Confirm the real column names.
3. Freeze `FEATURE_COLUMNS`, the label convention (`is_phishing`) and the interfaces from section 4.

**Hours 1-4: parallel work**

- **A:** write `extract_features`, unit tests, and the EDA notebook.
- **B:** build the training script and pipeline skeleton using a quick placeholder (the dataset's own numeric columns such as `URLLength`, `NoOfSubDomain`) so that only the input file changes later.
- **C:** build the Streamlit skeleton with a dummy model and the explanation display.

**Hours 4-5: handoff**

- A runs the extractor on the full dataset and pushes `features.csv`, then reviews B's and C's code against the feature contract.

**Hours 5-8: first real results**

- **B:** train Logistic Regression and Random Forest on `features.csv`, then SVM. Save the first metrics.
- **C:** swap in the first saved model and test the app end to end.
- **A:** run the leakage check, help B with the ablation data (dataset columns versus URL-only), start the report's dataset and methodology sections.

### Day 2

**Morning**

1. B tunes the models and finalises `best_model.joblib` and the comparison plots.
2. C re-tests the app with the final model and the real-world URL sheet, and notes failure cases.
3. A writes the dataset, features and limitations sections of the report.

**Afternoon**

1. Everyone checks that the repo runs from a fresh clone.
2. C finalises the report and slides with A's and B's sections and results.
3. Rehearse the demo twice: one legitimate URL, one phishing-looking URL, and one failure case explained honestly.

## 7. Evaluation plan

- **Split:** 80/20 `GroupShuffleSplit` by domain, with a fixed random seed.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix. Missing a phishing URL is worse than a false alarm, so highlight phishing recall.
- **Experiments:** (1) Logistic Regression vs Random Forest vs SVM; (2) URL-only features vs the dataset's own pre-extracted columns; (3) optional: add character n-gram TF-IDF features.
- **Sanity rule:** this dataset is known to be easy. If URL-only results are suspiciously close to 100%, look for leakage (shared domains across splits, features using the label) before celebrating.

## 8. Known limitations (include these in the report)

- **Dataset bias.** Legitimate URLs in the dataset may be mostly site homepages, while phishing URLs tend to have longer paths. The model may then flag real deep links on legitimate sites. Test this with the real-world URL sheet and report what you find.
- **URL-only view.** Page content, domain age and certificate details are not used, so a well-crafted phishing URL can slip through.
- **Time drift.** Phishing tactics change; a model trained on a 2023-2024 snapshot will age.
- **Safety.** The app never visits submitted URLs. It is a learning project and not a replacement for real security tools.

## 9. Day-1 sanity-check script

```python
import pandas as pd

df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")
print(df.shape)                      # expect (235795, 56)
print(df.columns.tolist())           # confirm exact column names
print(df["label"].value_counts())    # expect 134850 of label 1 and 100945 of label 0
print(df[["URL", "Domain", "label"]].sample(10, random_state=1))
print("duplicate URLs:", df["URL"].duplicated().sum())
print("unique domains:", df["Domain"].nunique())
```

Spot-check a few rows by eye: label 1 rows should look like normal well-known sites, label 0 rows like suspicious ones. If it looks reversed, flip the mapping in Module A.

## 10. Optional extensions (only if ahead of schedule)

- A character-level CNN or character n-gram model as the "deep learning" angle, compared against Random Forest.
- XGBoost or LightGBM as a stronger tabular baseline.
- SHAP values for per-URL explanations in the app.
- A typosquatting feature: edit distance between the URL's domain and a list of top sites, which fits the "website impersonation" theme.
