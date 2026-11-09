"""Run the model over tests/sample_urls.csv and record where it fails.

Run:  python -m src.realworld_check
Out:  results/realworld_check.csv
"""

from pathlib import Path

import pandas as pd

from src.explain import analyze_url, load_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pipe, meta = load_model()
    df = pd.read_csv(ROOT / "tests" / "sample_urls.csv")
    rows = []
    for _, r in df.iterrows():
        res = analyze_url(r["url"], pipe, meta)
        rows.append({
            "url": r["url"], "note": r["note"], "expected": "phishing" if r["expected_phishing"] else "legit",
            "p_phishing": round(res["proba_phishing"], 3),
            "model_verdict": "phishing" if res["model_is_phishing"] else "legit",
            "app_verdict": "phishing" if res["is_phishing"] else "legit",
            "allowlisted": res["known_domain"],
            "model_correct": res["model_is_phishing"] == bool(r["expected_phishing"]),
            "top_reason": res["reasons"][0]["text"] if res["reasons"] else "",
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "realworld_check.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(out[["url", "expected", "p_phishing", "model_verdict", "app_verdict"]].to_string(index=False))
    print(f"\nmodel alone: {out.model_correct.sum()}/{len(out)} correct; "
          f"app (with allowlist): {(out.app_verdict == out.expected).sum()}/{len(out)} correct")


if __name__ == "__main__":
    main()
