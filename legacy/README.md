# Legacy v1 (kept for comparison)

This folder contains the first version of Sightector, kept unchanged so its results can be reproduced:
the notebook, the pickled scikit-learn models and the original Flask server.

**Do not use it in production.** The project audit (see [docs/REPORT.md](../docs/REPORT.md) §2) found three defects:

1. `app.py` re-fits the TF-IDF vectorizer on every message (`fit_transform`), which discards the learned IDF weights.
2. The notebook duplicates the dataset (`pd.concat([df, df])`) **before** splitting, so test rows leak into training.
3. Word-only features and an aggressive stop-list cannot represent disguised words (`fu*ck`, `iiidiot`).

`python -m sightector evaluate` re-scores this model exactly as it was served, for the before/after comparison.
