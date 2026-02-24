from __future__ import annotations

from typing import List, Optional, Tuple
from .phi_detector import count_phi


def leakage_score(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return sum(count_phi(t) for t in texts) / len(texts)


def utility_proxy_redaction_inverse(texts: List[str]) -> float:
    """Fewer [REDACTED] tokens and fewer PHI leaks -> higher score."""
    redactions = sum(str(t).count("[REDACTED]") for t in texts)
    return 1.0 / (1.0 + redactions)


def utility_proxy_retention(texts: List[str]) -> float:
    return utility_proxy_redaction_inverse(texts)


def compute_delta_auroc(
    original_texts: List[str],
    masked_texts: List[str],
    labels: Optional[List[int]] = None,
) -> Tuple[float, float, float]:
    """
    Downstream model utility signal: AUROC delta on a patient re-identification task.

    ```
    The downstream task is binary patient re-identification (label = patient identity,
    0 or 1). This is the correct task because:
      - Real PHI (names, MRNs, dates) is genuinely informative for re-identification.
      - Synthetic replacement degrades re-identification toward chance (negative delta).
      - Hard redaction destroys it further (more negative delta).
      - PHI-presence labels collapse to 0 delta because structural features
        remain informative regardless of whether actual PHI values are replaced.

    Method (stratified train/test split):
      1. Stratify by label so both patients appear in train and test.
      2. Fit TF-IDF + logistic regression on ORIGINAL train texts.
      3. Score on ORIGINAL test texts  -> auc_orig
      4. Score on MASKED  test texts   -> auc_mask
      5. Return (auc_mask - auc_orig, auc_orig, auc_mask)
         Negative delta = masking reduced re-identification (good -- PHI protected).

    Returns (delta, auc_orig, auc_mask). Falls back to (0.0, 0.0, 0.0) on failure.
    Labels must be patient-identity labels (0/1), not PHI-presence labels.
    """
    n = len(original_texts)
    if n < 8 or len(masked_texts) < 8:
        return 0.0, 0.0, 0.0

    if labels is None or len(set(labels)) < 2:
        return 0.0, 0.0, 0.0

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=42)
        try:
            train_idx, test_idx = next(sss.split(original_texts, labels))
        except ValueError:
            return 0.0, 0.0, 0.0

        orig_train = [original_texts[i] for i in train_idx]
        orig_test = [original_texts[i] for i in test_idx]
        mask_test = [masked_texts[i] for i in test_idx]
        y_train = [labels[i] for i in train_idx]
        y_test = [labels[i] for i in test_idx]

        if len(set(y_test)) < 2:
            return 0.0, 0.0, 0.0

        vec = TfidfVectorizer(max_features=300, sublinear_tf=True, min_df=1)
        X_train = vec.fit_transform(orig_train)
        X_orig_test = vec.transform(orig_test)
        X_mask_test = vec.transform(mask_test)

        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)

        p_orig = clf.predict_proba(X_orig_test)[:, 1]
        p_mask = clf.predict_proba(X_mask_test)[:, 1]

        auc_orig = float(roc_auc_score(y_test, p_orig))
        auc_mask = float(roc_auc_score(y_test, p_mask))
        return round(auc_mask - auc_orig, 5), round(auc_orig, 5), round(auc_mask, 5)

    except Exception:
        return 0.0, 0.0, 0.0
