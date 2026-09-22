"""Pause predictor — gradient boosting (CatBoost), lab 2.

Same fixes as `pause_predictor_linear.py` (see its module docstring for the full
diagnosis): a hard rule override for strong punctuation, a tuned F1-maximizing
decision threshold instead of class balancing, log-space duration regression, and
the extended lexical/syntactic feature set from `features.py`.

Categorical features (`punct_class`, POS context) are passed to CatBoost directly,
by name (`cat_features=...`) -- no one-hot encoding needed, CatBoost handles
categoricals natively.

Run as a script to fit on `train` and score on both folds::

    python pause_predictor_catboost.py
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from features import (
    FeatureExtractor, CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURES,
    load_cached_features,
)

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'
FEATURES_CACHE_PATH = 'data/RUSLAN_pause_features.csv'
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Punctuation classes with P(pause) >= ~0.9 in the EDA (period/multi-dot/!/?);
# "closing" (~0.75) and "semicolon" (1 sample in the corpus) are deliberately left
# out -- not reliable enough to force without the model's judgement.
STRONG_PUNCT_CLASSES = frozenset({'period', 'ellipsis', 'exclaim', 'question'})


def _select_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Decision threshold maximizing F1 on the given (val) set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.nanargmax(f1s[:-1]))  # last P/R pair has no matching threshold
    return float(thresholds[best_idx])


class PausePredictorCatboost:
    """Predicts pause placement and duration with CatBoost, a hard rule for strong
    punctuation, and a tuned decision threshold.

    Same public contract as `pause_predictor.PausePredictor` (`predict`,
    `predict_durations`), plus `fit()`.
    """

    def __init__(self, use_pos: bool = True, strong_punct_classes: frozenset = STRONG_PUNCT_CLASSES,
                 val_size: float = 0.15, random_state: int = 42, verbose: bool = False):
        self.feature_extractor = FeatureExtractor(use_pos=use_pos)
        self.strong_punct_classes = strong_punct_classes
        self.val_size = val_size
        self.random_state = random_state

        self.clf = CatBoostClassifier(
            random_seed=random_state,
            loss_function='Logloss',
            eval_metric='F1',
            cat_features=CATEGORICAL_FEATURES,
            verbose=verbose,
        )
        self.reg = CatBoostRegressor(
            random_seed=random_state,
            loss_function='MAE',
            eval_metric='MAE',
            cat_features=CATEGORICAL_FEATURES,
            verbose=verbose,
        )
        self.threshold = 0.5
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> 'PausePredictorCatboost':
        """Fit both models.

        `train_df` must be `set == 'train'` rows, including every sentence's last
        word (features need full-sentence context). The last-word exclusion from
        training targets happens internally -- see `pause_predictor_linear.py` for
        the same note, it applies identically here.
        """
        train_df = train_df.reset_index(drop=True)
        if set(ALL_FEATURES).issubset(train_df.columns):
            feats = train_df.copy()
        else:
            feats = self.feature_extractor.extract_dataframe(train_df)
        # CatBoost categorical columns must not be float/NaN -- ensure plain strings.
        feats[CATEGORICAL_FEATURES] = feats[CATEGORICAL_FEATURES].astype(str)

        trainable = train_df.is_last_word.values == 0
        X_cls = feats.loc[trainable, FEATURE_COLUMNS]
        y_cls = train_df.is_pause_after.values[trainable]

        # Threshold picked on a held-out slice of train, then the final classifier
        # is refit on the full trainable set so no data is wasted for the model
        # itself -- only for threshold selection.
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_cls, y_cls, test_size=self.val_size, random_state=self.random_state, stratify=y_cls,
        )
        self.clf.fit(X_fit, y_fit)
        val_proba = self.clf.predict_proba(X_val)[:, 1]
        self.threshold = _select_threshold(y_val, val_proba)
        self.clf.fit(X_cls, y_cls)

        tp_mask = trainable & (train_df.is_pause_after.values == 1)
        X_reg = feats.loc[tp_mask, FEATURE_COLUMNS]
        y_reg = np.log1p(train_df.pause_duration.values[tp_mask])  # log-space target
        self.reg.fit(X_reg, y_reg)

        self._fitted = True
        return self

    def _classify(self, feats: pd.DataFrame) -> np.ndarray:
        """Probability-threshold classification + hard override for strong
        punctuation, shared by `predict` and `predict_batch`.
        """
        feats = feats.copy()
        feats[CATEGORICAL_FEATURES] = feats[CATEGORICAL_FEATURES].astype(str)
        X = feats[FEATURE_COLUMNS]

        proba = self.clf.predict_proba(X)[:, 1]
        is_pause = (proba >= self.threshold).astype(int)
        strong_mask = feats.punct_class.isin(self.strong_punct_classes).values
        is_pause[strong_mask] = 1
        return is_pause

    def _regress(self, feats: pd.DataFrame, is_pause: np.ndarray) -> np.ndarray:
        feats = feats.copy()
        feats[CATEGORICAL_FEATURES] = feats[CATEGORICAL_FEATURES].astype(str)

        n = len(feats)
        pause_duration = np.zeros(n, dtype=float)
        mask = is_pause.astype(bool)
        if mask.any():
            log_durations = self.reg.predict(feats.loc[mask, FEATURE_COLUMNS])
            durations = np.expm1(log_durations)
            durations = np.clip(durations, a_min=1e-3, a_max=None)
            pause_duration[mask] = durations
        return pause_duration

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same contract as `pause_predictor.PausePredictor.predict`."""
        if not self._fitted:
            raise RuntimeError('Call fit() before predict().')

        tokens = list(tokens)
        feats = self.feature_extractor.extract_sentence(tokens)
        is_pause = self._classify(feats)
        pause_duration = self._regress(feats, is_pause)
        return is_pause, pause_duration

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same contract as `pause_predictor.PausePredictor.predict_durations`."""
        def expand_is_pause(token, is_pause):
            return [token, '<SIL>'] if bool(is_pause) else [token]

        def expand_durations(pause_duration):
            return [-1., pause_duration] if pause_duration > 0. else [-1.]

        is_pause, durations = self.predict(tokens)
        tokens_w_pauses = np.concatenate([expand_is_pause(a, b) for a, b in zip(tokens, is_pause)])
        durations_w_pauses = np.concatenate([expand_durations(d) for d in durations]).astype(np.float32)
        return tokens_w_pauses, durations_w_pauses

    def predict_batch(self, feats_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized prediction directly from a frame that already carries
        `ALL_FEATURES` columns (e.g. loaded from the features cache).

        Skips feature extraction (no Natasha calls) entirely -- for fast bulk
        evaluation over an already-featurized train/test set.
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before predict_batch().')

        is_pause = self._classify(feats_df)
        pause_duration = self._regress(feats_df, is_pause)
        return is_pause, pause_duration


def _load_data_with_features() -> pd.DataFrame:
    """Load the metadata, attach cached features if available (built by
    `build_features_cache.py`), or fall back to extracting them now.
    """
    df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)

    cached = load_cached_features(df, FEATURES_CACHE_PATH)
    if cached is not None:
        print(f'[pause_predictor_catboost] Loaded features from {FEATURES_CACHE_PATH}')
        return cached

    print(f'[pause_predictor_catboost] No usable cache at {FEATURES_CACHE_PATH} -- '
          f'extracting features now (consider running build_features_cache.py first).')
    return FeatureExtractor(use_pos=True).extract_dataframe(df)


if __name__ == '__main__':
    from pause_predictor import calc_metrics

    df = _load_data_with_features()
    train_df = df[df.set == 'train'].reset_index(drop=True)
    test_df = df[df.set == 'test'].reset_index(drop=True)

    pp = PausePredictorCatboost(verbose=100).fit(train_df)
    from sklearn.metrics import precision_score, recall_score, average_precision_score

    # --- Диагностика 1: Вывод Feature Importance ---
    print("\n=== CatBoost Feature Importance ===")
    importances = pp.clf.get_feature_importance(prettified=True)
    print(importances)

    for name, part in [('train', train_df), ('test', test_df)]:
        evaluable = part[part.is_last_word == 0].copy()
        is_pause_hat, pause_duration_hat = pp.predict_batch(evaluable)
        evaluable['is_pause_hat'] = is_pause_hat
        evaluable['pause_duration_hat'] = pause_duration_hat
        
        print(f'\n-- {name} --')
        calc_metrics(evaluable)
        
        # --- Диагностика 2: Разбивка Precision/Recall по punct_class на тесте ---
        if name == 'test':
            print("\n=== Метрики по классам пунктуации (Test) ===")
            for cls in evaluable.punct_class.unique():
                sub = evaluable[evaluable.punct_class == cls]
                if sub.is_pause_after.sum() == 0:
                    continue
                prc = precision_score(sub.is_pause_after, sub.is_pause_hat, zero_division=0)
                rec = recall_score(sub.is_pause_after, sub.is_pause_hat, zero_division=0)
                print(f"{cls:<10} | Всего токенов: {sub.shape[0]:<6} | P = {prc:.3f} | R = {rec:.3f}")

