"""Предсказатель пауз на градиентном бустинге (CatBoost) — lab 2.

Та же схема, что и в `pause_predictor_linear.py` (полное обоснование — в его
докстринге): жёсткое правило для сильной пунктуации, длительность паузы
регрессируется напрямую в секундах (без лог-трансформации — она ухудшала MAE),
и набор фичей `punct_class` / `tokens_since_punct` / `tokens_to_punct`. Порог
принятия решения зафиксирован константой (без перебора по сетке — см. п.7
задания), т.к. подбор по F1 на валидации давал нестабильный выигрыш и усложнял
пайплайн.

Категориальные фичи (`punct_class`, POS-контекст) передаются в CatBoost
напрямую по имени (`cat_features=...`) — one-hot не нужен, CatBoost работает
с категориальными признаками нативно.

Запуск как скрипта обучает модель на `train` и считает метрики на обеих
выборках, плюс важность фичей и диагностику по классам пунктуации::

    python pause_predictor_catboost.py
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import f1_score, precision_score, recall_score

from features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureExtractor,
    load_cached_features,
)

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'
FEATURES_CACHE_PATH = 'data/RUSLAN_pause_features.csv'

# is_next_cconj / is_curr_noun_or_propn (п.8) считаются в features.py и уже
# входят в NUMERIC_FEATURES -- отдельно их тут добавлять не нужно.
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Классы пунктуации с P(пауза) >= ~0.9 по EDA (точка/многоточие/!/?);
# "closing" (~0.75) и "semicolon" (1 пример в корпусе) намеренно не форсируются --
# недостаточно надёжны, чтобы переопределять решение модели.
STRONG_PUNCT_CLASSES = frozenset({'period', 'ellipsis', 'exclaim', 'question'})

# Порог классификации зафиксирован (п.7 задания) -- без перебора по сетке.
DECISION_THRESHOLD = 0.4


class PausePredictorCatboost:
    """Предсказывает расположение и длительность пауз CatBoost'ом, с жёстким
    правилом для сильной пунктуации и фиксированным порогом решения.

    Тот же публичный контракт, что и `pause_predictor.PausePredictor`
    (`predict`, `predict_durations`), плюс `fit()`.
    """

    def __init__(self, use_pos: bool = True, strong_punct_classes: frozenset = STRONG_PUNCT_CLASSES,
                 threshold: float = DECISION_THRESHOLD, random_state: int = 42, verbose: bool = False):
        self.feature_extractor = FeatureExtractor(use_pos=use_pos)
        self.strong_punct_classes = strong_punct_classes
        self.threshold = threshold
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
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> 'PausePredictorCatboost':
        """Обучает классификатор и регрессор.

        `train_df` должен содержать строки `set == 'train'`, включая последнее
        слово каждого предложения (фичам нужен контекст всего предложения).
        Исключение последнего слова из таргетов происходит внутри метода.

        Args:
            train_df: Обучающая выборка (сырые строки метаданных или уже
                посчитанные фичи, если содержит все колонки `ALL_FEATURES`).

        Returns:
            self, с обученными `clf` и `reg`.
        """
        train_df = train_df.reset_index(drop=True)
        if set(ALL_FEATURES).issubset(train_df.columns):
            feats = train_df.copy()
        else:
            feats = self.feature_extractor.extract_dataframe(train_df)
        # Категориальные колонки CatBoost не должны быть float/NaN -- приводим к строкам.
        feats[CATEGORICAL_FEATURES] = feats[CATEGORICAL_FEATURES].astype(str)

        trainable = train_df.is_last_word.values == 0
        X_cls = feats.loc[trainable, FEATURE_COLUMNS]
        y_cls = train_df.is_pause_after.values[trainable]
        self.clf.fit(X_cls, y_cls)

        tp_mask = trainable & (train_df.is_pause_after.values == 1)
        X_reg = feats.loc[tp_mask, FEATURE_COLUMNS]
        y_reg = train_df.pause_duration.values[tp_mask]  # сырые секунды, без лог-трансформации
        self.reg.fit(X_reg, y_reg)

        self._fitted = True
        return self

    def _classify(self, feats: pd.DataFrame) -> np.ndarray:
        """Классификация по порогу вероятности + жёсткое правило для сильной
        пунктуации, общая для `predict` и `predict_batch`.

        Args:
            feats: Таблица фичей (уже с `pos_curr`/`pos_next1`).

        Returns:
            Массив `is_pause` (0/1) той же длины, что и `feats`.
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
        """Регрессия длительности паузы там, где `is_pause == 1`.

        Args:
            feats: Таблица фичей.
            is_pause: Маска предсказанных пауз (0/1).

        Returns:
            Массив длительностей (секунды), 0.0 там, где паузы нет.
        """
        feats = feats.copy()
        feats[CATEGORICAL_FEATURES] = feats[CATEGORICAL_FEATURES].astype(str)

        n = len(feats)
        pause_duration = np.zeros(n, dtype=float)
        mask = is_pause.astype(bool)
        if mask.any():
            durations = self.reg.predict(feats.loc[mask, FEATURE_COLUMNS])
            durations = np.clip(durations, a_min=1e-3, a_max=None)
            pause_duration[mask] = durations
        return pause_duration

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Тот же контракт, что `pause_predictor.PausePredictor.predict`."""
        if not self._fitted:
            raise RuntimeError('Call fit() before predict().')

        tokens = list(tokens)
        feats = self.feature_extractor.extract_sentence(tokens)
        is_pause = self._classify(feats)
        pause_duration = self._regress(feats, is_pause)
        return is_pause, pause_duration

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Тот же контракт, что `pause_predictor.PausePredictor.predict_durations`."""
        def expand_is_pause(token, is_pause):
            return [token, '<SIL>'] if bool(is_pause) else [token]

        def expand_durations(pause_duration):
            return [-1., pause_duration] if pause_duration > 0. else [-1.]

        is_pause, durations = self.predict(tokens)
        tokens_w_pauses = np.concatenate([expand_is_pause(a, b) for a, b in zip(tokens, is_pause)])
        durations_w_pauses = np.concatenate([expand_durations(d) for d in durations]).astype(np.float32)
        return tokens_w_pauses, durations_w_pauses

    def predict_batch(self, feats_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Векторизованное предсказание сразу из фрейма с колонками `ALL_FEATURES`
        (например, загруженного из кеша фичей).

        Пропускает извлечение фичей (без вызовов Natasha) -- для быстрой
        массовой оценки на уже посчитанном train/test.

        Args:
            feats_df: Таблица фичей.

        Returns:
            Кортеж `(is_pause, pause_duration)`, как в `predict`.
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before predict_batch().')

        is_pause = self._classify(feats_df)
        pause_duration = self._regress(feats_df, is_pause)
        return is_pause, pause_duration


def _load_data_with_features() -> pd.DataFrame:
    """Загружает метаданные и подключает закешированные фичи, если они есть
    (собраны `build_features_cache.py`), иначе считает фичи заново.

    Returns:
        Таблица метаданных с фичами.
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
    print(f'Decision threshold (fixed): {pp.threshold:.3f}')

    # --- Диагностика 1: Feature Importance ---
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

        # --- Диагностика 2: Precision/Recall по punct_class на тесте ---
        if name == 'test':
            print("\n=== Метрики по классам пунктуации (Test) ===")
            for cls in evaluable.punct_class.unique():
                sub = evaluable[evaluable.punct_class == cls]
                if sub.is_pause_after.sum() == 0:
                    continue
                prc = precision_score(sub.is_pause_after, sub.is_pause_hat, zero_division=0)
                rec = recall_score(sub.is_pause_after, sub.is_pause_hat, zero_division=0)
                print(f"{cls:<10} | Всего токенов: {sub.shape[0]:<6} | P = {prc:.3f} | R = {rec:.3f}")
