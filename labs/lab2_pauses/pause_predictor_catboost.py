"""Предсказатель пауз на градиентном бустинге (CatBoost).

Две независимые модели:

* Классификатор (`CatBoostClassifier`) -- есть ли пауза после токена
  (таргет `is_pause_after`). Обучается на Logloss: F1 недифференцируем, напрямую
  его не оптимизировать. F1 считается как `eval_metric` и определяет качество
  через порог `DECISION_THRESHOLD`. Жёсткое правило поверх классификатора
  форсирует паузу после сильной пунктуации (точка, многоточие, !, ?).

* Регрессор длительности (`CatBoostRegressor`, MAE) -- обучается ТОЛЬКО на строках
  с `is_pause_after == 1` (отсутствие паузы -- дело классификатора).

Последнее слово каждой записи исключается из таргетов обеих моделей (по протоколу
задачи: пауза после него -- стык аудиофайлов, а не естественная пауза), но остаётся
в данных: признакам нужен контекст всей записи.

Категориальные признаки передаются в CatBoost напрямую (`cat_features`). Слова,
редкие в train, в `prev_word`/`next_word` заменяются на `<UNK>` (словарь -- только по train).

Запуск обучает модель на `train` и печатает метрики на обеих выборках::

    uv run python pause_predictor_catboost.py

Разбор ошибок и важности признаков -- в `analyze_results.py`.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    apply_word_vocab,
    build_word_vocab,
    extract_sentence,
    load_features,
)
from pause_predictor import PausePredictor, calc_metrics

# Классы пунктуации с P(пауза) >= ~0.9 по данным EDA; "closing" (~0.75) и
# "semicolon" (единичные примеры) намеренно не форсируются -- недостаточно
# надёжны, чтобы переопределять решение модели.
STRONG_PUNCT_CLASSES = frozenset({'period', 'ellipsis', 'exclaim', 'question'})

# Порог зафиксирован: перебор по сетке на валидации давал нестабильный выигрыш по F1.
DECISION_THRESHOLD = 0.4
RANDOM_STATE = 42


class PausePredictorCatboost(PausePredictor):
    """Места пауз и их длительность -- CatBoost'ом, с жёстким правилом для сильной пунктуации."""

    def __init__(self, verbose: bool | int = False):
        self.word_vocab: frozenset | None = None  # заполняется в fit()
        self.clf = CatBoostClassifier(
            random_seed=RANDOM_STATE, loss_function='Logloss', eval_metric='F1',
            cat_features=CATEGORICAL_FEATURES, verbose=verbose,
        )
        self.reg = CatBoostRegressor(
            random_seed=RANDOM_STATE, loss_function='MAE', eval_metric='MAE',
            cat_features=CATEGORICAL_FEATURES, verbose=verbose,
        )

    def _prepare_X(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Матрица признаков: редкие слова -> `<UNK>`, категориальные -- строками."""
        X = apply_word_vocab(feats[ALL_FEATURES], self.word_vocab)
        X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype(str)
        return X

    def fit(self, train_df: pd.DataFrame) -> 'PausePredictorCatboost':
        """Обучает классификатор и регрессор.

        Args:
            train_df: Строки `train` вместе с признаками (`load_features`), включая
                последние слова предложений. Нужны колонки `ALL_FEATURES`, `label_raw`,
                `is_last_word`, `is_pause_after`, `pause_duration`.
        """
        train_df = train_df.reset_index(drop=True)
        self.word_vocab = build_word_vocab(train_df.label_raw.values)
        print(f'Словарь prev_word/next_word: {len(self.word_vocab)} слов, остальные -> <UNK>')
        X = self._prepare_X(train_df)

        trainable = train_df.is_last_word.values == 0  # последнее слово не участвует в таргетах
        self.clf.fit(X[trainable], train_df.is_pause_after.values[trainable])

        with_pause = trainable & (train_df.is_pause_after.values == 1)
        print(f'Регрессор: {with_pause.sum()} строк с паузой')
        self.reg.fit(X[with_pause], train_df.pause_duration.values[with_pause])
        return self

    def predict_batch(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Предсказание по фрейму с колонками `ALL_FEATURES` (без вызовов Natasha).

        Returns:
            `(is_pause, pause_duration)`, как в `PausePredictor.predict`.
        """
        if self.word_vocab is None:
            raise RuntimeError('Сначала вызовите fit().')

        feats = feats.reset_index(drop=True)
        X = self._prepare_X(feats)

        proba = self.clf.predict_proba(X)[:, 1]
        is_pause = (proba >= DECISION_THRESHOLD).astype(int)
        is_pause[feats.punct_class.isin(STRONG_PUNCT_CLASSES).values] = 1

        pause_duration = np.zeros(len(feats))
        mask = is_pause.astype(bool)
        if mask.any():
            # строго положительная длительность: на это полагается predict_durations
            pause_duration[mask] = np.clip(self.reg.predict(X[mask]), 1e-3, None)
        return is_pause, pause_duration

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_batch(extract_sentence(list(tokens)))


def predict_on(pp: PausePredictorCatboost, df: pd.DataFrame) -> pd.DataFrame:
    """Предсказания для фрейма с признаками, без последнего токена каждой записи (протокол оценки)."""
    part = df[df.is_last_word == 0].copy()
    part['is_pause_hat'], part['pause_duration_hat'] = pp.predict_batch(part)
    return part


if __name__ == '__main__':
    df = load_features()
    train_df = df[df.set == 'train']
    test_df = df[df.set == 'test']

    pp = PausePredictorCatboost(verbose=100).fit(train_df)
    print(f'Порог решения (фиксированный): {DECISION_THRESHOLD}')

    for name, part in [('train', train_df), ('test', test_df)]:
        print(f'\n-- {name} --')
        calc_metrics(predict_on(pp, part))
