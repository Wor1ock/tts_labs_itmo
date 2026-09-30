"""Базовый интерфейс предиктора пауз и общие метрики качества.

`PausePredictor` (`predict`, `predict_durations`) -- контракт, общий для всех
реализаций (например, `PausePredictorCatboost`); `calc_metrics` -- единый подсчёт метрик.

Запуск как скрипта считает метрики базового (пустого) предиктора::

    uv run python pause_predictor.py

Считаются precision, recall и F1 для `is_pause_after`, и MAE для `pause_duration`
только по истинно положительным срабатываниям. Последнее слово каждой записи
исключается из оценки.
"""
import csv

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score

from paths import PAUSE_METADATA_PATH


class PausePredictor:
    """Предсказывает расположение пауз в предложении и их длительность.

    Вход -- одно предложение как последовательность токенов `label_raw`
    (слово с последующей пунктуацией)::

        ["Я", "вышел", "из", "дома,", "когда", "стемнело."]
    """

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Для каждого токена: есть ли после него пауза и сколько она длится.

        Returns:
            `(is_pause, pause_duration)` длиной `len(tokens)`: `is_pause` -- 0/1,
            `pause_duration` -- секунды, 0.0 где паузы нет. Где `is_pause == 1`,
            длительность должна быть положительной: `predict_durations` на это полагается.
        """
        return np.zeros(len(tokens), int), np.zeros(len(tokens), float)

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вставляет предсказанные паузы в последовательность токенов.

        Формат для акустической модели. Наследуется всеми подклассами через `predict`.

        Returns:
            `(tokens_w_pauses, durations_w_pauses)`: токены с `"<SIL>"` после каждой
            паузы и по одной длительности на выходной токен: секунды для `"<SIL>"`,
            `-1.0` для слов (длительность слов оставлена акустической модели).
        """
        is_pause, durations = self.predict(tokens)

        out_tokens, out_durations = [], []
        for token, pause, duration in zip(tokens, is_pause, durations):
            out_tokens.append(token)
            out_durations.append(-1.)
            if pause:
                out_tokens.append('<SIL>')
                out_durations.append(duration)
        return np.array(out_tokens), np.array(out_durations, dtype=np.float32)


def calc_metrics(df: pd.DataFrame) -> None:
    """Печатает precision, recall, F1 для мест пауз и MAE длительности (только по TP).

    Args:
        df: Таблица со столбцами `is_pause_after`, `is_pause_hat`,
            `pause_duration`, `pause_duration_hat`.
    """
    rec = recall_score(df.is_pause_after, df.is_pause_hat)
    prc = precision_score(df.is_pause_after, df.is_pause_hat)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat)

    tp = df[(df.is_pause_after == 1) & (df.is_pause_hat == 1)]
    mae = mean_absolute_error(tp.pause_duration, tp.pause_duration_hat)
    print(f'PRC: {prc}, REC: {rec}, F1: {f1}; MAE: {mae};')


def test_pause_predictor() -> None:
    """Прогоняет предиктор по каждому предложению и печатает метрики train/test.

    Ожидает разметку от `prepare_training_data.py`: строки одной записи идут подряд.
    """
    pause_df = pd.read_csv(PAUSE_METADATA_PATH, sep='|', quoting=csv.QUOTE_NONE)
    pp = PausePredictor()

    predictions = [pp.predict(g.label_raw.values)
                   for _, g in tqdm.tqdm(pause_df.groupby('id', sort=False))]
    pause_df['is_pause_hat'] = np.concatenate([p[0] for p in predictions])
    pause_df['pause_duration_hat'] = np.concatenate([p[1] for p in predictions])

    print('Метрики, train; последние токены предложений исключены!')
    calc_metrics(pause_df[(pause_df.set == 'train') & (pause_df.is_last_word == 0)])

    print('\nМетрики, test; последние токены предложений исключены!')
    calc_metrics(pause_df[(pause_df.set == 'test') & (pause_df.is_last_word == 0)])


if __name__ == '__main__':
    test_pause_predictor()
