"""Скелет предсказателя пауз — lab 2.

Определяет базовый интерфейс `PausePredictor` (`predict`, `predict_durations`)
и функцию подсчёта метрик, общую для всех реализаций (linear, catboost).

Запуск как скрипта считает метрики базового (пустого) предиктора на подготовленных
данных::

    python pause_predictor.py

Считаются precision, recall и F1 для `is_pause_after`, и MAE для `pause_duration`
только по истинно положительным срабатываниям. Последнее слово каждого
высказывания исключается из оценки.
"""
import csv

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'


class PausePredictor():
    """Предсказывает расположение пауз в предложении и их длительность.

    На вход подаётся одно предложение как последовательность токенов
    `label_raw` — слова с последующей пунктуацией, по порядку::

        ["Я", "вышел", "из", "дома,", "когда", "стемнело."]
    """

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Решает для каждого токена, следует ли за ним пауза.

        Args:
            tokens: Токены одного предложения.

        Returns:
            Кортеж `(is_pause, pause_duration)`: `is_pause` (int, 1 если пауза
            следует за токеном) и `pause_duration` (float, секунды, 0.0 где
            паузы нет), оба длиной `len(tokens)`.

        Note:
            Там, где `is_pause` равен 1, длительность должна быть положительной:
            :meth:`predict_durations` на это полагается.
        """
        is_pause = np.zeros(len(tokens), int)
        pause_duration = np.zeros(len(tokens), float)

        return is_pause, pause_duration

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вставляет предсказанные паузы в последовательность токенов.

        Это формат, который потребляет акустическая модель в лабах 4 и 5.

        Args:
            tokens: Токены одного предложения.

        Returns:
            Кортеж `(tokens_w_pauses, durations_w_pauses)`: токены с `"<SIL>"`
            после каждой предсказанной паузы, и по одной длительности на
            выходной токен: секунды для `"<SIL>"`, `-1.0` для слов (оставлено
            акустической модели).
        """
        def expand_is_pause(token: str, is_pause: int) -> list[str]:
            if bool(is_pause):
                return [token, '<SIL>']
            return [token]

        def expand_durations(pause_duration: float) -> list[float]:
            if pause_duration > 0.:
                return [-1., pause_duration]
            return [-1.]

        is_pause, durations = self.predict(tokens)

        tokens_w_pauses = np.concatenate([expand_is_pause(a, b) for a, b in zip(tokens, is_pause)])
        durations_w_pauses = np.concatenate([expand_durations(dur) for dur in durations]).astype(np.float32)

        return tokens_w_pauses, durations_w_pauses


def calc_metrics(df: pd.DataFrame) -> None:
    """Печатает precision, recall и F1 для расположения пауз, и MAE для длительности.

    MAE считается только по строкам, где пауза есть и в разметке, и в предсказании.

    Args:
        df: Таблица со столбцами `is_pause_after`, `is_pause_hat`,
            `pause_duration`, `pause_duration_hat`.
    """
    rec = recall_score(df.is_pause_after, df.is_pause_hat)
    prc = precision_score(df.is_pause_after, df.is_pause_hat)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat)

    mae = mean_absolute_error(
        df[(df.is_pause_after == 1) & (df.is_pause_hat == 1)].pause_duration,
        df[(df.is_pause_after == 1) & (df.is_pause_hat == 1)].pause_duration_hat,
    )
    print(f'PRC: {prc}, REC: {rec}, F1: {f1}; MAE: {mae};')


def test_pause_predictor() -> None:
    """Прогоняет предиктор по каждому предложению и печатает метрики train/test.

    Ожидает разметку, записанную `prepare_training_data.py`: строки сгруппированы
    по высказыванию по порядку, каждое высказывание заканчивается строкой с
    `is_last_word`.
    """
    pause_df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)

    pp = PausePredictor()

    lens = {i: l for i, l in pause_df.groupby('id').count().reset_index(drop=False)[['id', 'label']].values}

    is_pause_after_hat = []
    pause_duration_hat = []

    idx = 0
    for i, is_last in tqdm.tqdm(pause_df[['id', 'is_last_word']].values):
        if not is_last:
            continue
        sentence = pause_df.iloc[idx:idx + lens[i]]
        idx += lens[i]

        is_pause_hat, pause_dur_hat = pp.predict(sentence.label_raw.values)
        is_pause_after_hat += list(is_pause_hat)
        pause_duration_hat += list(pause_dur_hat)

    pause_df['is_pause_hat'] = is_pause_after_hat
    pause_df['pause_duration_hat'] = pause_duration_hat

    print('Calculate metrics, traning fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set == 'train') & (pause_df.is_last_word == 0)])

    print('\nCalculate metrics, testing fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set == 'test') & (pause_df.is_last_word == 0)])


if __name__ == '__main__':
    test_pause_predictor()
