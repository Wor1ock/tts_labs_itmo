"""Построение кеша фичей для всего корпуса — lab 2.

Читает `data/RUSLAN_pause_metadata.csv`, извлекает все фичи (включая
POS-разметку Natasha — самую медленную часть) для каждой строки и записывает их
в `data/RUSLAN_pause_features.csv`. Запускается один раз; `pause_predictor.py`
и `pause_predictor_catboost.py` подхватят кеш автоматически при последующих
запусках и пропустят извлечение фичей::

    python build_features_cache.py

Перезапускать нужно каждый раз, когда меняется `RUSLAN_pause_metadata.csv`
(например, после повторного запуска `prepare_training_data.py` на новом
выравнивании) — скрипты-предикторы сами обнаруживают несовпадение кеша и
предупреждают об этом, но лучше не забывать перезапускать этот скрипт первым.
"""
import csv
import time

import pandas as pd

from features import FeatureExtractor, extract_and_cache_dataframe

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'
FEATURES_CACHE_PATH = 'data/RUSLAN_pause_features.csv'


def main() -> None:
    """Извлекает фичи для всего корпуса и сохраняет их в файл кеша."""
    df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)
    print(f'Loaded {len(df)} rows, {df.id.nunique()} sentences from {PAUSE_PREDICTOR_DATA}')

    extractor = FeatureExtractor(use_pos=True)

    t0 = time.time()
    extract_and_cache_dataframe(extractor, df, FEATURES_CACHE_PATH)
    print(f'Done in {time.time() - t0:.1f}s. Features cached at {FEATURES_CACHE_PATH}')


if __name__ == '__main__':
    main()
