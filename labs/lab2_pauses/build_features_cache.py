"""Build the features cache once, for the whole corpus — lab 2.

Reads `data/RUSLAN_pause_metadata.csv`, extracts all features (including Natasha
POS-tagging, the slow part) for every row, and writes them to
`data/RUSLAN_pause_features.csv`. Run this once; `pause_predictor_linear.py` and
`pause_predictor_catboost.py` will pick the cache up automatically on later runs and
skip extraction entirely::

    python build_features_cache.py

Re-run it whenever `RUSLAN_pause_metadata.csv` changes (e.g. you re-ran
`prepare_training_data.py` on new alignment) — the predictor scripts detect a
mismatched cache and warn you to do so themselves, but it doesn't hurt to just
remember to re-run this first.
"""
import csv
import time

import pandas as pd

from features import FeatureExtractor, extract_and_cache_dataframe

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'
FEATURES_CACHE_PATH = 'data/RUSLAN_pause_features.csv'


def main() -> None:
    df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)
    print(f'Loaded {len(df)} rows, {df.id.nunique()} sentences from {PAUSE_PREDICTOR_DATA}')

    extractor = FeatureExtractor(use_pos=True)

    t0 = time.time()
    extract_and_cache_dataframe(extractor, df, FEATURES_CACHE_PATH)
    print(f'Done in {time.time() - t0:.1f}s. Features cached at {FEATURES_CACHE_PATH}')


if __name__ == '__main__':
    main()
