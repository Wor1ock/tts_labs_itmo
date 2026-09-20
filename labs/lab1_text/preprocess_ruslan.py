"""Build the filtered and normalized RUSLAN metadata for lab 1.

Reads the corpus metadata (two columns), normalizes every utterance, drops
whatever the classifier rejects, and writes the result in LJSpeech-style
format (three columns: id, raw text, normalized text).

Run from the lab directory:

    python preprocess_ruslan.py
"""
import csv
import logging

import pandas as pd

from text_filter import TextFilter
from text_normalizer import TextNormalizer
C:\Users\vanya\Documents\tts_labs_itmo\data
C:\Users\vanya\Documents\tts_labs_itmo\labs\lab1_text
INPUT_PATH = "../../data/metadata_RUSLAN_22200.csv"
OUTPUT_PATH = "../../data/metadata_RUSLAN_22200_normalized.csv"

# quoting=csv.QUOTE_NONE is required in both directions: the corpus text contains
# « » „ " ' and pandas would otherwise read them as field delimiters.
CSV_KWARGS = {"sep": "|", "quoting": csv.QUOTE_NONE}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the normalize-then-filter pipeline and save the cleaned metadata."""
    logger.info("Reading corpus metadata from %s", INPUT_PATH)
    raw = pd.read_csv(INPUT_PATH, names=["id", "raw"], **CSV_KWARGS)
    logger.info("Loaded %d rows", len(raw))

    normalizer = TextNormalizer()
    text_filter = TextFilter()

    logger.info("Running text normalization")
    raw["nrm"] = raw["raw"].apply(normalizer.normalize)

    logger.info("Running filter classification")
    raw["is_valid"] = raw["nrm"].apply(text_filter.filter)

    clean = raw[raw["is_valid"] == 1]
    dropped = len(raw) - len(clean)
    logger.info(
        "Kept %d rows (%.2f%%), dropped %d rows (%.2f%%)",
        len(clean),
        len(clean) / len(raw) * 100,
        dropped,
        dropped / len(raw) * 100,
    )

    clean[["id", "raw", "nrm"]].to_csv(
        OUTPUT_PATH, index=False, header=False, **CSV_KWARGS
    )
    logger.info("Saved normalized metadata to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
