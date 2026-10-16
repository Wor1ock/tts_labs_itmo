"""Lab 3 data from the MFA alignment: tokens_phones.csv, tokens_phones_sil.csv, lexicon.csv, alignment_raw.csv.

Sentences with `spn` or with a word without phones are dropped whole; alignment_raw.csv keeps everything for the EDA.
"""
import unicodedata
from pathlib import Path

import pandas as pd
from praatio import textgrid
from tqdm import tqdm

import g2p

BASE = Path(__file__).parent
ROOT = BASE.parents[1]
ALIGN_DIR = ROOT / 'data/RUSLAN_align/v2'
NORMALIZED_META = ROOT / 'data/metadata_RUSLAN_22200_normalized.csv'
OUT_DIR = BASE / 'data'

SIL = g2p.SIL


def numbered(df):
    df = df.assign(token_id=df.groupby('id').cumcount())
    return df[['id', 'token_id', 'token', 'phones', 'set']]


def main() -> None:
    # the alignment was built on raw text; keep only utterances that survived the lab 1 filter
    with open(NORMALIZED_META, encoding='utf-8') as fh:
        keep = {line.split('|', 1)[0] for line in fh if line.strip()}

    rows = []
    for path in tqdm(sorted(ALIGN_DIR.glob('*.TextGrid'))):
        if path.stem not in keep:
            continue
        word_tier, phone_tier = textgrid.openTextgrid(str(path), True).tiers
        phones = [(p.start, p.end, p.label) for p in phone_tier.entries]
        fold = 'test' if g2p.utterance_fold(path.stem) == 0 else 'train'
        for entry in word_tier.entries:
            if entry.label:
                # tier boundaries can differ in the last decimal, so phones are matched by midpoint;
                # an empty phone label inside a word is dropped: only pauses between words become <SIL>
                token = unicodedata.normalize('NFC', entry.label)
                word_phones = ' '.join(
                    unicodedata.normalize('NFC', label).replace('ː', '')
                    for start, end, label in phones
                    if label and entry.start <= (start + end) / 2 < entry.end)
            else:
                token = word_phones = SIL
            rows.append({'id': path.stem, 'token': token, 'phones': word_phones, 'set': fold})

    raw = pd.DataFrame(rows)
    bad = (raw.phones == '') | raw.phones.str.split().map(lambda p: 'spn' in p)
    clean = raw[~raw.id.isin(set(raw.id[bad]))]
    print(f'dropped {raw.id[bad].nunique()} sentences with spn or a word without phones')

    # sentences are written without a leading pause, as in the task example
    leading = ~clean.id.duplicated() & (clean.token == SIL)
    print(f'{leading.sum()} of {clean.id.nunique()} sentences start with a pause, dropped')
    with_sil = numbered(clean[~leading])
    # PHONE_MAP is applied once, here, so lexicon.csv and the reference carry already-mapped phones
    with_sil = with_sil.assign(phones=with_sil.phones.map(lambda p: ' '.join(g2p.map_phones(p.split()))))
    tokens = numbered(with_sil[with_sil.token != SIL])
    print(f'{len(tokens)} tokens in {tokens.id.nunique()} sentences, '
          f'{(tokens.drop_duplicates("id").set == "test").mean():.1%} test')

    OUT_DIR.mkdir(exist_ok=True)
    g2p.write_pipe_csv(raw, OUT_DIR / 'alignment_raw.csv')
    g2p.write_pipe_csv(tokens, OUT_DIR / 'tokens_phones.csv')
    g2p.write_pipe_csv(with_sil, OUT_DIR / 'tokens_phones_sil.csv')

    # train fold only, otherwise test words end up in the dictionary
    lexicon = g2p.build_lexicon(tokens[tokens.set == 'train'])
    g2p.write_pipe_csv(lexicon, OUT_DIR / 'lexicon.csv')
    print(f'lexicon: {lexicon.token.nunique()} types, '
          f'{(lexicon.token.value_counts() > 1).sum()} with several pronunciations')


if __name__ == '__main__':
    main()
