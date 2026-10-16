"""Stage 4: data/metadata_phones_gt.csv (reference) and data/metadata_phones.csv (G2P), format id|text|phones."""
import re
from collections import Counter

import pandas as pd

import g2p

TEXTS_PATH = g2p.BASE.parents[1] / 'data/metadata_RUSLAN_22200_normalized.csv'   # first two columns: id|text


def main() -> None:
    tokens = g2p.load_tokens()
    tokens['fold'] = tokens.id.map(g2p.utterance_fold)   # fold 0 is the test set
    texts = g2p.read_pipe_csv(TEXTS_PATH, header=None, usecols=[0, 1])
    texts = dict(zip(texts[0], texts[1]))

    # a train sentence is transcribed by a lexicon built without its own fold, so its G2P errors look like test ones
    words = tokens[(tokens.token != g2p.SIL) & (tokens.set == 'train')]
    models = {0: g2p.G2P()}
    for fold in sorted(set(tokens.fold) - {0}):
        models[fold] = g2p.G2P(g2p.build_lexicon(words[words.fold != fold]))

    gt_rows, pred_rows = [], []
    dist, length, exact, sentences = Counter(), Counter(), Counter(), Counter()
    length_mismatch = 0
    for sent_id, sentence in tokens.groupby('id', sort=True):
        out = models[sentence.fold.iloc[0]].transcribe(sentence.token)
        gt = [p for phones in sentence.phones for p in phones.split()]
        pred = [p for phones in out for p in phones.split()]
        gt_rows.append((sent_id, texts[sent_id], ' '.join(gt)))
        pred_rows.append((sent_id, texts[sent_id], ' '.join(pred)))

        # the text comes from the lab 1 file, the words from the alignment: a cheap sync check
        n_text = sum(bool(re.search(r'\w', w)) for w in texts[sent_id].split())
        length_mismatch += n_text != (sentence.token != g2p.SIL).sum()

        fold = sentence.set.iloc[0]
        ref, hyp = [p for p in gt if p != g2p.SIL], [p for p in pred if p != g2p.SIL]
        dist[fold] += g2p.edit_distance(ref, hyp)
        length[fold] += len(ref)
        exact[fold] += ref == hyp
        sentences[fold] += 1

    print(f'sentences whose word count differs between text and alignment: {length_mismatch}')
    for fold in ('train', 'test'):
        print(f'{fold}: pred vs gt PER {dist[fold] / length[fold]:.4f}, '
              f'identical sentences {exact[fold] / sentences[fold]:.1%}')

    for name, rows in (('metadata_phones_gt', gt_rows), ('metadata_phones', pred_rows)):
        used = {p for _, _, phones in rows for p in phones.split()} - {g2p.SIL}
        assert used <= models[0].target, f'{name}: phones outside the target inventory: {used - models[0].target}'
        g2p.write_pipe_csv(pd.DataFrame(rows, columns=['id', 'text', 'phones']),
                            g2p.BASE / f'data/{name}.csv', header=False)
        n_sil = sum(p.split().count(g2p.SIL) for _, _, p in rows)
        print(f'{name}: {len(rows)} sentences, {n_sil} <SIL>')


if __name__ == '__main__':
    main()
