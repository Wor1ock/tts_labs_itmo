"""Grapheme-to-phoneme: lexicon -> MFA dictionary -> rules; `<SIL>` passes through unchanged.

`uv run python g2p.py` runs the `<SIL>` checks and prints the ablations and the error analysis.
"""
import csv
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
TOKENS_PATH = BASE / 'data/tokens_phones_sil.csv'
LEXICON_PATH = BASE / 'data/lexicon.csv'
MFA_DICT_PATH = BASE / 'data/russian_mfa.dict'
TARGET_PHONES_PATH = BASE.parents[1] / 'common/phones_mfa.txt'

SIL = '<SIL>'
log = logging.getLogger(__name__)

CSV_SEP = '|'
N_FOLDS = 5   # lab 2's split rule: test iff the utterance number divides by N_FOLDS


def read_pipe_csv(path, **kwargs):
    """Read a pipe-separated CSV in the format every script in this project writes."""
    return pd.read_csv(path, sep=CSV_SEP, quoting=csv.QUOTE_NONE, keep_default_na=False, **kwargs)


def write_pipe_csv(df, path, **kwargs):
    """Write a pipe-separated CSV matching read_pipe_csv's format."""
    df.to_csv(path, sep=CSV_SEP, quoting=csv.QUOTE_NONE, index=False, encoding='utf-8', **kwargs)


def utterance_fold(sentence_id):
    """Utterance number mod N_FOLDS; fold 0 is the test set (lab 2's split rule)."""
    return int(str(sentence_id).split('_')[0]) % N_FOLDS


# Beware: 'ɡ' is IPA U+0261, not the Latin 'g'.
# The first two entries adapt the MFA dictionary; the rest merge phones too rare to train an embedding
# (24, 10 and 1 corpus examples). The map is applied to the reference as well, so the metrics stay comparable.
PHONE_MAP = {'dzʲ': 'd̪ z̪', 'tsʲ': 't̪s̪', 'd̪z̪': 'd̪ z̪', 'ɣ': 'ɡ', 'ʑ': 'ʐ'}

# Final obstruent of these words assimilates to the next word. `к` is left out: k -> g is rare in the corpus.
ASSIMILATING_WORDS = frozenset({'в', 'с', 'из', 'без', 'от', 'под', 'над', 'об'})

# Spelling rewrites applied in order; they keep the vowel count, so the stress position stays valid.
REWRITES = (
    ('chto', r'^что', 'што'),
    ('tsya', r'ть?ся$', 'тса'),
    ('sch', r'[сз]ч', 'щ'),
    ('clusters', 'стн', 'сн'), ('clusters', 'здн', 'зн'), ('clusters', 'стл', 'сл'),
    ('clusters', 'вств', 'ств'), ('clusters', 'лнц', 'нц'),
    ('gk', r'г([кч])', r'х\1'),
    ('ogo', r'(?<=..)([ое])го$', r'\1во'),
)
RULES = tuple(dict.fromkeys([name for name, *_ in REWRITES] +
                            ['stress_nie', 'palatal', 'voicing', 'degemination']))
COMPONENTS = frozenset(RULES) | {'mfa', 'rules', 'assimilation'}

VOICED_TO_VOICELESS = {
    'b': 'p', 'bʲ': 'pʲ', 'v': 'f', 'vʲ': 'fʲ', 'd̪': 't̪', 'dʲ': 'tʲ',
    'z̪': 's̪', 'zʲ': 'sʲ', 'ɡ': 'k', 'ɟ': 'c', 'ʐ': 'ʂ',
}
VOICELESS_TO_VOICED = {v: k for k, v in VOICED_TO_VOICELESS.items()}
VOICELESS_OBSTRUENTS = frozenset(VOICELESS_TO_VOICED) | {'x', 'ç', 'tɕ', 'ɕ', 't̪s̪', 'tʂ'}
VOICED_OBSTRUENTS = frozenset(VOICED_TO_VOICELESS) | {'dʐ'}
# `v` does not voice the consonant before it (из вас -> iz vas)
_VOICING_TRIGGERS = VOICED_OBSTRUENTS - {'v', 'vʲ'}

_HARD = {
    'б': 'b', 'в': 'v', 'г': 'ɡ', 'д': 'd̪', 'ж': 'ʐ', 'з': 'z̪', 'й': 'j', 'к': 'k',
    'л': 'ɫ', 'м': 'm', 'н': 'n̪', 'п': 'p', 'р': 'r', 'с': 's̪', 'т': 't̪', 'ф': 'f',
    'х': 'x', 'ц': 't̪s̪', 'ч': 'tɕ', 'ш': 'ʂ', 'щ': 'ɕ',
}
# л and н carry no `ʲ`; velar stops become palatal c / ɟ / ç, as in the corpus
_SOFT = {
    'б': 'bʲ', 'в': 'vʲ', 'г': 'ɟ', 'д': 'dʲ', 'з': 'zʲ', 'к': 'c', 'л': 'ʎ', 'м': 'mʲ',
    'н': 'ɲ', 'п': 'pʲ', 'р': 'rʲ', 'с': 'sʲ', 'т': 'tʲ', 'ф': 'fʲ', 'х': 'ç',
}
_VOWEL_PHONES = frozenset('aeoiuɨɪəɐʊɛɵæʉ')
_RULE_PHONES = frozenset(_HARD.values()) | frozenset(_SOFT.values()) | _VOWEL_PHONES

_CONS_LETTERS = frozenset('бвгджзйклмнпрстфхцчшщ')
_VOWEL_LETTERS = frozenset('аеёиоуыэюя')
_SOFTENERS = frozenset('иеёюяь')
_ALWAYS_SOFT = frozenset('чщ')
_ALWAYS_HARD = frozenset('жшц')
# consonant letter -> letters after which it is softened by the next soft consonant
_ASSIM_SOFT = {'з': 'тдснз', 'с': 'тдснз', 'в': 'пбмф', 'г': 'д'}
_NIE_RE = re.compile(r'([аеяи])ни(?:е|я|и|ю|ем|ей|й|ям|ях)$')   # -ание, -ение: stress before ни

_LATIN = {'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г', 'h': 'х',
          'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п',
          'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
          'y': 'и', 'z': 'з'}


def normalize_token(token, keep_stress=False):
    token = unicodedata.normalize('NFC', str(token)).lower()
    if not keep_stress:
        token = token.replace('\u0301', '')
    return re.sub(r'[^\w\u0301-]', '', token).strip('-')


def map_phones(phones):
    return [new for phone in phones for new in PHONE_MAP.get(phone, phone).split()]


def load_mfa_dict():
    """Most probable pronunciation of each word; `spn` entries are skipped."""
    best = {}
    with open(MFA_DICT_PATH, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            word, *rest = line.rstrip('\n').split('\t')
            prob = float(rest[0]) if len(rest) > 1 else 1.0
            phones = map_phones(unicodedata.normalize('NFC', rest[-1]).replace('ː', '').split())
            word = normalize_token(word)
            if word and phones and 'spn' not in phones and (word not in best or prob > best[word][0]):
                best[word] = (prob, phones)
    return {word: phones for word, (_, phones) in best.items()}


def build_lexicon(tokens):
    """token | phones | frequency from a table of tokens without pauses."""
    return (tokens.groupby(['token', 'phones']).size().reset_index(name='frequency')
            .sort_values(['token', 'frequency'], ascending=[True, False]))


def rule_transcribe(word, off=frozenset()):
    """Rule-based transcription of a normalized word with stress marks kept; `off` names disabled rules."""
    letters, stress_ord, n_vowels = [], None, 0
    for ch in word:
        if ch == '\u0301':
            if letters and letters[-1] in _VOWEL_LETTERS:
                stress_ord = n_vowels - 1
            continue
        for c in _LATIN.get(ch, ch):
            if c in _CONS_LETTERS or c in _VOWEL_LETTERS or c in 'ьъ':
                letters.append(c)
                n_vowels += c in _VOWEL_LETTERS
            else:
                log.warning('rules: symbol %r skipped in %r', c, word)
    s = ''.join(letters)
    for name, pattern, repl in REWRITES:
        if name not in off:
            s = re.sub(pattern, repl, s)

    n = len(s)
    if not n:
        return []
    vowels_at = [i for i, ch in enumerate(s) if ch in _VOWEL_LETTERS]
    # the text carries no stress, so outside the explicit mark and `ё` this is a guess
    if not vowels_at:
        stressed_at = None
    elif stress_ord is not None and stress_ord < len(vowels_at):
        stressed_at = vowels_at[stress_ord]
    elif 'ё' in s:
        stressed_at = s.index('ё')
    elif len(vowels_at) == 1:
        stressed_at = vowels_at[0]
    elif 'stress_nie' not in off and (match := _NIE_RE.search(s)):
        stressed_at = match.start(1)
    else:
        stressed_at = vowels_at[-2]
    stressed_ord = vowels_at.index(stressed_at) if stressed_at is not None else None

    soft = [False] * n
    for i, ch in enumerate(s):
        if ch in _CONS_LETTERS and ch != 'й':
            if ch in _ALWAYS_SOFT:
                soft[i] = True
            elif ch not in _ALWAYS_HARD:
                soft[i] = i + 1 < n and s[i + 1] in _SOFTENERS
    if 'palatal' not in off:
        for i in range(n - 2, -1, -1):      # softening before a soft consonant: здесь, вперёд
            if s[i] == 'н' and s[i + 1] in 'чщ':
                soft[i] = True
            elif s[i] in _ASSIM_SOFT and s[i + 1] in _ASSIM_SOFT[s[i]] and soft[i + 1]:
                soft[i] = True

    phones = []
    for i, ch in enumerate(s):
        if ch in 'ьъ':
            continue
        if ch in _CONS_LETTERS:
            phones.append(_SOFT[ch] if soft[i] and ch in _SOFT else _HARD[ch])
            continue
        prev = s[i - 1] if i else ''
        nxt = s[i + 1] if i + 1 < n else ''
        iotated = (ch in 'яеёю' and (i == 0 or prev in _VOWEL_LETTERS or prev in 'ьъ')) \
            or (ch == 'и' and prev == 'ь')
        if iotated:
            phones.append('j')
        soft_ctx = iotated or prev == 'й' or (prev in _CONS_LETTERS and soft[i - 1])
        next_soft = nxt in _CONS_LETTERS and nxt != 'й' and soft[i + 1]
        stressed = i == stressed_at
        pretonic = stressed_ord is not None and vowels_at.index(i) == stressed_ord - 1
        hard_special = prev in _ALWAYS_HARD         # the vowel follows ж, ш or ц
        if ch == 'ы':
            vowel = 'ɨ'
        elif ch == 'и':
            vowel = 'ɨ' if hard_special else ('i' if stressed else 'ɪ')
        elif ch in 'ая':
            if stressed:
                vowel = 'æ' if soft_ctx and next_soft else 'a'
            elif soft_ctx:
                vowel = 'ɪ' if pretonic else 'ə'
            else:
                vowel = 'ɐ' if pretonic else 'ə'
        elif ch == 'ё':
            vowel = 'o' if hard_special else 'ɵ'
        elif ch == 'о':
            vowel = 'o' if stressed else ('ɐ' if pretonic else 'ə')
        elif ch in 'ую':
            vowel = ('ʉ' if soft_ctx and next_soft else 'u') if stressed else 'ʊ'
        elif ch == 'э':
            vowel = 'ɛ' if stressed else 'ɪ'
        elif hard_special:                          # е
            vowel = 'ɛ' if stressed else 'ɨ'
        elif stressed:
            vowel = 'e'
        else:
            vowel = 'e' if i == n - 1 else 'ɪ'      # final unstressed -е is written `e` in the MFA data
        phones.append(vowel)

    if 'voicing' not in off:
        for i in range(len(phones) - 1, -1, -1):
            if i == len(phones) - 1 or phones[i + 1] in VOICELESS_OBSTRUENTS:
                phones[i] = VOICED_TO_VOICELESS.get(phones[i], phones[i])
            elif phones[i + 1] in _VOICING_TRIGGERS:
                phones[i] = VOICELESS_TO_VOICED.get(phones[i], phones[i])
    if 'degemination' in off:
        return phones
    out = []
    for phone in phones:                            # the corpus has no length mark: сс -> s̪
        if not (out and phone == out[-1] and phone not in _VOWEL_PHONES):
            out.append(phone)
    return out


class G2P:
    """Sentence-level G2P. `disabled` switches off components ('mfa', 'rules', 'assimilation') and rules by name."""

    def __init__(self, lexicon=None, disabled=()):
        unknown = set(disabled) - COMPONENTS
        if unknown:
            raise ValueError(f'unknown components: {sorted(unknown)}')
        self.disabled = frozenset(disabled)
        with open(TARGET_PHONES_PATH, encoding='utf-8') as fh:
            self.target = frozenset(unicodedata.normalize('NFC', fh.read()).split())
        self._mfa = load_mfa_dict()

        if lexicon is None:
            lexicon = read_pipe_csv(LEXICON_PATH)
        counts = defaultdict(Counter)
        for word, phones, count in lexicon[['token', 'phones', 'frequency']].itertuples(index=False):
            counts[normalize_token(word)][tuple(phones.split())] += count
        self._lex = {word: variants.most_common() for word, variants in counts.items()}
        # variants matter only for words whose ending depends on the next word
        self._assim = {w: v for w, v in self._lex.items() if w in ASSIMILATING_WORDS and len(v) > 1}

        # fail on load, not silently at synthesis time
        used = set(_RULE_PHONES)
        used |= {p for phones in self._mfa.values() for p in phones}
        used |= {p for variants in self._lex.values() for phones, _ in variants for p in phones}
        if used - self.target:
            raise ValueError(f'phones outside {TARGET_PHONES_PATH.name}: {sorted(used - self.target)}')

    def source(self, token):
        """'lexicon', 'homograph' (several variants), 'mfa' or 'rules'."""
        key = normalize_token(token)
        if key in self._lex:
            return 'homograph' if len(self._lex[key]) > 1 else 'lexicon'
        return 'mfa' if 'mfa' not in self.disabled and key in self._mfa else 'rules'

    def lookup(self, token):
        """Phones of one word without context."""
        key = normalize_token(token)
        source = self.source(key)
        if source in ('lexicon', 'homograph'):
            return list(self._lex[key][0][0])
        if source == 'mfa':
            return list(self._mfa[key])
        if '-' in key:
            return [p for part in key.split('-') if part for p in self.lookup(part)]
        if 'rules' not in self.disabled and key:
            return rule_transcribe(normalize_token(token, keep_stress=True), self.disabled)
        return []

    def transcribe(self, tokens):
        """One string of space-separated phones per token; `<SIL>` stays `<SIL>`.

        Tokens go right to left: the ending of в / с / из ... depends on the next spoken word.
        """
        tokens = list(tokens)
        result, next_phone = [''] * len(tokens), None
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == SIL:
                result[i], next_phone = SIL, None
                continue
            key = normalize_token(tokens[i])
            if 'assimilation' not in self.disabled and key in self._assim:
                wanted = (VOICELESS_OBSTRUENTS if next_phone in VOICELESS_OBSTRUENTS
                          else VOICED_OBSTRUENTS if next_phone in _VOICING_TRIGGERS else ())
                variants = self._assim[key]     # sorted by frequency
                phones = list(next((p for p, _ in variants if p[-1] in wanted), variants[0][0]))
            else:
                phones = self.lookup(tokens[i])
            if not phones and 'rules' not in self.disabled:
                log.warning('no transcription for %r', tokens[i])
            result[i] = ' '.join(phones)
            next_phone = phones[0] if phones else next_phone
        return result


def load_tokens(path=TOKENS_PATH):
    return read_pipe_csv(path, dtype={'id': str, 'token': str, 'phones': str})


def edit_distance(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def per(*groups):
    """Corpus-level PER: summed edit distance over summed reference length."""
    length = sum(g['len'] for g in groups)
    return sum(g['dist'] for g in groups) / length if length else float('nan')


def evaluate(model, tokens):
    """Edit distance, reference length and word counts per group.

    'running' is a whole sentence without pauses; the other groups are `model.source` values and do not overlap.
    """
    stats = defaultdict(Counter)

    def add(group, ref, hyp):
        stats[group].update(dist=edit_distance(ref, hyp), len=len(ref), words=1, exact=int(ref == hyp))

    for _, sentence in tokens.groupby('id', sort=False):
        words = sentence.token.tolist()
        pred = [p.split() for p in model.transcribe(words)]
        ref = [p.split() for p in sentence.phones]
        add('running', [p for r in ref for p in r if p != SIL], [p for h in pred for p in h if p != SIL])
        for word, r, h in zip(words, ref, pred):
            if word != SIL:
                add(model.source(word), r, h)
    return stats


if __name__ == '__main__':
    tokens = load_tokens()
    train_tokens, test_tokens = (tokens[tokens.set == fold] for fold in ('train', 'test'))
    model = G2P()

    for case in ([], [SIL], [SIL, 'мама', SIL, SIL, 'мыла', SIL], ['в', SIL, 'школе']):
        out = model.transcribe(case)
        assert len(out) == len(case) and all((o == SIL) == (t == SIL) for t, o in zip(case, out)), case
        assert {p for o in out for p in o.split()} - {SIL} <= model.target
    print('<SIL> checks passed\n')

    configs = {
        'lexicon only': {'mfa', 'rules', 'assimilation'},
        '+ MFA': {'rules', 'assimilation'},
        '+ rules': {'assimilation'},
        '+ assimilation (full)': set(),
        'full without MFA': {'mfa'},
    }
    print(f'{"PER":<24}{"train":>8}{"test":>8}{"OOV":>8}{"residual":>10}   (OOV and residual: test; residual = words')
    print(f'{"":<24}{"":>34}   transcribed by rules, i.e. in no available dictionary)')
    for name, disabled in configs.items():
        m = G2P(disabled=disabled)
        train, test = evaluate(m, train_tokens), evaluate(m, test_tokens)
        print(f'{name:<24}{per(train["running"]):8.4f}{per(test["running"]):8.4f}'
              f'{per(test["mfa"], test["rules"]):8.4f}{per(test["rules"]):10.4f}')

    test = evaluate(model, test_tokens)
    labels = {'lexicon': 'in lexicon, 1 variant', 'homograph': 'in lexicon, >1 variants',
              'mfa': 'OOV, in MFA', 'rules': 'OOV, nowhere'}
    n_words = sum(test[g]['words'] for g in labels)
    n_errors = sum(test[g]['dist'] for g in labels)
    n_phones = sum(test[g]['len'] for g in labels)
    print('\ntest, full model:')
    for group, label in labels.items():
        s = test[group]
        print(f'{label:<24} words {s["words"] / n_words:6.1%}  PER {per(s):.4f}  '
              f'exact {s["exact"] / s["words"]:.4f}  share of errors {s["dist"] / n_errors:6.1%}')
    print(f'word-level PER {n_errors / n_phones:.4f}; '
          f'with homographs resolved perfectly {(n_errors - test["homograph"]["dist"]) / n_phones:.4f}')

    # rules alone on every train word (as if it were in no dictionary): the rules are judged on train
    train_words = (train_tokens[(train_tokens.token != SIL) & ~train_tokens.token.str.contains('-')]
                   .groupby('token').phones.agg(lambda s: s.mode()[0]))
    refs = [(normalize_token(word, keep_stress=True), phones.split())
            for word, phones in train_words.items()]
    ref_length = sum(len(r) for _, r in refs)
    print(f'\nrule ablation, PER{"rules on train words":>24}{"test residual":>16}')
    for rule in (None, *RULES):
        model.disabled = frozenset([rule] if rule else [])
        hyps = [rule_transcribe(w, model.disabled) for w, _ in refs]
        dists = [edit_distance(r, h) for (_, r), h in zip(refs, hyps)]
        if rule is None:
            worst = sorted(zip(dists, refs, hyps), key=lambda e: -e[0])[:20]
        print(f'{"all rules" if rule is None else "without " + rule:<26}{sum(dists) / ref_length:20.4f}'
              f'{per(evaluate(model, test_tokens)["rules"]):16.4f}')

    print('\nworst words for the rules on train (reference | rules):')
    for dist, (word, ref), hyp in worst:
        print(f'{word:<22}{dist:3d}  {" ".join(ref)}  |  {" ".join(hyp)}')

    running_per, residual_per = per(test['running']), per(test['rules'])
    print(f'\nacceptance: running PER {running_per:.4f} <= 0.025, residual PER {residual_per:.4f} <= 0.20')
    assert running_per <= 0.025, f'PER on test running text {running_per:.4f} exceeds the 0.025 threshold'
    assert residual_per <= 0.20, f'PER on test residual {residual_per:.4f} exceeds the 0.20 threshold'
