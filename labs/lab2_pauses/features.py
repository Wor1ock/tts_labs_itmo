"""Признаки для предсказателя пауз.

Точки входа:
    extract_sentence(tokens) -- инференс: признаки одного предложения (токены `label_raw`)
    extract_features(df)     -- обучение: признаки по фрейму формата RUSLAN_pause_metadata.csv
    load_features()          -- то же, но с кешем на диске (POS-разметка Natasha медленная)

Построить кеш заранее (пересчитывается сам, если устарел; чтобы форсировать -- удалите файл кеша)::

    uv run python features.py
"""
import csv
import re
from collections import Counter
from functools import lru_cache

import pandas as pd
from natasha import Doc, NewsEmbedding, NewsMorphTagger, Segmenter
from tqdm.auto import tqdm

from paths import FEATURES_CACHE_PATH, PAUSE_METADATA_PATH

# --- пунктуация ---------------------------------------------------------------------

NO_PUNCT = 'none'
OTHER_PUNCT = 'other'
_PUNCT_BY_LAST_CHAR = {
    '.': 'period', ',': 'comma', ';': 'semicolon', ':': 'colon', '!': 'exclaim',
    '?': 'question', '—': 'dash', '-': 'dash',
    ')': 'closing', ']': 'closing', '»': 'closing', '"': 'closing',
}
# Классы, закрывающие грамматическое предложение: по ним запись `id` режется на предложения.
SENTENCE_BOUNDARY_PUNCT = frozenset({'period', 'ellipsis', 'exclaim', 'question'})

_QUOTE_CHARS = frozenset('«»"“”')
_VOWELS = frozenset('аеёиоуыэюя')  # в русском слог = гласная
_NON_LETTERS_RE = re.compile(r'[^а-яё]')

PAD = '<PAD>'    # соседа нет (граница записи); в словах и в классах пунктуации
UNK_WORD = '<UNK>'
PAD_POS = 'NONE'
MIN_WORD_COUNT = 2  # слова, встречающиеся в train реже, -> UNK_WORD


def clean_letters(label: str) -> str:
    """Только кириллические буквы в нижнем регистре."""
    return _NON_LETTERS_RE.sub('', label.lower()) if isinstance(label, str) else ''


def punct_class(label_raw: str) -> str:
    """Класс пунктуации в конце токена (`comma`, `period`, ...; `none` -- нет, `other` -- прочее)."""
    if not isinstance(label_raw, str):
        return NO_PUNCT
    s = label_raw.strip()
    if s.endswith(('...', '…')):
        return 'ellipsis'
    if not s:
        return NO_PUNCT
    if s[-1] in _PUNCT_BY_LAST_CHAR:
        return _PUNCT_BY_LAST_CHAR[s[-1]]
    return NO_PUNCT if re.match(r'[а-яёa-z]', s[-1].lower()) else OTHER_PUNCT


# --- токенизация: пунктуация остаётся приклеенной к словам ---------------------------
# Открывающие символы уходят в ПРЕФИКС следующего слова («Дорал), всё остальное между
# словами (запятая, », —, …) -- в хвост предыдущего. Используется и при подготовке
# обучающих данных (`split_gap`), и в инференсе (`tokenize_text`).
OPENING_CHARS = '«„“‘([{"\''
_OPENING_TAIL_RE = re.compile('[' + re.escape(OPENING_CHARS) + ']+$')


def split_gap(gap: str) -> tuple[str, str]:
    """Промежуток между словами -> (хвост предыдущего слова, префикс следующего)."""
    m = _OPENING_TAIL_RE.search(gap)
    prefix = m.group() if m else ''
    return gap[:len(gap) - len(prefix)].strip(), prefix


def tokenize_text(text: str) -> list[str]:
    """Текст -> токены `label_raw` (слово + пунктуация вокруг), как в обучающих данных.

    Делит по пробелам, но отдельно стоящие знаки (`«`, `—`, `…`) приклеивает:
    открывающие -- к следующему слову, остальные -- к предыдущему.
    """
    tokens: list[str] = []
    prefix = ''
    for w in text.split():
        if set(w) <= set(OPENING_CHARS):
            prefix += w
        elif tokens and not any(c.isalnum() for c in w):
            tokens[-1] += ' ' + w
        else:
            tokens.append(prefix + w)
            prefix = ''
    return tokens


# --- POS-теггинг (Natasha) ----------------------------------------------------------

@lru_cache(maxsize=1)
def _natasha():
    """Тяжёлые модели грузим один раз на процесс."""
    return Segmenter(), NewsMorphTagger(NewsEmbedding())


def pos_tags(words: list[str]) -> list[str]:
    """POS-теги слов (по контексту всего предложения, не пословно), len(result) == len(words).

    Если токенизация Natasha не совпала 1:1, теги обрезаются/добиваются `X`,
    чтобы одно странное предложение не роняло прогон по корпусу.
    """
    if not words:
        return []
    segmenter, tagger = _natasha()
    doc = Doc(' '.join(w for w in words if w))
    doc.segment(segmenter)
    doc.tag_morph(tagger)
    tags = [t.pos for t in doc.tokens if t.pos != 'PUNCT']
    return (tags + ['X'] * len(words))[:len(words)]


# --- признаки -----------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    'punct_class', 'pos_prev2', 'pos_prev1', 'pos_curr', 'pos_next1', 'pos_next2',
    'prev_word', 'next_word', 'next_punct_class',
]
NUMERIC_FEATURES = [
    'word_len', 'n_syllables', 'rel_pos', 'pos_in_sentence', 'sent_len',
    'tokens_since_punct', 'tokens_to_punct', 'tokens_to_end',
    'is_next_cconj', 'next_is_stop', 'is_curr_noun_or_propn', 'is_curr_propn', 'has_quote_mark',
    'is_not_last_sentence', 'n_sentences_in_id',
    'tokens_to_next_comma', 'commas_left_in_clause',
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def extract_sentence(tokens: list[str]) -> pd.DataFrame:
    """Признаки одного предложения: по строке на токен `label_raw` (например "дома,").

    `is_last_word` признаком намеренно нет: он восстанавливается из `rel_pos == 1.0`.
    Колонка остаётся в метаданных только чтобы исключать последнее слово из
    обучения и оценки.
    """
    n = len(tokens)
    if n == 0:
        return pd.DataFrame(columns=ALL_FEATURES)

    clean = [clean_letters(t) for t in tokens]
    pcs = [punct_class(t) for t in tokens]
    pos = [PAD_POS] * 2 + pos_tags(clean) + [PAD_POS] * 2  # окно +/-2 вокруг токена

    # Вперёд: расстояние от последней пунктуации и номер грамматического предложения.
    # Один `id` -- не обязательно одно предложение, режем по сильной пунктуации.
    since = [0] * n
    sent_idx = [0] * n
    last_punct, sent = -1, 0
    for i, pc in enumerate(pcs):
        since[i] = i - last_punct - 1
        sent_idx[i] = sent
        if pc != NO_PUNCT:
            last_punct = i
        if pc in SENTENCE_BOUNDARY_PUNCT:
            sent += 1
    n_sent = sent_idx[-1] + 1

    # Назад: расстояния до ближайшей пунктуации / запятой / конца предложения.
    # commas_left_in_clause: 0 у запятой -- граница клаузы (последняя запятая перед
    # точкой), >=1 -- перечисление (впереди ещё запятые).
    to_punct, to_comma, commas_left, to_end = [0] * n, [0] * n, [0] * n, [0] * n
    next_punct = next_comma = n
    sent_end = n - 1
    commas_ahead = 0
    for i in range(n - 1, -1, -1):
        pc = pcs[i]
        if pc in SENTENCE_BOUNDARY_PUNCT:
            sent_end = i
        if pc != NO_PUNCT:
            next_punct = i
        if pc == 'comma':
            next_comma = i
        to_punct[i] = next_punct - i
        to_comma[i] = next_comma - i
        to_end[i] = sent_end - i
        commas_left[i] = commas_ahead
        if pc == 'comma':
            commas_ahead += 1
        if pc in SENTENCE_BOUNDARY_PUNCT:
            commas_ahead = 0

    feats = pd.DataFrame({
        'punct_class': pcs,
        'word_len': [len(w) for w in clean],
        'n_syllables': [sum(ch in _VOWELS for ch in w) for w in clean],
        'rel_pos': [i / max(n - 1, 1) for i in range(n)],
        'pos_in_sentence': list(range(n)),
        'sent_len': n,
        'tokens_since_punct': since,
        'tokens_to_punct': to_punct,
        'tokens_to_end': to_end,
        # редкие слова заменяются на <UNK> уже при обучении (`apply_word_vocab`);
        # в кеше лежат сырые слова, пустая строка -- токен без кириллицы
        'prev_word': [PAD] + clean[:-1],
        'next_word': clean[1:] + [PAD],
        'next_punct_class': pcs[1:] + [PAD],
        'has_quote_mark': [int(isinstance(t, str) and any(c in _QUOTE_CHARS for c in t)) for t in tokens],
        'is_not_last_sentence': [int(s < n_sent - 1) for s in sent_idx],
        'n_sentences_in_id': n_sent,
        'tokens_to_next_comma': to_comma,
        'commas_left_in_clause': commas_left,
        'pos_prev2': pos[0:n],
        'pos_prev1': pos[1:n + 1],
        'pos_curr': pos[2:n + 2],
        'pos_next1': pos[3:n + 3],
        'pos_next2': pos[4:n + 4],
    })
    feats['is_next_cconj'] = (feats.pos_next1 == 'CCONJ').astype(int)
    feats['next_is_stop'] = feats.pos_next1.isin(['ADP', 'CCONJ', 'SCONJ']).astype(int)
    feats['is_curr_noun_or_propn'] = feats.pos_curr.isin(['NOUN', 'PROPN']).astype(int)
    feats['is_curr_propn'] = (feats.pos_curr == 'PROPN').astype(int)
    return feats[ALL_FEATURES]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки по фрейму с колонками `id`, `label_raw`; возвращает `df` + колонки признаков.

    Предложение нужно передавать целиком, включая последнее слово, иначе
    неверно посчитаются признаки позиции и контекста.
    """
    df = df.reset_index(drop=True)
    # Строки одного `id` идут подряд (так пишет prepare_training_data.py), поэтому
    # порядок групп совпадает с порядком строк.
    parts = [extract_sentence(g.label_raw.tolist())
             for _, g in tqdm(df.groupby('id', sort=False), desc='Извлечение признаков')]
    return pd.concat([df, pd.concat(parts, ignore_index=True)], axis=1)


def load_features(metadata_path=PAUSE_METADATA_PATH, cache_path=FEATURES_CACHE_PATH) -> pd.DataFrame:
    """Метаданные + признаки. Берёт кеш, если он совпадает с метаданными, иначе пересчитывает и пишет кеш."""
    df = pd.read_csv(metadata_path, sep='|', quoting=csv.QUOTE_NONE)

    # escapechar нужен: токены могут содержать кавычку, а csv с QUOTE_NONE без него на ней падает.
    csv_kwargs = dict(sep='|', quoting=csv.QUOTE_NONE, escapechar='\\')

    if cache_path.exists():
        cached = pd.read_csv(cache_path, keep_default_na=False, **csv_kwargs)
        if (set(cached.columns) == set(ALL_FEATURES) | {'id'}
                and len(cached) == len(df) and (cached.id.values == df.id.values).all()):
            return pd.concat([df, cached[ALL_FEATURES]], axis=1)
        print(f'Кеш {cache_path} устарел (другие признаки или данные) -- пересчитываю.')

    result = extract_features(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result[['id'] + ALL_FEATURES].to_csv(cache_path, index=False, **csv_kwargs)
    return result


# --- словарь слов -------------------------------------------------------------------
# `prev_word`/`next_word` -- категориальные признаки с огромной кардинальностью.
# Слова-одиночки ничему не учат, только раздувают память и переобучают модель.
# Словарь строится ТОЛЬКО по train (иначе в test утекает информация); всё вне
# словаря -- и редкие train-слова, и незнакомые слова при инференсе, и токены без
# кириллицы -- превращается в `<UNK>`.

def build_word_vocab(labels, min_count: int = MIN_WORD_COUNT) -> frozenset:
    """Слова (только буквы) из `labels`, встречающиеся не реже `min_count` раз."""
    counts = Counter(clean_letters(t) for t in labels)
    counts.pop('', None)
    return frozenset(w for w, c in counts.items() if c >= min_count)


def apply_word_vocab(feats: pd.DataFrame, vocab) -> pd.DataFrame:
    """Копия `feats`, где слова вне `vocab` в `prev_word`/`next_word` заменены на `<UNK>` (`<PAD>` остаётся)."""
    feats = feats.copy()
    for col in ('prev_word', 'next_word'):
        s = feats[col].astype(str)
        feats[col] = s.where(s.isin(vocab) | (s == PAD), UNK_WORD)
    return feats


if __name__ == '__main__':
    df = load_features()
    print(f'Признаки готовы: {len(df)} строк, кеш -- {FEATURES_CACHE_PATH}')
