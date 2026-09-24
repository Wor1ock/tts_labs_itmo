"""Извлечение признаков для предсказателя пауз — lab 2.

Для каждого токена предложения строит признаки, на которых обучаются
`PausePredictorLinear` и `PausePredictorCatboost`:

    punct_class     -- тип пунктуации сразу после токена (точка, запятая, ...)
                        -- самый сильный признак (~37% важности в CatBoost)
    word_len        -- число кириллических букв в токене
    rel_pos         -- относительная позиция в предложении, 0.0 (первое) .. 1.0 (последнее)
    pos_in_sentence -- абсолютная позиция, с нуля
    sent_len        -- число токенов в предложении
    is_last_word    -- 1 для последнего слова предложения (это признак; сама
                        *метка* для этой строки всё равно исключается из
                        обучения/оценки по протоколу лабы -- эта фильтрация
                        происходит в модулях предикторов, не здесь)
    pos_prev2, pos_prev1, pos_curr, pos_next1, pos_next2
                    -- POS-теги текущего токена и соседей в окне +/-2,
                       размечены Natasha (учитывает контекст всей
                       последовательности, а не разметка слова изолированно)
    tokens_since_punct
                    -- сколько токенов назад была последняя пунктуация (0, если
                       *предыдущий* токен был с пунктуацией; иначе считается от
                       начала предложения) -- насколько глубоко мы внутри
                       непрерывного, без запятых, отрезка предложения
    tokens_to_punct -- сколько токенов до *следующей* пунктуации (0, если сам
                       текущий токен с пунктуацией; до конца предложения, если
                       дальше пунктуации нет) -- та же идея, но вперёд
    is_next_cconj   -- 1, если POS следующего токена -- сочинительный союз
                       (CCONJ); маркер границы клауз перед союзом
    is_curr_noun_or_propn
                    -- 1, если POS текущего токена -- существительное или
                       имя собственное (NOUN/PROPN); маркер конца именной группы

Две точки входа:
    FeatureExtractor.extract_sentence(tokens)   -- инференс, одно предложение
    FeatureExtractor.extract_dataframe(df)      -- обучение, батчем по формату
                                                    RUSLAN_pause_metadata.csv

Кеширование: извлечение POS-признаков по всему корпусу медленное (Natasha
вызывается на каждое предложение). `extract_and_cache_dataframe` /
`load_cached_features` внизу модуля позволяют посчитать признаки один раз (см.
`build_features_cache.py`), а `pause_predictor_linear.py` /
`pause_predictor_catboost.py` — загрузить результат вместо пересчёта на
каждом запуске.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import pandas as pd

try:
    from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter
    _NATASHA_AVAILABLE = True
except ImportError:  # pragma: no cover - окружение без natasha
    _NATASHA_AVAILABLE = False


# --- классификация пунктуации ----------------------------------------------------
# Те же категории, что использовались в EDA (report.md / eda.ipynb §5) -- чтобы
# распределения признаков совпадали с анализом, который их мотивировал.

_PUNCT_PATTERNS = [
    (re.compile(r'\.\.\.$|…$'), 'ellipsis'),
    (re.compile(r'[.]$'), 'period'),
    (re.compile(r'[,]$'), 'comma'),
    (re.compile(r'[;]$'), 'semicolon'),
    (re.compile(r'[:]$'), 'colon'),
    (re.compile(r'[!]$'), 'exclaim'),
    (re.compile(r'[?]$'), 'question'),
    (re.compile(r'[—-]$'), 'dash'),
    (re.compile(r'[)\]»"]$'), 'closing'),
]
NO_PUNCT = 'none'
OTHER_PUNCT = 'other'

_LETTERS_RE = re.compile(r'[^а-яё]')

UNK_POS = 'UNK'    # несовпадение токенизации Natasha -- запасное значение
PAD_POS = 'NONE'   # контекст за границами предложения (начало/конец)
CONTEXT_WINDOW = 2  # слов до/после

# POS-теги, из которых считаются производные бинарные признаки
_CCONJ_TAG = 'CCONJ'
_NOUN_LIKE_TAGS = frozenset({'NOUN', 'PROPN'})


def punct_class(label_raw: str) -> str:
    """Определяет тип пунктуации после токена по его форме `label_raw`.

    Args:
        label_raw: Токен с исходной пунктуацией (например, "дома,").

    Returns:
        Название класса пунктуации (`period`, `comma`, ... , `NO_PUNCT` или
        `OTHER_PUNCT`).
    """
    if not isinstance(label_raw, str):
        return NO_PUNCT
    s = label_raw.strip()
    for pattern, name in _PUNCT_PATTERNS:
        if pattern.search(s):
            return name
    return NO_PUNCT if re.search(r'[а-яёa-z]$', s.lower()) else OTHER_PUNCT


def clean_letters(label: str) -> str:
    """Оставляет только кириллические буквы (убирает пунктуацию/цифры/пробелы)."""
    if not isinstance(label, str):
        return ''
    return _LETTERS_RE.sub('', label.lower())


def word_len(label: str) -> int:
    """Число кириллических букв в токене."""
    return len(clean_letters(label))


def punct_distances(punct_classes: list[str]) -> tuple[list[int], list[int]]:
    """Расстояние (в токенах) до ближайшей пунктуации, назад и вперёд.

    `tokens_since_punct[i]` -- сколько токенов прошло с последнего
    пунктуированного токена *до* i (0 значит, что непосредственно предыдущий
    токен был с пунктуацией); от начала предложения, если пунктуации раньше не было.

    `tokens_to_punct[i]` -- сколько токенов до следующего пунктуированного
    токена *начиная с* i (0 значит, что сам токен с пунктуацией); до конца
    предложения, если дальше пунктуации нет.

    Args:
        punct_classes: Классы пунктуации всех токенов предложения, по порядку.

    Returns:
        Кортеж `(tokens_since_punct, tokens_to_punct)`, оба длиной `len(punct_classes)`.
    """
    n = len(punct_classes)
    since = [0] * n
    to = [0] * n

    last_punct_idx = -1  # виртуальная граница пунктуации до начала предложения
    for i in range(n):
        since[i] = i - last_punct_idx - 1
        if punct_classes[i] != NO_PUNCT:
            last_punct_idx = i

    next_punct_idx = n  # виртуальная граница пунктуации после конца предложения
    for i in range(n - 1, -1, -1):
        if punct_classes[i] != NO_PUNCT:
            next_punct_idx = i
        to[i] = next_punct_idx - i

    return since, to


# --- POS-теггинг (Natasha) --------------------------------------------------------

class PosTagger:
    """Обёртка над морфологическим пайплайном Natasha.

    `NewsEmbedding` -- тяжёлая модель (скачивается при первом использовании,
    затем кешируется локально) -- собирается один раз на экземпляр `PosTagger`
    и переиспользуется, а не пересобирается на каждое предложение.
    """

    def __init__(self):
        if not _NATASHA_AVAILABLE:
            raise ImportError(
                "natasha is required for POS features. Install with `pip install natasha`."
            )
        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        self._emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._emb)

    def tag_sentence(self, tokens: list[str]) -> list[str]:
        """POS-разметка предложения из "чистых" токенов (только буквы, без пунктуации).

        Собирает из токенов строку и прогоняет через сегментацию + морфологический
        тэггер Natasha (именно это делает разметку *контекстно-зависимой* -- не
        пословный lookup), затем выравнивает полученные теги обратно на входные
        токены по порядку.

        Если токенизация Natasha не совпадает 1:1 со входом (редко -- например,
        необычный токен разбился иначе), несовпадение не роняет вызов, а
        дополняется/обрезается `UNK_POS`, чтобы одно странное предложение не
        обрушивало прогон по всему корпусу.

        Args:
            tokens: Слова предложения без пунктуации.

        Returns:
            Список POS-тегов, той же длины, что и `tokens`.
        """
        if not tokens:
            return []

        text = ' '.join(t for t in tokens if t)
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        word_tags = [t.pos for t in doc.tokens if t.pos != 'PUNCT']

        n = len(tokens)
        if len(word_tags) == n:
            return word_tags
        if len(word_tags) > n:
            return word_tags[:n]
        return word_tags + [UNK_POS] * (n - len(word_tags))


@lru_cache(maxsize=1)
def get_pos_tagger() -> PosTagger:
    """Синглтон на процесс -- не даёт повторно грузить модель эмбеддингов."""
    return PosTagger()


def pos_context(pos_tags: list[str], idx: int, window: int = CONTEXT_WINDOW) -> dict:
    """POS-контекст +/- `window` вокруг позиции `idx`, за границами -- `PAD_POS`.

    Args:
        pos_tags: POS-теги всего предложения.
        idx: Позиция токена, для которого строится контекст.
        window: Размер окна в каждую сторону.

    Returns:
        Словарь с ключами `pos_prev2, pos_prev1, pos_curr, pos_next1, pos_next2`
        (для window=2).
    """
    feats = {}
    for offset in range(-window, window + 1):
        j = idx + offset
        if offset == 0:
            key = 'pos_curr'
        else:
            key = f'pos_{"prev" if offset < 0 else "next"}{abs(offset)}'
        feats[key] = pos_tags[j] if 0 <= j < len(pos_tags) else PAD_POS
    return feats


def derived_pos_features(pos_ctx: dict) -> dict:
    """Бинарные признаки, производные от POS-контекста токена.

    Args:
        pos_ctx: Словарь POS-контекста из `pos_context` (нужны ключи `pos_curr`,
            `pos_next1`).

    Returns:
        Словарь с ключами `is_next_cconj` (следующий токен -- союз CCONJ) и
        `is_curr_noun_or_propn` (текущий токен -- NOUN/PROPN), значения 0/1.
    """
    return {
        'is_next_cconj': int(pos_ctx.get('pos_next1') == _CCONJ_TAG),
        'is_curr_noun_or_propn': int(pos_ctx.get('pos_curr') in _NOUN_LIKE_TAGS),
    }


CATEGORICAL_FEATURES = ['punct_class', 'pos_prev2', 'pos_prev1', 'pos_curr', 'pos_next1', 'pos_next2']
NUMERIC_FEATURES = [
    'word_len', 'rel_pos', 'pos_in_sentence', 'sent_len',
    'tokens_since_punct', 'tokens_to_punct',
    'is_next_cconj', 'is_curr_noun_or_propn',
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class FeatureExtractor:
    """Строит таблицу признаков по токенам, общую для обеих реализаций предиктора.

    `use_pos=False` полностью пропускает Natasha (например, для быстрой проверки
    без POS-контекстных признаков, или если natasha не установлена) -- POS-колонки
    тогда заполняются `UNK_POS`, а производные от них признаки — нулями.
    """

    use_pos: bool = True
    _tagger: Optional[PosTagger] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.use_pos:
            self._tagger = get_pos_tagger()

    def extract_sentence(self, tokens: list[str]) -> pd.DataFrame:
        """Извлекает признаки для одного предложения, заданного как токены `label_raw`.

        `tokens` -- ровно то, что получает `PausePredictor.predict`: сырые токены
        с пунктуацией, например ["Я", "вышел", "из", "дома,", "когда", "стемнело."]

        Args:
            tokens: Токены одного предложения, по порядку.

        Returns:
            Таблица признаков, по одной строке на токен.
        """
        n = len(tokens)
        clean_tokens = [clean_letters(t) for t in tokens]
        punct_classes = [punct_class(t) for t in tokens]
        since_punct, to_punct = punct_distances(punct_classes)

        pos_tags = self._tagger.tag_sentence(clean_tokens) if self.use_pos else [UNK_POS] * n

        rows = []
        for i, tok in enumerate(tokens):
            row = {
                'punct_class': punct_classes[i],
                'word_len': word_len(tok),
                'rel_pos': i / max(n - 1, 1),
                'pos_in_sentence': i,
                'sent_len': n,
                'is_last_word': int(i == n - 1),
                'tokens_since_punct': since_punct[i],
                'tokens_to_punct': to_punct[i],
            }
            pos_ctx = pos_context(pos_tags, i)
            row.update(pos_ctx)
            row.update(derived_pos_features(pos_ctx))
            rows.append(row)

        return pd.DataFrame(rows)

    def extract_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Батчевое извлечение признаков по фрейму формата `RUSLAN_pause_metadata.csv`.

        Ожидает колонки `id`, `label_raw`, строки сгруппированы и упорядочены по
        `id` (как их пишет `prepare_training_data.py`). Передавать нужно *всё*
        предложение целиком для каждого `id`, включая последнее слово -- иначе
        неверно посчитаются признаки позиции/контекста; исключение строки
        последнего слова из обучения/оценки -- ответственность вызывающего кода
        (модули предикторов), не этой функции.

        Args:
            df: Фрейм с колонками `id`, `label_raw`.

        Returns:
            `df` с добавленными колонками признаков; порядок строк сохраняется.
        """
        df = df.reset_index(drop=True)
        feat_rows: list = [None] * len(df)

        for _, group in df.groupby('id', sort=False):
            tokens = group.label_raw.tolist()
            feats = self.extract_sentence(tokens)
            for local_i, orig_i in enumerate(group.index):
                feat_rows[orig_i] = feats.iloc[local_i]

        feat_df = pd.DataFrame(feat_rows).reset_index(drop=True)
        return pd.concat([df, feat_df], axis=1)


# --- кеширование признаков ---------------------------------------------------------
# Извлечение POS-признаков по всему корпусу (~250к токенов, один вызов Natasha на
# предложение) занимает время. `build_features_cache.py` считает их один раз и
# пишет файл кеша; `pause_predictor_linear.py` / `pause_predictor_catboost.py`
# затем загружают этот кеш вместо пересчёта на каждом запуске.

FEATURE_CACHE_COLUMNS = ['id'] + ALL_FEATURES


def extract_and_cache_dataframe(
    extractor: FeatureExtractor, df: pd.DataFrame, cache_path: str
) -> pd.DataFrame:
    """Извлекает признаки для всего `df` и записывает их в `cache_path`.

    Args:
        extractor: Настроенный `FeatureExtractor`.
        df: Фрейм с колонками `id`, `label_raw`.
        cache_path: Путь для файла кеша.

    Returns:
        `df` с прикреплёнными колонками признаков (как в `extract_dataframe`) --
        можно использовать вместо неё, если результат нужно ещё и сохранить.
    """
    result = extractor.extract_dataframe(df)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    result[FEATURE_CACHE_COLUMNS].to_csv(cache_path, sep='|', index=False, quoting=csv.QUOTE_NONE)
    return result


def load_cached_features(df: pd.DataFrame, cache_path: str) -> Optional[pd.DataFrame]:
    """Загружает кеш признаков, если он есть и совпадает с `df` построчно (по `id`).

    Args:
        df: Текущий фрейм метаданных (без признаков).
        cache_path: Путь к файлу кеша.

    Returns:
        `df` с признаками из кеша, либо `None` (вместо исключения), если кеш
        отсутствует, устарел или посчитан для другого набора признаков
        (не хватает добавленной колонки или есть удалённая) -- тогда вызывающий
        код должен пересчитать признаки сам, а не молча использовать устаревшие.
    """
    if not os.path.exists(cache_path):
        return None

    df = df.reset_index(drop=True)
    cached = pd.read_csv(cache_path, sep='|', quoting=csv.QUOTE_NONE)

    if set(ALL_FEATURES) != set(cached.columns) - {'id'}:
        print(f'[features] Cache at {cache_path} has a different feature set than the '
              f'current code -- ignoring it, will recompute. Re-run build_features_cache.py '
              f'to refresh it.')
        return None

    if len(cached) != len(df) or not (cached.id.values == df.id.values).all():
        print(f'[features] Cache at {cache_path} does not match the current data '
              f'(different length or row order) -- ignoring it, will recompute.')
        return None

    return pd.concat([df, cached[ALL_FEATURES].reset_index(drop=True)], axis=1)
