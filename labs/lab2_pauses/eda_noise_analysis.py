"""Анализ шума разметки MFA в паузах класса `none` — lab 2, доп. EDA.

Строит графики:
  1. Гистограмма длительностей пауз (none, 0-500мс) — поиск пика тех. шума
  2. Доля пауз по последней букве предшествующего слова — стыки звуков
  3. Тепловая карта корреляций топ-30 слов / POS с фактом паузы
  4. Вероятность паузы в зависимости от длины безпунктуационной синтагмы
  5. Доля пауз до/после части речи
  6. Длительность паузы по классам пунктуации (boxplot)

Запуск::

    python eda_noise_analysis.py
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from features import FeatureExtractor, NO_PUNCT, clean_letters, punct_class

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'


def load_data() -> pd.DataFrame:
    """Загружает разметку пауз и добавляет колонку `punct_class`.

    Returns:
        Таблица без последнего слова каждого предложения (пауза на стыке
        файлов не релевантна для анализа).
    """
    df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)
    df = df[df.is_last_word == 0].copy()
    df['punct_class'] = df.label_raw.apply(punct_class)
    return df


# --- 1. Гистограмма длительностей пауз класса `none` (поиск пика шума) ----------

def plot_none_pause_duration_hist(df: pd.DataFrame, ax=None) -> pd.DataFrame:
    """Строится на строках `punct_class=='none'` с `pause_duration > 0`.

    Ожидаемый результат: аномальный пик в районе 0-50мс -- признак того, что MFA
    иногда вставляет короткий "силентный" интервал на стыке звуков/слов, который
    физически не является паузой диктора, а является артефактом выравнивания.
    Такие точки -- кандидаты на переразметку `is_pause_after=0`.

    Args:
        df: Таблица с колонкой `punct_class`.
        ax: Ось matplotlib для отрисовки; если None, берётся текущая.

    Returns:
        Подвыборка `none`-пауз, по которой строился график.
    """
    sub = df[(df.punct_class == NO_PUNCT) & (df.pause_duration > 0)]
    durations_ms = sub.pause_duration.values * 1000

    ax = ax or plt.gca()
    ax.hist(durations_ms, bins=50, range=(0, 500), color='steelblue', edgecolor='black')
    ax.axvline(50, color='red', linestyle='--', label='Порог шума (50мс)')
    ax.set_xlabel('Длительность паузы, мс')
    ax.set_ylabel('Количество пауз')
    ax.set_title(f'Паузы без пунктуации (none), n={len(sub)}')
    ax.legend()
    return sub


# --- 2. Доля пауз по последней букве предшествующего слова ---------------------

def plot_pause_rate_by_last_letter(df: pd.DataFrame, ax=None, min_count: int = 30) -> pd.DataFrame:
    """Строится на всех строках класса `none` (пауза есть/нет).

    Берётся последняя кириллическая буква слова перед паузой (по `label`, а не
    `label_raw`, чтобы игнорировать пунктуацию/регистр) и считается
    `P(is_pause_after=1 | последняя буква = X)` по всем вхождениям этой буквы.

    Ожидаемый результат: повышенная доля пауз после глухих смычных/взрывных
    согласных (к, п, т, с) относительно гласных и звонких -- MFA хуже находит
    границу слова на таких стыках и "съедает" часть звука в паузу.

    Args:
        df: Таблица с колонкой `punct_class`.
        ax: Ось matplotlib для отрисовки.
        min_count: Минимальное число вхождений буквы, чтобы попасть в график.

    Returns:
        Таблица `pause_rate`/`count` по буквам, отсортированная по убыванию доли.
    """
    sub = df[df.punct_class == NO_PUNCT].copy()
    sub['last_letter'] = sub.label.apply(lambda w: clean_letters(w)[-1:] if clean_letters(w) else '')
    sub = sub[sub.last_letter != '']

    stats = sub.groupby('last_letter').agg(
        pause_rate=('is_pause_after', 'mean'),
        count=('is_pause_after', 'size'),
    )
    stats = stats[stats['count'] >= min_count].sort_values('pause_rate', ascending=False)

    ax = ax or plt.gca()
    colors = ['crimson' if l in 'кпстфхцчшщ' else 'steelblue' for l in stats.index]
    ax.bar(stats.index, stats.pause_rate * 100, color=colors)
    ax.set_xlabel('Последняя буква слова перед паузой')
    ax.set_ylabel('% случаев паузы')
    ax.set_title('Доля пауз (none) по последней букве слова\n(красным -- глухие/взрывные согласные)')
    ax.tick_params(axis='x', rotation=0)
    return stats


# --- 3. Тепловая карта корреляций топ-слов / POS с паузой -----------------------

def plot_correlation_heatmap(df: pd.DataFrame, ax=None, top_n: int = 30, use_pos: bool = True) -> pd.DataFrame:
    """Строится на строках класса `none`.

    One-hot по `top_n` самым частым словам (`label`) + категориальные
    `pos_curr`, `pos_next1` (тоже one-hot) против бинарного `is_pause_after`.
    Считается корреляция Пирсона каждого признака с фактом паузы.

    Ожидаемый результат: отдельные слова/POS с заметно повышенной корреляцией --
    кандидаты на "пропущенную пунктуацию" в исходном тексте (например, вводные
    слова, союзы, после которых автор текста не поставил запятую, но диктор
    интонационно паузу сделал) или устойчивые синтаксические границы (конец
    именной группы перед глаголом и т.п.).

    Args:
        df: Таблица с колонкой `punct_class`.
        ax: Ось matplotlib (для heatmap).
        top_n: Сколько самых частых слов брать в one-hot.
        use_pos: Добавлять ли one-hot по `pos_curr`/`pos_next1` (требует
            фичи Natasha -- считаются на лету, если их ещё нет).

    Returns:
        Таблица корреляций с `is_pause_after`, отсортированная по модулю.
    """
    sub = df[df.punct_class == NO_PUNCT].copy()

    if use_pos:
        if not {'pos_curr', 'pos_next1'}.issubset(sub.columns):
            sub = FeatureExtractor(use_pos=True).extract_dataframe(sub)

    top_words = sub.label.value_counts().head(top_n).index
    word_ohe = pd.get_dummies(sub.label.where(sub.label.isin(top_words)), prefix='word')

    frames = [word_ohe]
    if use_pos:
        frames.append(pd.get_dummies(sub.pos_curr, prefix='pos_curr'))
        frames.append(pd.get_dummies(sub.pos_next1, prefix='pos_next1'))

    feat_matrix = pd.concat(frames, axis=1)
    feat_matrix['is_pause_after'] = sub.is_pause_after.values

    corr = feat_matrix.corr()[['is_pause_after']].drop('is_pause_after')
    corr = corr.reindex(corr.is_pause_after.abs().sort_values(ascending=False).index)

    ax = ax or plt.gca()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Корреляция Пирсона'})
    ax.set_title(f'Корреляция топ-{top_n} слов и POS с фактом паузы (none)')
    ax.set_xlabel('')
    return corr


# --- 4. Вероятность паузы от длины синтагмы (tokens_since_punct) ---------------

def plot_pause_prob_by_syntagma_length(df: pd.DataFrame, ax=None, max_len: int = 25) -> pd.DataFrame:
    """Строится на строках класса `none` по признаку `tokens_since_punct`.

    Для каждого значения "сколько токенов прошло с последней пунктуации"
    считается доля строк с `is_pause_after=1`.

    Ожидаемый результат: рост вероятности паузы с ростом длины синтагмы --
    физиологическая потребность диктора сделать вдох, если фраза без знаков
    препинания слишком длинная. Подтверждает, что часть пауз в `none` -- не
    шум, а системный эффект длины фразы (в отличие от графика 1, где шум --
    случайный технический артефакт).

    Args:
        df: Таблица с колонкой `punct_class`.
        ax: Ось matplotlib для отрисовки (создаётся вторая ось для n).
        max_len: Максимальная длина синтагмы, включаемая в график.

    Returns:
        Таблица `pause_rate`/`count` по длине синтагмы.
    """
    sub = df[df.punct_class == NO_PUNCT].copy()
    if 'tokens_since_punct' not in sub.columns:
        sub = FeatureExtractor(use_pos=False).extract_dataframe(sub)

    sub = sub[sub.tokens_since_punct <= max_len]
    stats = sub.groupby('tokens_since_punct').agg(
        pause_rate=('is_pause_after', 'mean'),
        count=('is_pause_after', 'size'),
    )

    ax = ax or plt.gca()
    ax.bar(stats.index, stats.pause_rate, color='seagreen')
    ax2 = ax.twinx()
    ax2.plot(stats.index, stats['count'], color='gray', alpha=0.5, marker='o', markersize=3, label='Кол-во токенов (n)')
    ax2.set_ylabel('n (серая линия)')

    ax.set_xlabel('Токенов с последней пунктуации (tokens_since_punct)')
    ax.set_ylabel('Доля пауз (P(is_pause_after=1))')
    ax.set_title('Вероятность паузы vs длина синтагмы без пунктуации')
    return stats


def plot_pause_rate_by_pos(df: pd.DataFrame, ax=None, min_count: int = 30, position: str = 'after') -> pd.DataFrame:
    """Доля пауз в зависимости от части речи, до или после которой стоит пауза.

    `position='after'` -- POS текущего токена (`pos_curr`): "после какой части
    речи чаще ставится пауза" -- например, ожидаемо высокая доля после NOUN
    (граница именной группы) или деепричастий.
    `position='before'` -- POS следующего токена (`pos_next1`, т.е. POS слова,
    что идёт сразу после паузы): "перед какой частью речи чаще ставится пауза" --
    например, повышенная доля перед союзами (CCONJ) как маркер границы клауз.

    Строится только на строках `punct_class == 'none'` -- осмысленные знаки
    препинания и так почти всегда сопровождаются паузой, интересен именно
    безпунктуационный случай.

    Args:
        df: Таблица с колонкой `punct_class`.
        ax: Ось matplotlib для отрисовки.
        min_count: Минимальное число вхождений POS, чтобы попасть в график.
        position: `'after'` или `'before'`.

    Returns:
        Таблица `pause_rate`/`count` по значению POS.
    """
    assert position in ('after', 'before')
    sub = df[df.punct_class == NO_PUNCT].copy()
    if not {'pos_curr', 'pos_next1'}.issubset(sub.columns):
        sub = FeatureExtractor(use_pos=True).extract_dataframe(sub)

    pos_col = 'pos_curr' if position == 'after' else 'pos_next1'
    stats = sub.groupby(pos_col).agg(
        pause_rate=('is_pause_after', 'mean'),
        count=('is_pause_after', 'size'),
    )
    stats = stats[stats['count'] >= min_count].sort_values('pause_rate', ascending=False)

    ax = ax or plt.gca()
    ax.bar(stats.index, stats.pause_rate * 100, color='darkorange' if position == 'after' else 'teal')
    ax.set_xlabel(f'{"Текущая" if position == "after" else "Следующая"} часть речи ({pos_col})')
    ax.set_ylabel('% случаев паузы')
    title = 'после' if position == 'after' else 'перед'
    ax.set_title(f'Доля пауз (none) {title} частью речи')
    ax.tick_params(axis='x', rotation=45)
    return stats


# --- 6. Длительность паузы по классам пунктуации (boxplot) ---------------------

def plot_pause_duration_by_punct_class(df: pd.DataFrame, ax=None, min_count: int = 5) -> pd.DataFrame:
    """Boxplot длительности паузы по классам пунктуации, для всех строк с паузой.

    В отличие от графика 1 (гистограмма только по `none` в узком диапазоне
    0-500мс, для поиска шума), здесь сравнивается распределение длительности
    паузы между всеми классами пунктуации -- ожидаемо: `none` и запятая дают
    короткие паузы, точка/многоточие/`!`/`?` -- заметно более длинные (конец
    высказывания, вдох перед следующей фразой).

    Args:
        df: Таблица с колонками `punct_class`, `pause_duration`.
        ax: Ось matplotlib для отрисовки.
        min_count: Минимальное число пауз в классе, чтобы попасть в график.

    Returns:
        Таблица со средней/медианной длительностью и числом пауз по классам.
    """
    sub = df[df.pause_duration > 0].copy()
    sub['pause_duration_ms'] = sub.pause_duration * 1000

    counts = sub.punct_class.value_counts()
    classes = counts[counts >= min_count].sort_values(ascending=False).index.tolist()
    sub = sub[sub.punct_class.isin(classes)]

    order = sub.groupby('punct_class').pause_duration_ms.median().sort_values(ascending=False).index

    ax = ax or plt.gca()
    sns.boxplot(data=sub, x='punct_class', y='pause_duration_ms', order=order, ax=ax, showfliers=False)
    ax.set_xlabel('Класс пунктуации')
    ax.set_ylabel('Длительность паузы, мс')
    ax.set_title('Длительность паузы по классам пунктуации')
    ax.tick_params(axis='x', rotation=30)

    stats = sub.groupby('punct_class').pause_duration_ms.agg(['mean', 'median', 'count']).reindex(order)
    return stats


def main():
    df = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    print('--- 1. Гистограмма длительностей ---')
    hist_sub = plot_none_pause_duration_hist(df, ax=axes[0, 0])
    noise_share = (hist_sub.pause_duration * 1000 < 50).mean()
    print(f'Доля пауз < 50мс среди none-пауз: {noise_share:.1%}')

    print('\n--- 2. Доля пауз по последней букве ---')
    letter_stats = plot_pause_rate_by_last_letter(df, ax=axes[0, 1])
    print(letter_stats.head(10))

    print('\n--- 3. Корреляция топ-слов/POS с паузой ---')
    corr = plot_correlation_heatmap(df, ax=axes[1, 0])
    print(corr.head(10))

    print('\n--- 4. Вероятность паузы vs длина синтагмы ---')
    syntagma_stats = plot_pause_prob_by_syntagma_length(df, ax=axes[1, 1])
    print(syntagma_stats.head(10))

    plt.tight_layout()
    plt.savefig('data/noise_analysis.png', dpi=150)
    print('\nСохранено в data/noise_analysis.png')
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
    print('\n--- 5а. Доля пауз ПОСЛЕ части речи ---')
    pos_after_stats = plot_pause_rate_by_pos(df, ax=axes2[0], position='after')
    print(pos_after_stats)

    print('\n--- 5б. Доля пауз ПЕРЕД частью речи ---')
    pos_before_stats = plot_pause_rate_by_pos(df, ax=axes2[1], position='before')
    print(pos_before_stats)

    plt.tight_layout()
    plt.savefig('data/pos_before_after.png', dpi=150)
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    print('\n--- 6. Длительность паузы по классам пунктуации ---')
    duration_stats = plot_pause_duration_by_punct_class(df, ax=ax3)
    print(duration_stats)

    plt.tight_layout()
    plt.savefig('data/pause_duration_by_class.png', dpi=150)
    print('Сохранено в data/pause_duration_by_class.png')
    plt.show()


if __name__ == '__main__':
    main()
