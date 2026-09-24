"""Построение обучающих данных для предсказателя пауз — lab 2.

Соединяет нормализованные метаданные (lab 1) с выравниванием MFA (TextGrid) и
записывает результат по одной строке на слово в ``data/RUSLAN_pause_metadata.csv``::

    id|label|label_raw|duration|is_last_word|is_pause_after|pause_duration|set

Записи, чей id оканчивается на 0 или 5, уходят в ``test``, остальные — в ``train``.

Запуск из директории лабы::

    python prepare_training_data.py
"""
import csv
import os

import pandas as pd
import tqdm
from praatio import textgrid

from features import punct_class as _punct_class

RUSLAN_META = '../../data/metadata_RUSLAN_22200_normalized.csv'
ALIGN_DIR = '../../data/RUSLAN_align/v2'
RESULT_PATH = 'data/RUSLAN_pause_metadata.csv'

NOISE_PAUSE_THRESHOLD_MS = 100


def read_text_grids(ruslan: pd.DataFrame, align_root: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Читает MFA TextGrid каждого файла из `ruslan`.

    Args:
        ruslan: Метаданные со столбцами `id` и `nrm`.
        align_root: Директория с файлами `{id}.TextGrid`.

    Returns:
        Интервалы слов (`label`, `duration`, `id`; тишина — пустая строка),
        интервалы фонем (та же схема; тишина — `"<SIL>"`), и список строк
        фонемной последовательности (пробел между фонемами), по одной на
        строку метаданных (`""`, если TextGrid не найден).
    """
    word_docs = []
    phn_docs = []
    phoneme_sequences = []
    for wave_id, text in tqdm.tqdm(ruslan[['id', 'nrm']].values):
        try:
            tg = textgrid.openTextgrid(os.path.join(align_root, wave_id + '.TextGrid'), True)
        except:
            phoneme_sequences.append('')
            continue
        words, phones = tg.tiers

        words = words.entries
        for i in words:
            word_docs.append({'label': i.label, 'duration': i.end - i.start, 'id': wave_id})

        phoneme_sequences.append(' '.join([p.label for p in phones.entries]))
        phones = phones.entries
        for i in phones:
            if i.label == '':
                phn_docs.append({'label': '<SIL>', 'duration': i.end - i.start, 'id': wave_id})
            else:
                phn_docs.append({'label': i.label, 'duration': i.end - i.start, 'id': wave_id})
    word_df = pd.DataFrame(word_docs)
    phn_df = pd.DataFrame(phn_docs)
    return word_df, phn_df, phoneme_sequences


def align_text_and_textgrid(tokens: pd.DataFrame, text: str) -> pd.DataFrame:
    """Восстанавливает исходный текст (с пунктуацией и регистром) для каждого слова.

    MFA-метки в нижнем регистре и без пунктуации. `label_raw` возвращает регистр и
    добавляет пунктуацию, идущую после слова; текст до первого слова уходит в первое
    слово. Интервалы тишины получают `"<SIL>"`.

    Args:
        tokens: Интервалы слов одного файла (как из `read_text_grids`).
        text: Нормализованный текст того же файла.

    Returns:
        `tokens` со столбцом `label_raw`, либо `tokens` без изменений, если слово
        не найдено в `text` (такие файлы отбрасываются позже).
    """
    raw_tokens = []
    text_lower = text.lower()
    previous_word = -1
    for t, d, i in tokens[['label', 'duration', 'id']].values:
        if t == '':  # пустой токен -- тишина
            raw_tokens.append('<SIL>')
            continue
        splits = text_lower.split(t, maxsplit=1)
        if len(splits) == 1:  # слово не найдено в тексте
            print(f'Error aligning f{i}!')
            print(t, text, tokens)
            return tokens
        if previous_word == -1:
            raw_tokens.append(text[:len(splits[0] + t)].strip())
        else:
            raw_tokens[previous_word] += text[:len(splits[0])].strip()
            raw_tokens.append(text[len(splits[0]):len(splits[0] + t)])
        text_lower = splits[1]
        text = text[len(splits[0] + t):]
        previous_word = len(raw_tokens) - 1
    if len(text) and (previous_word >= 0):
        raw_tokens[previous_word] += text.strip()
    tokens['label_raw'] = raw_tokens
    return tokens


def add_pause_labels(align: pd.DataFrame) -> pd.DataFrame:
    """Помечает, после каких слов следует пауза, и её длительность.

    Добавляет `is_last_word`, `is_pause_after` и `pause_duration` (в секундах).

    Технический мусор на границах файла размечается как отсутствие паузы:
    тишина ДО первого слова никому не приписывается, тишина ПОСЛЕ последнего
    слова (хвост записи) явно не приписывается последнему слову.

    Args:
        align: Интервалы слов одного файла, по порядку.

    Returns:
        `align` с тремя добавленными столбцами.
    """
    n = len(align)
    is_last_word = []
    pause_after = []
    pause_duration = []
    last_word = -1
    for idx, (label, dur) in enumerate(align[['label', 'duration']].values):
        if label == '':
            is_trailing_silence = (idx == n - 1)  # хвост записи после последнего слова
            if last_word >= 0 and not is_trailing_silence:
                pause_after[last_word] = True
                pause_duration[last_word] = dur
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
        else:
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
            last_word = idx
    if last_word >= 0:
        is_last_word[last_word] = True
    align['is_last_word'] = is_last_word
    align['is_pause_after'] = pause_after
    align['pause_duration'] = pause_duration
    return align


def zero_out_noise_pauses(pause_df: pd.DataFrame, threshold_ms: float = NOISE_PAUSE_THRESHOLD_MS) -> pd.DataFrame:
    """Зануляет короткие паузы класса `none` — шум выравнивания MFA.

    Строка (слово) остаётся в датасете, зануляется только факт/длительность паузы
    после него. Затрагивает только `punct_class == 'none'` — паузы на месте
    пунктуации считаются осмысленными вне зависимости от длительности.

    Args:
        pause_df: Таблица пауз с колонками `label_raw`, `pause_duration`.
        threshold_ms: Порог в миллисекундах, ниже которого пауза класса `none`
            считается шумом.

    Returns:
        Копия `pause_df` с занулёнными шумными паузами.
    """
    pc = pause_df.label_raw.apply(_punct_class)
    noise_mask = (
        (pc == 'none')
        & (pause_df.pause_duration > 0)
        & (pause_df.pause_duration * 1000 < threshold_ms)
    )
    print(f'Зануляю {noise_mask.sum()} шумных пауз '
          f'(<{threshold_ms:.0f}мс, punct_class=none)')
    pause_df = pause_df.copy()
    pause_df.loc[noise_mask, 'is_pause_after'] = 0
    pause_df.loc[noise_mask, 'pause_duration'] = 0.0
    return pause_df


def main() -> None:
    """Читает метаданные и выравнивания, размечает паузы, делит на train/test и сохраняет."""
    ruslan = pd.read_csv(f'{RUSLAN_META}', sep='|', names=['id', 'raw', 'nrm'], quoting=csv.QUOTE_NONE)
    word_df, _, _ = read_text_grids(ruslan, ALIGN_DIR)

    word_df.label = word_df.label.str.replace('‐', '-')
    word_df.label = word_df.label.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‐', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('’', "'")
    ruslan.nrm = ruslan.nrm.str.replace('\\((.*?)\\)', '[bracketed]', regex=True)
    ruslan.nrm = ruslan.nrm.str.replace('\\<(.*?)\\>', '[bracketed]', regex=True)

    aligns = []
    for n, i in tqdm.tqdm(ruslan[['nrm', 'id']].values):
        tokens = word_df[word_df.id == i]
        aligns.append(align_text_and_textgrid(tokens, n))

    aligns = [add_pause_labels(a) for a in aligns]

    pause_df = pd.concat(aligns)
    pause_df = pause_df[pause_df.label_raw != '<SIL>']
    pause_df = pause_df[pause_df.label_raw.notna()]
    pause_df = pause_df.reset_index(drop=True)

    pause_df.is_last_word = pause_df.is_last_word.astype(int)
    pause_df.is_pause_after = pause_df.is_pause_after.astype(int)

    pause_df = zero_out_noise_pauses(pause_df)

    pause_df['set'] = 'train'
    pause_df.loc[pause_df.id.str.split('_', expand=True)[0].astype(int) % 5 == 0, 'set'] = 'test'
    pause_df.to_csv(f'{RESULT_PATH}', sep='|', index=False, header=True, quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    main()
