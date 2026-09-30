"""Построение обучающих данных для предсказателя пауз.

Соединяет нормализованные метаданные (текст) с выравниванием MFA (TextGrid) и
записывает результат по одной строке на слово в `RUSLAN_pause_metadata.csv`::

    id|label|label_raw|duration|is_last_word|is_pause_after|pause_duration|set

Записи, чей id делится на 5 (оканчивается на 0 или 5), уходят в `test`, остальные -- в `train`.

Запуск из директории лабы::

    uv run python prepare_training_data.py
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from praatio import textgrid

from features import punct_class, split_gap
from paths import PAUSE_METADATA_PATH, RUSLAN_ALIGN_DIR, RUSLAN_METADATA_PATH

DOWN_NOISE_PAUSE_THRESHOLD_MS = 100
UPPER_NOISE_PAUSE_THRESHOLD_MS = 400


def read_word_intervals(ruslan: pd.DataFrame, align_root: Path) -> pd.DataFrame:
    """Интервалы слов из MFA TextGrid каждого файла (`label`, `duration`, `id`; тишина -- пустой `label`).

    Файлы без TextGrid пропускаются. Ярус фонем (второй) понадобится в лабе 3.
    """
    docs = []
    missing = 0
    for wave_id in tqdm.tqdm(ruslan.id):
        path = Path(align_root) / f'{wave_id}.TextGrid'
        if not path.exists():
            missing += 1
            continue
        words = textgrid.openTextgrid(str(path), True).tiers[0]
        docs += [{'label': w.label, 'duration': w.end - w.start, 'id': wave_id} for w in words.entries]
    if missing:
        print(f'Нет TextGrid для {missing} файлов')
    return pd.DataFrame(docs)


def align_text_and_textgrid(tokens: pd.DataFrame, text: str) -> pd.DataFrame:
    """Восстанавливает исходный текст (регистр и пунктуацию) для каждого слова.

    MFA-метки в нижнем регистре и без пунктуации. `label_raw` возвращает регистр и
    добавляет пунктуацию, идущую после слова; текст до первого слова уходит в первое
    слово. Интервалы тишины получают `"<SIL>"`.

    Returns:
        `tokens` со столбцом `label_raw`; если слово не найдено в `text`, `tokens`
        возвращается без него (такие файлы отбрасываются позже).
    """
    raw_tokens = []
    text_lower = text.lower()
    previous_word = -1
    for label, wave_id in tokens[['label', 'id']].values:
        if label == '':
            raw_tokens.append('<SIL>')
            continue
        splits = text_lower.split(label, maxsplit=1)
        if len(splits) == 1:
            print(f'Ошибка выравнивания {wave_id}: слово "{label}" не найдено в тексте "{text}"')
            return tokens
        gap_len = len(splits[0])
        gap, word = text[:gap_len], text[gap_len:gap_len + len(label)]
        if previous_word == -1:
            raw_tokens.append((gap + word).strip())
        else:
            # Открывающие «( уходят в префикс следующего слова, остальное -- хвост предыдущего.
            tail, prefix = split_gap(gap)
            raw_tokens[previous_word] += tail
            raw_tokens.append(prefix + word)
        text_lower = splits[1]
        text = text[gap_len + len(label):]
        previous_word = len(raw_tokens) - 1
    if text and previous_word >= 0:
        raw_tokens[previous_word] += text.strip()
    tokens['label_raw'] = raw_tokens
    return tokens


def add_pause_labels(align: pd.DataFrame) -> pd.DataFrame:
    """Добавляет `is_last_word`, `is_pause_after` и `pause_duration` (секунды) для слов одного файла.

    Технический мусор на границах файла паузой не считается: тишина ДО первого
    слова никому не приписывается, тишина ПОСЛЕ последнего слова (хвост записи)
    не приписывается последнему слову.
    """
    is_word = (align.label != '').to_numpy()
    pos = np.arange(len(align))

    # пауза после слова -- следующий за ним интервал тишины, если он не хвост записи
    next_is_pause = np.zeros(len(align), bool)
    next_is_pause[:-1] = ~is_word[1:] & (pos[:-1] + 1 != len(align) - 1)
    pause = is_word & next_is_pause

    align['is_pause_after'] = pause.astype(int)
    align['pause_duration'] = np.where(pause, align.duration.shift(-1, fill_value=0.), 0.)
    align['is_last_word'] = (pos == (pos[is_word].max() if is_word.any() else -1)).astype(int)
    return align


def zero_out_noise_pauses(
    pause_df: pd.DataFrame,
    down_threshold_ms: float = DOWN_NOISE_PAUSE_THRESHOLD_MS,
    upper_threshold_ms: float = UPPER_NOISE_PAUSE_THRESHOLD_MS,
) -> pd.DataFrame:
    """Зануляет слишком короткие/длинные паузы класса `none` -- шум выравнивания MFA.

    Слово остаётся в датасете, зануляется только факт и длительность паузы после
    него. Паузы на месте пунктуации считаются осмысленными при любой длительности.
    """
    ms = pause_df.pause_duration * 1000
    noise_mask = (
        (pause_df.label_raw.apply(punct_class) == 'none')
        & (ms > 0)
        & ((ms < down_threshold_ms) | (ms > upper_threshold_ms))
    )
    print(f'Зануляю {noise_mask.sum()} шумных пауз '
          f'(<{down_threshold_ms:.0f} и >{upper_threshold_ms:.0f}мс, punct_class=none)')
    pause_df = pause_df.copy()
    pause_df.loc[noise_mask, ['is_pause_after', 'pause_duration']] = 0
    return pause_df


def main() -> None:
    ruslan = pd.read_csv(RUSLAN_METADATA_PATH, sep='|', names=['id', 'raw', 'nrm'], quoting=csv.QUOTE_NONE)
    word_df = read_word_intervals(ruslan, RUSLAN_ALIGN_DIR)

    # приводим разные дефисы и апострофы к одному виду, скобочные вставки -- к заглушке
    word_df.label = word_df.label.str.replace('[‐‑]', '-', regex=True)
    ruslan.nrm = (
        ruslan.nrm.str.replace('[‐‑]', '-', regex=True)
        .str.replace('’', "'")
        .str.replace(r'\(.*?\)|<.*?>', '[bracketed]', regex=True)
    )

    words_by_id = dict(list(word_df.groupby('id')))
    aligns = []
    for wave_id, text in tqdm.tqdm(ruslan[['id', 'nrm']].values):
        if wave_id in words_by_id:
            aligns.append(add_pause_labels(align_text_and_textgrid(words_by_id[wave_id].copy(), text)))

    pause_df = pd.concat(aligns)
    pause_df = pause_df[pause_df.label_raw.notna() & (pause_df.label_raw != '<SIL>')].reset_index(drop=True)
    pause_df = zero_out_noise_pauses(pause_df)

    is_test = pause_df.id.str.split('_').str[0].astype(int) % 5 == 0
    pause_df['set'] = np.where(is_test, 'test', 'train')
    pause_df.to_csv(PAUSE_METADATA_PATH, sep='|', index=False, header=True, quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    main()
