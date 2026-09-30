"""Анализ качества обученного предиктора пауз (`PausePredictorCatboost`).

Разделы (графики выводятся на экран, таблицы -- в консоль):
    1. Train vs test -- сигнал переобучения.
    2. Важность признаков классификатора и регрессора.
    3. Ошибки классификации по классу пунктуации и позиции в предложении.
    4. Худшие по MAE предсказания длительности.
    5. HTML-страница "текст + аудио" для проверки на слух -- не технический ли это
       шум (монтажный стык, обрезанное слово), прежде чем придумывать новые признаки.

Запуск из директории лабы::

    uv run python analyze_results.py
"""
import html
import webbrowser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error

from features import load_features
from pause_predictor import calc_metrics
from pause_predictor_catboost import PausePredictorCatboost, predict_on
from paths import AUDIO_REVIEW_HTML, RUSLAN_AUDIO_DIR

N_AUDIO_EXAMPLES = 30  # сколько случайных FN/FP показать на странице
OVERFIT_F1_GAP_THRESHOLD = 0.05
OVERFIT_MAE_GAP_THRESHOLD_S = 0.02


# --- 1. Train vs test: сигнал переобучения ----------------------------------------

def section_overfit_check(train_eval: pd.DataFrame, test_eval: pd.DataFrame) -> None:
    print('\n=== 1. Train vs Test (переобучение) ===')
    rows = {}
    for name, part in [('train', train_eval), ('test', test_eval)]:
        tp = part[(part.is_pause_after == 1) & (part.is_pause_hat == 1)]
        rows[name] = {
            'f1': f1_score(part.is_pause_after, part.is_pause_hat),
            'mae_ms': mean_absolute_error(tp.pause_duration, tp.pause_duration_hat) * 1000,
        }
    summary = pd.DataFrame(rows).T
    print(summary)

    gap_f1 = summary.loc['train', 'f1'] - summary.loc['test', 'f1']
    gap_mae_s = (summary.loc['test', 'mae_ms'] - summary.loc['train', 'mae_ms']) / 1000
    print(f'Разрыв F1 (train-test): {gap_f1:.3f}; разрыв MAE (test-train): {gap_mae_s * 1000:.1f}мс')
    if gap_f1 > OVERFIT_F1_GAP_THRESHOLD or gap_mae_s > OVERFIT_MAE_GAP_THRESHOLD_S:
        print('Заметный разрыв train/test -- признак переобучения.')
    else:
        print('Разрыв train/test невелик -- модель не выглядит переобученной.')

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    summary.f1.plot(kind='bar', ax=axes[0], rot=0, title='F1 (is_pause_after)', ylim=(0, 1))
    summary.mae_ms.plot(kind='bar', ax=axes[1], rot=0, title='MAE длительности паузы, мс')
    plt.tight_layout()
    plt.show()


# --- 2. Важность признаков ----------------------------------------------------------

def section_feature_importance(pp: PausePredictorCatboost, top_n: int = 15) -> None:
    print('\n=== 2. Важность признаков ===')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, model) in zip(axes, [('Классификатор (is_pause_after)', pp.clf),
                                         ('Регрессор (pause_duration)', pp.reg)]):
        importance = model.get_feature_importance(prettified=True)
        print(f'\n{title}:\n', importance)
        top = importance.head(top_n).iloc[::-1]
        ax.barh(top['Feature Id'], top['Importances'])
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


# --- 3. Ошибки классификации --------------------------------------------------------

def section_error_breakdown(test_eval: pd.DataFrame) -> None:
    print('\n=== 3. Ошибки классификации: пунктуация и позиция в предложении ===')
    test_eval = test_eval.copy()
    test_eval['error_type'] = np.select(
        [
            (test_eval.is_pause_after == 1) & (test_eval.is_pause_hat == 0),
            (test_eval.is_pause_after == 0) & (test_eval.is_pause_hat == 1),
        ],
        ['false_negative', 'false_positive'],
        default='correct',
    )
    test_eval['rel_pos_bin'] = pd.cut(test_eval.rel_pos, bins=10, labels=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (key, title) in zip(axes, [
        ('punct_class', 'по классу пунктуации'),
        ('rel_pos_bin', 'по позиции в предложении (0 = начало, 9 = конец)'),
    ]):
        counts = test_eval.groupby([key, 'error_type']).size().unstack(fill_value=0)
        print(f'\nОшибки {title}:\n', counts)
        counts[[c for c in ('false_negative', 'false_positive') if c in counts.columns]].plot(
            kind='bar', stacked=True, ax=ax, title=f'Ошибки {title} (test)')
        ax.set_ylabel('количество токенов')
    plt.tight_layout()
    plt.show()


# --- 4. Худшие по MAE предсказания длительности -------------------------------------

def section_worst_duration_errors(test_eval: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    print(f'\n=== 4. Топ-{top_n} худших предсказаний длительности (test, TP) ===')
    tp = test_eval[(test_eval.is_pause_after == 1) & (test_eval.is_pause_hat == 1)].copy()
    tp['abs_error_ms'] = (tp.pause_duration - tp.pause_duration_hat).abs() * 1000
    worst = tp.sort_values('abs_error_ms', ascending=False).head(top_n)
    print(worst[['id', 'label_raw', 'punct_class', 'pause_duration',
                 'pause_duration_hat', 'abs_error_ms']].to_string(index=False))
    return worst


# --- 5. Аудио-обзор ошибок -----------------------------------------------------------

def _render_error_cards(df: pd.DataFrame, detail_fn) -> str:
    """Плашка на каждую строку: id, токен, класс пунктуации, пояснение и плеер.

    Строки без .wav файла на диске пропускаются.
    """
    cards = []
    for _, row in df.iterrows():
        audio_path = RUSLAN_AUDIO_DIR / f'{row.id}.wav'
        if not audio_path.exists():
            continue
        cards.append(f'''
        <div class="card">
          <div class="card-header">{row.id} — <code>{html.escape(row.label_raw)}</code> ({row.punct_class})</div>
          <div class="card-body">{detail_fn(row)}</div>
          <audio controls preload="none" src="{audio_path.resolve().as_uri()}"></audio>
        </div>''')
    return '\n'.join(cards) if cards else '<p><em>Нет доступных .wav файлов для этой группы.</em></p>'


def section_audio_review(worst_duration: pd.DataFrame, test_eval: pd.DataFrame) -> None:
    """Собирает HTML-страницу "текст + аудио" по худшим примерам и случайным FN/FP и открывает её в браузере."""
    print('\n=== 5. Аудио-обзор ошибок ===')
    if not RUSLAN_AUDIO_DIR.is_dir():
        print(f'Директория с аудио {RUSLAN_AUDIO_DIR} не найдена -- пропускаю аудио-обзор.')
        return

    fn_pool = test_eval[(test_eval.is_pause_after == 1) & (test_eval.is_pause_hat == 0)]
    fp_pool = test_eval[(test_eval.is_pause_after == 0) & (test_eval.is_pause_hat == 1)]
    fn_examples = fn_pool.sample(min(N_AUDIO_EXAMPLES, len(fn_pool)), random_state=42)
    fp_examples = fp_pool.sample(min(N_AUDIO_EXAMPLES, len(fp_pool)), random_state=42)

    duration_cards = _render_error_cards(
        worst_duration,
        lambda r: f'Факт: {r.pause_duration * 1000:.0f}мс, предсказано: {r.pause_duration_hat * 1000:.0f}мс',
    )
    fn_cards = _render_error_cards(
        fn_examples,
        lambda r: f'Пауза в разметке есть ({r.pause_duration * 1000:.0f}мс), модель её не поставила',
    )
    fp_cards = _render_error_cards(fp_examples, lambda r: 'Модель поставила паузу, в разметке её нет')

    page = f'''<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Аудио-обзор ошибок предиктора пауз</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }}
  h2 {{ margin-top: 32px; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; }}
  .card-header {{ font-weight: 600; margin-bottom: 4px; }}
  .card-body {{ color: #555; margin-bottom: 8px; font-size: 14px; }}
  audio {{ width: 100%; }}
</style></head>
<body>
<h1>Аудио-обзор ошибок предиктора пауз</h1>
<p>Текст токена и аудио файла целиком -- чтобы на слух проверить, не технический ли это шум
(монтажный стык, обрезанное слово), прежде чем придумывать новые признаки.</p>
<h2>Худшие по MAE (test)</h2>
{duration_cards}
<h2>Пропущенные паузы -- false negative (случайные {len(fn_examples)} из {len(fn_pool)})</h2>
{fn_cards}
<h2>Лишние паузы -- false positive (случайные {len(fp_examples)} из {len(fp_pool)})</h2>
{fp_cards}
</body></html>'''

    AUDIO_REVIEW_HTML.parent.mkdir(parents=True, exist_ok=True)
    AUDIO_REVIEW_HTML.write_text(page, encoding='utf-8')
    print(f'Сохранено: {AUDIO_REVIEW_HTML}')
    webbrowser.open(AUDIO_REVIEW_HTML.resolve().as_uri())


def main() -> None:
    df = load_features()
    train_df = df[df.set == 'train']
    test_df = df[df.set == 'test']

    pp = PausePredictorCatboost().fit(train_df)
    train_eval = predict_on(pp, train_df)
    test_eval = predict_on(pp, test_df)

    print('\n-- train --')
    calc_metrics(train_eval)
    print('\n-- test --')
    calc_metrics(test_eval)

    section_overfit_check(train_eval, test_eval)
    section_feature_importance(pp)
    section_error_breakdown(test_eval)
    worst_duration = section_worst_duration_errors(test_eval)
    section_audio_review(worst_duration, test_eval)


if __name__ == '__main__':
    main()
