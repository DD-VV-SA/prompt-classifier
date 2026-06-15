# Malicious Prompt Classifier

Классификатор вредоносных текстовых запросов на основе анализа диалогового
контекста, разработанный в рамках курсовой работы по дисциплине
«Машинное обучение» (ОГУ, ИМИТ, кафедра математики и цифровых технологий).

> **Тема:** Исследование методов классификации вредоносных текстовых запросов
> на основе анализа диалогового контекста.
>
> **Автор:** Даваян Д. А., группа 25ИСТ(м)ИИП
> **Руководитель:** С. Т. Дусакаева, канд. пед. наук, доцент

## Постановка задачи

Бинарная классификация текстовых запросов (`benign` / `malicious`) к большим
языковым моделям с учётом диалогового контекста. Решение применимо как
компонент системы фильтрации запросов перед их подачей в LLM.

## Реализация

- **Векторизация:** мультиязычная Sentence-BERT модель
  `paraphrase-multilingual-MiniLM-L12-v2` (384-мерные эмбеддинги для prompt
  и context, конкатенация в 768-мерный вектор).
- **Классификаторы:** четыре алгоритмически различных модели —
  логистическая регрессия, LinearSVC с калибрацией Платта, градиентный
  бустинг CatBoost, стекинговый ансамбль.
- **Датасет:** [neuralchemy/Prompt-injection-dataset][ds]
  (конфигурация `core`, 6 274 размеченных записи, 31 категория атак,
  group-aware разбиение).

[ds]: https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset

## Основные результаты

| Модель           | F1-macro (test) | ROC-AUC | Δ train-val |
| ---------------- | --------------- | ------- | ----------- |
| LogReg           | 0,9431          | 0,9911  | 0,019       |
| LinearSVC (Platt)| **0,9527**      | 0,9919  | 0,033       |
| CatBoost         | 0,9430          | 0,9900  | 0,062       |
| Stacking         | 0,9453          | 0,9918  | 0,035       |

Лучшая модель — **LinearSVC с калибрацией Платта** на пространстве
`prompt + context` (F1-macro = 0,9527, ROC-AUC = 0,9919). Переобучения
не наблюдается ни у одной модели (Δ train-val ≤ 0,06).

## Установка

### Через `uv` (рекомендуется)

```bash
git clone https://github.com/davayan/malicious-prompt-classifier.git
cd malicious-prompt-classifier
uv sync
```

### Через `pip`

```bash
git clone https://github.com/davayan/malicious-prompt-classifier.git
cd malicious-prompt-classifier
python -m venv .venv
source .venv/bin/activate          # на Windows: .venv\Scripts\activate
pip install -e .
```

### Через conda / mamba

```bash
git clone https://github.com/davayan/malicious-prompt-classifier.git
cd malicious-prompt-classifier
conda env create -f environment.yml
conda activate malicious-prompt
```

## Загрузка датасета

Эксперимент использует три CSV-файла конфигурации `core` датасета
`neuralchemy/Prompt-injection-dataset`. Их можно подготовить одной командой:

```bash
python -c "
import pandas as pd
base = 'hf://datasets/neuralchemy/Prompt-injection-dataset/core'
for split in ['train', 'validation', 'test']:
    df = pd.read_parquet(f'{base}/{split}-00000-of-00001.parquet')
    df.to_csv(f'core_{split}.csv', index=False)
    print(f'core_{split}.csv: {len(df)} строк')
"
```

После выполнения в текущей директории появятся `core_train.csv` (4 391),
`core_validation.csv` (941), `core_test.csv` (942).

## Запуск эксперимента

```bash
python experiment_v4.py
```

Время работы — около 10–15 минут на стандартном CPU (без графического
ускорителя). Скрипт выводит подробный лог в консоль и сохраняет:

```
.
├── results_v4.json              # все метрики train / val / test
├── console_output_v4.txt        # полный лог запуска
└── figs_v4/                     # семь PNG-графиков
    ├── fig1_models_x_features.png   # сравнение моделей × пространство признаков
    ├── fig2_train_val_test.png      # метрики train / val / test
    ├── fig3_confusion_all.png       # матрицы ошибок всех 4 моделей
    ├── fig4_confusion_best.png      # матрица ошибок лучшей модели
    ├── fig5_distribution.png        # распределение классов и категорий
    ├── fig6_pipeline.png            # схема пайплайна
    └── fig7_attack_types.png        # точность по типам атак
```

## Архитектура пайплайна

```
        ┌─────────┐      ┌──────────────┐
        │ Prompt  │─────▶│ Sentence-BERT│──┐
        └─────────┘      │ (384 dim)    │  │
                         └──────────────┘  │   ┌────────────────┐
                                           ├──▶│ Конкатенация   │
        ┌─────────┐      ┌──────────────┐  │   │ (768 dim)      │
        │ Context │─────▶│ Sentence-BERT│──┘   └────────────────┘
        │ (2 реп.)│      │ (384 dim)    │              │
        └─────────┘      └──────────────┘              ▼
                                            ┌─────────────────────┐
                                            │  LogReg / LinearSVC │
                                            │  CatBoost / Stacking│
                                            └─────────────────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │ Soft Voting / Stack │
                                            │  malicious / benign │
                                            └─────────────────────┘
```

## Ключевые методологические особенности

1. **Group-aware разбиение** на train / validation / test обеспечивается
   самим датасетом (поле `group_id`). Скрипт явно проверяет отсутствие
   пересечений по `group_id` между подвыборками и завершается с ошибкой
   при их обнаружении.
2. **Формирование диалогового контекста внутри подвыборки.** Контекст
   моделируется из двух случайных реплик той же подвыборки (для train —
   только из train, для val — только из val, для test — только из test).
   Это исключает любые утечки информации между обучением, валидацией
   и тестом.
3. **Ablation-тест устойчивости к шуму.** Сравнение трёх пространств
   признаков (`prompt`, `context`, `prompt + context`) показывает,
   что модель сохраняет F1-macro ≈ 0,95 при том, что 71 % контекстов
   является нерелевантным (имеют смешанные или противоположные классы).

## Структура репозитория

```
.
├── experiment_v4.py     # основной скрипт эксперимента
├── pyproject.toml       # метаданные проекта и зависимости
├── environment.yml      # альтернативная установка через conda
├── README.md            # этот файл
├── .gitignore           # исключения git
└── LICENSE              # лицензия MIT
```

## Цитирование

При использовании результатов в научных работах ссылайтесь на курсовую работу:

> Даваян Д. А. Исследование методов классификации вредоносных текстовых
> запросов на основе анализа диалогового контекста: курсовая работа /
> Оренбург. гос. ун-т; рук. С. Т. Дусакаева. — Оренбург, 2026. — 33 с.

## Лицензия

[MIT](LICENSE)
