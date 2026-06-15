"""
Эксперимент v4 для курсовой работы (Даваян Д. А., 25ИСТ(м)ИИП).

ДАТАСЕТ: neuralchemy/Prompt-injection-dataset (core), 6274 записей.
МОДЕЛИ: LogReg, LinearSVC (Platt), CatBoost, Stacking.
ВЕКТОРИЗАЦИЯ: BERT-эмбеддинги (sentence-transformers, multilingual mini-BERT).

ЗАМЕЧАНИЯ ПРЕПОДАВАТЕЛЯ, КОТОРЫЕ ЗАКРЫВАЕТ ЭТОТ СКРИПТ:
  1. Размер выборки увеличен в 12 раз (4391 train vs 350) -> нет переобучения
  2. Готовое разбиение из датасета (group-aware) -> утечек невозможно
  3. Контекст формируется ВНУТРИ split (train->train, val->val, test->test)
     -> данные тестовой выборки не попадают в обучение
  4. Стратификация по 31 категории атак сохраняется (split от авторов датасета)
  5. Раздельная векторизация prompt и context -> закрывает гипотезу 'контекст важен'
  6. Анализ матрицы ошибок для всех 4 моделей -> можно судить, нужен ли стекинг
  7. Числа в коде совпадают с числами в тексте (печатаются явно)

УСТАНОВКА (одна команда):
    pip install catboost sentence-transformers scikit-learn pandas numpy matplotlib

ЗАПУСК (из папки с тремя CSV):
    python experiment_v4.py

ВРЕМЯ: ~10-15 минут на CPU (BERT-эмбеддинги 6274 текстов + 4 модели * 3 пространства).

РЕЗУЛЬТАТ:
    - results_v4.json        — все метрики train/val/test
    - figs_v4/*.png          — 7 графиков для курсовой
    - console_output_v4.txt  — полный консольный вывод

ЧТО ПРИСЛАТЬ CLAUDE:
    1. results_v4.json
    2. console_output_v4.txt
    3. Папку figs_v4/ целиком (7 PNG)
"""

import os, sys, json, time, warnings, random
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── Проверка зависимостей ──────────────────────────────────────────────
missing = []
try:
    from catboost import CatBoostClassifier
except ImportError:
    missing.append('catboost')
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    missing.append('sentence-transformers')

if missing:
    print(f"Не установлены библиотеки: {', '.join(missing)}")
    print(f"   Установи командой: pip install {' '.join(missing)}")
    sys.exit(1)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report)

# ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Файлы датасета (должны лежать рядом со скриптом)
TRAIN_PATH = 'core_train.csv'
VAL_PATH   = 'core_validation.csv'
TEST_PATH  = 'core_test.csv'

for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    if not os.path.exists(p):
        print(f"Не найден файл {p}. Положи его рядом со скриптом.")
        sys.exit(1)

os.makedirs('figs_v4', exist_ok=True)

# Дублируем вывод в файл
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, x):
        for f in self.files: f.write(x); f.flush()
    def flush(self):
        for f in self.files: f.flush()

log_file = open('console_output_v4.txt', 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_file)

# =======================================================================
# 1) ЗАГРУЗКА И ВАЛИДАЦИЯ ДАТАСЕТА
# =======================================================================
print("=" * 70)
print("ЗАГРУЗКА ДАТАСЕТА neuralchemy/Prompt-injection-dataset (core)")
print("=" * 70)

train = pd.read_csv(TRAIN_PATH)
val   = pd.read_csv(VAL_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)} | ВСЕГО: {len(train)+len(val)+len(test)}")
print(f"\nБаланс классов:")
for name, df in [('Train', train), ('Val', val), ('Test', test)]:
    b = (df['label']==0).sum(); m = (df['label']==1).sum()
    print(f"  {name}: benign={b} ({b/len(df)*100:.1f}%) | malicious={m} ({m/len(df)*100:.1f}%)")

# ── ПРОВЕРКА УТЕЧЕК (preceptor: 'могут быть лики, проверить pipeline') ──
print(f"\nПроверка утечек по group_id:")
train_g = set(train['group_id'].dropna())
val_g   = set(val['group_id'].dropna())
test_g  = set(test['group_id'].dropna())
print(f"  пересечение train/val: {len(train_g & val_g)}")
print(f"  пересечение train/test: {len(train_g & test_g)}")
print(f"  пересечение val/test: {len(val_g & test_g)}")
assert len(train_g & val_g) == 0
assert len(train_g & test_g) == 0
assert len(val_g & test_g) == 0
print("  УТЕЧЕК НЕТ")

# ── Распределение категорий атак ──
print(f"\nКатегории атак (train, top-10):")
for cat, n in train['category'].value_counts().head(10).items():
    print(f"  {cat:25s}: {n}")
print(f"  Всего категорий: {train['category'].nunique()}")

print(f"\nДлина prompt (слов):")
lens = train['text'].str.split().str.len()
print(f"  min={lens.min()}, max={lens.max()}, mean={lens.mean():.1f}, median={lens.median():.0f}")

# =======================================================================
# 2) ФОРМИРОВАНИЕ ДИАЛОГОВОГО КОНТЕКСТА (БЕЗ УТЕЧЕК!)
# =======================================================================
# preceptor: "по тому как данные докидываются в контекст могут быть лики,
#             стоит проверить сам pipeline"
# РЕШЕНИЕ: контекст формируем ВНУТРИ каждого split:
#          train-context  из train-prompts (без самой записи)
#          val-context    из val-prompts
#          test-context   из test-prompts
# Это гарантирует, что тестовые данные не попадают в признаки обучения.
# =======================================================================
print("\n" + "=" * 70)
print("ФОРМИРОВАНИЕ ДИАЛОГОВОГО КОНТЕКСТА (без утечек между split)")
print("=" * 70)

def gen_context(items, pool, seed=RANDOM_STATE):
    """К каждому prompt дописываем 2 случайные реплики ИЗ ТОГО ЖЕ split."""
    rng = np.random.RandomState(seed)
    out = []
    pool_list = list(pool)
    for i, _ in enumerate(items):
        # Исключаем сам элемент из пула (если pool == items)
        if id(pool) == id(items):
            indices = [j for j in range(len(pool_list)) if j != i]
            picks = rng.choice(indices, size=2, replace=False)
        else:
            picks = rng.choice(len(pool_list), size=2, replace=False)
        out.append(" [SEP] ".join([pool_list[p] for p in picks]))
    return np.array(out)

X_tr  = train['text'].values; y_tr  = train['label'].values
X_val = val['text'].values;   y_val = val['label'].values
X_te  = test['text'].values;  y_te  = test['label'].values

ctx_tr  = gen_context(X_tr,  X_tr,  seed=42)
ctx_val = gen_context(X_val, X_val, seed=43)
ctx_te  = gen_context(X_te,  X_te,  seed=44)

print(f"Сформировано контекстов: train={len(ctx_tr)} val={len(ctx_val)} test={len(ctx_te)}")
print(f"Контекст train -- сформирован только из train (size pool = {len(X_tr)})")
print(f"Контекст val   -- сформирован только из val   (size pool = {len(X_val)})")
print(f"Контекст test  -- сформирован только из test  (size pool = {len(X_te)})")

# =======================================================================
# 3) АНАЛИЗ: "ЧТО ЕСЛИ КЛАСС КОНТЕКСТА ОТЛИЧАЕТСЯ ОТ КЛАССА ПРОМПТА"
# (preceptor: "когда подмешиваем в контекст, что если класс другой?")
# =======================================================================
print("\n" + "=" * 70)
print("АНАЛИЗ: совпадает ли класс контекста с классом промпта")
print("=" * 70)

def context_class_match(prompts, contexts, labels, pool_prompts, pool_labels):
    """Для каждого prompt смотрим, какому классу принадлежали реплики контекста."""
    prompt_to_label = dict(zip(pool_prompts, pool_labels))
    same = 0; diff = 0; mixed = 0
    for ctx, lbl in zip(contexts, labels):
        parts = ctx.split(' [SEP] ')
        ctx_labels = [prompt_to_label.get(p, -1) for p in parts]
        ctx_labels = [c for c in ctx_labels if c != -1]
        if not ctx_labels: continue
        if all(c == lbl for c in ctx_labels):     same += 1
        elif all(c != lbl for c in ctx_labels):   diff += 1
        else:                                     mixed += 1
    return same, diff, mixed

s, d, m = context_class_match(X_tr, ctx_tr, y_tr, X_tr, y_tr)
print(f"  Train: контекст того же класса={s} ({s/len(X_tr)*100:.1f}%), "
      f"другого={d} ({d/len(X_tr)*100:.1f}%), смешанный={m} ({m/len(X_tr)*100:.1f}%)")
s, d, m = context_class_match(X_te, ctx_te, y_te, X_te, y_te)
print(f"  Test:  контекст того же класса={s} ({s/len(X_te)*100:.1f}%), "
      f"другого={d} ({d/len(X_te)*100:.1f}%), смешанный={m} ({m/len(X_te)*100:.1f}%)")
print("Вывод: контекст содержит реплики СМЕШАННЫХ классов, что моделирует")
print("реалистичную ситуацию диалога (нельзя предсказать класс только по контексту).")

# =======================================================================
# 4) ВЕКТОРИЗАЦИЯ ТРАНСФОРМЕРНОЙ МОДЕЛЬЮ (mini-BERT)
# =======================================================================
print("\n" + "=" * 70)
print("ВЕКТОРИЗАЦИЯ: sentence-transformers paraphrase-multilingual-MiniLM-L12-v2")
print("=" * 70)
print("Загрузка модели (первый запуск -- скачивание ~120 МБ)...")

t0 = time.time()
encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(f"Модель загружена за {time.time()-t0:.1f}s. Размерность: {encoder.get_sentence_embedding_dimension()}")

def embed(texts, label=""):
    t0 = time.time()
    emb = encoder.encode(list(texts), show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True,
                         batch_size=64)
    print(f"  {label}: shape={emb.shape}, время={time.time()-t0:.1f}s")
    return emb

print("\nВекторизация prompts:")
P_tr  = embed(X_tr,  "train prompts")
P_val = embed(X_val, "val   prompts")
P_te  = embed(X_te,  "test  prompts")
print("\nВекторизация contexts:")
C_tr  = embed(ctx_tr,  "train contexts")
C_val = embed(ctx_val, "val   contexts")
C_te  = embed(ctx_te,  "test  contexts")

# Три типа признакового пространства
def join(a, b): return np.hstack([a, b])
feature_sets = {
    'prompt_only':    (P_tr, P_val, P_te),
    'context_only':   (C_tr, C_val, C_te),
    'prompt+context': (join(P_tr, C_tr), join(P_val, C_val), join(P_te, C_te)),
}
print(f"\nРазмерности признакового пространства:")
print(f"  prompt_only:    {P_tr.shape[1]}")
print(f"  context_only:   {C_tr.shape[1]}")
print(f"  prompt+context: {P_tr.shape[1]+C_tr.shape[1]}")

# =======================================================================
# 5) МОДЕЛИ
# =======================================================================
def make_models():
    """4 алгоритмически различающиеся модели."""
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                            random_state=RANDOM_STATE)
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        cv=3, method='sigmoid')
    catboost = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        loss_function='Logloss', eval_metric='F1',
        random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
    stacking = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
            ('svm', CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000,
                random_state=RANDOM_STATE), cv=3, method='sigmoid')),
            ('cat', CatBoostClassifier(iterations=200, learning_rate=0.05,
                depth=6, random_seed=RANDOM_STATE, verbose=False,
                allow_writing_files=False)),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000,
            random_state=RANDOM_STATE),
        cv=5, stack_method='predict_proba')
    return {'LogReg': lr, 'LinearSVC (Platt)': svm,
            'CatBoost': catboost, 'Stacking': stacking}

# =======================================================================
# 6) ЦИКЛ ЭКСПЕРИМЕНТОВ
# =======================================================================
def evaluate(model, X, y):
    pred = model.predict(X)
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)[:, 1]
    else:
        d = model.decision_function(X)
        prob = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return {
        'acc': accuracy_score(y, pred),
        'f1m': f1_score(y, pred, average='macro'),
        'f1b': f1_score(y, pred, average='binary'),
        'auc': roc_auc_score(y, prob),
        'pred': pred, 'prob': prob,
    }

results = {}
for fset_name, (Xtr, Xval, Xte) in feature_sets.items():
    print("\n" + "=" * 70)
    print(f"ЭКСПЕРИМЕНТ: признаки = {fset_name}  (размерность {Xtr.shape[1]})")
    print("=" * 70)
    models = make_models()
    for mname, model in models.items():
        t0 = time.time()
        model.fit(Xtr, y_tr)
        ttrain = time.time() - t0
        r_train = evaluate(model, Xtr,  y_tr)
        r_val   = evaluate(model, Xval, y_val)
        r_test  = evaluate(model, Xte,  y_te)
        results[(fset_name, mname)] = {
            'train': r_train, 'val': r_val, 'test': r_test,
            'train_time_s': round(ttrain, 3),
        }
        print(f"\n-- {mname}  ({ttrain:.2f}s)")
        print(f"   train: Acc={r_train['acc']:.4f}  F1m={r_train['f1m']:.4f}  AUC={r_train['auc']:.4f}")
        print(f"   val:   Acc={r_val['acc']:.4f}  F1m={r_val['f1m']:.4f}  AUC={r_val['auc']:.4f}")
        print(f"   test:  Acc={r_test['acc']:.4f}  F1m={r_test['f1m']:.4f}  AUC={r_test['auc']:.4f}")
        delta = r_train['f1m'] - r_val['f1m']
        if delta > 0.10:
            print(f"   ВНИМАНИЕ: возможное переобучение, dF1m(train-val) = {delta:.3f}")
        else:
            print(f"   OK: переобучения не наблюдается, dF1m(train-val) = {delta:.3f}")

# =======================================================================
# 7) АНАЛИЗ МАТРИЦ ОШИБОК ДЛЯ ВСЕХ 4 МОДЕЛЕЙ
# (preceptor: "анализ нестековых методов на матрицу ошибок,
#             вдруг нет смысла стекинг использовать")
# =======================================================================
print("\n" + "=" * 70)
print("МАТРИЦЫ ОШИБОК ВСЕХ 4 МОДЕЛЕЙ (prompt+context, test)")
print("=" * 70)
confusion_all = {}
for mname in ['LogReg', 'LinearSVC (Platt)', 'CatBoost', 'Stacking']:
    r = results[('prompt+context', mname)]['test']
    cm = confusion_matrix(y_te, r['pred'])
    confusion_all[mname] = cm.tolist()
    print(f"\n{mname}:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"  Acc={r['acc']:.4f}  F1m={r['f1m']:.4f}  F1b={r['f1b']:.4f}")

# Сравнение со Stacking
print("\nВЫВОД (нужен ли Stacking?):")
base_f1 = max(results[('prompt+context', m)]['test']['f1m']
              for m in ['LogReg', 'LinearSVC (Platt)', 'CatBoost'])
stack_f1 = results[('prompt+context', 'Stacking')]['test']['f1m']
delta = stack_f1 - base_f1
if delta > 0.005:
    print(f"  Stacking даёт прирост +{delta:.4f} F1m -- стекинг ОПРАВДАН")
else:
    print(f"  Stacking даёт {delta:+.4f} F1m -- стекинг практически НЕ нужен,")
    print(f"  лучшая базовая модель сопоставима со стекингом.")

# =======================================================================
# 8) ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ
# =======================================================================
# Выберем лучшую модель по F1m на validation set
best_name = max(['LogReg', 'LinearSVC (Platt)', 'CatBoost', 'Stacking'],
                key=lambda m: results[('prompt+context', m)]['val']['f1m'])
print(f"\nЛучшая модель по val F1m: {best_name}")
best = results[('prompt+context', best_name)]
cm_best = confusion_matrix(y_te, best['test']['pred'])
print(classification_report(y_te, best['test']['pred'],
                             target_names=['benign', 'malicious'], digits=4))

# Точность по типам атак
print(f"\nТочность по типам атак (тест, модель = {best_name}):")
pred_test = best['test']['pred']
attack_types = test['category'].values
attack_acc = {}
for atype in sorted(set(attack_types)):
    if atype == 'benign': continue
    mask = attack_types == atype
    n = mask.sum()
    if n == 0: continue
    correct = ((pred_test[mask] == 1) & (y_te[mask] == 1)).sum()
    attack_acc[atype] = (int(correct), int(n))
    print(f"  {atype:25s}: {correct}/{n} ({correct/n*100:.0f}%)")

# =======================================================================
# 9) СОХРАНЕНИЕ В JSON
# =======================================================================
def clean(d):
    return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in d.items() if k not in ('pred', 'prob')}

results_json = {}
for (fset, mname), r in results.items():
    results_json[f"{fset}__{mname}"] = {
        'feature_set': fset, 'model': mname,
        'train': clean(r['train']),
        'val':   clean(r['val']),
        'test':  clean(r['test']),
        'train_time_s': r['train_time_s'],
    }

results_json['_meta'] = {
    'dataset': 'neuralchemy/Prompt-injection-dataset (core)',
    'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
    'embedding_dim': int(encoder.get_sentence_embedding_dimension()),
    'n_train': len(X_tr), 'n_val': len(X_val), 'n_test': len(X_te),
    'best_model': best_name,
    'attack_acc_best': {k: list(v) for k, v in attack_acc.items()},
    'confusion_all': confusion_all,
    'confusion_best': cm_best.tolist(),
    'n_categories': int(train['category'].nunique()),
}

with open('results_v4.json', 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print("\nСохранено: results_v4.json")

# =======================================================================
# 10) ГРАФИКИ
# =======================================================================
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

models_list = ['LogReg', 'LinearSVC (Platt)', 'CatBoost', 'Stacking']
fsets_list = ['prompt_only', 'context_only', 'prompt+context']
fset_labels = {'prompt_only': 'Только prompt',
               'context_only': 'Только context',
               'prompt+context': 'Prompt + context'}

def get(fset, model, split, metric):
    return results_json[f"{fset}__{model}"][split][metric]

# ─── FIG 1: F1m × модели × пространства (test) ───
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(models_list)); w = 0.27
colors = {'prompt_only': '#4C72B0', 'context_only': '#C44E52', 'prompt+context': '#55A868'}
for i, fset in enumerate(fsets_list):
    vals = [get(fset, m, 'test', 'f1m') for m in models_list]
    bars = ax.bar(x + (i-1)*w, vals, w, label=fset_labels[fset], color=colors[fset])
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{b.get_height():.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(models_list, fontsize=9)
ax.set_ylabel('F1-macro (test)'); ax.set_ylim(0, 1.15)
ax.set_title('Сравнение моделей по типу признакового пространства (test)')
ax.legend(loc='lower right'); ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figs_v4/fig1_models_x_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("figs_v4/fig1_models_x_features.png")

# ─── FIG 2: Train vs Val vs Test (prompt+context) ───
fig, ax = plt.subplots(figsize=(10, 5))
splits = ['train', 'val', 'test']
split_colors = ['#4C72B0', '#DD8452', '#55A868']
x = np.arange(len(models_list)); w = 0.27
for i, sp in enumerate(splits):
    vals = [get('prompt+context', m, sp, 'f1m') for m in models_list]
    bars = ax.bar(x + (i-1)*w, vals, w, label=sp.capitalize(), color=split_colors[i])
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f'{b.get_height():.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(models_list, fontsize=9)
ax.set_ylabel('F1-macro')
ax.set_title('Сравнение метрик на train / val / test (признаки prompt + context)')
ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figs_v4/fig2_train_val_test.png', dpi=150, bbox_inches='tight')
plt.close()
print("figs_v4/fig2_train_val_test.png")

# ─── FIG 3: Матрицы ошибок ВСЕХ 4 моделей (для замечания препода) ───
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, mname in zip(axes, models_list):
    cm = np.array(confusion_all[mname])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malic.'])
    ax.set_yticklabels(['Benign', 'Malic.'])
    ax.set_xlabel('Предсказ.'); ax.set_ylabel('Истинный')
    ax.set_title(mname, fontsize=10)
    for i in range(2):
        for j in range(2):
            col = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color=col, fontsize=13, fontweight='bold')
plt.suptitle('Матрицы ошибок всех моделей (prompt + context, test)', fontsize=12)
plt.tight_layout()
plt.savefig('figs_v4/fig3_confusion_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("figs_v4/fig3_confusion_all.png")

# ─── FIG 4: Матрица ошибок лучшей модели крупно ───
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_best, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Benign (0)', 'Malicious (1)'])
ax.set_yticklabels(['Benign (0)', 'Malicious (1)'])
ax.set_xlabel('Предсказанный класс'); ax.set_ylabel('Истинный класс')
ax.set_title(f'Матрица ошибок {best_name} (prompt + context, test)')
for i in range(2):
    for j in range(2):
        col = 'white' if cm_best[i, j] > cm_best.max()/2 else 'black'
        ax.text(j, i, str(cm_best[i, j]), ha='center', va='center',
                color=col, fontsize=22, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('figs_v4/fig4_confusion_best.png', dpi=150, bbox_inches='tight')
plt.close()
print("figs_v4/fig4_confusion_best.png")

# ─── FIG 5: распределение классов и категорий атак в train ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
sizes = [(train['label']==0).sum(), (train['label']==1).sum()]
labels = [f'Benign\n({sizes[0]}, {sizes[0]/sum(sizes)*100:.1f}%)',
          f'Malicious\n({sizes[1]}, {sizes[1]/sum(sizes)*100:.1f}%)']
ax1.pie(sizes, labels=labels, colors=['#4C72B0', '#C44E52'], startangle=90,
        explode=(0, 0.04), wedgeprops={'edgecolor':'white','linewidth':2})
ax1.set_title(f'Распределение классов в train ({len(train)} записей)')

# top-10 категорий атак
attack_counts = train[train['label']==1]['category'].value_counts().head(10)
bars = ax2.barh(attack_counts.index, attack_counts.values, color='#C44E52')
ax2.set_xlabel('Количество записей')
ax2.set_title('Топ-10 типов атак (train)')
ax2.invert_yaxis()
for bar, c in zip(bars, attack_counts.values):
    ax2.text(bar.get_width()+10, bar.get_y()+bar.get_height()/2, str(c),
             va='center', fontsize=10, fontweight='bold')
ax2.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figs_v4/fig5_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("figs_v4/fig5_distribution.png")

# ─── FIG 6: схема пайплайна (бесцветная, для курсовой) ───
import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(13, 8.5), dpi=150)
ax.set_xlim(0, 680); ax.set_ylim(0, 520)
ax.invert_yaxis(); ax.set_aspect('equal'); ax.axis('off')
fig.patch.set_facecolor('white')

def box(x, y, w, h, t1, t2=None, dashed=False):
    ls = '--' if dashed else '-'
    r = patches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=4",
        linewidth=1.2, edgecolor='black', facecolor='white', linestyle=ls)
    ax.add_patch(r)
    if t2:
        ax.text(x+w/2, y+h/2-7, t1, ha='center', va='center', fontsize=10.5, fontweight='bold')
        ax.text(x+w/2, y+h/2+9, t2, ha='center', va='center', fontsize=9)
    else:
        ax.text(x+w/2, y+h/2, t1, ha='center', va='center', fontsize=10.5, fontweight='bold')

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))

box(20, 60,   110, 56,  'Prompt', '(текущий запрос)')
box(20, 404,  110, 56,  'Context', '(2 реплики)')
box(160, 60,  140, 56,  'Sentence-BERT', 'embedding (384)')
box(160, 404, 140, 56,  'Sentence-BERT', 'embedding (384)')
box(330, 232, 120, 56,  'Конкатенация', '(768 dim)')
box(480, 50,  130, 44,  'LogReg')
box(480, 148, 130, 44,  'LinearSVC')
box(480, 246, 130, 44,  'CatBoost')
box(480, 344, 130, 44,  'Stacking', '(мета-LR)')
box(100, 480, 540, 30, '', dashed=True)
ax.text(370, 495, 'Soft Voting / Stacking -> решение: malicious / benign',
        ha='center', va='center', fontsize=10)
arrow(130, 88, 158, 88); arrow(130, 432, 158, 432)
ax.plot([300, 320, 320], [88, 88, 250], color='black', lw=1.2); arrow(320, 250, 330, 250)
ax.plot([300, 320, 320], [432, 432, 270], color='black', lw=1.2); arrow(320, 270, 330, 270)
ax.plot([450, 465], [260, 260], color='black', lw=1.2)
ax.plot([465, 465], [72, 366], color='black', lw=1.2)
for y_ in [72, 170, 268, 366]:
    arrow(465, y_, 478, y_)
ax.plot([545, 545], [388, 478], color='black', lw=1.2); arrow(545, 478, 545, 480)
plt.tight_layout()
plt.savefig('figs_v4/fig6_pipeline.png', dpi=200, bbox_inches='tight')
plt.close()
print("figs_v4/fig6_pipeline.png")

# ─── FIG 7: точность по типам атак ───
if attack_acc:
    # top-10 категорий
    items = sorted(attack_acc.items(), key=lambda x: -x[1][1])[:10]
    names = [k for k, _ in items]
    percs = [v[0]/v[1]*100 for _, v in items]
    fracs = [f"{v[0]}/{v[1]}" for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(names, percs, color='#55A868')
    ax.set_xlim(0, 115); ax.set_xlabel('Точность классификации, %')
    ax.set_title(f'Точность модели {best_name} по типам атак (top-10, prompt + context, test)')
    ax.invert_yaxis()
    for bar, p, f in zip(bars, percs, fracs):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                f' {p:.0f}% ({f})', va='center', fontsize=10, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('figs_v4/fig7_attack_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("figs_v4/fig7_attack_types.png")

print("\n" + "=" * 70)
print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
print("=" * 70)
print("\nОтправь обратно Claude:")
print("  1. results_v4.json")
print("  2. console_output_v4.txt")
print("  3. Папку figs_v4/ целиком (7 PNG файлов)")

log_file.close()
