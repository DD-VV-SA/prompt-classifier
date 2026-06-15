import json
import random
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_dataset():
    train = pd.read_csv("core_train.csv")
    val = pd.read_csv("core_validation.csv")
    test = pd.read_csv("core_test.csv")

    train_g = set(train["group_id"].dropna())
    val_g = set(val["group_id"].dropna())
    test_g = set(test["group_id"].dropna())
    assert not (train_g & val_g), "group_id leak: train/val"
    assert not (train_g & test_g), "group_id leak: train/test"
    assert not (val_g & test_g), "group_id leak: val/test"

    return train, val, test


def gen_context(items, pool, seed):
    rng = np.random.RandomState(seed)
    pool_list = list(pool)
    out = []
    for i, _ in enumerate(items):
        if id(pool) == id(items):
            indices = [j for j in range(len(pool_list)) if j != i]
            picks = rng.choice(indices, size=2, replace=False)
        else:
            picks = rng.choice(len(pool_list), size=2, replace=False)
        out.append(" [SEP] ".join([pool_list[p] for p in picks]))
    return np.array(out)


def context_class_match(contexts, labels, pool_prompts, pool_labels):
    prompt_to_label = dict(zip(pool_prompts, pool_labels))
    same = diff = mixed = 0
    for ctx, lbl in zip(contexts, labels):
        parts = ctx.split(" [SEP] ")
        ctx_labels = [prompt_to_label.get(p, -1) for p in parts]
        ctx_labels = [c for c in ctx_labels if c != -1]
        if not ctx_labels:
            continue
        if all(c == lbl for c in ctx_labels):
            same += 1
        elif all(c != lbl for c in ctx_labels):
            diff += 1
        else:
            mixed += 1
    return same, diff, mixed


def make_models():
    lr = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        cv=3,
        method="sigmoid",
    )
    catboost = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    stacking = StackingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(
                    C=1.0, max_iter=1000, random_state=RANDOM_STATE
                ),
            ),
            (
                "svm",
                CalibratedClassifierCV(
                    LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
                    cv=3,
                    method="sigmoid",
                ),
            ),
            (
                "cat",
                CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE
        ),
        cv=5,
        stack_method="predict_proba",
    )
    return {
        "LogReg": lr,
        "LinearSVC (Platt)": svm,
        "CatBoost": catboost,
        "Stacking": stacking,
    }


def evaluate(model, X, y):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        d = model.decision_function(X)
        prob = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return {
        "acc": accuracy_score(y, pred),
        "f1m": f1_score(y, pred, average="macro"),
        "f1b": f1_score(y, pred, average="binary"),
        "auc": roc_auc_score(y, prob),
        "pred": pred,
        "prob": prob,
    }


def main():
    train, val, test = load_dataset()

    X_tr, y_tr = train["text"].values, train["label"].values
    X_val, y_val = val["text"].values, val["label"].values
    X_te, y_te = test["text"].values, test["label"].values

    print(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")
    print(
        f"Train balance: benign={(y_tr == 0).sum()}, "
        f"malicious={(y_tr == 1).sum()}"
    )
    print(f"Categories in train: {train['category'].nunique()}")

    ctx_tr = gen_context(X_tr, X_tr, seed=42)
    ctx_val = gen_context(X_val, X_val, seed=43)
    ctx_te = gen_context(X_te, X_te, seed=44)

    s, d, m = context_class_match(ctx_tr, y_tr, X_tr, y_tr)
    print(
        f"Context match (train): same={s} ({s / len(X_tr) * 100:.1f}%), "
        f"diff={d} ({d / len(X_tr) * 100:.1f}%), "
        f"mixed={m} ({m / len(X_tr) * 100:.1f}%)"
    )
    s, d, m = context_class_match(ctx_te, y_te, X_te, y_te)
    print(
        f"Context match (test):  same={s} ({s / len(X_te) * 100:.1f}%), "
        f"diff={d} ({d / len(X_te) * 100:.1f}%), "
        f"mixed={m} ({m / len(X_te) * 100:.1f}%)"
    )

    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def embed(texts):
        return encoder.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
        )

    P_tr, P_val, P_te = embed(X_tr), embed(X_val), embed(X_te)
    C_tr, C_val, C_te = embed(ctx_tr), embed(ctx_val), embed(ctx_te)

    feature_sets = {
        "prompt_only": (P_tr, P_val, P_te),
        "context_only": (C_tr, C_val, C_te),
        "prompt+context": (
            np.hstack([P_tr, C_tr]),
            np.hstack([P_val, C_val]),
            np.hstack([P_te, C_te]),
        ),
    }

    results = {}
    print(
        f"\n{'feature_set':<16} {'model':<20} "
        f"{'train':>7} {'val':>7} {'test':>7} {'auc':>7}"
    )
    for fset_name, (Xtr, Xval, Xte) in feature_sets.items():
        for mname, model in make_models().items():
            t0 = time.time()
            model.fit(Xtr, y_tr)
            r_train = evaluate(model, Xtr, y_tr)
            r_val = evaluate(model, Xval, y_val)
            r_test = evaluate(model, Xte, y_te)
            results[(fset_name, mname)] = {
                "train": r_train,
                "val": r_val,
                "test": r_test,
                "train_time_s": round(time.time() - t0, 3),
            }
            print(
                f"{fset_name:<16} {mname:<20} "
                f"{r_train['f1m']:>7.4f} {r_val['f1m']:>7.4f} "
                f"{r_test['f1m']:>7.4f} {r_test['auc']:>7.4f}"
            )

    print(
        f"\n{'model':<20} {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5} {'F1m':>7}"
    )
    confusion_all = {}
    for mname in ["LogReg", "LinearSVC (Platt)", "CatBoost", "Stacking"]:
        r = results[("prompt+context", mname)]["test"]
        cm = confusion_matrix(y_te, r["pred"])
        confusion_all[mname] = cm.tolist()
        print(
            f"{mname:<20} {cm[0, 0]:>5} {cm[0, 1]:>5} "
            f"{cm[1, 0]:>5} {cm[1, 1]:>5} {r['f1m']:>7.4f}"
        )

    best_name = max(
        ["LogReg", "LinearSVC (Platt)", "CatBoost", "Stacking"],
        key=lambda m: results[("prompt+context", m)]["val"]["f1m"],
    )
    best = results[("prompt+context", best_name)]
    cm_best = confusion_matrix(y_te, best["test"]["pred"])
    print(f"\nBest model (by val F1m): {best_name}")
    print(
        classification_report(
            y_te,
            best["test"]["pred"],
            target_names=["benign", "malicious"],
            digits=4,
        )
    )

    pred_test = best["test"]["pred"]
    attack_types = test["category"].values
    attack_acc = {}
    print("Per-category accuracy (test, malicious only):")
    for atype in sorted(set(attack_types)):
        if atype == "benign":
            continue
        mask = attack_types == atype
        n = mask.sum()
        if n == 0:
            continue
        correct = ((pred_test[mask] == 1) & (y_te[mask] == 1)).sum()
        attack_acc[atype] = (int(correct), int(n))
        print(
            f"  {atype:<25} {correct:>3}/{n:<3} "
            f"({correct / n * 100:>5.1f}%)"
        )

    def clean(d):
        return {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in d.items()
            if k not in ("pred", "prob")
        }

    results_json = {
        f"{fset}__{mname}": {
            "feature_set": fset,
            "model": mname,
            "train": clean(r["train"]),
            "val": clean(r["val"]),
            "test": clean(r["test"]),
            "train_time_s": r["train_time_s"],
        }
        for (fset, mname), r in results.items()
    }
    results_json["_meta"] = {
        "dataset": "neuralchemy/Prompt-injection-dataset (core)",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": int(encoder.get_sentence_embedding_dimension()),
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "n_test": len(X_te),
        "best_model": best_name,
        "attack_acc_best": {k: list(v) for k, v in attack_acc.items()},
        "confusion_all": confusion_all,
        "confusion_best": cm_best.tolist(),
    }

    with open("results_v4.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
