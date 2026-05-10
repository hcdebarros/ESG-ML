"""Pipeline de treino com tracking MLflow.

Estrategia:
    - Carrega o dataset de 55 registros (10 originais + 45 sinteticos).
    - Remove as 9 questoes criticas das features (anti-leakage).
    - 4 modelos: KNN, DecisionTree, RandomForest, LogisticRegression.
    - 3 estrategias de validacao: Holdout (70/30 estratificado),
      StratifiedKFold (5-fold), LeaveOneOut.
    - MLflow registra: parametros, metricas (accuracy, f1_macro, precision_macro,
      recall_macro), confusion matrix (artefato PNG) e modelo serializado.
    - O melhor modelo (por f1_macro em CV) e salvo em models/best_model.joblib
      com metadados em models/feature_columns.json.

Uso:
    python -m src.train
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def load_training_data():
    """Carrega 55 registros e separa X / y, removendo colunas criticas."""
    df = pd.read_csv(config.PROCESSED_FULL)
    drop_cols = config.CRITICAL_QUESTIONS + ["esg_score", config.TARGET_COLUMN, "origem"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[config.TARGET_COLUMN]
    return X, y, df


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def build_models():
    """Retorna dict nome -> (Pipeline, params para log)."""
    return {
        "knn": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))]),
            {"n_neighbors": 5, "weights": "distance"},
        ),
        "decision_tree": (
            Pipeline([("clf", DecisionTreeClassifier(
                max_depth=5, min_samples_split=4, random_state=config.RANDOM_STATE))]),
            {"max_depth": 5, "min_samples_split": 4},
        ),
        "random_forest": (
            Pipeline([("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_split=3,
                random_state=config.RANDOM_STATE, n_jobs=-1))]),
            {"n_estimators": 200, "max_depth": 8, "min_samples_split": 3},
        ),
        "logistic_regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(
                          max_iter=2000, C=1.0,
                          random_state=config.RANDOM_STATE))]),
            {"C": 1.0, "max_iter": 2000},
        ),
    }


# ---------------------------------------------------------------------------
# Avaliacao
# ---------------------------------------------------------------------------
def evaluate_holdout(model, X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=config.RANDOM_STATE)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "holdout_accuracy": accuracy_score(y_te, y_pred),
        "holdout_f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "holdout_precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "holdout_recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
    }, y_te, y_pred


def evaluate_kfold(model, X, y, k: int = 5):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_STATE)
    scores_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    scores_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return {
        "cv5_accuracy_mean": float(np.mean(scores_acc)),
        "cv5_accuracy_std": float(np.std(scores_acc)),
        "cv5_f1_macro_mean": float(np.mean(scores_f1)),
        "cv5_f1_macro_std": float(np.std(scores_f1)),
    }


def evaluate_loocv(model, X, y):
    loo = LeaveOneOut()
    scores_acc = cross_val_score(model, X, y, cv=loo, scoring="accuracy", n_jobs=-1)
    return {
        "loo_accuracy": float(np.mean(scores_acc)),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run principal
# ---------------------------------------------------------------------------
def run_training():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    X, y, _ = load_training_data()
    classes = sorted(y.unique().tolist())
    print("[train] features:", X.shape, "| classes:", classes)
    print("[train] distribuicao y:", y.value_counts().to_dict())

    leaderboard = []
    artefatos_modelos = {}

    for name, (model, params) in build_models().items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("modelo", name)
            mlflow.log_params(params)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_amostras", X.shape[0])

            holdout_metrics, y_te, y_pred = evaluate_holdout(model, X, y)
            cv_metrics = evaluate_kfold(model, X, y, k=5)
            loo_metrics = evaluate_loocv(model, X, y)

            metrics = {**holdout_metrics, **cv_metrics, **loo_metrics}
            mlflow.log_metrics(metrics)

            cm = confusion_matrix(y_te, y_pred, labels=classes)
            cm_path = config.REPORTS_DIR / "figures" / f"cm_{name}.png"
            plot_confusion_matrix(cm, classes, cm_path,
                                  f"Matriz de Confusao - {name} (holdout)")
            mlflow.log_artifact(str(cm_path), artifact_path="figures")

            report_str = classification_report(y_te, y_pred, zero_division=0)
            report_path = config.REPORTS_DIR / "figures" / f"report_{name}.txt"
            report_path.write_text(report_str, encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            # Modelo treinado em todo o dataset (deploy-ready)
            full_model = model
            full_model.fit(X, y)
            mlflow.sklearn.log_model(full_model, "model")

            artefatos_modelos[name] = full_model
            leaderboard.append({"modelo": name, **metrics})
            print(f"[train] {name}: cv5_f1={cv_metrics['cv5_f1_macro_mean']:.3f} "
                  f"loo={loo_metrics['loo_accuracy']:.3f} "
                  f"holdout_f1={holdout_metrics['holdout_f1_macro']:.3f}")

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(
        "cv5_f1_macro_mean", ascending=False).reset_index(drop=True)
    leaderboard_path = config.REPORTS_DIR / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)
    print("\n[train] leaderboard:")
    print(leaderboard_df.to_string(index=False))

    best_name = leaderboard_df.iloc[0]["modelo"]
    best_model = artefatos_modelos[best_name]
    joblib.dump(best_model, config.MODELS_DIR / "best_model.joblib")
    feature_meta = {
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "classes": classes,
    }
    (config.MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(feature_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[train] melhor modelo: {best_name} -> models/best_model.joblib")


if __name__ == "__main__":
    run_training()
