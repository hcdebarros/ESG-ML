"""Pre-processamento dos dados originais (10 registros).

Etapas:
    1. Le o XLSX bruto em data/raw/.
    2. Remove colunas PII (CNPJ, email, nome, etc.) e textos livres.
    3. Renomeia colunas para nomes curtos (ver `config.COLUMN_RENAME`).
    4. Codifica respostas Sim/Nao -> 1/0; ordena `porte` e `faturamento`.
    5. Constroi `esg_score` e o target `maturidade_esg`. NAO remove as
       questoes criticas: a remocao acontece apenas em `src.train`
       (anti-leakage). Manter as criticas aqui permite que o sintetizador
       (SDV) preserve a relacao entre essas features e o target.
    6. Persiste em data/processed/fornecedores_base.csv.

Uso:
    python -m src.preprocess
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Forms")
    return df


def drop_pii_and_text(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in config.PII_COLUMNS + config.TEXT_FREE_COLUMNS if c in df.columns]
    return df.drop(columns=to_drop)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=config.COLUMN_RENAME)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in config.ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype("int64")
    yn_cols = [c for c in df.columns if c not in config.ORDINAL_MAPS]
    for col in yn_cols:
        df[col] = df[col].map(config.YES_NO_MAP).fillna(0).astype("int64")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score = df[config.CRITICAL_QUESTIONS].sum(axis=1)
    df["esg_score"] = score.astype("int64")
    df[config.TARGET_COLUMN] = pd.cut(
        score,
        bins=config.TARGET_BINS,
        labels=config.TARGET_LABELS,
        include_lowest=True,
    ).astype(str)
    return df


def preprocess(input_path: Path = config.RAW_FILE,
               output_path: Path = config.PROCESSED_BASE) -> pd.DataFrame:
    df = load_raw(input_path)
    df = drop_pii_and_text(df)
    df = rename_columns(df)
    df = encode_features(df)
    df = build_target(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print("[preprocess] gravou {} registros e {} colunas em {}".format(
        len(df), df.shape[1], output_path))
    print("[preprocess] distribuicao do target:")
    print(df[config.TARGET_COLUMN].value_counts())
    return df


if __name__ == "__main__":
    preprocess()
