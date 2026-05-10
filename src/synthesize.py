"""Geracao de dados sinteticos com algoritmo de ML.

Estrategia (dataset com 10 registros, classes desbalanceadas):
    1. SDV `GaussianCopulaSynthesizer` aprende a distribuicao conjunta dos 10
       registros base e gera 30 amostras sinteticas "naturais".
    2. Como os originais sao majoritariamente fornecedores de alta/media
       maturidade ESG, geramos 15 amostras adicionais via PERTURBACAO
       CONTROLADA: copiamos registros base e invertimos K (4-7) das 9 questoes
       criticas para 'Nao'. Isso garante representacao da classe minoritaria
       (Baixa) sem fugir do padrao realista do questionario.
    3. Recalculamos score + target apos unir originais + sinteticos.

A justificativa metodologica (por que GaussianCopula + por que perturbacao)
esta no relatorio SBC e no notebook de EDA.

Uso:
    python -m src.synthesize
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config


# ---------------------------------------------------------------------------
# 1. SDV GaussianCopula
# ---------------------------------------------------------------------------
def synthesize_with_sdv(base_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Treina GaussianCopulaSynthesizer e gera `n_samples` amostras."""
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    # Trabalhamos em features apenas — score/target sao recalculados depois.
    train_df = base_df.drop(columns=["esg_score", config.TARGET_COLUMN], errors="ignore")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    # Forcamos os campos Sim/Nao como categoricos para o SDV nao tratar como continuos.
    yn_cols = [c for c in train_df.columns
               if c not in config.ORDINAL_MAPS]
    for col in yn_cols:
        try:
            metadata.update_column(column_name=col, sdtype="categorical")
        except Exception:
            # Se o SDV ja tiver detectado, ignora
            pass

    synth = GaussianCopulaSynthesizer(metadata, default_distribution="beta")
    synth.fit(train_df)
    samples = synth.sample(num_rows=n_samples)
    samples = samples.reset_index(drop=True)
    return samples


# ---------------------------------------------------------------------------
# 2. Perturbacao controlada para classe minoritaria
# ---------------------------------------------------------------------------
def perturb_to_low_maturity(base_df: pd.DataFrame, n_samples: int,
                            min_flips: int = 4, max_flips: int = 7,
                            random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Cria amostras com baixa maturidade ESG.

    Para cada amostra:
        - escolhe aleatoriamente um registro base
        - inverte K questoes criticas (Sim->Nao) com K em [min_flips, max_flips]
        - aplica leve ruido nas demais features Sim/Nao (5% de chance de flip)

    Resultado: amostras coerentes com a estrutura original mas score baixo.
    """
    rng = np.random.default_rng(random_state)
    base_features = base_df.drop(columns=["esg_score", config.TARGET_COLUMN], errors="ignore")

    samples = []
    for _ in range(n_samples):
        # 1) base aleatoria
        row = base_features.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0].copy()

        # 2) flip de K criticas para 0
        k = int(rng.integers(min_flips, max_flips + 1))
        flips = rng.choice(config.CRITICAL_QUESTIONS, size=k, replace=False)
        for col in flips:
            row[col] = 0

        # 3) leve ruido nas demais Sim/Nao (5% chance)
        non_critical_yn = [c for c in base_features.columns
                           if c not in config.ORDINAL_MAPS
                           and c not in config.CRITICAL_QUESTIONS]
        for col in non_critical_yn:
            if rng.random() < 0.05:
                row[col] = 1 - int(row[col])

        samples.append(row)

    return pd.DataFrame(samples).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Recalculo de target apos uniao
# ---------------------------------------------------------------------------
def recompute_target(df: pd.DataFrame) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def synthesize(base_path: Path = config.PROCESSED_BASE,
               output_full: Path = config.PROCESSED_FULL,
               output_synth: Path = config.PROCESSED_SYNTH,
               n_sdv: int = 30,
               n_perturbed: int = 15) -> pd.DataFrame:
    if (n_sdv + n_perturbed) != config.N_SYNTHETIC:
        raise ValueError(
            f"Total sintetico precisa ser {config.N_SYNTHETIC}; recebeu "
            f"{n_sdv + n_perturbed}"
        )

    base = pd.read_csv(base_path)

    # 1) SDV
    print(f"[synthesize] treinando GaussianCopula em {len(base)} registros...")
    sdv_samples = synthesize_with_sdv(base, n_samples=n_sdv)
    sdv_samples["origem"] = "sintetico_sdv"

    # 2) Perturbacao
    print(f"[synthesize] gerando {n_perturbed} amostras perturbadas (classe Baixa)...")
    perturbed = perturb_to_low_maturity(base, n_samples=n_perturbed)
    perturbed["origem"] = "sintetico_perturbado"

    # 3) Originais
    base_out = base.drop(columns=["esg_score", config.TARGET_COLUMN], errors="ignore").copy()
    base_out["origem"] = "original"

    # 4) Tipos consistentes (SDV pode retornar como object)
    feature_cols = base_out.columns.drop("origem")
    for c in feature_cols:
        sdv_samples[c] = sdv_samples[c].astype(base_out[c].dtype)
        perturbed[c] = perturbed[c].astype(base_out[c].dtype)

    full = pd.concat([base_out, sdv_samples, perturbed], ignore_index=True)
    full = recompute_target(full)

    output_full.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(output_full, index=False, encoding="utf-8")

    # Salva apenas os sinteticos para inspecao (auditoria)
    synth_only = full[full["origem"] != "original"].reset_index(drop=True)
    synth_only.to_csv(output_synth, index=False, encoding="utf-8")

    print(f"[synthesize] gravou {len(full)} registros em {output_full}")
    print(f"[synthesize] distribuicao do target apos uniao:")
    print(full[config.TARGET_COLUMN].value_counts())
    print(f"[synthesize] distribuicao por origem:")
    print(pd.crosstab(full["origem"], full[config.TARGET_COLUMN]))

    return full


if __name__ == "__main__":
    synthesize()
