"""Dashboard Streamlit - Maturidade ESG de Fornecedores.

5 abas:
    1. Visao geral - distribuicao do dataset.
    2. Comparacao de modelos - leaderboard + matrizes de confusao.
    3. Previsoes - simulacao interativa (input manual de respostas).
    4. Importancia das features - quando o modelo expoe `feature_importances_`.
    5. Sobre - rastreabilidade do projeto.

Uso:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src import config

st.set_page_config(
    page_title="ESG - Maturidade de Fornecedores",
    page_icon=":seedling:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Carregamento (cache)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_FULL)


@st.cache_data
def load_leaderboard() -> pd.DataFrame:
    p = config.REPORTS_DIR / "leaderboard.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_resource
def load_model():
    model = joblib.load(config.MODELS_DIR / "best_model.joblib")
    meta = json.loads((config.MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    return model, meta


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("ESG - Fornecedores")
st.sidebar.caption("ML I + Projeto 3 - CESAR School")

aba = st.sidebar.radio(
    "Navegue",
    ("Visao Geral", "Comparacao de Modelos", "Simulador de Previsao",
     "Importancia das Features", "Sobre"),
)

df = load_data()
leaderboard = load_leaderboard()
try:
    model, meta = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_err = str(e)


# ---------------------------------------------------------------------------
# 1. Visao Geral
# ---------------------------------------------------------------------------
if aba == "Visao Geral":
    st.title(":seedling: Maturidade ESG de Fornecedores")
    st.write(
        "Solucao de ML para classificar fornecedores em **Baixa / Media / Alta** "
        "maturidade socioambiental, com base em respostas de questionario."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros totais", len(df))
    c2.metric("Originais", int((df["origem"] == "original").sum()))
    c3.metric("Sinteticos (SDV)", int((df["origem"] == "sintetico_sdv").sum()))
    c4.metric("Sinteticos (perturbados)", int((df["origem"] == "sintetico_perturbado").sum()))

    st.subheader("Distribuicao do target")
    counts = df[config.TARGET_COLUMN].value_counts().reindex(["Baixa", "Media", "Alta"]).fillna(0)
    fig = px.bar(
        x=counts.index, y=counts.values,
        labels={"x": "Maturidade ESG", "y": "Frequencia"},
        color=counts.index,
        color_discrete_map={"Baixa": "#d65f5f", "Media": "#dfa44c", "Alta": "#5fa75f"},
        text=counts.values,
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribuicao por origem")
    cross = pd.crosstab(df["origem"], df[config.TARGET_COLUMN]).reindex(
        columns=["Baixa", "Media", "Alta"]).fillna(0).astype(int)
    fig2 = px.bar(
        cross.reset_index().melt(id_vars="origem", var_name="maturidade", value_name="freq"),
        x="origem", y="freq", color="maturidade",
        color_discrete_map={"Baixa": "#d65f5f", "Media": "#dfa44c", "Alta": "#5fa75f"},
        barmode="stack",
    )
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Ver amostra do dataset"):
        st.dataframe(df.head(20))


# ---------------------------------------------------------------------------
# 2. Comparacao de Modelos
# ---------------------------------------------------------------------------
elif aba == "Comparacao de Modelos":
    st.title(":bar_chart: Comparacao de Modelos")

    if leaderboard.empty:
        st.warning("Leaderboard nao encontrado. Rode `python -m src.train` para gerar.")
    else:
        st.subheader("Leaderboard (ordenado por F1 macro - CV 5-fold)")
        st.dataframe(
            leaderboard.style.format({c: "{:.3f}" for c in leaderboard.columns
                                      if c != "modelo"}),
            use_container_width=True,
        )

        st.subheader("Comparacao visual")
        metric_cols = ["holdout_f1_macro", "cv5_f1_macro_mean", "loo_accuracy"]
        long_df = leaderboard.melt(id_vars="modelo", value_vars=metric_cols,
                                   var_name="metrica", value_name="valor")
        fig = px.bar(long_df, x="modelo", y="valor", color="metrica", barmode="group",
                     text=long_df["valor"].round(3))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Matrizes de confusao (holdout)")
        fig_dir = config.REPORTS_DIR / "figures"
        cols = st.columns(2)
        for i, modelo in enumerate(leaderboard["modelo"].tolist()):
            cm_path = fig_dir / f"cm_{modelo}.png"
            if cm_path.exists():
                # use_column_width e o parametro compativel com todas as
                # versoes recentes do Streamlit (use_container_width foi
                # introduzido em st.image so a partir da 1.36).
                cols[i % 2].image(Image.open(cm_path), caption=modelo,
                                  use_column_width=True)


# ---------------------------------------------------------------------------
# 3. Simulador de Previsao
# ---------------------------------------------------------------------------
elif aba == "Simulador de Previsao":
    st.title(":mag: Simulador - Classifique um novo fornecedor")
    if not model_loaded:
        st.error(f"Modelo nao carregado: {model_err}")
        st.stop()

    st.write(f"**Modelo em producao:** `{meta['model_name']}`")
    st.caption("Preencha as respostas abaixo para obter a maturidade ESG predita.")

    porte_labels = {0: "Microempresa", 1: "Pequeno", 2: "Medio", 3: "Grande"}
    fat_labels = {-1: "Prefere nao informar",
                  0: "R$ 500 mil - R$ 1,5 mi",
                  1: "R$ 1,5 mi - R$ 3 mi",
                  2: "Acima de R$ 3 mi"}

    inputs = {}
    c1, c2 = st.columns(2)
    with c1:
        inputs["porte"] = st.selectbox("Porte", options=list(porte_labels), format_func=lambda x: porte_labels[x])
        inputs["faturamento"] = st.selectbox("Faturamento", options=list(fat_labels), format_func=lambda x: fat_labels[x])
        inputs["sancao_admin"] = int(st.toggle("Recebeu sancao administrativa nos ultimos 5 anos?"))
        inputs["tem_certificacoes"] = int(st.toggle("Possui certificacoes (ISO 14001, 45001, etc)?"))
        inputs["treina_sustentabilidade"] = int(st.toggle("Realiza treinamentos de sustentabilidade?"))
        inputs["gestao_agua"] = int(st.toggle("Praticas de gestao de agua?"))
        inputs["eficiencia_energetica"] = int(st.toggle("Praticas de eficiencia energetica?"))
        inputs["gestao_residuos"] = int(st.toggle("Praticas de gestao de residuos?"))

    with c2:
        inputs["pegada_carbono"] = int(st.toggle("Calcula pegada de carbono?"))
        inputs["auditoria_gee"] = int(st.toggle("Inventario GEE auditado?"))
        inputs["voluntariado"] = int(st.toggle("Programas de voluntariado?"))
        inputs["saude_mental"] = int(st.toggle("Programas de saude mental?"))
        inputs["clausulas_contratos"] = int(st.toggle("Clausulas ESG em contratos?"))
        inputs["treina_compradores"] = int(st.toggle("Treina time de compras em sustentabilidade?"))
        inputs["diversidade_fornecedores"] = int(st.toggle("Programa de diversidade na cadeia?"))

    feature_cols = meta["feature_columns"]
    X_new = pd.DataFrame([[inputs[c] for c in feature_cols]], columns=feature_cols)

    if st.button("Classificar fornecedor", type="primary"):
        pred = model.predict(X_new)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]

        col_pred, col_proba = st.columns([1, 2])
        with col_pred:
            color = {"Baixa": "#d65f5f", "Media": "#dfa44c", "Alta": "#5fa75f"}.get(pred, "#888")
            st.markdown(
                f"<div style='padding:1rem;background:{color};color:white;"
                f"border-radius:0.5rem;text-align:center;font-size:1.3rem;'>"
                f"<b>Maturidade prevista:</b><br><span style='font-size:2rem'>{pred}</span></div>",
                unsafe_allow_html=True,
            )

        if proba is not None:
            with col_proba:
                proba_df = pd.DataFrame({"classe": model.classes_, "probabilidade": proba})
                proba_df = proba_df.sort_values("probabilidade", ascending=False)
                fig = px.bar(proba_df, x="classe", y="probabilidade",
                             color="classe",
                             color_discrete_map={"Baixa": "#d65f5f", "Media": "#dfa44c", "Alta": "#5fa75f"},
                             text=proba_df["probabilidade"].round(2))
                fig.update_layout(yaxis_range=[0, 1], height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 4. Importancia das Features
# ---------------------------------------------------------------------------
elif aba == "Importancia das Features":
    st.title(":dna: Importancia das Features")
    if not model_loaded:
        st.error(f"Modelo nao carregado: {model_err}")
        st.stop()

    feature_cols = meta["feature_columns"]
    clf = model.named_steps.get("clf", model)

    if hasattr(clf, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_cols,
            "importancia": clf.feature_importances_,
        }).sort_values("importancia", ascending=True)
        fig = px.bar(imp, x="importancia", y="feature", orientation="h", height=500)
        st.plotly_chart(fig, use_container_width=True)
    elif hasattr(clf, "coef_"):
        coef = clf.coef_
        # multiclass -> media absoluta
        imp_vals = np.mean(np.abs(coef), axis=0)
        imp = pd.DataFrame({
            "feature": feature_cols,
            "importancia (|coef| medio)": imp_vals,
        }).sort_values("importancia (|coef| medio)", ascending=True)
        fig = px.bar(imp, x="importancia (|coef| medio)", y="feature",
                     orientation="h", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"O modelo `{meta['model_name']}` nao expoe importancia direta. "
                "Considere usar Random Forest ou Decision Tree para esta visualizacao.")


# ---------------------------------------------------------------------------
# 5. Sobre
# ---------------------------------------------------------------------------
else:
    st.title("Sobre o projeto")
    st.markdown("""
**Disciplina:** Machine Learning I + Projeto 3 (CESAR School)

**Pipeline:**
1. `src/preprocess.py` - tratamento dos 10 registros originais (PII removidos, encoding, target).
2. `src/synthesize.py` - SDV GaussianCopula (30) + perturbacao controlada (15) -> 45 sinteticos.
3. `src/train.py` - 4 modelos (KNN, Decision Tree, Random Forest, Logistic Regression);
   3 estrategias de validacao (Holdout, Stratified 5-Fold, Leave-One-Out); MLflow tracking.
4. `app/dashboard.py` - este dashboard Streamlit.
5. `Dockerfile` + `docker-compose.yml` - reprodutibilidade total.

**Anti-leakage:** as 9 questoes criticas usadas para construir o target sao removidas
das features de treino. O modelo aprende a inferir maturidade ESG a partir de praticas
operacionais correlatas (porte, certificacoes, treinamentos, clausulas em contratos, etc.).

**Para inspecionar experimentos no MLflow UI:**
```
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```
""")
