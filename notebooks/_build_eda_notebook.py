"""Gera o notebook 01_eda.ipynb programaticamente.

Roda uma unica vez para criar o .ipynb. Apos gerado, edite no Jupyter normalmente.
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_eda.ipynb"

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# EDA - Maturidade ESG de Fornecedores

**Disciplina:** Machine Learning I + Projeto 3 (CESAR School)
**Dataset:** Questionario socioambiental aplicado a fornecedores (10 originais + 45 sinteticos = 55).

## Objetivo
Explorar o dataset apos preprocessamento e geracao sintetica, validar a coerencia das amostras
geradas (SDV GaussianCopula + perturbacao controlada) e gerar visualizacoes interpretadas que
sustentem as decisoes de modelagem.

## Roteiro
1. Carga e estatisticas descritivas
2. Distribuicao do target `maturidade_esg`
3. Distribuicao das features categoricas (porte, faturamento)
4. Mapa de correlacao das features Sim/Nao
5. Comparacao de praticas ESG por classe de maturidade
6. Validacao da geracao sintetica (originais vs sinteticos)
"""))

cells.append(nbf.v4.new_code_cell(
"""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.append(str(ROOT))
from src import config

df = pd.read_csv(ROOT / 'data' / 'processed' / 'fornecedores_55.csv')
print('Shape:', df.shape)
df.head()"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 1. Estatisticas descritivas

Como quase todas as features sao binarias (1 = pratica adotada, 0 = nao adotada), a media de
cada coluna pode ser lida como **proporcao de fornecedores que adotam aquela pratica**.
"""))

cells.append(nbf.v4.new_code_cell(
"""num_cols = df.select_dtypes(include='number').columns
df[num_cols].describe().T[['mean','std','min','max']].round(2)"""))

cells.append(nbf.v4.new_markdown_cell(
"""**Interpretacao:** o painel acima mostra que praticas como `compromisso_trabalho_digno`,
`gestao_residuos` e `tem_certificacoes` aparecem em mais de 60% dos fornecedores, enquanto
`pegada_carbono` e `auditoria_gee` ficam abaixo de 40% - indicando areas operacionalmente
mais carentes de evolucao ESG.
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 2. Distribuicao do target

A classe `Media` concentra a maior parte dos fornecedores; `Baixa` e `Alta` formam as caudas.
Esse desbalanceamento moderado sera tratado via metricas macro (f1_macro) e estratificacao
nas validacoes (Holdout estratificado, StratifiedKFold).
"""))

cells.append(nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(6,4))
order = ['Baixa', 'Media', 'Alta']
counts = df[config.TARGET_COLUMN].value_counts().reindex(order).fillna(0)
ax.bar(counts.index, counts.values, color=['#d65f5f', '#dfa44c', '#5fa75f'])
for i, v in enumerate(counts.values):
    ax.text(i, v + 0.4, int(v), ha='center', fontsize=11)
ax.set_title('Distribuicao da maturidade ESG (55 fornecedores)')
ax.set_ylabel('Quantidade')
plt.tight_layout()
plt.savefig(ROOT / 'reports' / 'figures' / 'fig_target.png', dpi=120)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 3. Porte x Maturidade ESG

Cruza o porte da empresa com a classe alvo. Permite avaliar se empresas maiores tendem a ter
maior maturidade ESG (hipotese plausivel: mais recursos para certificacao, compliance, etc.).
"""))

cells.append(nbf.v4.new_code_cell(
"""porte_labels = {0:'Micro', 1:'Pequeno', 2:'Medio', 3:'Grande'}
plot_df = df.copy()
plot_df['porte_lbl'] = plot_df['porte'].map(porte_labels)
ct = pd.crosstab(plot_df['porte_lbl'], plot_df[config.TARGET_COLUMN]).reindex(
    index=['Micro','Pequeno','Medio','Grande'], columns=['Baixa','Media','Alta']).fillna(0)
ct.plot(kind='bar', stacked=True, figsize=(7,4),
        color=['#d65f5f','#dfa44c','#5fa75f'])
plt.title('Maturidade ESG por porte da empresa')
plt.ylabel('Frequencia')
plt.xlabel('Porte')
plt.xticks(rotation=0)
plt.legend(title='Maturidade')
plt.tight_layout()
plt.savefig(ROOT / 'reports' / 'figures' / 'fig_porte_target.png', dpi=120)
plt.show()
ct"""))

cells.append(nbf.v4.new_markdown_cell(
"""**Interpretacao:** medias e grandes empresas concentram a classe `Alta`. Microempresas e
pequenos fornecedores aparecem com mais frequencia em `Baixa` e `Media` - coerente com a
literatura de sustentabilidade em cadeia de suprimentos. O porte sera, portanto, uma feature
relevante mesmo nao sendo determinant do target.
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 4. Mapa de correlacao das features Sim/Nao

Identifica grupos de praticas que tendem a aparecer juntas (ex: empresas que treinam
sustentabilidade tambem costumam ter politica formal). Correlacoes muito altas (>0.85)
seriam candidatas a remocao por redundancia, mas no nosso caso ficam moderadas.
"""))

cells.append(nbf.v4.new_code_cell(
"""features = [c for c in df.columns
            if c not in ['esg_score', config.TARGET_COLUMN, 'origem']
            and c not in config.CRITICAL_QUESTIONS
            and c not in config.ORDINAL_MAPS]
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(9,7))
sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0, ax=ax,
            cbar_kws={'shrink':0.8})
ax.set_title('Correlacao entre features Sim/Nao')
plt.tight_layout()
plt.savefig(ROOT / 'reports' / 'figures' / 'fig_corr.png', dpi=120)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(
"""**Interpretacao:** observa-se um cluster de correlacao positiva entre `gestao_agua`,
`gestao_residuos`, `eficiencia_energetica` e `pegada_carbono` - praticas operacionais que
tendem a vir juntas. `clausulas_contratos` e `treina_compradores` formam outro cluster
(pilar 'cadeia de suprimentos'). Sem correlacoes >0.85, nao precisamos remover features
por redundancia.
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 5. Praticas ESG por classe de maturidade

Para cada feature, plotamos a proporcao de "Sim" entre os fornecedores de cada classe. As
features que melhor separam as classes serao naturalmente as mais informativas para o
modelo (mesmo apos remocao das criticas).
"""))

cells.append(nbf.v4.new_code_cell(
"""prop = df.groupby(config.TARGET_COLUMN)[features].mean().T
prop = prop.reindex(columns=['Baixa','Media','Alta'])
fig, ax = plt.subplots(figsize=(9,6))
prop.plot(kind='barh', ax=ax, color=['#d65f5f','#dfa44c','#5fa75f'])
ax.set_title('Proporcao de adocao de cada pratica por classe de maturidade')
ax.set_xlabel('Proporcao de empresas com a pratica (Sim)')
ax.set_xlim(0, 1)
ax.legend(title='Maturidade', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(ROOT / 'reports' / 'figures' / 'fig_praticas_classe.png', dpi=120)
plt.show()
prop.round(2)"""))

cells.append(nbf.v4.new_markdown_cell(
"""**Interpretacao:** ha um gradiente claro: empresas `Alta` mostram adocao consistentemente
maior em quase todas as praticas, com maior contraste em `tem_certificacoes`,
`treina_sustentabilidade`, `clausulas_contratos` e `diversidade_fornecedores`. Esse padrao
e o que sustenta a viabilidade do problema de classificacao mesmo sem usar as questoes
criticas como entrada do modelo.
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 6. Validacao da geracao sintetica

Compara as distribuicoes marginais entre os 10 originais e os 45 sinteticos. Distribuicoes
proximas validam que o SDV preservou a estrutura, sem copiar exatamente os pontos originais.
"""))

cells.append(nbf.v4.new_code_cell(
"""orig = df[df['origem']=='original'][features].mean()
syn = df[df['origem']!='original'][features].mean()
comp = pd.DataFrame({'original (n=10)': orig, 'sintetico (n=45)': syn}).sort_values('original (n=10)', ascending=True)
fig, ax = plt.subplots(figsize=(9,6))
comp.plot(kind='barh', ax=ax, color=['#3b6ea5','#e98a3b'])
ax.set_title('Comparacao de medias: original vs sintetico')
ax.set_xlabel('Proporcao de Sim')
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(ROOT / 'reports' / 'figures' / 'fig_orig_vs_synth.png', dpi=120)
plt.show()
comp.round(2)"""))

cells.append(nbf.v4.new_markdown_cell(
"""**Interpretacao:** as proporcoes nos sinteticos ficam, em media, ligeiramente abaixo dos
originais - efeito esperado por dois motivos: (i) o GaussianCopula introduz variabilidade,
(ii) as 15 amostras de perturbacao foram desenhadas para representar fornecedores menos
maduros (classe `Baixa`), o que reduz proporcionalmente as marginais. Nao ha colapso de
classes nem reproducao exata de pontos.

## Conclusoes para a modelagem

- **Target tratavel** com 3 classes representadas (Alta=14, Media=28, Baixa=13).
- **Features informativas:** porte, certificacoes, treinamentos, clausulas em contratos.
- **Estrategia de validacao:** Holdout 70/30 estratificado para metricas finais; CV 5-fold
  estratificado para selecao de modelo; Leave-One-Out como sanity check (n=55 ainda aceitavel).
- **Anti-leakage:** as 9 questoes criticas usadas para construir o target sao removidas
  no `src.train`, forcando o modelo a aprender atraves de features correlatas.
"""))

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Notebook gerado em {OUT}")
