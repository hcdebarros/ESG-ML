# ESG Score - Maturidade Socioambiental de Fornecedores

> Solucao completa de Machine Learning para classificar fornecedores em
> **Baixa / Media / Alta** maturidade ESG, a partir de respostas de
> questionario socioambiental.

## Equipe

| Nome | GitHub |
|------|--------|
| Helder Barros | @___ |
| _(membro 2)_  | @___ |
| _(membro 3)_  | @___ |
| _(membro 4)_  | @___ |

> Substituir os placeholders pelos nomes e usuarios reais do GitHub antes da entrega.

## Disciplina e instituicao

- **Disciplina:** Machine Learning I
- **Projeto:** Projeto 3 (integracao com a disciplina)
- **Instituicao:** CESAR School
- **Periodo:** 2026.1
- **Unidade:** 2a (AV2)

## Solucao

A solucao constroi um classificador supervisionado que prediz a **maturidade
ESG** de um fornecedor (Baixa / Media / Alta) a partir de praticas
operacionais declaradas no questionario (porte, faturamento, certificacoes,
treinamentos, gestao de agua/energia/residuos, diversidade, clausulas em
contratos, etc.). O target e construido a partir de **9 questoes criticas**
(compliance, trabalho digno, politica ESG formal, compras sustentaveis, etc.)
que sao removidas das features de treino para evitar leakage; o modelo
aprende, portanto, a inferir maturidade atraves dos sinais correlatos.

Como o dataset original tem apenas **10 registros**, a etapa 2 do pipeline
gera **45 amostras sinteticas** combinando:
- `SDV GaussianCopulaSynthesizer` (30 amostras) - aprende a distribuicao
  conjunta dos originais e gera dados realistas;
- **Perturbacao controlada** (15 amostras) - cria fornecedores de baixa
  maturidade invertendo aleatoriamente 4-7 das questoes criticas, garantindo
  representacao da classe minoritaria.

> **Google Sites do projeto:** _(link a ser adicionado pela equipe)_

## Arquitetura

```
.
+- data/
|   +- raw/                   # XLSX original (nao versionado)
|   +- processed/             # CSV tratados (10 base, 45 sint, 55 final)
+- notebooks/
|   +- 01_eda.ipynb           # EDA com 6 visualizacoes interpretadas
+- src/
|   +- config.py              # paths, mapeamentos, parametros
|   +- preprocess.py          # tratamento + target
|   +- synthesize.py          # SDV + perturbacao
|   +- train.py               # 4 modelos + holdout/CV/LOO + MLflow
+- app/
|   +- dashboard.py           # Streamlit (5 abas)
+- mlruns/                    # tracking MLflow
+- models/                    # melhor modelo + metadata
+- reports/                   # leaderboard + figuras
+- Dockerfile
+- docker-compose.yml
+- requirements.txt
+- Makefile
+- README.md
```

## Como executar

### Opcao A: Docker (recomendado para reprodutibilidade)

```bash
docker compose up --build
```

Apos a inicializacao:
- Dashboard Streamlit: <http://localhost:8501>
- MLflow UI: <http://localhost:5000>

### Opcao B: Local (Python 3.10+)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Rodar pipeline completo (preprocess -> synth -> train)
make all
# ou separado:
python -m src.preprocess
python -m src.synthesize
python -m src.train

# 3. Subir dashboard
streamlit run app/dashboard.py
# em outro terminal:
mlflow ui --backend-store-uri ./mlruns --port 5000
```

### EDA

Para reabrir/editar o notebook:

```bash
jupyter lab notebooks/01_eda.ipynb
```

## Modelos avaliados

| Modelo | Hiperparametros principais |
|--------|---------------------------|
| KNN | n_neighbors=5, weights=distance |
| Decision Tree | max_depth=5, min_samples_split=4 |
| Random Forest | n_estimators=200, max_depth=8 |
| Logistic Regression (extra) | C=1.0, max_iter=2000 |

Cada modelo e avaliado com **3 estrategias de validacao**:
1. **Holdout** estratificado (70/30) - metricas finais reportaveis
2. **StratifiedKFold** (5-fold) - selecao de modelo por f1_macro medio
3. **Leave-One-Out** - sanity check (n=55 ainda viavel)

Metricas registradas no MLflow: accuracy, f1_macro, precision_macro,
recall_macro, alem da matriz de confusao como artefato. O leaderboard
final e gravado em `reports/leaderboard.csv` e o melhor modelo (treinado
em todo o dataset) em `models/best_model.joblib`.

## Cobertura dos requisitos

| Requisito | Onde |
|-----------|------|
| EDA + estatisticas + tratamento | `notebooks/01_eda.ipynb`, `src/preprocess.py` |
| >= 5 visualizacoes interpretadas | `notebooks/01_eda.ipynb` (6 figuras) |
| Holdout + CV + LOO + justificativa | `src/train.py` + relatorio SBC |
| KNN + DT + RF + 1 extra | `src/train.py` |
| MLflow com params/metrics/artefatos/modelo | `src/train.py` + `mlruns/` |
| Dashboard interativo com previsoes/metricas | `app/dashboard.py` |
| Containerizacao Docker + instrucoes | `Dockerfile`, `docker-compose.yml` |
| Relatorio no template SBC | `reports/relatorio_sbc.docx` |

## Notas sobre os dados

Os 10 registros originais do dataset sao **dados ficticios** criados para
modelagem (campo `Confirma que todas as informacoes...` indica isso). Nenhum
PII real e processado; ainda assim, colunas como CNPJ, email e nome do
respondente sao removidas no preprocessamento por padrao de governanca.
