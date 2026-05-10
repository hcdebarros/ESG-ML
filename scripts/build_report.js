// Gera relatorio SBC (.docx) do projeto ESG Fornecedores.
// Uso: node scripts/build_report.js
const fs = require('fs');
const path = require('path');

const NODE_MODULES = '/sessions/fervent-adoring-edison/.local/lib/node/node_modules';
const docx = require(path.join(NODE_MODULES, 'docx'));
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, Footer
} = docx;

const ROOT = path.resolve(__dirname, '..');
const FIG = path.join(ROOT, 'reports', 'figures');
const OUT = path.join(ROOT, 'reports', 'relatorio_sbc.docx');

const FONT = "Times New Roman";

function txt(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: 22, ...opts });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    children: Array.isArray(text) ? text : [txt(text)],
    ...opts,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, font: FONT, size: 28, bold: true })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, font: FONT, size: 24, bold: true })],
  });
}

function center(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [new TextRun({ text, font: FONT, ...opts })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { after: 60 },
    alignment: AlignmentType.JUSTIFIED,
    children: [txt(text)],
  });
}

function img(filename, w = 480, h = 320) {
  const fullPath = path.join(FIG, filename);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 80 },
    children: [new ImageRun({
      type: filename.endsWith('.jpg') ? 'jpg' : 'png',
      data: fs.readFileSync(fullPath),
      transformation: { width: w, height: h },
      altText: { title: filename, description: filename, name: filename },
    })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text, font: FONT, size: 18, italics: true })],
  });
}

// =========================================================================
// TABELA: leaderboard
// =========================================================================
const leaderRaw = fs.readFileSync(path.join(ROOT, 'reports', 'leaderboard.csv'), 'utf-8');
const leaderRows = leaderRaw.trim().split('\n').map(r => r.split(','));
const leaderHeader = leaderRows[0];
const leaderData = leaderRows.slice(1);
const colsToShow = ['modelo', 'holdout_f1_macro', 'cv5_f1_macro_mean', 'cv5_f1_macro_std', 'loo_accuracy'];
const colIdx = colsToShow.map(c => leaderHeader.indexOf(c));

function makeLeaderTable() {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "AAAAAA" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const headerLabels = ['Modelo', 'F1 macro (holdout)', 'F1 macro (CV5 medio)', 'F1 macro (CV5 desvio)', 'Acuracia (LOO)'];

  const w = 9000;
  const cw = [2000, 1900, 1900, 1700, 1500];

  const headerRow = new TableRow({
    tableHeader: true,
    children: headerLabels.map((label, i) => new TableCell({
      borders,
      width: { size: cw[i], type: WidthType.DXA },
      shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: label, font: FONT, size: 20, bold: true })],
      })],
    })),
  });

  const dataRows = leaderData.map(row => new TableRow({
    children: colIdx.map((idx, i) => {
      const val = row[idx];
      const display = (i === 0) ? val : Number(val).toFixed(3);
      return new TableCell({
        borders,
        width: { size: cw[i], type: WidthType.DXA },
        margins: { top: 60, bottom: 60, left: 120, right: 120 },
        children: [new Paragraph({
          alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          children: [new TextRun({ text: display, font: FONT, size: 20 })],
        })],
      });
    }),
  }));

  return new Table({
    width: { size: w, type: WidthType.DXA },
    columnWidths: cw,
    rows: [headerRow, ...dataRows],
  });
}

// =========================================================================
// MONTA O DOCUMENTO
// =========================================================================
const children = [];

// Titulo
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [new TextRun({
    text: "Classificacao de Maturidade ESG de Fornecedores: uma solucao de Machine Learning com geracao sintetica de dados e MLOps",
    font: FONT, size: 32, bold: true,
  })],
}));

// Autores
children.push(center("Helder Barros, [Membro 2], [Membro 3], [Membro 4]", { size: 22 }));
children.push(center("CESAR School - Machine Learning I + Projeto 3", { size: 22 }));
children.push(center("Recife, PE - Brasil", { size: 22 }));
children.push(center("{helder.barros, membro2, membro3, membro4}@cesar.school.br", { size: 22 }));
children.push(new Paragraph({ children: [new TextRun({ text: "" })] }));

// Resumo
children.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 120 },
  children: [
    new TextRun({ text: "Resumo. ", font: FONT, size: 22, bold: true }),
    new TextRun({
      text: "Este trabalho apresenta uma solucao completa de Machine Learning para classificar fornecedores em tres niveis de maturidade socioambiental (Baixa, Media, Alta) a partir de respostas a um questionario ESG. Dado o tamanho reduzido do conjunto original (10 registros), 45 amostras sinteticas foram geradas combinando o algoritmo GaussianCopula da biblioteca SDV (30 amostras) com perturbacao controlada (15 amostras) para representar a classe minoritaria. Quatro classificadores foram comparados (KNN, Decision Tree, Random Forest e Regressao Logistica) sob tres estrategias de validacao (Holdout estratificado, StratifiedKFold com k=5 e Leave-One-Out). Os experimentos foram rastreados com MLflow, e a solucao final e disponibilizada em um dashboard interativo Streamlit, sendo toda a stack reproduzida via Docker.",
      font: FONT, size: 22,
    }),
  ],
}));

children.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
  children: [
    new TextRun({ text: "Abstract. ", font: FONT, size: 22, bold: true, italics: true }),
    new TextRun({
      text: "This work presents an end-to-end Machine Learning solution to classify suppliers into three levels of socio-environmental maturity (Low, Medium, High) from ESG questionnaire answers. Given the small original dataset (10 records), 45 synthetic samples were generated by combining the GaussianCopula algorithm (30 samples) with controlled perturbation (15 samples) to represent the minority class. Four classifiers (KNN, Decision Tree, Random Forest, Logistic Regression) were compared under three validation strategies (stratified Holdout, StratifiedKFold k=5 and Leave-One-Out). Experiments were tracked with MLflow and the final solution is delivered as an interactive Streamlit dashboard, with the whole stack reproduced via Docker.",
      font: FONT, size: 22, italics: true,
    }),
  ],
}));

// 1. Introducao
children.push(h1("1. Introducao"));
children.push(p("A avaliacao da maturidade ambiental, social e de governanca (ESG) de fornecedores e um tema de crescente relevancia para grandes empresas que precisam mitigar riscos ao longo da cadeia de suprimentos. Tradicionalmente, esse processo depende de auditorias manuais e julgamento humano sobre questionarios extensos, o que limita a escala e a consistencia da classificacao."));
children.push(p("Este trabalho propoe uma solucao automatizada baseada em Machine Learning que recebe as respostas de um questionario ESG e atribui ao fornecedor uma classe de maturidade (Baixa, Media ou Alta). O escopo cobre todo o ciclo da disciplina Machine Learning I (CESAR School): leitura e analise exploratoria dos dados, tratamento, construcao de target, geracao de dados sinteticos via algoritmo de ML, treinamento e comparacao de modelos, rastreamento de experimentos com MLflow, dashboard interativo e conteinerizacao via Docker."));
children.push(p("O dataset de partida foi construido a partir de respostas reais (anonimizadas) de 10 fornecedores. Como esse volume e insuficiente para treinar e validar modelos com seguranca, uma etapa central da solucao e a geracao de 45 amostras sinteticas, totalizando 55 registros. A geracao combina dois algoritmos: o GaussianCopulaSynthesizer da biblioteca SDV, que aprende as distribuicoes marginais e dependencias dos dados originais, e uma rotina de perturbacao controlada que cria fornecedores menos maduros para garantir representacao da classe minoritaria."));

// 2. Dados e EDA
children.push(h1("2. Dados e Analise Exploratoria"));
children.push(p("O dataset original possui 50 colunas, sendo 10 metadados (identificadores, datas, e-mail, nome) e 40 questoes do questionario ESG. As variaveis de identificacao pessoal (PII), como CNPJ, e-mail, razao social e nome do respondente, foram removidas no preprocessamento (src/preprocess.py). Campos de texto livre tambem foram descartados, conforme decisao metodologica de focar nas variaveis estruturadas com sinal claro."));
children.push(p("Apos a limpeza, restaram 24 features de interesse. As respostas Sim/Nao foram codificadas em 1/0; \"Nao se aplica\" tambem foi mapeado para 0. As variaveis ordinais porte (Microempresa a Grande porte) e faturamento (R$ 500 mil a Acima de R$ 3 mi) foram mapeadas em escala inteira preservando a ordem natural."));

children.push(h2("2.1 Construcao do target sem leakage"));
children.push(p("O target maturidade_esg e construido somando-se a quantidade de respostas \"Sim\" entre 9 questoes consideradas criticas pela equipe (compromisso com trabalho digno, politica formal de responsabilidade socioambiental, compliance, levantamento de aspectos e impactos, diversidade de colaboradores, politica de compras sustentaveis, criterio para fornecedores, auditoria de fornecedores e processos para legislacao ambiental). O score esg_score (0 a 9) e binarizado em tres classes: Baixa (0-3), Media (4-6) e Alta (7-9)."));
children.push(p("Para evitar data leakage, essas 9 questoes criticas sao removidas das features de treino em src/train.py. Isso obriga o modelo a inferir a maturidade ESG a partir das praticas operacionais correlatas (porte, certificacoes, treinamentos, gestao de agua/energia/residuos, clausulas em contratos, diversidade na cadeia, etc.), aproximando o cenario de aplicacao real, em que tais perguntas criticas podem nao estar disponiveis para todos os fornecedores."));

children.push(h2("2.2 Visualizacoes-chave"));
children.push(p("A Figura 1 mostra a distribuicao final do target apos a geracao sintetica. A classe Media concentra a maioria dos casos, com Baixa e Alta nas caudas. O desbalanceamento e moderado e foi tratado via metricas macro (f1_macro) e estratificacao no Holdout e na validacao cruzada."));
children.push(img('fig_target.png', 360, 240));
children.push(caption("Figura 1. Distribuicao da maturidade ESG no dataset final (n=55)."));

children.push(p("A Figura 2 cruza o porte da empresa com a classe alvo. Empresas medias e grandes concentram a classe Alta, enquanto micro e pequenas aparecem com mais frequencia em Baixa e Media - resultado coerente com a literatura de sustentabilidade em cadeia de suprimentos, em que organizacoes maiores tendem a ter mais recursos para programas formais e certificacoes."));
children.push(img('fig_porte_target.png', 420, 260));
children.push(caption("Figura 2. Maturidade ESG por porte da empresa."));

children.push(p("A Figura 3 apresenta a correlacao entre as 15 features que efetivamente alimentam os modelos. Observa-se um cluster operacional (gestao de agua, residuos, energia e pegada de carbono) e um cluster de cadeia de suprimentos (clausulas em contratos, treinamento de compradores, diversidade na cadeia). Nenhuma correlacao acima de 0.85 foi observada, justificando a manutencao de todas as features."));
children.push(img('fig_corr.png', 460, 360));
children.push(caption("Figura 3. Mapa de correlacao das features Sim/Nao."));

children.push(p("A Figura 4 plota a proporcao de adocao de cada pratica por classe de maturidade. Ha um gradiente claro: empresas Alta apresentam adocao consistentemente maior em quase todas as praticas, com maior contraste em tem_certificacoes, treina_sustentabilidade, clausulas_contratos e diversidade_fornecedores. Esse padrao sustenta a viabilidade do problema de classificacao mesmo sem usar as questoes criticas como entrada."));
children.push(img('fig_praticas_classe.png', 460, 320));
children.push(caption("Figura 4. Proporcao de adocao por pratica e classe de maturidade."));

// 3. Geracao de dados sinteticos
children.push(h1("3. Geracao de Dados Sinteticos"));
children.push(p("Com apenas 10 registros originais, treinar e validar um classificador de tres classes seria pouco confiavel. A solucao adotada foi gerar 45 amostras sinteticas combinando dois mecanismos complementares."));

children.push(h2("3.1 GaussianCopula (SDV)"));
children.push(p("O GaussianCopulaSynthesizer da biblioteca SDV (Synthetic Data Vault) modela cada variavel marginalmente e captura as dependencias entre elas atraves de uma copula gaussiana. Apos ajuste sobre os 10 registros originais, foram amostradas 30 instancias sinteticas. A biblioteca foi escolhida por trabalhar bem com dados tabulares mistos (categoricos + ordinais), ser interpretavel e nao exigir treinamento intensivo como redes neurais (CTGAN), o que e adequado para um conjunto base muito pequeno."));

children.push(h2("3.2 Perturbacao controlada"));
children.push(p("Como os 10 registros originais sao todos de fornecedores razoavelmente maduros (score >= 4), o GaussianCopula tende a reproduzir o mesmo perfil. Para garantir representacao da classe Baixa, foram geradas 15 amostras adicionais por perturbacao: cada amostra parte de uma copia aleatoria de um registro original, tem K (4 a 7) das 9 questoes criticas invertidas para Nao, e recebe ruido leve (5% de chance de flip) nas demais features Sim/Nao. Essa estrategia simula fornecedores com menor estruturacao ESG mantendo coerencia com o vocabulario do questionario original."));

children.push(h2("3.3 Validacao da geracao"));
children.push(p("A Figura 5 compara as proporcoes medias de \"Sim\" entre os 10 originais e os 45 sinteticos. As marginais sinteticas ficam, em media, ligeiramente abaixo das originais - efeito esperado dado que (i) o GaussianCopula introduz variabilidade, e (ii) as 15 amostras perturbadas representam fornecedores menos maduros, reduzindo proporcionalmente as marginais. Nao ha colapso de classes nem reproducao exata de pontos, e a distribuicao final do target tem 14 amostras Alta, 28 Media e 13 Baixa."));
children.push(img('fig_orig_vs_synth.png', 460, 320));
children.push(caption("Figura 5. Comparacao de medias por feature: original vs sintetico."));

// 4. Modelagem
children.push(h1("4. Modelagem e Validacao"));
children.push(h2("4.1 Modelos avaliados"));
children.push(p("Quatro modelos foram comparados, todos encapsulados em pipelines do scikit-learn:"));
children.push(bullet("KNN (k=5, weights=distance) com StandardScaler. Captura padroes locais e e simples de interpretar."));
children.push(bullet("Decision Tree (max_depth=5, min_samples_split=4). Baseline interpretavel."));
children.push(bullet("Random Forest (200 arvores, max_depth=8). Reduz variancia da arvore unica e expoe importancia de features."));
children.push(bullet("Logistic Regression (C=1.0, max_iter=2000) com StandardScaler. Modelo linear como referencia, importante quando o sinal e quase aditivo."));

children.push(h2("4.2 Estrategias de validacao"));
children.push(p("Tres estrategias foram aplicadas a cada modelo:"));
children.push(bullet("Holdout estratificado 70/30: separa um conjunto fixo de teste (n=17 nas amostras finais) para metricas reportaveis e geracao de matrizes de confusao."));
children.push(bullet("StratifiedKFold (k=5): usado como criterio de selecao de modelo, com a media e o desvio do f1_macro entre folds."));
children.push(bullet("Leave-One-Out (LOO): com n=55 ainda viavel computacionalmente, fornece um sanity check pessimista (alta variancia entre folds), util pelo tamanho pequeno do dataset."));
children.push(p("A escolha do f1_macro como metrica principal se justifica pelo desbalanceamento das classes (Media tem mais que o dobro das outras): metricas macro evitam que o modelo otimize apenas a classe majoritaria. Holdout foi escolhido como reportavel, mas decisoes de modelo se basearam na CV5 (mais robusta a sorteios)."));

children.push(h2("4.3 Anti-leakage"));
children.push(p("Conforme detalhado em 2.1, as 9 questoes criticas usadas para construir o target sao removidas das features no momento do treino. O modelo aprende, portanto, a partir de 15 features (porte, faturamento, sancao_admin, tem_certificacoes, treina_sustentabilidade, gestao_agua, eficiencia_energetica, gestao_residuos, pegada_carbono, auditoria_gee, voluntariado, saude_mental, clausulas_contratos, treina_compradores, diversidade_fornecedores)."));

// 5. MLOps
children.push(h1("5. MLOps - MLflow e Reprodutibilidade"));
children.push(p("Todos os experimentos sao rastreados com MLflow. Para cada modelo, e registrado:"));
children.push(bullet("Parametros: hiperparametros do classificador e dimensao da matriz de features."));
children.push(bullet("Metricas: accuracy, f1_macro, precision_macro, recall_macro nos tres regimes de validacao (holdout, cv5, loo)."));
children.push(bullet("Artefatos: matriz de confusao em PNG e classification_report em texto."));
children.push(bullet("Modelo: o modelo treinado no dataset completo e salvo via mlflow.sklearn.log_model, permitindo reproducao sem reexecutar o treino."));

children.push(p("O backend do MLflow e um diretorio local (mlruns/), e a UI pode ser aberta em http://localhost:5000 com mlflow ui --backend-store-uri ./mlruns. O melhor modelo (criterio cv5_f1_macro_mean) e adicionalmente serializado em models/best_model.joblib junto com models/feature_columns.json contendo nome do modelo, ordem das features e classes - garantindo deploy seguro do dashboard."));

children.push(p("A reprodutibilidade e fechada com Docker. O Dockerfile constroi uma imagem Python 3.10-slim com todas as dependencias listadas em requirements.txt (pandas, scikit-learn, sdv, mlflow, streamlit, matplotlib, etc.). O docker-compose.yml orquestra tres servicos: pipeline (executa preprocess + synthesize + train), mlflow (UI na porta 5000) e dashboard (Streamlit na porta 8501). Um simples docker compose up --build entrega a solucao completa em ambiente isolado."));

// 6. Resultados
children.push(h1("6. Resultados"));
children.push(p("A Tabela 1 apresenta o leaderboard final, ordenado por f1_macro medio na CV de 5 folds."));
children.push(makeLeaderTable());
children.push(caption("Tabela 1. Leaderboard dos modelos avaliados."));

children.push(p("A Regressao Logistica liderou no f1_macro de CV (0.454), seguida do KNN (0.429). O Random Forest obteve a melhor acuracia em LOO (0.527), mas com f1_macro inferior - sinal de que tende a privilegiar a classe majoritaria. A Arvore de Decisao apresentou desempenho mais fraco e maior variancia, esperado dado o tamanho do dataset e a presenca de classes minoritarias."));

children.push(p("Os valores absolutos modestos (f1_macro entre 0.30 e 0.45) sao consistentes com o cenario do problema: 55 amostras, tres classes e remocao deliberada das 9 features que constroem o target. A intencao do projeto e mostrar pipeline e metodologia, nao otimizar para um leaderboard inflado por leakage. As matrizes de confusao por modelo (registradas como artefatos no MLflow) confirmam que os erros se concentram na fronteira Media/Alta - o que tambem e plausivel: ha sobreposicao de praticas operacionais entre fornecedores intermediarios e maduros."));

children.push(h2("6.1 Dashboard interativo"));
children.push(p("O dashboard Streamlit (app/dashboard.py) expoe a solucao para o usuario final em cinco abas: (i) Visao Geral com a distribuicao do dataset e a composicao por origem; (ii) Comparacao de Modelos com o leaderboard, comparacao visual de metricas e matrizes de confusao; (iii) Simulador de Previsao, em que o usuario insere as respostas de um novo fornecedor e recebe a classe predita junto com as probabilidades por classe; (iv) Importancia das Features, que apresenta feature_importances_ ou |coef| medio conforme o melhor modelo; e (v) Sobre, com a rastreabilidade do projeto. O dashboard carrega o melhor modelo via models/best_model.joblib e respeita exatamente a ordem de features de feature_columns.json."));

// 7. Consideracoes finais
children.push(h1("7. Consideracoes Finais"));
children.push(p("A solucao entregue cobre os requisitos da especificacao do projeto: leitura e EDA estruturadas, no minimo cinco visualizacoes interpretadas, multiplas estrategias de validacao com justificativa, treinamento de quatro modelos (incluindo KNN, Decision Tree e Random Forest exigidos), tracking de experimentos com MLflow, dashboard interativo e ambiente containerizado."));
children.push(p("O ponto metodologicamente mais delicado foi lidar com o tamanho extremamente reduzido do dataset original. A combinacao de GaussianCopula com perturbacao controlada se mostrou eficaz para criar diversidade, balancear o target e preservar a coerencia do questionario. Em trabalhos futuros, valeria a pena: (i) coletar mais respostas reais para validar empiricamente os modelos contra dados nao sinteticos; (ii) avaliar synthesizers baseados em redes (CTGAN, TVAE) com transferencia de aprendizado; e (iii) incluir analise de fairness verificando se a classificacao desfavorece sistematicamente fornecedores menores ou de regioes especificas."));

// Referencias
children.push(h1("Referencias"));
children.push(p("Patki, N., Wedge, R., Veeramachaneni, K. (2016). The Synthetic Data Vault. IEEE International Conference on Data Science and Advanced Analytics."));
children.push(p("Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830."));
children.push(p("Zaharia, M. et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow. IEEE Data Engineering Bulletin."));
children.push(p("Streamlit Inc. (2024). Streamlit Documentation. https://docs.streamlit.io"));
children.push(p("Docker Inc. (2024). Docker Compose Documentation. https://docs.docker.com/compose/"));

// =========================================================================
// DOCUMENTO
// =========================================================================
const doc = new Document({
  creator: "Claude (CESAR School ML I + P3)",
  title: "Relatorio SBC - ESG Fornecedores",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "Pagina ", font: FONT, size: 18 }),
          new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 }),
        ],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("Relatorio gerado em", OUT, "(", buf.length, "bytes )");
});
