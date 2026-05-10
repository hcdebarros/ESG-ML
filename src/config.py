"""Configuracoes centrais do projeto.

Centraliza paths, nomes de colunas, listas de colunas criticas e parametros
para que os scripts (`preprocess`, `synthesize`, `train`, `predict`) e o
dashboard usem a mesma fonte de verdade.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_FILE = RAW_DIR / "fornecedores.xlsx"
PROCESSED_BASE = PROCESSED_DIR / "fornecedores_base.csv"        # 10 registros tratados
PROCESSED_SYNTH = PROCESSED_DIR / "fornecedores_sinteticos.csv"  # 45 registros gerados
PROCESSED_FULL = PROCESSED_DIR / "fornecedores_55.csv"          # 10 + 45 finais

# ----------------------------------------------------------------------
# Mapeamentos de colunas
# ----------------------------------------------------------------------
# Colunas de identificacao / PII a remover
PII_COLUMNS = [
    "ID",
    "Hora de início",
    "Hora de conclusão",
    "Hora da última modificação",
    "Email",
    "Nome",
    "Qual é a razão social e o nome fantasia da sua empresa?",
    "Qual é o CNPJ da empresa?",
    "Informe seu nome completo:",
    "Qual é o seu e-mail?",
]

# Campos de texto livre a descartar (definicao do usuario)
TEXT_FREE_COLUMNS = [
    "Discorra brevemente sobre o motivo da sanção:",
    "Favor especificar abaixo e enviar evidências conforme instruções.",
    "Explique abaixo as áreas abrangidas, os temas que são abordados e a periodicidade dos treinamentos:",
    "Explique quais são essas práticas:",
    "Explique abaixo sobre essas práticas:",
    "Explique abaixo sobre essas práticas:2",
    "Explique abaixo sobre essas práticas:3",
    "Explique abaixo sobre essas práticas:4",
    "Quais combustíveis são utilizados nos veículos da empresa? Escreve abaixo a percentagem média de utilização dos combustíveis, caso seja só um, escreva o nome do combustível e 100%. Exemplo: (Diese...",
    "Detalhe as práticas estruturadas implementadas pela empresa para garantir inclusão e diversidade entre colaboradores.",
    "Descreva as ações ou programas de voluntariado realizados com a comunidade.",
    "Informe as\xa0ações que sua empresa realiza para cuidar da saúde mental e bem-estar dos colaboradores.",
    "Se sim, escreva qual(is) critério(s) utilizado(s):",
    "Você gostaria de nos contar mais alguma coisa sobre sua empresa ou fazer algum comentário final? ",
    "Confirma que todas as informações fornecidas estão corretas?",
    "Qual cargo ou função atual na empresa?",
]

# Mapeamento "amigavel" coluna_original -> short name (para facilitar leitura/uso)
COLUMN_RENAME = {
    "Quantos colaboradores a empresa possui atualmente?": "porte",
    "Qual é o faturamento médio anual da empresa?": "faturamento",
    "Existem processos internos para garantir o cumprimento da legislação ambiental e de segurança do trabalho?": "processos_legislacao",
    "Nos últimos cinco anos, a empresa recebeu alguma sanção administrativa de natureza ambiental e de segurança do trabalho?": "sancao_admin",
    "A empresa possui compromisso formal com o trabalho digno, incluindo a não utilização de trabalho infantil, forçado ou análogo à escravidão, e o cumprimento integral da legislação trabalhista vigente?": "compromisso_trabalho_digno",
    "A empresa possui certificações relacionadas à gestão ambiental, segurança do trabalho ou conformidade legal (ex: ISO 14001, ISO 45001, OHSAS 18001, FSC, entre outras)?": "tem_certificacoes",
    "A empresa possui uma política formal implementada relacionada à responsabilidade sociolambiental (como política ambiental, de sustentabilidade ou equivalente)?": "politica_socioambiental",
    "A empresa realiza treinamentos sobre temas de sustentabilidade com seus colaboradores? ": "treina_sustentabilidade",
    "A empresa possui práticas formais para garantir o compliance do negócio, como políticas anticorrupção, cumprimento de normas legais e promoção da transparência?": "compliance_formal",
    "A empresa realiza o levantamento de seus aspectos e impactos ambientais e estabelece controle para eles?": "aspectos_impactos",
    "A empresa possui práticas em sua operação para gestão e uso responsável de água? ": "gestao_agua",
    "A empresa possui práticas em sua operação para gestão e eficiência energética?": "eficiencia_energetica",
    "A empresa possui práticas em sua operação para gerenciamento de resíduos?": "gestao_residuos",
    "A empresa possui práticas implementadas para calcular ou estimar a pegada de carbono de seus produtos e serviços?": "pegada_carbono",
    "Caso tenha inventário de GEE, o mesmo é auditado por empresa terceira parte?": "auditoria_gee",
    "A empresa possui práticas estruturadas para promover a inclusão e a diversidade entre seus colaboradores (ex.: equidade de gênero, raça, pessoas com deficiência, LGBTQIA+, etc)?": "diversidade_colaboradores",
    "A empresa realiza ações ou programas de voluntariado com envolvimento da comunidade local?": "voluntariado",
    "A empresa possui programas ou práticas estruturadas voltadas à saúde mental e bem-estar emocional dos colaboradores (ex.: apoio psicológico, campanhas internas, parcerias com profissionais ou plat...": "saude_mental",
    "Sua empresa possui uma política formal de Compras Sustentáveis, separada do Código de Conduta para Fornecedores, que inclua objetivos qualitativos e metas quantitativas para questões ambientais e ...": "politica_compras_sustentaveis",
    "A empresa possui algum critério socioambiental estabelecido para selecionar sua cadeia de fornecedores?": "criterio_fornecedores",
    "Os contratos firmados com fornecedores incluem cláusulas específicas sobre requisitos ambientais, sociais, trabalhistas e de direitos humanos?": "clausulas_contratos",
    "Os profissionais de compras da sua empresa recebem treinamentos regulares sobre temas de sustentabilidade aplicados à cadeia de suprimentos?": "treina_compradores",
    "A empresa possui um programa ou práticas estruturadas para promover a diversidade na sua cadeia de fornecedores?": "diversidade_fornecedores",
    "Sua empresa realiza auditorias (presenciais ou remotas) nos fornecedores para verificar o cumprimento de requisitos de sustentabilidade?": "auditoria_fornecedores",
}

# ----------------------------------------------------------------------
# Definicao do TARGET (sem leakage)
# ----------------------------------------------------------------------
# Questoes "criticas" usadas APENAS para construir o target.
# Sao removidas das features para o modelo nao prever de forma deterministica.
CRITICAL_QUESTIONS = [
    "processos_legislacao",
    "compromisso_trabalho_digno",
    "politica_socioambiental",
    "compliance_formal",
    "aspectos_impactos",
    "diversidade_colaboradores",
    "politica_compras_sustentaveis",
    "criterio_fornecedores",
    "auditoria_fornecedores",
]

TARGET_COLUMN = "maturidade_esg"
TARGET_LABELS = ["Baixa", "Media", "Alta"]

# Bins do score (numero de "Sim" entre as 9 questoes criticas, 0..9)
# 0-3 -> Baixa | 4-6 -> Media | 7-9 -> Alta
TARGET_BINS = [-0.5, 3.5, 6.5, 9.5]

# ----------------------------------------------------------------------
# Tratamento de respostas
# ----------------------------------------------------------------------
# Mapeamento de respostas Sim/Nao
YES_NO_MAP = {
    "Sim": 1,
    "Sim (por favor, envie as evidências conforme orientações fornecidas)": 1,
    "Não": 0,
    "Não se aplica": 0,
}

# Colunas que sao categoricas multi-classe (porte, faturamento)
ORDINAL_MAPS = {
    "porte": {
        "Até 9 (Microempresa)": 0,
        "10 a 99 (Pequeno porte)": 1,
        "100 a 999 (Médio porte)": 2,
        "1.000 ou mais (Grande porte)": 3,
    },
    "faturamento": {
        "De R$ 500 mil a R$ 1,5 milhões": 0,
        "De R$ 1,5 milhões a R$ 3 milhões": 1,
        "Acima de R$ 3 milhões": 2,
        "Prefere não informar": -1,
    },
}

# ----------------------------------------------------------------------
# Parametros de geracao sintetica
# ----------------------------------------------------------------------
N_BASE_RECORDS = 10
N_SYNTHETIC = 45
N_TOTAL = N_BASE_RECORDS + N_SYNTHETIC  # 55
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# MLflow
# ----------------------------------------------------------------------
MLFLOW_TRACKING_URI = f"file:{(MLRUNS_DIR).as_posix()}"
MLFLOW_EXPERIMENT = "esg_fornecedores"
