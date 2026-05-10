# Imagem unica que serve para preprocessamento, treino, MLflow UI e Streamlit.
# Use docker-compose.yml para subir os servicos lado a lado.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias do sistema (matplotlib + libomp p/ sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Camada de dependencias - instalada em chunks para reduzir pico de memoria
# (o "Bus error" classico no Docker Desktop com WSL2 e quase sempre OOM
# durante a compilacao de scipy/numpy/sdv em paralelo).
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.13.0
RUN pip install --no-cache-dir scikit-learn==1.4.2 joblib==1.4.2 openpyxl==3.1.2
RUN pip install --no-cache-dir matplotlib==3.8.4 seaborn==0.13.2 plotly==5.22.0
RUN pip install --no-cache-dir mlflow==2.13.2
RUN pip install --no-cache-dir streamlit==1.35.0
RUN pip install --no-cache-dir sdv==1.13.1
RUN pip install --no-cache-dir pyyaml==6.0.1

# Codigo
COPY src/ ./src/
COPY app/ ./app/
COPY notebooks/ ./notebooks/
COPY data/ ./data/

# Diretorios de saida
RUN mkdir -p mlruns models reports/figures

EXPOSE 8501 5000

# Comando default: roda o pipeline completo e sobe o Streamlit.
# Servicos especificos sao orquestrados pelo docker-compose.yml.
CMD ["bash", "-lc", "python -m src.preprocess && python -m src.synthesize && python -m src.train && streamlit run app/dashboard.py --server.address=0.0.0.0 --server.port=8501"]
