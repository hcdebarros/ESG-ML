.PHONY: install preprocess synth train all dashboard mlflow docker clean

install:
	pip install -r requirements.txt

preprocess:
	python -m src.preprocess

synth:
	python -m src.synthesize

train:
	python -m src.train

all: preprocess synth train

dashboard:
	streamlit run app/dashboard.py

mlflow:
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

docker:
	docker compose up --build

clean:
	rm -rf mlruns models/best_model.joblib reports/figures/*.png reports/leaderboard.csv data/processed/*.csv
