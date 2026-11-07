"""Estágio `train`: treina o modelo, registra o experimento no MLflow e
cria uma nova versão no Model Registry.

A promoção (alias @champion) NÃO é automática: é uma decisão de governança.

Validação temporal: os últimos `validation_days` dias do treino são separados
para medir o modelo ANTES de olhar o teste (metrics_val.json). É nessa métrica
que se comparam hiperparâmetros. Depois, o modelo final é retreinado com o
treino inteiro, para não desperdiçar os dados mais recentes.
"""
import json
import subprocess
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from bike import MODEL_NAME, regression_metrics, time_split, tracking_uri, xy

params = yaml.safe_load(open("params.yaml"))
hp = params["train"]


def data_version() -> str:
    """Hash do dado bruto, lido do arquivo .dvc (liga o modelo à versão dos dados)."""
    dvc_file = Path("data/raw/bike.csv.dvc")
    if not dvc_file.exists():
        return "nao-versionado"
    return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_commit() -> str:
    return git("rev-parse", "--short", "HEAD") or "desconhecido"


def git_dirty() -> bool:
    """True se há arquivos rastreados alterados e ainda não commitados.

    No fluxo `dvc repro` -> `git commit`, o treino roda ANTES do commit: o HEAD
    ainda é o commit anterior (com outros parâmetros/código). Com git_dirty=true,
    o git_commit indica só a BASE; o retrato fiel do treino é o commit seguinte,
    que contém o dvc.lock com os hashes de tudo que foi usado.

    Só olha o que afeta o modelo (código, parâmetros, pipeline e ponteiros de dados):
    notebooks abertos ou o dvc.lock (que o próprio `dvc repro` reescreve) não contam.
    """
    relevantes = ["src", "params.yaml", "dvc.yaml", "data"]
    return bool(git("status", "--porcelain", "--untracked-files=no", "--", *relevantes))


train = pd.read_csv("data/split/train.csv", parse_dates=["dteday"])
X, y = xy(train)

model = HistGradientBoostingRegressor(
    learning_rate=hp["learning_rate"],
    max_iter=hp["max_iter"],
    max_leaf_nodes=hp["max_leaf_nodes"],
    random_state=hp["random_state"],
    early_stopping=hp["early_stopping"],
)

# --- Validação temporal (mesma lógica do split treino/teste, aplicada ao treino) ---
fit_part, val_part = time_split(train, test_days=hp["validation_days"])
X_fit, y_fit = xy(fit_part)
X_val, y_val = xy(val_part)
val_model = clone(model).fit(X_fit, y_fit)
val_metrics = regression_metrics(y_val, val_model.predict(X_val))

mlflow.set_tracking_uri(tracking_uri())
mlflow.set_experiment(params["mlflow"]["experiment"])

with mlflow.start_run() as run:
    mlflow.log_params(hp)
    mlflow.set_tags({
        "data_version": data_version(),
        "git_commit": git_commit(),
        "git_dirty": str(git_dirty()).lower(),
        "train_rows": len(train),
        "train_until": f"{train['dteday'].max():%Y-%m-%d}",
        "val_from": f"{val_part['dteday'].min():%Y-%m-%d}",
    })
    mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

    # Modelo final: mesmos hiperparâmetros, treino completo
    model.fit(X, y)

    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(X, model.predict(X)),
        input_example=X.head(3),
        registered_model_name=MODEL_NAME,
        # Formato tradicional (pickle). O padrão novo do MLflow (skops) é mais seguro,
        # mas exige listar "tipos confiáveis" para modelos de árvore (tema da Aula 4).
        serialization_format="cloudpickle",
    )
    mlflow.MlflowClient().update_model_version(
        MODEL_NAME, info.registered_model_version, description="Treino do pipeline (dvc repro)")

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.joblib")
json.dump(
    {"run_id": run.info.run_id, "model_name": MODEL_NAME,
     "model_version": info.registered_model_version},
    open("models/run.json", "w"), indent=2,
)
json.dump(val_metrics, open("metrics_val.json", "w"), indent=2)
print(f"run_id={run.info.run_id} | {MODEL_NAME} versão {info.registered_model_version} | "
      f"val_rmse={val_metrics['rmse']:.2f} (validação {val_part['dteday'].min():%Y-%m-%d} a "
      f"{val_part['dteday'].max():%Y-%m-%d})")
