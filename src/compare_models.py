"""Estágio `compare`: o modelo recém-treinado (challenger) deve substituir o @champion?

Sem argumentos (o que o `dvc repro` roda):
    carrega o @champion do Registry e o challenger (a versão criada pelo `train`,
    lida de models/run.json), mede os dois no MESMO teste (data/split/test.csv),
    decide pela regra do params.yaml e grava reports/comparacao.json.
    Não altera o Registry.

Com --promover (rodado à mão, depois de revisar o comparacao.json):
    aplica a decisão gravada: tags `decisao` e `motivo` na versão do challenger e,
    se a decisão for promover, tag `previous_champion` + alias @champion.
    Promover continua sendo um ato explícito de governança.
"""
import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from bike import MODEL_NAME, regression_metrics, tracking_uri, xy

OUT = Path("reports/comparacao.json")


def champion_atual(client: MlflowClient):
    """Versão apontada por @champion, ou None se o alias ainda não existe."""
    try:
        return client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None


def medir(version: str, X, y) -> dict:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    return regression_metrics(y, model.predict(X))


def comparar(client: MlflowClient) -> dict:
    margem = yaml.safe_load(open("params.yaml"))["compare"]["margem_minima"]
    test = pd.read_csv("data/split/test.csv", parse_dates=["dteday"])
    X, y = xy(test)

    chall_version = str(json.load(open("models/run.json"))["model_version"])
    m_chall = medir(chall_version, X, y)
    champ = champion_atual(client)

    resultado = {
        "teste": {"de": f"{test['dteday'].min():%Y-%m-%d}",
                  "ate": f"{test['dteday'].max():%Y-%m-%d}", "linhas": len(test)},
        "margem_minima": margem,
        "challenger": {"versao": chall_version, **m_chall},
    }

    if champ is None:
        decisao, motivo = "promover", "não existe @champion: o challenger vira o primeiro"
    elif champ.version == chall_version:
        decisao, motivo = "manter", "o challenger já é o @champion"
    else:
        m_champ = medir(champ.version, X, y)
        ganho = 1 - m_chall["rmse"] / m_champ["rmse"]
        resultado["champion"] = {"versao": champ.version, **m_champ}
        resultado["ganho_rmse"] = ganho
        comparacao = (f"RMSE {m_chall['rmse']:.1f} (v{chall_version}) x {m_champ['rmse']:.1f} "
                      f"(v{champ.version}) no teste {resultado['teste']['de']} a {resultado['teste']['ate']}")
        if ganho >= margem:
            decisao, motivo = "promover", f"{comparacao}: ganho de {ganho:.1%} >= margem de {margem:.0%}"
        else:
            decisao, motivo = "manter", f"{comparacao}: ganho de {ganho:.1%} < margem de {margem:.0%}"

    resultado["decisao"] = decisao
    resultado["motivo"] = motivo
    return resultado


def promover(client: MlflowClient) -> None:
    """Aplica a decisão gravada no comparacao.json (não mede de novo)."""
    r = json.load(open(OUT))
    chall = r["challenger"]["versao"]
    champ = champion_atual(client)
    champ_versao_gravada = r.get("champion", {}).get("versao")

    # Proteção: a decisão só vale para o champion contra o qual foi tomada
    if champ is not None and champ.version not in (champ_versao_gravada, chall):
        raise SystemExit(f"O @champion mudou (agora v{champ.version}, a comparação foi contra "
                         f"v{champ_versao_gravada}). Rode `dvc repro` de novo antes de promover.")

    client.set_model_version_tag(MODEL_NAME, chall, "decisao", r["decisao"])
    client.set_model_version_tag(MODEL_NAME, chall, "motivo", r["motivo"])

    if r["decisao"] == "promover" and (champ is None or champ.version != chall):
        if champ is not None:
            client.set_model_version_tag(MODEL_NAME, chall, "previous_champion", champ.version)
        client.set_model_version_tag(MODEL_NAME, chall, "validation_status", "approved")
        client.set_registered_model_alias(MODEL_NAME, "champion", chall)
        anterior = f"v{champ.version}" if champ else "nenhum"
        print(f"✅ Promovido: @champion agora é v{chall} (antes: {anterior})")
    else:
        if r["decisao"] == "manter" and (champ is None or champ.version != chall):
            client.set_model_version_tag(MODEL_NAME, chall, "validation_status", "rejected")
        print(f"⏸️ Mantido: @champion continua v{champ.version if champ else '-'} | {r['motivo']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--promover", action="store_true",
                        help="aplica a decisão gravada em reports/comparacao.json")
    args = parser.parse_args()

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()

    if args.promover:
        promover(client)
        return

    resultado = comparar(client)
    OUT.parent.mkdir(exist_ok=True)
    json.dump(resultado, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
