# Datathon FIAP MLE - Passos Mágicos

Projeto de Machine Learning Engineering para estimar risco de defasagem escolar (`y=1`) no ano `t+1` usando apenas variáveis observáveis no ano `t`.

O repositório implementa o ciclo ponta a ponta solicitado no Datathon: preparação de dados, treino temporal, avaliação, API de inferência, testes automatizados e containerização.

## Objetivo de negócio

Apoiar decisões preventivas da Associação Passos Mágicos, priorizando alunos com maior probabilidade de defasagem futura para direcionamento pedagógico antecipado.

## Visão geral da solução

Pipeline temporal:
1. Ingestão de dados brutos (`data/raw`).
2. Pré-processamento e normalização (`data/interim`).
3. Construção de pares `t -> t+1` e geração do alvo (`data/processed`).
4. Treinamento e comparação de modelos (artefatos em `artifacts/models` e `artifacts/metrics`).
5. Exposição via API FastAPI (`/predict`, `/train`, `/infra`, `/leaderboard`).
6. Avaliação offline e rastreabilidade (métricas, ponteiros `latest_*.json`, leaderboard).

## Regras de modelagem temporal e anti-vazamento

- A predição é sempre `t -> t+1`.
- Features de entrada incluem apenas dados do ano `t`.
- Colunas de alvo/futuro são bloqueadas na borda da API.
- O alvo binário é derivado de `IAN(t+1)` conforme regra do projeto.

## Estrutura principal

- `src/data`: carga, limpeza e padronização.
- `src/features`: montagem de pares temporais.
- `src/pipelines`: execução de build, treino e avaliação.
- `src/models`: treino, tuning, comparação e utilidades de modelo.
- `src/api`: aplicação FastAPI, schemas e roteadores.
- `tests`: suíte de testes unitários/integrados.
- `artifacts`: modelos, métricas, leaderboard e ponteiros.

## Tecnologias

- Python 3.12
- pandas, numpy, pyarrow, scikit-learn
- xgboost, catboost
- FastAPI + Uvicorn
- joblib para serialização
- pytest + coverage
- Docker

## Como executar localmente

Pré-requisitos:
- Python 3.12+
- `pip`
- Docker (opcional)

Instalação:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Execução do pipeline

1) Construir datasets processados:

```bash
python -m src.pipelines.build_dataset \
  --raw-dir data/raw \
  --interim-dir data/interim \
  --processed-dir data/processed
```

2) Treinar baseline temporal:

```bash
python -m src.pipelines.train \
  --processed-dir data/processed \
  --artifacts-dir artifacts/models
```

3) Comparar modelos e gerar leaderboard:

```bash
python -m src.models.compare
```

4) Avaliar artefato em dataset:

```bash
python -m src.pipelines.evaluate \
  --dataset-path data/processed/pair_2023_2024.parquet \
  --model-path artifacts/models/latest_logreg.json \
  --output-dir artifacts/evaluation
```

## API (FastAPI)

Subir API:

```bash
uvicorn src.api.main:app --reload
```

Documentação:
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`

Endpoints principais:
- `GET /health` e `GET /infra/health`
- `GET /smoke` e `GET /infra/smoke`
- `GET /model`, `GET /predict/model`, `GET /infra/model`
- `POST /predict`
- `POST /predict/batch`
- `POST /train`
- `GET /leaderboard`

Exemplo de inferência:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "logreg",
    "features": {
      "fase": 5,
      "turma": "A",
      "ian": 7.0,
      "ida": 6.5,
      "ieg": 6.8,
      "ipp": 5.9,
      "idade": 13
    }
  }'
```

## Docker

Build:

```bash
docker build -t passos-magicos-api .
```

Run:

```bash
docker run --rm -p 8000:8000 passos-magicos-api
```

## Testes

Executar suíte:

```bash
pytest -q
```

Cobertura:

```bash
pytest --cov=src --cov-report=term-missing
```

## CI/CD (GitHub Actions)

Workflow implementado em:
- `.github/workflows/ci-cd.yml`

Fluxo:
1. Executa testes e coverage em `push` e `pull_request` para `main`.
2. Faz build da imagem Docker após testes passarem.
3. Em `push` na `main`, dispara deploy no Render via Deploy Hook.

Configuração necessária no GitHub (`Settings -> Secrets and variables -> Actions`):
- `RENDER_DEPLOY_HOOK_URL`: URL do Deploy Hook do seu serviço no Render.

## Rastreabilidade de artefatos

- Modelos versionados por `run_id`.
- Métricas salvas em JSON por execução.
- Ponteiros `latest_*.json` para resolver modelo ativo por chave.
- Leaderboard consolidado para comparação entre experimentos.

## Critérios do Datathon e status

- Pipeline completo de ML (pré-processamento -> treino -> avaliação -> inferência): atendido.
- Serialização e carregamento de modelo para produção: atendido.
- API de predição com contrato e validações: atendido.
- Testes automatizados e cobertura mínima: atendido.
- Containerização para execução isolada: atendido.
- Documentação técnica de execução e uso: atendido.

## O que ainda pode evoluir (não bloqueante)

- Deploy gerenciado em nuvem (Cloud Run/Render/Heroku) com IaC.
- Monitoramento contínuo em produção com agendamento e alertas de drift.

## Resultado atual

O projeto está tecnicamente consistente para entrega acadêmica e demonstra, de forma reprodutível, os componentes exigidos para um fluxo MLE de ponta a ponta no contexto do problema proposto.
