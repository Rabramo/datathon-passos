# Datathon FIAP MLE - Passos Mágicos

Este projeto foi desenvolvido no contexto do Datathon de Machine Learning Engineering da FIAP com foco no desafio da Associação Passos Mágicos, organização social que atua no apoio educacional de crianças e adolescentes em situação de vulnerabilidade.

No cenário do projeto, a principal necessidade é identificar com antecedência alunos que podem entrar em defasagem de aprendizagem, para que a equipe pedagógica consiga intervir antes que o problema se consolide. Em vez de atuar apenas de forma reativa (quando a defasagem já aconteceu), a proposta é habilitar uma atuação pró-ativa, com priorização de acompanhamento, reforço e suporte individual.

Para isso, a solução estima o risco de defasagem escolar no ano `t+1` (`y=1`) usando apenas variáveis disponíveis no ano `t`, respeitando a lógica temporal e evitando vazamento de informação futura. O repositório implementa o ciclo ponta a ponta do problema: preparação de dados, treino temporal, avaliação, API de inferência, testes automatizados, deploy e monitoramento contínuo.

## Acesso rápido em produção (Render)

API em produção:
- `https://datathon-passos-api-ra363736.onrender.com`

Swagger em produção (prioritário):
- `https://datathon-passos-api-ra363736.onrender.com/docs`

OpenAPI:
- `https://datathon-passos-api-ra363736.onrender.com/openapi.json`

Healthcheck:
- `https://datathon-passos-api-ra363736.onrender.com/health`

Observação:
- A rota raiz `/` retorna um payload simples com links úteis da API.

### Opções disponíveis no Swagger (GET e POST)

GET:
- `GET /`:
  payload inicial com links úteis da API.
- `GET /health`:
  liveness básico da aplicação.
- `GET /infra/health`:
  healthcheck da camada de infraestrutura.
- `GET /infra/model`:
  metadados do modelo carregado (threshold, features esperadas e caminho do artefato).
- `GET /infra/smoke`:
  verificação rápida da API + modelo, com opção `dry_run`.
- `GET /predict/model`:
  inspeção de metadados do modelo selecionado para inferência.
- `GET /predict/feature-descriptions`:
  dicionário de features com descrição e indicação de variável derivada.
- `GET /leaderboard`:
  consulta do ranking de modelos (JSON ou CSV), com ordenação e filtros.

POST:
- `POST /predict`:
  inferência individual de risco de defasagem (`model_key` via query param).
- `POST /predict/batch`:
  inferência em lote.
- `POST /predict/features/select`:
  seleção de subconjunto de features aceitas na inferência.
- `POST /train`:
  disparo de treinamento temporal com escolha de modelo e conjunto de variáveis.

## Uso com Docker (recomendado)

Build:

```bash
docker build -t passos-magicos-api .
```

Run:

```bash
docker run --rm -p 8000:8000 passos-magicos-api
```

Swagger local via Docker:
- `http://127.0.0.1:8000/docs`

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

## Stack técnico consolidado por categoria

1. Desenvolvimento
- Python
- Estrutura modular em `src/`
- Testes automatizados com `pytest`
- Cobertura com `coverage`

2. ML/MLOps
- Pipeline de treino temporal
- Feature engineering
- Avaliação de modelos
- Versionamento de artefatos
- Monitoramento de drift
- Relatórios automáticos de drift
- Alertas automáticos

3. API
- FastAPI
- OpenAPI/Swagger
- Serving via endpoints HTTP

4. DevOps / Plataforma
- Docker
- GitHub
- GitHub Actions
- Render
- Secrets no GitHub Actions
- Deploy hook
- Webhooks para alertas

5. Observabilidade operacional
- Drift monitoring agendado
- Execução manual e agendada
- Artifact upload
- Exit code crítico em caso de alerta

## Execução local (desenvolvimento)

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
- `GET /`
- `GET /health` e `GET /infra/health`
- `GET /smoke` e `GET /infra/smoke`
- `GET /model`, `GET /predict/model`, `GET /predict/feature-descriptions`, `GET /infra/model`
- `POST /predict`
- `POST /predict/batch`
- `POST /train`
- `GET /leaderboard`

Exemplo de inferência:

```bash
curl -X POST "http://127.0.0.1:8000/predict?model_key=logreg" \
  -H "Content-Type: application/json" \
  -d '{
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

## Validação em produção (Render)

```bash
curl -sS https://datathon-passos-api-ra363736.onrender.com/
curl -sS https://datathon-passos-api-ra363736.onrender.com/health
curl -sS https://datathon-passos-api-ra363736.onrender.com/docs > /dev/null
curl -sS https://datathon-passos-api-ra363736.onrender.com/openapi.json > /dev/null
curl -sS https://datathon-passos-api-ra363736.onrender.com/predict/feature-descriptions
```

Exemplo de predição em produção:

```bash
curl -X POST "https://datathon-passos-api-ra363736.onrender.com/predict?model_key=logreg" \
  -H "Content-Type: application/json" \
  -d '{
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

## Testes

Executar suíte:

```bash
pytest -q
```

Cobertura:

```bash
pytest --cov=src --cov-report=term-missing
```

Cobertura atual (última execução local):

```text
-------- coverage: platform darwin, python 3.12.12-final-0 ----------
Name                               Stmts   Miss  Cover   Missing
----------------------------------------------------------------
src/__init__.py                        0      0   100%
src/api/__init__.py                    0      0   100%
src/api/app.py                        45      4    91%   76-77, 92-93
src/api/feature_descriptions.py       13      4    69%   154-164
src/api/leaderboard.py                45     31    31%   33-48, 66-101
src/api/main.py                       36      4    89%   56, 104, 119, 138
src/api/model_loader.py               81     13    84%   103, 106, 110, 114, 124, 134, 143-144, 157-160, 173
src/api/predict.py                     3      3     0%   1-5
src/api/routers/__init__.py            0      0   100%
src/api/routers/infra.py              67      5    93%   23, 29-30, 119-120
src/api/routers/leaderboard.py        45     31    31%   37-56, 91-137
src/api/routers/predict.py           168     72    57%   80, 166-167, 172-185, 218-222, 244-259, 330-331, 335, 346, 362, 365, 372-373, 415-476
src/api/routers/train.py              42      5    88%   27, 32, 54-55, 88
src/api/schemas.py                    95      1    99%   22
src/data/__init__.py                   0      0   100%
src/data/build_pairs.py               42      1    98%   42
src/data/load.py                      40      0   100%
src/data/load_processed_pairs.py      24      0   100%
src/data/preprocess.py               186     17    91%   123, 145, 163-167, 177-178, 182-184, 197-198, 200, 233, 297, 353
src/data/validate.py                  38      0   100%
src/features/__init__.py               0      0   100%
src/features/build_pairs.py           42      3    93%   41, 44, 65
src/features/preprocess.py            76      0   100%
src/models/evaluate.py                52      6    88%   35-38, 66, 82-83
src/models/train.py                   74      9    88%   26, 41, 55, 57, 80, 83, 96, 116, 121
src/models/train_api.py               88     13    85%   41, 45-49, 52-56, 68, 129
src/pipelines/build_dataset.py        69     15    78%   71, 94, 96, 98, 111-118, 122-123, 134
src/pipelines/evaluate.py            167     32    81%   50, 60, 87, 94, 98, 109, 119, 140, 143, 155, 172, 186, 193, 196-200, 225-226, 334-373, 377-393, 397
src/pipelines/train.py               128     20    84%   172-176, 184, 189-193, 235-240, 244-245, 254
src/utils/io.py                        4      0   100%
----------------------------------------------------------------
TOTAL                               1670    289    83%
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

## Monitoramento contínuo de drift

Workflow implementado em:
- `.github/workflows/drift-monitoring.yml`

Funcionamento:
1. Execução agendada diariamente (cron) e também manual (`workflow_dispatch`).
2. Cálculo de drift por feature com PSI em `src/monitoring/drift.py`.
3. Geração de relatórios em `artifacts/monitoring` (JSON + CSV).
4. Upload do relatório como artifact do GitHub Actions.
5. Alerta externo via webhook quando há drift crítico.

Secret opcional para alerta:
- `DRIFT_ALERT_WEBHOOK_URL`: webhook (Slack/Discord/Teams) para alertas.

## Rastreabilidade de artefatos

- Modelos versionados por `run_id`.
- Métricas salvas em JSON por execução.
- Ponteiros `latest_*.json` para resolver modelo ativo por chave.
- Leaderboard consolidado para comparação entre experimentos.
