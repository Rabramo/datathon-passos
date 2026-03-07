# src/api/feature_descriptions.py
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

router = APIRouter(
    tags=["Análise de Risco de Defasagem"],
)

# Observações:
# - Campos marcados como derivada=True não vêm do dicionário; são features engenheiradas.
# - Se você padronizou nomes diferentes nos CSVs/pipeline, ajuste as chaves aqui.

FEATURE_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "fase": {
        "descricao": "Fase (nível de aprendizado do aluno na Passos Mágicos).",
        "derivada": False,
    },
    "turma": {
        "descricao": "Turma do aluno (identificador da turma dentro da fase).",
        "derivada": False,
    },
    "ano_nasc": {
        "descricao": "Ano de nascimento do aluno.",
        "derivada": False,
    },
    "genero": {
        "descricao": "Gênero do aluno (conforme codificação no dataset).",
        "derivada": False,
    },
    "ano_ingresso": {
        "descricao": "Ano de ingresso do aluno na Passos Mágicos.",
        "derivada": False,
    },
    "instituicao_de_ensino": {
        "descricao": "Instituição de ensino (escola) do aluno no ano de referência.",
        "derivada": False,
    },
    "no_av": {
        "descricao": "Quantidade de avaliações do aluno no ano (QTDE_AVAL).",
        "derivada": False,
    },
    "rec_av1": {
        "descricao": "Recomendação da Equipe de Avaliação 1 (REC_AVAL_1).",
        "derivada": False,
    },
    "rec_av2": {
        "descricao": "Recomendação da Equipe de Avaliação 2 (REC_AVAL_2).",
        "derivada": False,
    },
    "rec_av3": {
        "descricao": "Recomendação da Equipe de Avaliação 3 (REC_AVAL_3).",
        "derivada": False,
    },
    "rec_av4": {
        "descricao": "Recomendação da Equipe de Avaliação 4 (REC_AVAL_4).",
        "derivada": False,
    },
    "iaa": {
        "descricao": "Indicador de Autoavaliação: média das notas de autoavaliação no ano (IAA).",
        "derivada": False,
    },
    "ieg": {
        "descricao": "Indicador de Engajamento: média das notas de engajamento no ano (IEG).",
        "derivada": False,
    },
    "ips": {
        "descricao": "Indicador Psicossocial: média das notas psicossociais no ano (IPS).",
        "derivada": False,
    },
    "rec_psicologia": {
        "descricao": "Recomendação da equipe de psicologia (REC_PSICO).",
        "derivada": False,
    },
    "ida": {
        "descricao": "Indicador de Aprendizagem: média do indicador de aprendizagem no ano (IDA).",
        "derivada": False,
    },
    "matem": {
        "descricao": "Média/nota de Matemática no ano (NOTA_MAT).",
        "derivada": False,
    },
    "portug": {
        "descricao": "Média/nota de Português no ano (NOTA_PORT).",
        "derivada": False,
    },
    "ingles": {
        "descricao": "Média/nota de Inglês no ano (NOTA_ING).",
        "derivada": False,
    },
    "indicado": {
        "descricao": "Indica se o aluno foi indicado para bolsa no ano (INDICADO_BOLSA).",
        "derivada": False,
    },
    "atingiu_pv": {
        "descricao": "Indica se o aluno atingiu Ponto de Virada no ano (PONTO_VIRADA).",
        "derivada": False,
    },
    "ipv": {
        "descricao": "Indicador de Ponto de Virada: média das notas de PV no ano (IPV).",
        "derivada": False,
    },
    "ian": {
        "descricao": "Indicador de Adequação ao Nível: média das notas no ano (IAN).",
        "derivada": False,
    },
    "fase_ideal": {
        "descricao": "Nível/Fase ideal do aluno no ano (NIVEL_IDEAL).",
        "derivada": False,
    },
    "defas": {
        "descricao": "Nível de defasagem no ano (DEFASAGEM).",
        "derivada": False,
    },
    "pedra": {
        "descricao": "Classificação do aluno baseada no INDE (Quartzo/Ágata/Ametista/Topázio).",
        "derivada": False,
    },
    "idade": {
        "descricao": "Idade do aluno no ano (IDADE_ALUNO).",
        "derivada": False,
    },
    "ipp": {
        "descricao": "Indicador Psicopedagógico: média das notas psicopedagógicas no ano (IPP).",
        "derivada": False,
    },
    "tenure": {
        "descricao": "Anos na Passos Mágicos no ano (ANOS_NA_PM).",
        "derivada": False,
    },
    # Derivadas (engenharia de features)
    "gap_fase": {
        "descricao": "Diferença entre fase_ideal e fase (ex.: fase_ideal - fase).",
        "derivada": True,
    },
    "pedra_ord": {
        "descricao": "Versão ordinal da classificação pedra (mapeamento interno do projeto).",
        "derivada": True,
    },
    "rec_av_count": {
        "descricao": "Contagem de recomendações REC_AVAL_1..4 preenchidas/não nulas.",
        "derivada": True,
    },
}

@router.get(
    "/feature-descriptions",
    summary="Descrições das features (contrato de entrada)",
)
def feature_descriptions() -> Dict[str, Any]:
    # formato pronto para consumo em front/swagger
    items: List[Dict[str, Any]] = []
    for name, info in FEATURE_DESCRIPTIONS.items():
        items.append(
            {
                "feature": name,
                "descricao": info["descricao"],
                "derivada": bool(info.get("derivada", False)),
            }
        )

    return {
        "status": "ok",
        "n_features": len(items),
        "items": items,
    }

def get_feature_descriptions_map() -> dict[str, dict[str, object]]:

    return FEATURE_DESCRIPTIONS