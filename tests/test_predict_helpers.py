from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.routers.predict import (
    _colunas_esperadas,
    _colunas_esperadas_do_modelo,
    _extrair_features,
    _normalizar_model_key,
    _obter_threshold,
    _validar_sem_vazamento_ou_target,
)


def test_extrair_features_quando_payload_tem_dict_features():
    payload = SimpleNamespace(
        features={
            "idade": 12,
            "sexo": "F",
            "renda": 1000.0,
        }
    )

    feats = _extrair_features(payload)

    assert feats == {
        "idade": 12,
        "sexo": "F",
        "renda": 1000.0,
    }


def test_extrair_features_quando_payload_flat_tipo_request_model():
    payload = SimpleNamespace(
        features=None,
        model_dump=lambda **kwargs: {
            "idade": 12,
            "sexo": "F",
            "renda": 1000.0,
        },
    )

    feats = _extrair_features(payload)

    assert feats == {
        "idade": 12,
        "sexo": "F",
        "renda": 1000.0,
    }


def test_validar_sem_vazamento_ou_target_ok():
    feats = {
        "idade": 12,
        "renda_familiar": 1000,
        "frequencia": 0.9,
    }

    _validar_sem_vazamento_ou_target(feats)


@pytest.mark.parametrize(
    "feats,chave_esperada",
    [
        ({"y": 1}, "y"),
        ({"target": 1}, "target"),
        ({"feature_t1": 10}, "feature_t1"),
        ({"nota_2024_t1": 8.5}, "nota_2024_t1"),
    ],
)
def test_validar_sem_vazamento_ou_target_levanta_http_422(feats, chave_esperada):
    with pytest.raises(HTTPException) as exc_info:
        _validar_sem_vazamento_ou_target(feats)

    exc = exc_info.value
    assert exc.status_code == 422
    assert chave_esperada in exc.detail["forbidden_keys"]


def test_validar_sem_vazamento_ou_target_permita_risco_defasagem_se_nao_proibido():
    feats = {"risco_defasagem": 1}
    _validar_sem_vazamento_ou_target(feats)


def test_colunas_esperadas_do_modelo_a_partir_de_feature_names_in():
    model = SimpleNamespace(feature_names_in_=["f1", "f2", "f3"])

    cols = _colunas_esperadas_do_modelo(model)

    assert cols == ["f1", "f2", "f3"]


def test_colunas_esperadas_do_modelo_retorna_none_quando_atributo_ausente():
    model = SimpleNamespace()

    cols = _colunas_esperadas_do_modelo(model)

    assert cols is None


def test_colunas_esperadas_prefere_raw_features_do_meta():
    model = SimpleNamespace(feature_names_in_=["model_f1"])
    meta = {"raw_features": ["meta_f1", "meta_f2"]}

    cols = _colunas_esperadas(model, meta)

    assert cols == ["meta_f1", "meta_f2"]


def test_colunas_esperadas_faz_fallback_para_modelo():
    model = SimpleNamespace(feature_names_in_=["model_f1", "model_f2"])
    meta = {}

    cols = _colunas_esperadas(model, meta)

    assert cols == ["model_f1", "model_f2"]


def test_obter_threshold_a_partir_do_meta():
    assert _obter_threshold({"threshold": 0.7}) == 0.7
    assert _obter_threshold({"threshold": "0.65"}) == 0.65


def test_obter_threshold_fallback_para_default():
    assert _obter_threshold({}) == 0.5
    assert _obter_threshold({"threshold": "invalido"}) == 0.5
    assert _obter_threshold({}, default=0.3) == 0.3


@pytest.mark.parametrize(
    "raw,esperado",
    [
        (None, None),
        ("default", "default"),
        ("logreg", "logreg"),
        ("tree", "tree"),
    ],
)
def test_normalizar_model_key(raw, esperado):
    assert _normalizar_model_key(raw) == esperado