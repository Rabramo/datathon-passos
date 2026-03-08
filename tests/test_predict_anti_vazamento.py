from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.routers.predict import _validar_sem_vazamento_ou_target


def test_rejeita_chave_de_target() -> None:
    with pytest.raises(HTTPException) as exc:
        _validar_sem_vazamento_ou_target({"IAN": 10, "fase": "F1"})
    assert exc.value.status_code == 422


def test_rejeita_chave_de_futuro() -> None:
    with pytest.raises(HTTPException) as exc:
        _validar_sem_vazamento_ou_target({"idade_t1": 13, "fase": "F1"})
    assert exc.value.status_code == 422


def test_aceita_features_do_ano_t() -> None:
    _validar_sem_vazamento_ou_target({"fase": "F1", "ian": 7.5, "idade": 12})