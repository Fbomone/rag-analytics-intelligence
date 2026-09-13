import math

import numpy as np
import pytest

from utils.data import normalizar_nombre, parsear_monto


@pytest.mark.parametrize("entrada, esperado", [
    ("$ 1.234,50", 1234.5),
    ("US$ 48.300", 48300.0),
    ("32.000", 32000.0),
    ("1.250.000", 1250000.0),
    ("27100.0", 27100.0),
    ("39.14", 39.14),
    ("2,5", 2.5),
    ("1,234.56", 1234.56),
    ("1,500", 1500.0),
    ("-$ 200,00", -200.0),
    (1500, 1500.0),
    (12.75, 12.75),
    (np.int64(3), 3.0),
])
def test_parsear_monto_valores(entrada, esperado):
    assert parsear_monto(entrada) == pytest.approx(esperado)


@pytest.mark.parametrize("entrada", [None, "", "   ", "abc", "-", float("nan"), np.nan])
def test_parsear_monto_vacios_devuelven_nan(entrada):
    assert math.isnan(parsear_monto(entrada))


@pytest.mark.parametrize("entrada, esperado", [
    ("  LARA CABRERA", "Lara Cabrera"),
    ("luciana quiroga  ", "Luciana Quiroga"),
    ("Uma   Medina", "Uma Medina"),
    ("Ok Nombre", "Ok Nombre"),
])
def test_normalizar_nombre(entrada, esperado):
    assert normalizar_nombre(entrada) == esperado


@pytest.mark.parametrize("entrada", [None, "", "   ", float("nan")])
def test_normalizar_nombre_vacios(entrada):
    assert normalizar_nombre(entrada) != normalizar_nombre(entrada)  # NaN
