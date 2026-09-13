"""Integración: el generador es reproducible y el ETL detecta la suciedad sembrada."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from utils import data as D


@pytest.fixture(scope="module")
def datos():
    return D.cargar_datos()


def _generador():
    spec = importlib.util.spec_from_file_location("generar_datos", D.RAIZ / "scripts" / "generar_datos.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_generador_reproducible():
    gen = _generador()
    precios = D.cargar_referencia().precios
    clientes, vendedores, mecanicos = [f"C{i}" for i in range(20)], [f"V{i}" for i in range(5)], [f"M{i}" for i in range(5)]
    pesos = np.full(20, 1 / 20)

    def correr():
        rng = np.random.default_rng(gen.SEED)
        return (gen.generar_ventas(rng, precios, clientes, pesos, vendedores),
                gen.generar_servicios(rng, clientes, pesos, mecanicos))

    (v1, s1), (v2, s2) = correr(), correr()
    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(s1, s2)


def test_tamanios(datos):
    assert len(datos.ventas) == 60 and len(datos.servicios) == 200


def test_alertas_detectan_errores_sembrados(datos):
    alertas = {a.titulo: len(a.filas) for a in D.detectar_alertas(datos)}
    assert 3 <= alertas["Orden cobrada sin monto"] <= 5
    assert alertas["Fecha de ingreso fuera de rango"] + alertas["Fecha de venta fuera de rango"] == 2
    assert alertas["Modelos sin precio de referencia"] > 0
    assert alertas["Nombres con formato inconsistente"] > 0


def test_prorrateo_global_cuadra(datos):
    pendiente = datos.ventas.loc[datos.ventas["pendiente_precio"], "factura_sin_iva"].sum()
    asignado = datos.unidades["factura_asignada"].sum()
    assert asignado + pendiente == pytest.approx(datos.ventas["factura_sin_iva"].sum())
