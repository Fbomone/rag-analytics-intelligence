import math

import pandas as pd
import pytest

from utils.data import (Referencia, expandir_unidades, limpiar_ventas, prorratear_factura,
                        split_modelos)

CATALOGO = ["Corolla Cross", "Yaris", "Hilux", "Territory"]
PRECIOS = {"Corolla Cross": 32000.0, "Yaris": 19000.0, "Hilux": 48000.0}


# ── split de modelos ──

@pytest.mark.parametrize("celda, esperado", [
    ("Corolla Cross, Yaris", ["Corolla Cross", "Yaris"]),
    ("Corolla Cross,Yaris", ["Corolla Cross", "Yaris"]),
    ("Hilux, Hilux", ["Hilux", "Hilux"]),                  # no deduplica
    ("  corolla   cross ,YARIS", ["Corolla Cross", "Yaris"]),
    ("Hilux,", ["Hilux"]),
    ("Taos", ["Taos"]),                                     # fuera de catálogo → título
    ("", []),
    (None, []),
    (float("nan"), []),
])
def test_split_modelos(celda, esperado):
    assert split_modelos(celda, CATALOGO) == esperado


# ── prorrateo ──

def test_prorrateo_proporcional_al_precio():
    partes = prorratear_factura(51000, ["Corolla Cross", "Yaris"], PRECIOS)
    assert partes == pytest.approx([32000, 19000])


def test_prorrateo_conserva_el_total_con_precio_distinto_al_de_referencia():
    partes = prorratear_factura(45900, ["Corolla Cross", "Yaris"], PRECIOS)
    assert sum(partes) == pytest.approx(45900)
    assert partes[0] / partes[1] == pytest.approx(32000 / 19000)


def test_prorrateo_modelo_repetido_se_divide_en_partes_iguales():
    assert prorratear_factura(90000, ["Hilux", "Hilux"], PRECIOS) == pytest.approx([45000, 45000])


def test_prorrateo_con_modelo_sin_precio_no_reparte():
    partes = prorratear_factura(70000, ["Corolla Cross", "Territory"], PRECIOS)
    assert len(partes) == 2 and all(math.isnan(p) for p in partes)


def test_prorrateo_monto_vacio_y_lista_vacia():
    assert all(math.isnan(p) for p in prorratear_factura(float("nan"), ["Yaris"], PRECIOS))
    assert prorratear_factura(1000, [], PRECIOS) == []


# ── expansión a unidades ──

@pytest.fixture
def ventas():
    crudo = pd.DataFrame({
        "id_operacion": ["OP-1", "OP-2", "OP-3", "OP-4"],
        "fecha_venta": ["2025-01-10", "2025-02-10", "2025-03-10", "2025-04-10"],
        "cliente": ["A", "B", "C", "D"],
        "modelo": ["Corolla Cross, Yaris", "Hilux, Hilux", "Yaris, Territory", "yaris"],
        "condicion": ["Nuevo"] * 4,
        "estado": ["Entregado"] * 4,
        "vendedor": ["V"] * 4,
        "factura_sin_iva": ["51000", "$ 96.000,00", "60000", None],
        "comision_pct": [5, 5, 5, 5],
        "comision_monto": [2550, None, 3000, None],
        "cobrado": ["si", "no", "SI", "no"],
    })
    return limpiar_ventas(crudo, Referencia(PRECIOS, ["Territory"]), hoy=pd.Timestamp("2026-01-01"))


def test_limpiar_ventas_tipa_y_recalcula_comision(ventas):
    assert ventas["factura_sin_iva"].tolist()[:3] == [51000, 96000, 60000]
    assert ventas.loc[1, "comision_monto"] == pytest.approx(4800)   # recalculada
    assert ventas["cobrado"].tolist() == [True, False, True, False]
    assert ventas["unidades"].tolist() == [2, 2, 2, 1]
    assert ventas["pendiente_precio"].tolist() == [False, False, True, False]


def test_expandir_unidades_una_fila_por_unidad(ventas):
    unidades = expandir_unidades(ventas, PRECIOS)
    assert len(unidades) == ventas["unidades"].sum() == 7
    assert (unidades["modelo"] == "Hilux").sum() == 2
    assert (unidades["modelo"] == "Yaris").sum() == 3


def test_expandir_unidades_asignado_mas_pendiente_igual_total(ventas):
    unidades = expandir_unidades(ventas, PRECIOS)
    pendiente = ventas.loc[ventas["pendiente_precio"], "factura_sin_iva"].sum()
    assert unidades["factura_asignada"].sum() + pendiente == pytest.approx(ventas["factura_sin_iva"].sum())
    op3 = unidades[unidades["id_operacion"] == "OP-3"]
    assert op3["factura_asignada"].isna().all()
    assert op3.set_index("modelo").loc["Yaris", "tiene_precio"]
