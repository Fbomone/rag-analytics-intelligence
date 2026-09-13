import numpy as np
import pandas as pd
import pytest

from utils import data as D


@pytest.fixture
def servicios_crudo():
    return pd.DataFrame({
        "id_orden": ["OT-1", "OT-2", "OT-3", "OT-4"],
        "fecha_ingreso": ["2025-01-05", "2025-01-06", "1925-01-07", "2025-01-08"],
        "fecha_entrega": ["2025-01-05", "", "", ""],
        "cliente": ["  ANA  PAZ", "Ana Paz", "Beto", "Caro"],
        "servicio": ["Lavado premium", "Chapa y pintura", "Mecánica general", "Cambio de aceite"],
        "vehiculo": ["Onix"] * 4,
        "estado": ["Cobrado", "Facturado esperando cobro", "Trabajo realizado", "Presupuesto enviado"],
        "horas_trabajo": ["2", "10,5", 4.0, 1.0],
        "precio_hora": [18, 45, 40, 35],
        "monto_total": ["$ 40,00", "500", None, 50],
        "mecanico_1": ["Lara", "Lara", "Delfi", np.nan],
        "horas_1": [2.0, 8.0, "2,5", np.nan],
        "mecanico_2": [np.nan, "Delfi", "  ", np.nan],
        "horas_2": [np.nan, 9.0, np.nan, np.nan],
        "mecanico_3": [np.nan, "Uma", np.nan, np.nan],
        "horas_3": [np.nan, 5.5, np.nan, np.nan],
    })


def test_desagregar_mecanicos_formato_largo(servicios_crudo):
    largo = D.desagregar_mecanicos(servicios_crudo)
    assert len(largo) == 5                                      # 1 + 3 + 1 (el nombre en blanco se descarta)
    assert set(largo.columns) >= {"id_orden", "mecanico", "horas", "n_mecanicos", "modalidad"}
    ot2 = largo[largo["id_orden"] == "OT-2"]
    assert ot2["n_mecanicos"].eq(3).all() and ot2["modalidad"].eq("Acompañado").all()
    assert largo.set_index("id_orden").loc["OT-1", "modalidad"] == "Solo"
    assert largo.set_index("id_orden").loc["OT-3", "horas"] == pytest.approx(2.5)


def test_horas_por_mecanico_pueden_superar_horas_trabajo(servicios_crudo):
    servicios = D.limpiar_servicios(servicios_crudo, hoy=pd.Timestamp("2026-01-01"))
    largo = D.desagregar_mecanicos(servicios)
    ot2_mecanicos = largo.loc[largo["id_orden"] == "OT-2", "horas"].sum()
    ot2_real = servicios.set_index("id_orden").loc["OT-2", "horas_trabajo"]
    assert ot2_mecanicos == pytest.approx(22.5) and ot2_real == pytest.approx(10.5)


def test_desagregar_sin_columnas_de_mecanicos():
    vacio = D.desagregar_mecanicos(pd.DataFrame({"id_orden": ["OT-1"]}))
    assert vacio.empty and "mecanico" in vacio.columns


def test_limpiar_servicios(servicios_crudo):
    s = D.limpiar_servicios(servicios_crudo, hoy=pd.Timestamp("2026-01-01"))
    assert s["cliente"].tolist()[:2] == ["Ana Paz", "Ana Paz"]
    assert s["monto_total"].tolist()[:2] == [40.0, 500.0]
    assert s["fecha_valida"].tolist() == [True, True, False, True]
    assert s["estado_cobro"].tolist()[:3] == [D.COBRO_COBRADO, D.COBRO_POR_COBRAR, D.COBRO_POR_FACTURAR]
    assert pd.isna(s.loc[3, "estado_cobro"])


def test_kpis_comision_no_es_facturacion():
    ventas = pd.DataFrame({
        "factura_sin_iva": [30000.0, 20000.0], "comision_monto": [1500.0, 1000.0],
        "estado_cobro": [D.COBRO_COBRADO, D.COBRO_POR_COBRAR], "unidades": [1, 2],
    })
    servicios = pd.DataFrame({
        "monto_total": [300.0, 200.0, 100.0], "horas_trabajo": [5.0, 3.0, 1.0],
        "estado_cobro": [D.COBRO_COBRADO, D.COBRO_POR_COBRAR, D.COBRO_POR_FACTURAR],
    })
    k = D.calcular_kpis(ventas, servicios)
    assert k["ingreso_total"] == 3100                            # 600 taller + 2500 comisiones
    assert k["volumen_intermediado"] == 50000                    # aparte, no suma
    assert k["participacion_taller"] + k["participacion_autos"] == pytest.approx(1)
    assert k["participacion_taller"] == pytest.approx(600 / 3100)
    assert (k["cobrado"], k["por_cobrar"], k["por_facturar"]) == (1800, 1200, 100)
    assert k["unidades"] == 3 and k["horas_taller"] == 9


def test_kpis_sin_datos_no_divide_por_cero():
    vacio_v = pd.DataFrame(columns=["factura_sin_iva", "comision_monto", "estado_cobro", "unidades"])
    vacio_s = pd.DataFrame(columns=["monto_total", "horas_trabajo", "estado_cobro"])
    k = D.calcular_kpis(vacio_v, vacio_s)
    assert k["ingreso_total"] == 0 and k["participacion_taller"] == 0
