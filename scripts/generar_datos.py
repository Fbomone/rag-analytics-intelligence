"""Generador de datos sintéticos para AutoPulse.

Produce dos CSV en ``data/``:

* ``ventas_autos.csv``      — operaciones de venta de autos a comisión (~60 filas).
* ``servicios_taller.csv``  — órdenes de trabajo del taller (~200 filas).

Los datos son 100% ficticios y reproducibles (semilla fija). Se ensucian a
propósito con errores típicos de carga manual (montos vacíos en filas
cobradas, fechas fuera de rango, nombres mal tipeados, montos con formato
es-AR) para que el panel de Calidad de Datos tenga algo real que detectar.

Uso:
    python scripts/generar_datos.py
"""

from __future__ import annotations

import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = RAIZ / "data"

INICIO = date(2024, 1, 1)
FIN = date(2026, 8, 31)
N_VENTAS = 60
N_SERVICIOS = 200

# Precio "real" de los modelos que NO tienen referencia cargada. Sólo se usa
# para que sus facturas sean verosímiles; no se publica en precios_referencia.json.
PRECIOS_OCULTOS = {"Territory": 36000, "Taos": 31000}

PESO_MODELO = {
    "Corolla Cross": 1.3, "Hilux": 0.9, "Yaris": 1.2, "Amarok": 0.6, "Onix": 1.4,
    "Tracker": 1.0, "Kicks": 1.0, "Territory": 0.35, "Taos": 0.35,
}
FACTOR_CONDICION = {"Nuevo": 1.0, "Usado": 0.72}
FACTOR_PAGO = {"Contado": 0.97, "Financiado": 1.06, "Permuta + saldo": 1.0, "Cheques": 1.03}
PROB_PAGO = {"Contado": 0.30, "Financiado": 0.35, "Permuta + saldo": 0.25, "Cheques": 0.10}
PROB_LEAD = {"Instagram": 0.28, "Referido": 0.22, "Web": 0.18, "Showroom": 0.17, "MercadoLibre": 0.15}

# horas (min, max), USD por hora, % de materiales sobre mano de obra,
# días de trabajo (min, max) y probabilidad de 1/2/3 mecánicos.
SERVICIOS = {
    "Lavado premium":          dict(peso=1.6, horas=(1, 2.5), precio_hora=18, materiales=0.10, dias=(0, 1),  mecanicos=(0.80, 0.20, 0.00)),
    "Detailing completo":      dict(peso=0.7, horas=(4, 9),   precio_hora=30, materiales=0.15, dias=(1, 3),  mecanicos=(0.40, 0.50, 0.10)),
    "Pulido y encerado":       dict(peso=0.6, horas=(3, 6),   precio_hora=28, materiales=0.12, dias=(1, 2),  mecanicos=(0.60, 0.40, 0.00)),
    "Mecánica general":        dict(peso=1.0, horas=(2, 10),  precio_hora=40, materiales=0.35, dias=(1, 5),  mecanicos=(0.50, 0.40, 0.10)),
    "Cambio de aceite":        dict(peso=1.2, horas=(0.5, 1.5), precio_hora=35, materiales=0.90, dias=(0, 1), mecanicos=(0.90, 0.10, 0.00)),
    "Chapa y pintura":         dict(peso=0.6, horas=(8, 30),  precio_hora=45, materiales=0.40, dias=(4, 12), mecanicos=(0.20, 0.50, 0.30)),
    "Alineación y balanceo":   dict(peso=0.8, horas=(1, 2),   precio_hora=38, materiales=0.05, dias=(0, 1),  mecanicos=(0.85, 0.15, 0.00)),
    "Diagnóstico electrónico": dict(peso=0.5, horas=(1, 3),   precio_hora=50, materiales=0.00, dias=(0, 2),  mecanicos=(0.70, 0.30, 0.00)),
}
ESTETICA = {"Lavado premium", "Detailing completo", "Pulido y encerado"}
MECANICA = {"Mecánica general", "Cambio de aceite", "Alineación y balanceo", "Diagnóstico electrónico"}

ESTADOS_SERVICIO = ["Cobrado", "Facturado esperando cobro", "Trabajo realizado",
                    "Presupuesto enviado", "Cancelado", "En proceso"]
VEHICULOS_EXTRA = ["Gol Trend", "Cronos", "208", "Ranger", "Sandero", "Etios", "Fiesta", "Duster"]


# ─────────────────────────────── helpers ────────────────────────────────────

def _meses() -> list[date]:
    """Primer día de cada mes entre INICIO y FIN."""
    return [p.to_timestamp().date() for p in pd.period_range(INICIO, FIN, freq="M")]


def _peso_tendencia(i: int) -> float:
    """Crecimiento leve: el último mes pesa ~2.2x el primero."""
    return 1 + 0.04 * i


def _estacionalidad(servicio: str, mes: int) -> float:
    """Más estética en verano (dic-feb), más mecánica al volver de vacaciones."""
    if servicio in ESTETICA:
        return {12: 2.0, 1: 2.0, 2: 2.0, 3: 1.3, 11: 1.3, 6: 0.6, 7: 0.6}.get(mes, 1.0)
    if servicio in MECANICA:
        return {2: 1.9, 3: 1.9, 8: 1.5, 1: 0.6}.get(mes, 1.0)
    return 1.0


def _dia_aleatorio(rng: np.random.Generator, mes: date) -> date:
    ultimo = calendar.monthrange(mes.year, mes.month)[1]
    return mes.replace(day=int(rng.integers(1, ultimo + 1)))


def _medias_horas(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Horas redondeadas a múltiplos de 0.5."""
    return max(0.5, round(rng.uniform(lo, hi) * 2) / 2)


def _formato_ar(monto: float) -> str:
    """Monto como lo exporta una planilla es-AR: '$ 1.234,50'."""
    entero, dec = f"{monto:,.2f}".split(".")
    return f"$ {entero.replace(',', '.')},{dec}"


def _ensuciar_nombre(nombre: str, variante: int) -> str:
    """Errores típicos de tipeo: espacios de más y mayúsculas inconsistentes."""
    if variante == 0:
        return f"  {nombre.upper()}"
    if variante == 1:
        return f"{nombre.lower()}  "
    return nombre.replace(" ", "   ", 1)


def _elegir(rng: np.random.Generator, probs: dict[str, float]) -> str:
    claves = list(probs)
    p = np.array([probs[k] for k in claves], dtype=float)
    return str(rng.choice(claves, p=p / p.sum()))


# ─────────────────────────────── ventas ─────────────────────────────────────

def generar_ventas(rng: np.random.Generator, precios: dict[str, float],
                   clientes: list[str], pesos_clientes: np.ndarray,
                   vendedores: list[str]) -> pd.DataFrame:
    """Genera las operaciones de venta de autos a comisión."""
    meses = _meses()
    w = np.array([_peso_tendencia(i) for i in range(len(meses))])
    fechas = sorted(_dia_aleatorio(rng, meses[i])
                    for i in rng.choice(len(meses), size=N_VENTAS, p=w / w.sum()))

    catalogo = list(PESO_MODELO)
    p_modelo = np.array(list(PESO_MODELO.values()))
    p_modelo = p_modelo / p_modelo.sum()
    precio_real = {**precios, **PRECIOS_OCULTOS}

    # Celdas forzadas para que los casos borde existan siempre.
    forzados = {
        2: ["Corolla Cross", "Yaris"],
        8: ["Territory"],
        17: ["Hilux", "Hilux"],          # mismo modelo repetido = 2 unidades
        33: ["Corolla Cross", "Taos"],   # mezcla con y sin precio
        41: ["Onix", "tracker"],         # minúscula inconsistente
    }

    filas = []
    for i, fecha in enumerate(fechas):
        if i in forzados:
            unidades = forzados[i]
        else:
            n = int(rng.choice([1, 2, 3], p=[0.75, 0.20, 0.05]))
            unidades = [str(m) for m in rng.choice(catalogo, size=n, p=p_modelo)]

        condicion = "Usado" if rng.random() < 0.35 else "Nuevo"
        forma_pago = _elegir(rng, PROB_PAGO)
        base = sum(precio_real[u.title() if u.islower() else u] for u in unidades)
        factura = round(base * FACTOR_CONDICION[condicion] * FACTOR_PAGO[forma_pago]
                        * rng.uniform(0.96, 1.04), -2)
        comision_pct = float(rng.choice(np.arange(3.0, 8.01, 0.5)))

        estado = _elegir(rng, {"Entregado": 0.80, "Reservado": 0.14, "Devuelto": 0.06})
        p_cobro = {"Entregado": 0.85, "Reservado": 0.30, "Devuelto": 0.0}[estado]
        separador = "," if i == 41 else ", "

        filas.append({
            "id_operacion": f"OP-{i + 1:03d}",
            "fecha_venta": fecha.isoformat(),
            "cliente": str(rng.choice(clientes, p=pesos_clientes)),
            "modelo": separador.join(unidades),
            "condicion": condicion,
            "origen_lead": _elegir(rng, PROB_LEAD),
            "forma_pago": forma_pago,
            "estado": estado,
            "vendedor": str(rng.choice(vendedores, p=[0.30, 0.25, 0.20, 0.15, 0.10])),
            "factura_sin_iva": factura,
            "comision_pct": comision_pct,
            "comision_monto": round(factura * comision_pct / 100, 2),
            "cobrado": "si" if rng.random() < p_cobro else "no",
        })

    df = pd.DataFrame(filas).astype({"factura_sin_iva": object, "comision_monto": object})

    # ── Suciedad intencional ──
    df.loc[12, ["factura_sin_iva", "comision_monto"]] = [None, None]   # cobrado sin monto
    df.loc[12, ["estado", "cobrado"]] = ["Entregado", "si"]
    df.loc[27, "fecha_venta"] = "2052-06-18"                           # typo de año
    for idx, variante in [(5, 0), (38, 2)]:
        df.loc[idx, "cliente"] = _ensuciar_nombre(df.loc[idx, "cliente"], variante)
    for idx in (9, 22, 47):                                            # formato planilla es-AR
        df.loc[idx, "factura_sin_iva"] = _formato_ar(float(df.loc[idx, "factura_sin_iva"]))
    return df


# ─────────────────────────────── servicios ──────────────────────────────────

def _estado_por_antiguedad(rng: np.random.Generator, fecha: date) -> str:
    """Las órdenes recientes todavía están abiertas; las viejas, mayormente cobradas."""
    dias = (FIN - fecha).days
    if dias <= 25:
        p = [0.15, 0.15, 0.20, 0.15, 0.05, 0.30]
    elif dias <= 90:
        p = [0.50, 0.20, 0.10, 0.08, 0.07, 0.05]
    else:
        p = [0.74, 0.07, 0.03, 0.07, 0.09, 0.00]
    return str(rng.choice(ESTADOS_SERVICIO, p=p))


def generar_servicios(rng: np.random.Generator, clientes: list[str],
                      pesos_clientes: np.ndarray, mecanicos: list[str]) -> pd.DataFrame:
    """Genera las órdenes de trabajo del taller con estacionalidad y tendencia."""
    meses = _meses()
    nombres_serv = list(SERVICIOS)
    w_mes = np.array([
        _peso_tendencia(i) * sum(SERVICIOS[s]["peso"] * _estacionalidad(s, m.month) for s in nombres_serv)
        for i, m in enumerate(meses)
    ])
    idx_meses = sorted(rng.choice(len(meses), size=N_SERVICIOS, p=w_mes / w_mes.sum()))
    vehiculos = list(PESO_MODELO) + VEHICULOS_EXTRA

    filas = []
    for n, i_mes in enumerate(idx_meses):
        mes = meses[i_mes]
        p_serv = np.array([SERVICIOS[s]["peso"] * _estacionalidad(s, mes.month) for s in nombres_serv])
        servicio = str(rng.choice(nombres_serv, p=p_serv / p_serv.sum()))
        spec = SERVICIOS[servicio]

        ingreso = _dia_aleatorio(rng, mes)
        estado = _estado_por_antiguedad(rng, ingreso)
        horas = _medias_horas(rng, *spec["horas"])
        precio_hora = spec["precio_hora"]
        monto = round(horas * precio_hora * (1 + spec["materiales"]) * rng.uniform(0.95, 1.08), 2)

        entrega = ""
        if estado in {"Cobrado", "Facturado esperando cobro", "Trabajo realizado"}:
            fin = ingreso + timedelta(days=int(rng.integers(spec["dias"][0], spec["dias"][1] + 1)))
            entrega = min(fin, FIN).isoformat()

        fila = {
            "id_orden": f"OT-{n + 1:03d}",
            "fecha_ingreso": ingreso.isoformat(),
            "fecha_entrega": entrega,
            "cliente": str(rng.choice(clientes, p=pesos_clientes)),
            "servicio": servicio,
            "vehiculo": str(rng.choice(vehiculos)),
            "estado": estado,
            "horas_trabajo": horas,
            "precio_hora": precio_hora,
            "monto_total": monto,
        }

        asignados: list[tuple[str, float]] = []
        if estado not in {"Presupuesto enviado", "Cancelado"}:
            k = int(rng.choice([1, 2, 3], p=spec["mecanicos"]))
            equipo = rng.choice(mecanicos, size=k, replace=False)
            if k == 1:
                asignados = [(str(equipo[0]), horas)]
            else:
                # Trabajan en simultáneo: la suma individual supera horas_trabajo.
                asignados = [(str(m), _medias_horas(rng, horas * 0.5, horas)) for m in equipo]
        for j in range(3):
            nombre, h = asignados[j] if j < len(asignados) else ("", "")
            fila[f"mecanico_{j + 1}"] = nombre
            fila[f"horas_{j + 1}"] = h
        filas.append(fila)

    df = pd.DataFrame(filas).astype({"monto_total": object, "horas_trabajo": object})

    # ── Suciedad intencional ──
    cobradas = df.index[df["estado"] == "Cobrado"].tolist()
    for idx in rng.choice(cobradas, size=4, replace=False):             # cobrado sin monto
        df.loc[idx, "monto_total"] = None
    df.loc[73, "fecha_ingreso"] = "1925-" + df.loc[73, "fecha_ingreso"][5:]  # typo de año
    for idx, variante in [(11, 0), (64, 1), (120, 2), (171, 0)]:
        df.loc[idx, "cliente"] = _ensuciar_nombre(df.loc[idx, "cliente"], variante)
    candidatos = [i for i in df.index if pd.notna(df.loc[i, "monto_total"])]
    for idx in rng.choice(candidatos, size=6, replace=False):          # formato planilla es-AR
        df.loc[idx, "monto_total"] = _formato_ar(float(df.loc[idx, "monto_total"]))
    for idx in (30, 95, 150):                                           # coma decimal
        df.loc[idx, "horas_trabajo"] = str(df.loc[idx, "horas_trabajo"]).replace(".", ",")
    return df


# ─────────────────────────────── main ───────────────────────────────────────

def main() -> None:
    """Genera ambos CSV en data/ a partir de la semilla fija."""
    rng = np.random.default_rng(SEED)
    fake = Faker("es_AR")
    Faker.seed(SEED)

    with open(DATA_DIR / "precios_referencia.json", encoding="utf-8") as f:
        precios = json.load(f)["precios"]

    personas = list(dict.fromkeys(f"{fake.first_name()} {fake.last_name()}" for _ in range(140)))
    vendedores, mecanicos, clientes = personas[:5], personas[5:10], personas[10:120]
    # Pocos clientes concentran muchas visitas (distribución tipo Pareto).
    pesos_clientes = 1 / np.arange(1, len(clientes) + 1) ** 0.8
    pesos_clientes = pesos_clientes / pesos_clientes.sum()

    ventas = generar_ventas(rng, precios, clientes, pesos_clientes, vendedores)
    servicios = generar_servicios(rng, clientes, pesos_clientes, mecanicos)

    DATA_DIR.mkdir(exist_ok=True)
    ventas.to_csv(DATA_DIR / "ventas_autos.csv", index=False, encoding="utf-8")
    servicios.to_csv(DATA_DIR / "servicios_taller.csv", index=False, encoding="utf-8")
    print(f"ventas_autos.csv:     {len(ventas)} filas")
    print(f"servicios_taller.csv: {len(servicios)} filas")


if __name__ == "__main__":
    main()
