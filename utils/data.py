"""Carga, limpieza (ETL) y unificación de los datos de AutoPulse.

Este módulo no depende de Streamlit: las páginas lo envuelven con
``st.cache_data`` y los tests lo importan directamente.

Regla de negocio central
------------------------
* **Servicios de taller**: el ingreso es el ``monto_total`` facturado.
* **Venta de autos**: la factura pasa por AutoPulse pero el ingreso real es la
  ``comision_monto``. La ``factura_sin_iva`` es *volumen intermediado* y nunca
  se suma al ingreso.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = RAIZ / "data"

FECHA_MIN_VALIDA = pd.Timestamp("2015-01-01")

ESTADOS_SERVICIO = [
    "Cobrado", "Facturado esperando cobro", "Trabajo realizado",
    "Presupuesto enviado", "En proceso", "Cancelado",
]
ESTADOS_SERVICIO_DEFAULT = ["Cobrado", "Facturado esperando cobro", "Trabajo realizado"]
CATEGORIA_SERVICIO = {
    "Lavado premium": "Estética", "Detailing completo": "Estética", "Pulido y encerado": "Estética",
    "Mecánica general": "Mecánica", "Cambio de aceite": "Mecánica",
    "Alineación y balanceo": "Mecánica", "Diagnóstico electrónico": "Mecánica",
    "Chapa y pintura": "Chapa y pintura",
}
ESTADOS_VENTA = ["Entregado", "Reservado", "Devuelto"]
ESTADOS_VENTA_DEFAULT = ["Entregado", "Reservado"]

UNIDAD_TALLER = "Servicios de taller"
UNIDAD_AUTOS = "Venta de autos"
PENDIENTE_PRECIO = "Pendiente de precio"

COBRO_COBRADO = "Cobrado"
COBRO_POR_COBRAR = "Por cobrar"
COBRO_POR_FACTURAR = "Por facturar"
ESTADOS_COBRO = [COBRO_COBRADO, COBRO_POR_COBRAR, COBRO_POR_FACTURAR]

_COBRO_SERVICIO = {
    "Cobrado": COBRO_COBRADO,
    "Facturado esperando cobro": COBRO_POR_COBRAR,
    "Trabajo realizado": COBRO_POR_FACTURAR,
}
_RE_MILES_PUNTO = re.compile(r"^-?\d{1,3}(\.\d{3})+$")
_RE_MILES_COMA = re.compile(r"^-?\d{1,3}(,\d{3})+$")
_RE_MECANICO = re.compile(r"^mecanico_(\d+)$")


# ─────────────────────────── parseo elemental ───────────────────────────────

def _es_nulo(valor) -> bool:
    return valor is None or (not isinstance(valor, str) and pd.isna(valor))


def parsear_monto(valor) -> float:
    """Convierte un monto o cantidad escrito a mano en ``float``.

    Acepta números ya tipados y textos exportados desde planillas en formato
    es-AR o en-US. Devuelve ``NaN`` si el valor está vacío o no es numérico.

    >>> parsear_monto("$ 1.234,50")
    1234.5
    >>> parsear_monto("2,5")
    2.5
    >>> parsear_monto("32.000")
    32000.0
    >>> parsear_monto("27100.0")
    27100.0

    Ambigüedad resuelta: un único separador seguido de exactamente tres dígitos
    (``"1.500"`` o ``"1,500"``) se interpreta como separador de miles.
    """
    if _es_nulo(valor):
        return np.nan
    if isinstance(valor, (int, float, np.integer, np.floating)) and not isinstance(valor, bool):
        return float(valor)

    texto = re.sub(r"[^\d,.\-]", "", str(valor))
    if texto in {"", "-"}:
        return np.nan
    if "," in texto and "." in texto:
        # El separador que aparece último es el decimal.
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif "," in texto:
        texto = texto.replace(",", "") if _RE_MILES_COMA.match(texto) else texto.replace(",", ".")
    elif _RE_MILES_PUNTO.match(texto):
        texto = texto.replace(".", "")
    try:
        return float(texto)
    except ValueError:
        return np.nan


def normalizar_nombre(valor) -> str | float:
    """Colapsa espacios y unifica mayúsculas: ``'  JUAN   perez '`` → ``'Juan Perez'``."""
    if _es_nulo(valor):
        return np.nan
    limpio = " ".join(str(valor).split())
    return limpio.title() if limpio else np.nan


def split_modelos(celda, catalogo: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Separa una celda de modelos por coma. Cada aparición es una unidad.

    No deduplica: ``"Hilux, Hilux"`` son dos unidades. Los nombres se llevan a
    su forma canónica del ``catalogo`` sin importar mayúsculas; si no están en
    el catálogo se devuelven en formato título.

    >>> split_modelos("Corolla Cross,  yaris", ["Corolla Cross", "Yaris"])
    ['Corolla Cross', 'Yaris']
    """
    if _es_nulo(celda):
        return []
    canon = {m.lower(): m for m in catalogo}
    modelos = []
    for parte in str(celda).split(","):
        nombre = " ".join(parte.split())
        if nombre:
            modelos.append(canon.get(nombre.lower(), nombre.title()))
    return modelos


def prorratear_factura(monto: float, modelos: list[str], precios: dict[str, float]) -> list[float]:
    """Reparte ``monto`` entre las unidades en proporción a su precio de referencia.

    Devuelve un valor por unidad (mismo orden que ``modelos``). Si alguna unidad
    no tiene precio de referencia —o el monto está vacío— no hay base objetiva
    para repartir, así que todas las unidades quedan en ``NaN`` y la factura
    completa se reporta como *pendiente de precio*.

    >>> prorratear_factura(51000, ["Corolla Cross", "Yaris"], {"Corolla Cross": 32000, "Yaris": 19000})
    [32000.0, 19000.0]
    """
    if not modelos:
        return []
    if _es_nulo(monto) or any(m not in precios for m in modelos):
        return [np.nan] * len(modelos)
    base = sum(precios[m] for m in modelos)
    return [float(monto) * precios[m] / base for m in modelos]


# ─────────────────────────── referencia de precios ──────────────────────────

@dataclass(frozen=True)
class Referencia:
    """Precios de referencia por modelo y modelos conocidos sin precio."""

    precios: dict[str, float]
    sin_precio: list[str] = field(default_factory=list)

    @property
    def catalogo(self) -> list[str]:
        """Todos los modelos conocidos, con y sin precio."""
        return list(self.precios) + [m for m in self.sin_precio if m not in self.precios]


def cargar_referencia(ruta: Path = DATA_DIR / "precios_referencia.json") -> Referencia:
    """Lee ``precios_referencia.json``."""
    with open(ruta, encoding="utf-8") as f:
        crudo = json.load(f)
    return Referencia(
        precios={k: float(v) for k, v in crudo.get("precios", {}).items()},
        sin_precio=list(crudo.get("modelos_sin_precio", [])),
    )


# ─────────────────────────── limpieza por fuente ────────────────────────────

def _rango_valido(fechas: pd.Series, hoy: pd.Timestamp | None) -> pd.Series:
    hoy = hoy or pd.Timestamp.today().normalize()
    return fechas.between(FECHA_MIN_VALIDA, hoy) & fechas.notna()


def limpiar_ventas(crudo: pd.DataFrame, referencia: Referencia,
                   hoy: pd.Timestamp | None = None) -> pd.DataFrame:
    """Tipa y normaliza ``ventas_autos.csv``.

    Agrega ``modelos`` (lista), ``unidades``, ``pendiente_precio``,
    ``fecha_valida`` y ``estado_cobro``. Si falta ``comision_monto`` pero hay
    factura y porcentaje, la recalcula.
    """
    df = crudo.copy()
    df["cliente"] = df["cliente"].map(normalizar_nombre)
    df["vendedor"] = df["vendedor"].map(normalizar_nombre)
    df["fecha_venta"] = pd.to_datetime(df["fecha_venta"], errors="coerce")
    df["fecha_valida"] = _rango_valido(df["fecha_venta"], hoy)
    for col in ("factura_sin_iva", "comision_pct", "comision_monto"):
        df[col] = df[col].map(parsear_monto).astype(float)

    recalcular = df["comision_monto"].isna() & df["factura_sin_iva"].notna() & df["comision_pct"].notna()
    df.loc[recalcular, "comision_monto"] = df["factura_sin_iva"] * df["comision_pct"] / 100

    df["cobrado"] = df["cobrado"].astype(str).str.strip().str.lower().isin({"si", "sí", "true", "1"})
    df["modelos"] = df["modelo"].map(lambda c: split_modelos(c, referencia.catalogo))
    df["unidades"] = df["modelos"].map(len)
    df["pendiente_precio"] = df["modelos"].map(lambda ms: any(m not in referencia.precios for m in ms))
    df["estado_cobro"] = np.where(df["cobrado"], COBRO_COBRADO, COBRO_POR_COBRAR)
    return df


def limpiar_servicios(crudo: pd.DataFrame, hoy: pd.Timestamp | None = None) -> pd.DataFrame:
    """Tipa y normaliza ``servicios_taller.csv``.

    Agrega ``fecha_valida`` y ``estado_cobro`` (``NaN`` para estados que no
    generan ingreso: presupuestos, órdenes en proceso y canceladas).
    """
    df = crudo.copy()
    df["cliente"] = df["cliente"].map(normalizar_nombre)
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    df["fecha_entrega"] = pd.to_datetime(df["fecha_entrega"], errors="coerce")
    df["fecha_valida"] = _rango_valido(df["fecha_ingreso"], hoy)
    numericas = ["horas_trabajo", "precio_hora", "monto_total"] + [c for c in df if c.startswith("horas_")]
    for col in dict.fromkeys(numericas):
        df[col] = df[col].map(parsear_monto).astype(float)
    for col in (c for c in df if _RE_MECANICO.match(c)):
        df[col] = df[col].map(normalizar_nombre)
    df["estado_cobro"] = df["estado"].map(_COBRO_SERVICIO)
    return df


# ─────────────────────────── transformaciones ───────────────────────────────

def expandir_unidades(ventas: pd.DataFrame, precios: dict[str, float]) -> pd.DataFrame:
    """Una fila por unidad vendida, con factura y comisión prorrateadas.

    ``ventas`` debe venir de :func:`limpiar_ventas`. Las unidades de
    operaciones con algún modelo sin precio tienen ``factura_asignada`` y
    ``comision_asignada`` en ``NaN``: cuentan como unidad, pero su dinero se
    reporta en el bucket *pendiente de precio*.
    """
    filas = []
    for op in ventas.to_dict("records"):
        modelos = op["modelos"]
        facturas = prorratear_factura(op["factura_sin_iva"], modelos, precios)
        comisiones = prorratear_factura(op["comision_monto"], modelos, precios)
        for modelo, fac, com in zip(modelos, facturas, comisiones):
            filas.append({
                "id_operacion": op["id_operacion"],
                "fecha_venta": op["fecha_venta"],
                "fecha_valida": op["fecha_valida"],
                "modelo": modelo,
                "tiene_precio": modelo in precios,
                "pendiente_precio": op["pendiente_precio"],
                "factura_asignada": fac,
                "comision_asignada": com,
                "condicion": op["condicion"],
                "estado": op["estado"],
                "cobrado": op["cobrado"],
                "vendedor": op["vendedor"],
            })
    columnas = ["id_operacion", "fecha_venta", "fecha_valida", "modelo", "tiene_precio",
                "pendiente_precio", "factura_asignada", "comision_asignada", "condicion",
                "estado", "cobrado", "vendedor"]
    return pd.DataFrame(filas, columns=columnas)


def desagregar_mecanicos(servicios: pd.DataFrame) -> pd.DataFrame:
    """Pasa los pares ``mecanico_N`` / ``horas_N`` a formato largo.

    Resultado: una fila por orden + mecánico con ``horas``, ``n_mecanicos`` en
    la orden y ``modalidad`` (``"Solo"`` / ``"Acompañado"``). Las horas por
    mecánico pueden sumar más que ``horas_trabajo`` porque trabajan en
    simultáneo; por eso el total de horas del taller NO se calcula desde acá.
    """
    base = [c for c in ("id_orden", "fecha_ingreso", "fecha_valida", "servicio", "estado")
            if c in servicios]
    indices = sorted(int(m.group(1)) for c in servicios if (m := _RE_MECANICO.match(c)))
    partes = [
        servicios[base + [f"mecanico_{n}", f"horas_{n}"]]
        .rename(columns={f"mecanico_{n}": "mecanico", f"horas_{n}": "horas"})
        for n in indices
    ]
    columnas = base + ["mecanico", "horas", "n_mecanicos", "modalidad"]
    if not partes:
        return pd.DataFrame(columns=columnas)

    largo = pd.concat(partes, ignore_index=True)
    largo["mecanico"] = largo["mecanico"].map(normalizar_nombre)
    largo = largo[largo["mecanico"].notna()].copy()
    largo["horas"] = largo["horas"].map(parsear_monto).astype(float)
    largo["n_mecanicos"] = largo.groupby("id_orden")["mecanico"].transform("size").astype(int)
    largo["modalidad"] = np.where(largo["n_mecanicos"] == 1, "Solo", "Acompañado")
    return largo.sort_values(["id_orden", "mecanico"]).reset_index(drop=True)[columnas]


def unificar_ingresos(ventas: pd.DataFrame, servicios: pd.DataFrame) -> pd.DataFrame:
    """Tabla única de movimientos de ambas unidades de negocio.

    Columnas: ``fecha``, ``fecha_valida``, ``unidad``, ``id``, ``cliente``,
    ``concepto``, ``estado``, ``estado_cobro``, ``ingreso`` y ``volumen``.

    * Taller: ``ingreso`` = ``volumen`` = ``monto_total``.
    * Autos:  ``ingreso`` = ``comision_monto``; ``volumen`` = ``factura_sin_iva``.
    """
    taller = pd.DataFrame({
        "fecha": servicios["fecha_ingreso"],
        "fecha_valida": servicios["fecha_valida"],
        "unidad": UNIDAD_TALLER,
        "id": servicios["id_orden"],
        "cliente": servicios["cliente"],
        "concepto": servicios["servicio"],
        "estado": servicios["estado"],
        "estado_cobro": servicios["estado_cobro"],
        "ingreso": servicios["monto_total"],
        "volumen": servicios["monto_total"],
    })
    autos = pd.DataFrame({
        "fecha": ventas["fecha_venta"],
        "fecha_valida": ventas["fecha_valida"],
        "unidad": UNIDAD_AUTOS,
        "id": ventas["id_operacion"],
        "cliente": ventas["cliente"],
        "concepto": ventas["modelo"],
        "estado": ventas["estado"],
        "estado_cobro": ventas["estado_cobro"],
        "ingreso": ventas["comision_monto"],
        "volumen": ventas["factura_sin_iva"],
    })
    return pd.concat([taller, autos], ignore_index=True)


# ─────────────────────────── filtros y KPIs ─────────────────────────────────

def _en_periodo(fechas: pd.Series, validas: pd.Series, desde, hasta) -> pd.Series:
    fin = pd.Timestamp(hasta) + pd.Timedelta(days=1)
    return validas & (fechas >= pd.Timestamp(desde)) & (fechas < fin)


def filtrar_servicios(servicios: pd.DataFrame, desde, hasta, estados: list[str]) -> pd.DataFrame:
    """Órdenes con fecha de ingreso válida dentro del período y en ``estados``."""
    mask = _en_periodo(servicios["fecha_ingreso"], servicios["fecha_valida"], desde, hasta)
    return servicios[mask & servicios["estado"].isin(estados)]


def filtrar_ventas(ventas: pd.DataFrame, desde, hasta, estados: list[str]) -> pd.DataFrame:
    """Operaciones con fecha de venta válida dentro del período y en ``estados``."""
    mask = _en_periodo(ventas["fecha_venta"], ventas["fecha_valida"], desde, hasta)
    return ventas[mask & ventas["estado"].isin(estados)]


def calcular_kpis(ventas: pd.DataFrame, servicios: pd.DataFrame) -> dict[str, float]:
    """KPIs de negocio sobre datos ya filtrados.

    ``ingreso_total`` = servicios + comisiones. ``volumen_intermediado`` es la
    facturación de autos y queda fuera del ingreso y de la participación.
    """
    ingreso_taller = float(servicios["monto_total"].sum())
    comisiones = float(ventas["comision_monto"].sum())
    ingreso_total = ingreso_taller + comisiones

    def cobro(estado: str) -> float:
        return float(servicios.loc[servicios["estado_cobro"] == estado, "monto_total"].sum()
                     + ventas.loc[ventas["estado_cobro"] == estado, "comision_monto"].sum())

    return {
        "ingreso_total": ingreso_total,
        "ingreso_taller": ingreso_taller,
        "comisiones": comisiones,
        "volumen_intermediado": float(ventas["factura_sin_iva"].sum()),
        "participacion_taller": ingreso_taller / ingreso_total if ingreso_total else 0.0,
        "participacion_autos": comisiones / ingreso_total if ingreso_total else 0.0,
        "cobrado": cobro(COBRO_COBRADO),
        "por_cobrar": cobro(COBRO_POR_COBRAR),
        "por_facturar": cobro(COBRO_POR_FACTURAR),
        "ordenes": int(len(servicios)),
        "operaciones": int(len(ventas)),
        "unidades": int(ventas["unidades"].sum()),
        "horas_taller": float(servicios["horas_trabajo"].sum()),
    }


def serie_mensual(df: pd.DataFrame, col_fecha: str, col_valor: str,
                  por: str | None = None) -> pd.DataFrame:
    """Suma ``col_valor`` por mes (y opcionalmente por ``por``)."""
    claves = [df[col_fecha].dt.to_period("M").dt.to_timestamp().rename("mes")]
    if por:
        claves.append(df[por])
    return df.groupby(claves)[col_valor].sum().reset_index()


# ─────────────────────────── calidad de datos ───────────────────────────────

def perfilar(df: pd.DataFrame, n_ejemplos: int = 3) -> pd.DataFrame:
    """Perfil por columna: dtype, % de nulos, cantidad de únicos y ejemplos."""
    filas = []
    for col in df.columns:
        serie = df[col]
        vacios = serie.isna() | (serie.astype(str).str.strip() == "")
        no_nulos = serie[~vacios]
        filas.append({
            "columna": col,
            "dtype": str(serie.dtype),
            "% nulos": round(100 * vacios.mean(), 1),
            "únicos": int(no_nulos.nunique()),
            "ejemplos": " · ".join(map(str, no_nulos.drop_duplicates().head(n_ejemplos))),
        })
    return pd.DataFrame(filas)


@dataclass
class Alerta:
    """Hallazgo de calidad de datos. No interrumpe la app."""

    nivel: str          # "error" | "warning" | "info"
    fuente: str
    titulo: str
    detalle: str
    filas: pd.DataFrame


def detectar_alertas(datos: "DatosAutoPulse") -> list[Alerta]:
    """Reglas de calidad sobre los datos cargados. Devuelve sólo las que tienen hallazgos."""
    v, s = datos.ventas, datos.servicios
    vc, sc = datos.ventas_crudo, datos.servicios_crudo
    alertas: list[Alerta] = []

    def agregar(nivel, fuente, titulo, detalle, mask, df, columnas):
        if mask.any():
            alertas.append(Alerta(nivel, fuente, titulo, detalle, df.loc[mask, columnas]))

    agregar("error", "servicios_taller.csv", "Orden cobrada sin monto",
            "Estado 'Cobrado' pero monto_total vacío o 0: ese ingreso no se está contabilizando.",
            (s["estado"] == "Cobrado") & (s["monto_total"].fillna(0) == 0), sc,
            ["id_orden", "fecha_ingreso", "cliente", "servicio", "estado", "monto_total"])
    agregar("error", "ventas_autos.csv", "Operación cobrada sin monto",
            "cobrado = si pero factura o comisión vacía o 0.",
            v["cobrado"] & (v["factura_sin_iva"].fillna(0) == 0), vc,
            ["id_operacion", "fecha_venta", "cliente", "modelo", "factura_sin_iva", "comision_monto", "cobrado"])

    fin = pd.Timestamp.today().normalize()
    rango = f"entre {FECHA_MIN_VALIDA:%d/%m/%Y} y hoy"
    agregar("warning", "servicios_taller.csv", "Fecha de ingreso fuera de rango",
            f"Se excluye de los análisis por período (rango válido: {rango}).",
            ~s["fecha_valida"], sc, ["id_orden", "fecha_ingreso", "cliente", "servicio", "monto_total"])
    agregar("warning", "ventas_autos.csv", "Fecha de venta fuera de rango",
            f"Se excluye de los análisis por período (rango válido: {rango}).",
            ~v["fecha_valida"], vc, ["id_operacion", "fecha_venta", "cliente", "modelo", "factura_sin_iva"])
    agregar("warning", "servicios_taller.csv", "Entrega anterior al ingreso",
            "fecha_entrega < fecha_ingreso.",
            s["fecha_entrega"].notna() & s["fecha_valida"] & (s["fecha_entrega"] < s["fecha_ingreso"]), sc,
            ["id_orden", "fecha_ingreso", "fecha_entrega"])

    sin_precio = sorted({m for ms in v["modelos"] for m in ms if m not in datos.referencia.precios})
    agregar("warning", "ventas_autos.csv", "Modelos sin precio de referencia",
            f"Modelos: {', '.join(sin_precio)}. Cuentan como unidad, pero su factura no se reparte por "
            "modelo. Cargar el precio en data/precios_referencia.json.",
            v["pendiente_precio"], vc, ["id_operacion", "fecha_venta", "modelo", "factura_sin_iva"])

    for fuente, crudo, cols in (("ventas_autos.csv", vc, ["cliente", "vendedor"]),
                                ("servicios_taller.csv", sc, ["cliente"])):
        mask = pd.Series(False, index=crudo.index)
        for col in cols:
            texto = crudo[col].astype(str)
            mask |= crudo[col].notna() & (texto != texto.map(normalizar_nombre))
        agregar("info", fuente, "Nombres con formato inconsistente",
                "Espacios de más o mayúsculas mezcladas. Se normalizan automáticamente.",
                mask, crudo, [c for c in ["id_operacion", "id_orden"] if c in crudo] + cols)

    for fuente, crudo, cols in (("ventas_autos.csv", vc, ["factura_sin_iva", "comision_monto"]),
                                ("servicios_taller.csv", sc, ["monto_total", "horas_trabajo"])):
        mask = pd.Series(False, index=crudo.index)
        for col in cols:
            texto = crudo[col].astype(str).str.strip()
            mask |= crudo[col].notna() & ~texto.str.fullmatch(r"-?\d+(\.\d+)?")
        agregar("info", fuente, "Números cargados como texto",
                "Formato de planilla (ej. '$ 1.234,50' o '2,5'). Se parsean automáticamente.",
                mask, crudo, [c for c in ["id_operacion", "id_orden"] if c in crudo] + cols)
    return alertas


# ─────────────────────────── punto de entrada ───────────────────────────────

@dataclass
class DatosAutoPulse:
    """Todas las tablas crudas y procesadas que consume el dashboard."""

    referencia: Referencia
    ventas_crudo: pd.DataFrame
    servicios_crudo: pd.DataFrame
    ventas: pd.DataFrame
    servicios: pd.DataFrame
    unidades: pd.DataFrame
    mecanicos: pd.DataFrame
    ingresos: pd.DataFrame


def cargar_datos(data_dir: Path = DATA_DIR) -> DatosAutoPulse:
    """Lee los CSV y el JSON de ``data_dir`` y corre todo el ETL."""
    referencia = cargar_referencia(data_dir / "precios_referencia.json")
    ventas_crudo = pd.read_csv(data_dir / "ventas_autos.csv")
    servicios_crudo = pd.read_csv(data_dir / "servicios_taller.csv")
    ventas = limpiar_ventas(ventas_crudo, referencia)
    servicios = limpiar_servicios(servicios_crudo)
    return DatosAutoPulse(
        referencia=referencia,
        ventas_crudo=ventas_crudo,
        servicios_crudo=servicios_crudo,
        ventas=ventas,
        servicios=servicios,
        unidades=expandir_unidades(ventas, referencia.precios),
        mecanicos=desagregar_mecanicos(servicios),
        ingresos=unificar_ingresos(ventas, servicios),
    )
