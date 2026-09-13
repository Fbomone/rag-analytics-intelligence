"""Motor RAG del Asistente IA de AutoPulse (Cohere API v2).

Pipeline:

1. **Contexto** — :func:`construir_bloques` resume los datos filtrados en
   bloques temáticos de texto (métricas agregadas, tops por dimensión, series
   mensuales y un diccionario de datos con las reglas de negocio). Nunca se
   manda el CSV crudo al LLM.
2. **Recuperación** — :class:`Recuperador` elige los bloques relevantes con
   embeddings de Cohere y similitud coseno. Si el embedding falla, cae a un
   matching por keywords. El diccionario y el resumen general van siempre.
3. **Generación** — :class:`AsistenteRAG` arma los mensajes (system prompt +
   historial + contexto + pregunta) y llama a ``ClientV2.chat``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils import data as D
from utils import format as F

MODELO_CHAT_DEFAULT = "command-a-plus-05-2026"
MODELO_EMBED_DEFAULT = "embed-v4.0"
MAX_MENSAJES_HISTORIAL = 8
BLOQUES_FIJOS = ("diccionario", "resumen")
ARCHIVO_DEMO = D.DATA_DIR / "demo_respuestas.json"

PREGUNTAS_SUGERIDAS = [
    "¿Qué servicio dejó más margen este año?",
    "¿Qué vendedor tuvo mejor desempeño?",
    "¿Cuánto tengo pendiente de cobro?",
    "¿Qué mecánico trabajó más horas?",
    "¿Qué modelo de auto vendió más unidades?",
]
PREGUNTAS_DEMO = PREGUNTAS_SUGERIDAS[:3]

SYSTEM_PROMPT = """Sos un analista de negocios de AutoPulse, una empresa con dos unidades: \
venta de autos a comisión y un taller de servicios (lavado, detailing, mecánica, chapa y pintura).

Reglas:
- Respondé en español, de forma concisa (idealmente menos de 150 palabras) y directa: primero la respuesta, después el sustento.
- SIEMPRE citá los números concretos del contexto en los que te basás (montos en US$, cantidades, período).
- Usá SOLO la información del CONTEXTO. Si el dato no está, decilo explícitamente ("no tengo ese dato") \
y aclarás qué información haría falta. Nunca inventes cifras ni supongas costos.
- En venta de autos el ingreso es la comisión, no la factura del vehículo. No sumes volumen intermediado al ingreso.
- Si la pregunta es ambigua (por ejemplo "margen" sin datos de costos), explicá la limitación y ofrecé la mejor aproximación disponible en el contexto.
- Podés usar listas cortas o una tabla pequeña si ayuda a leer los números."""


# ─────────────────────────── configuración ──────────────────────────────────

def obtener_api_key() -> str | None:
    """Lee ``COHERE_API_KEY`` del entorno o del ``.env`` en la raíz del proyecto."""
    load_dotenv(D.RAIZ / ".env", override=False)
    clave = (os.getenv("COHERE_API_KEY") or "").strip()
    return clave or None


def cargar_demo() -> list[dict[str, Any]]:
    """Respuestas precalculadas para el modo demo (sin API key)."""
    if not ARCHIVO_DEMO.exists():
        return []
    with open(ARCHIVO_DEMO, encoding="utf-8") as f:
        return json.load(f).get("respuestas", [])


# ─────────────────────────── 1. construcción del contexto ───────────────────

@dataclass(frozen=True)
class Bloque:
    """Un resumen temático indexable."""

    id: str
    titulo: str
    texto: str
    keywords: tuple[str, ...] = ()

    def como_contexto(self) -> str:
        """Texto del bloque tal como lo recibe el modelo."""
        return f"## {self.titulo}\n{self.texto}"


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def _h(horas: float) -> str:
    return f"{F.numero(horas, 1)} h"


def _bloque_diccionario(datos: D.DatosAutoPulse, hoy: pd.Timestamp, ultimo: pd.Timestamp) -> Bloque:
    precios = ", ".join(f"{m} {F.moneda(p)}" for m, p in datos.referencia.precios.items())
    sin_precio = ", ".join(datos.referencia.sin_precio) or "ninguno"
    texto = f"""Empresa ficticia AutoPulse (datos sintéticos). Todos los montos están en US$.
Fecha de hoy: {hoy:%d/%m/%Y}. Último dato disponible: {ultimo:%d/%m/%Y}.

REGLA DE NEGOCIO CLAVE
- Ingreso AutoPulse = monto de servicios de taller + comisiones de venta de autos.
- En venta de autos la factura del vehículo (factura_sin_iva) es VOLUMEN INTERMEDIADO, NO ingreso. El ingreso es comision_monto.
- En el taller todo lo facturado (monto_total) es ingreso.

VENTA DE AUTOS (ventas_autos.csv)
- modelo: puede listar varios separados por coma; cada aparición es una unidad ("Hilux, Hilux" = 2 unidades).
- estado: Entregado / Reservado / Devuelto. Los análisis incluyen Entregado y Reservado.
- comision_pct (3–8 %), comision_monto (ingreso), cobrado (si/no), vendedor, origen_lead, forma_pago, condicion (Nuevo/Usado).
- Precio de referencia por modelo: {precios}. Sin precio: {sin_precio}.
- Una factura con varios modelos se reparte según el precio de referencia. Si algún modelo no tiene precio, la factura entera queda "pendiente de precio" y no se asigna a ningún modelo.

SERVICIOS DE TALLER (servicios_taller.csv)
- servicio, vehiculo, cliente, estado: Cobrado, Facturado esperando cobro, Trabajo realizado, Presupuesto enviado, En proceso, Cancelado.
- horas_trabajo = horas reales de la orden. mecanico_N / horas_N = horas imputadas por cada mecánico; pueden sumar más que horas_trabajo porque trabajan en simultáneo.
- Categorías: Estética (lavado premium, detailing, pulido), Mecánica (mecánica general, cambio de aceite, alineación, diagnóstico), Chapa y pintura.

COBRANZA
- Cobrado. Por cobrar = facturado sin cobrar (taller "Facturado esperando cobro"; autos con cobrado = no).
- Por facturar = taller "Trabajo realizado" (terminado, sin facturar).

LIMITACIONES
- No hay datos de costos: NO se puede calcular margen real. La mejor aproximación es el ingreso por hora de trabajo y el ingreso total por servicio.
- No hay datos de stock, clientes potenciales ni objetivos de venta."""
    return Bloque("diccionario", "Diccionario de datos y reglas de negocio", texto)


def _bloque_resumen(k: dict, ventas: pd.DataFrame, servicios: pd.DataFrame,
                    periodo: str, estados: Sequence[str]) -> Bloque:
    lineas = [
        f"Período analizado: {periodo}.",
        f"Estados de taller incluidos: {', '.join(estados) or 'ninguno'}. Autos: Entregado y Reservado.",
        f"- Ingreso AutoPulse: {F.moneda(k['ingreso_total'])}",
        f"  - Servicios de taller: {F.moneda(k['ingreso_taller'])} ({F.porcentaje(k['participacion_taller'])})",
        f"  - Comisiones de autos: {F.moneda(k['comisiones'])} ({F.porcentaje(k['participacion_autos'])})",
        f"- Volumen intermediado de autos (no es ingreso): {F.moneda(k['volumen_intermediado'])}",
        f"- Órdenes de taller: {k['ordenes']} · horas reales trabajadas: {_h(k['horas_taller'])}",
        f"- Operaciones de autos: {k['operaciones']} · unidades vendidas: {k['unidades']}",
        "",
        "Ingreso por año:",
    ]
    anio_s = servicios.groupby(servicios["fecha_ingreso"].dt.year)
    anio_v = ventas.groupby(ventas["fecha_venta"].dt.year)
    taller, comis = anio_s["monto_total"].sum(), anio_v["comision_monto"].sum()
    meses = pd.concat([servicios["fecha_ingreso"], ventas["fecha_venta"]]).dt.to_period("M")
    for anio in sorted(set(taller.index) | set(comis.index)):
        t, c = float(taller.get(anio, 0)), float(comis.get(anio, 0))
        n_meses = meses[meses.dt.year == anio].nunique()
        lineas.append(f"- {anio} ({n_meses} meses con datos): total {F.moneda(t + c)} "
                      f"= taller {F.moneda(t)} + comisiones {F.moneda(c)}")
    return Bloque("resumen", "Resumen general del período", "\n".join(lineas))


def _bloque_servicios(servicios: pd.DataFrame, periodo: str, top_n: int) -> Bloque:
    g = (servicios.groupby("servicio")
         .agg(monto=("monto_total", "sum"), ordenes=("id_orden", "size"), horas=("horas_trabajo", "sum"))
         .sort_values("monto", ascending=False))
    lineas = [f"Período: {periodo}.", "Ingreso por tipo de servicio (mayor a menor):"]
    for s, r in g.iterrows():
        lineas.append(f"- {s} ({D.CATEGORIA_SERVICIO.get(s, '—')}): {F.moneda(r.monto)}, {int(r.ordenes)} órdenes, "
                      f"{_h(r.horas)}, {F.moneda(_div(r.monto, r.horas))}/hora, "
                      f"ticket promedio {F.moneda(_div(r.monto, r.ordenes))}")

    lineas.append("\nIngreso por servicio por año (ingreso · órdenes · US$/hora):")
    por_anio = servicios.assign(anio=servicios["fecha_ingreso"].dt.year)
    for anio, grupo in por_anio.groupby("anio"):
        ga = (grupo.groupby("servicio")
              .agg(monto=("monto_total", "sum"), ordenes=("id_orden", "size"), horas=("horas_trabajo", "sum"))
              .sort_values("monto", ascending=False))
        detalle = "; ".join(f"{s} {F.moneda(r.monto)} · {int(r.ordenes)} · {F.moneda(_div(r.monto, r.horas))}/h"
                            for s, r in ga.iterrows())
        lineas.append(f"- {anio}: {detalle}")

    veh = servicios.groupby("vehiculo")["monto_total"].agg(["sum", "size"]).nlargest(top_n, "sum")
    lineas.append(f"\nTop {top_n} vehículos atendidos por ingreso:")
    lineas += [f"- {v}: {F.moneda(r['sum'])} ({int(r['size'])} órdenes)" for v, r in veh.iterrows()]
    cli = servicios.groupby("cliente")["monto_total"].agg(["sum", "size"]).nlargest(top_n, "sum")
    lineas.append(f"\nTop {top_n} clientes del taller por ingreso:")
    lineas += [f"- {c}: {F.moneda(r['sum'])} ({int(r['size'])} órdenes)" for c, r in cli.iterrows()]
    return Bloque("servicios", "Servicios de taller", "\n".join(lineas), keywords=(
        "servicio", "taller", "lavado", "detailing", "pulido", "estetica", "mecanica", "aceite", "chapa",
        "pintura", "alineacion", "balanceo", "diagnostico", "margen", "rentable", "rentabilidad",
        "ganancia", "hora", "vehiculo", "cliente", "ticket"))


def _bloque_ventas(ventas: pd.DataFrame, unidades: pd.DataFrame, precios: dict[str, float],
                   periodo: str) -> Bloque:
    comisiones, volumen = ventas["comision_monto"].sum(), ventas["factura_sin_iva"].sum()
    lineas = [
        f"Período: {periodo}. Operaciones Entregadas y Reservadas.",
        f"- Comisiones (ingreso): {F.moneda(comisiones)} · volumen intermediado: {F.moneda(volumen)} · "
        f"comisión efectiva: {F.porcentaje(_div(comisiones, volumen))}",
        f"- Operaciones: {len(ventas)} · unidades: {len(unidades)}",
        "\nUnidades por modelo (factura y comisión prorrateadas por precio de referencia):",
    ]
    g = unidades.groupby("modelo").agg(
        unid=("id_operacion", "size"),
        factura=("factura_asignada", "sum"),
        comision=("comision_asignada", "sum"),
    ).sort_values("unid", ascending=False)
    for m, r in g.iterrows():
        if m in precios:
            lineas.append(f"- {m}: {int(r.unid)} unidades, factura asignada {F.moneda(r.factura)}, "
                          f"comisión asignada {F.moneda(r.comision)}")
        else:
            lineas.append(f"- {m}: {int(r.unid)} unidades, SIN precio de referencia (monto no asignable al modelo)")
    pend = ventas[ventas["pendiente_precio"]]
    lineas.append(f"- Pendiente de precio: {len(pend)} operaciones, factura {F.moneda(pend['factura_sin_iva'].sum())}, "
                  f"comisión {F.moneda(pend['comision_monto'].sum())}")

    for col, titulo in (("forma_pago", "Forma de pago"), ("origen_lead", "Canal de lead"), ("condicion", "Condición")):
        agg = ventas.groupby(col).agg(ops=("id_operacion", "size"), com=("comision_monto", "sum"),
                                      vol=("factura_sin_iva", "sum")).sort_values("ops", ascending=False)
        lineas.append(f"\n{titulo}:")
        lineas += [f"- {c}: {int(r.ops)} operaciones, comisión {F.moneda(r.com)}, volumen {F.moneda(r.vol)}"
                   for c, r in agg.iterrows()]

    anual = ventas.groupby(ventas["fecha_venta"].dt.year).agg(
        ops=("id_operacion", "size"), unid=("unidades", "sum"), com=("comision_monto", "sum"))
    lineas.append("\nPor año:")
    lineas += [f"- {a}: {int(r.ops)} operaciones, {int(r.unid)} unidades, comisiones {F.moneda(r.com)}"
               for a, r in anual.iterrows()]
    return Bloque("ventas", "Venta de autos", "\n".join(lineas), keywords=(
        "auto", "autos", "venta", "vendio", "modelo", "unidad", "comision", "factura", "volumen",
        "precio", "pago", "financiado", "contado", "permuta", "cheque", "lead", "canal", "instagram",
        "referido", "mercadolibre", "showroom", "web", "usado", "nuevo", "hilux", "corolla", "yaris",
        "amarok", "onix", "tracker", "kicks", "territory", "taos"))


def _bloque_vendedores(ventas: pd.DataFrame, periodo: str) -> Bloque:
    v = ventas.assign(com_cobrada=ventas["comision_monto"].where(ventas["cobrado"], 0))
    g = v.groupby("vendedor").agg(
        ops=("id_operacion", "size"), unid=("unidades", "sum"), vol=("factura_sin_iva", "sum"),
        com=("comision_monto", "sum"), cobrada=("com_cobrada", "sum"), pct=("comision_pct", "mean"),
    ).sort_values("com", ascending=False)
    lineas = [f"Período: {periodo}. Ranking de vendedores por comisiones generadas:"]
    for i, (nombre, r) in enumerate(g.iterrows(), 1):
        lineas.append(
            f"{i}. {nombre}: comisiones {F.moneda(r.com)} (cobradas {F.moneda(r.cobrada)}, por cobrar "
            f"{F.moneda(r.com - r.cobrada)}), {int(r.ops)} operaciones, {int(r.unid)} unidades, volumen {F.moneda(r.vol)}, "
            f"comisión promedio por operación {F.moneda(_div(r.com, r.ops))}, % comisión promedio {F.numero(r.pct, 1)} %")
    if not v.empty:
        lineas.append("\nComisiones por vendedor por año:")
        pivot = v.pivot_table(index="vendedor", columns=v["fecha_venta"].dt.year, values="comision_monto",
                              aggfunc="sum", fill_value=0)
        for nombre, fila in pivot.iterrows():
            lineas.append(f"- {nombre}: " + "; ".join(f"{a} {F.moneda(x)}" for a, x in fila.items()))
    return Bloque("vendedores", "Desempeño de vendedores", "\n".join(lineas), keywords=(
        "vendedor", "vendedores", "desempeño", "desempeno", "ranking", "mejor", "peor", "equipo",
        "comercial", "vendio"))


def _bloque_mecanicos(mecanicos: pd.DataFrame, servicios: pd.DataFrame, periodo: str) -> Bloque:
    lineas = [f"Período: {periodo}.",
              f"Horas reales del taller (horas_trabajo): {_h(servicios['horas_trabajo'].sum())}. "
              f"Horas imputadas por mecánicos: {_h(mecanicos['horas'].sum())} "
              "(suman más porque varios trabajan en simultáneo).",
              "\nHoras imputadas por mecánico (mayor a menor):"]
    g = mecanicos.groupby("mecanico").agg(
        horas=("horas", "sum"), ordenes=("id_orden", "nunique"),
        solo=("modalidad", lambda s: int((s == "Solo").sum())),
    ).sort_values("horas", ascending=False)
    for nombre, r in g.iterrows():
        ordenes, solo = int(r.ordenes), int(r.solo)
        acomp = ordenes - solo
        lineas.append(f"- {nombre}: {_h(r.horas)} en {ordenes} órdenes ({solo} solo, {acomp} acompañado, "
                      f"{F.porcentaje(_div(acomp, r.ordenes), 0)} acompañado)")
    if not mecanicos.empty:
        pivot = mecanicos.pivot_table(index="mecanico", columns=mecanicos["fecha_ingreso"].dt.year,
                                      values="horas", aggfunc="sum", fill_value=0)
        lineas.append("\nHoras por mecánico por año:")
        for nombre, fila in pivot.iterrows():
            lineas.append(f"- {nombre}: " + "; ".join(f"{a} {_h(x)}" for a, x in fila.items()))
        top_serv = (mecanicos.groupby(["mecanico", "servicio"])["horas"].sum()
                    .reset_index().sort_values("horas", ascending=False).groupby("mecanico").head(2))
        lineas.append("\nServicios donde más horas imputa cada mecánico:")
        for nombre, grupo in top_serv.groupby("mecanico"):
            lineas.append(f"- {nombre}: " + ", ".join(f"{s} ({_h(h)})" for s, h in zip(grupo.servicio, grupo.horas)))
    return Bloque("mecanicos", "Trabajo por mecánico", "\n".join(lineas), keywords=(
        "mecanico", "mecanicos", "horas", "trabajo", "solo", "acompanado", "productividad",
        "personal", "empleado"))


def _bloque_cobranza(ventas: pd.DataFrame, servicios: pd.DataFrame, k: dict, periodo: str, top_n: int) -> Bloque:
    lineas = [
        f"Período: {periodo}.",
        f"- Cobrado: {F.moneda(k['cobrado'])}",
        f"- Por cobrar (facturado sin cobrar): {F.moneda(k['por_cobrar'])}",
        f"- Por facturar (trabajos terminados sin facturar): {F.moneda(k['por_facturar'])}",
        f"- Total pendiente (por cobrar + por facturar): {F.moneda(k['por_cobrar'] + k['por_facturar'])}",
        "\nDetalle por unidad:",
    ]
    for estado in D.ESTADOS_COBRO:
        s = servicios[servicios["estado_cobro"] == estado]
        v = ventas[ventas["estado_cobro"] == estado]
        lineas.append(f"- {estado}: taller {F.moneda(s['monto_total'].sum())} ({len(s)} órdenes); "
                      f"comisiones {F.moneda(v['comision_monto'].sum())} ({len(v)} operaciones)")

    def listar(df, titulo, fmt):
        if not df.empty:
            lineas.append(f"\n{titulo}:")
            lineas.extend(fmt(r) for r in df.itertuples())

    listar(servicios[servicios["estado"] == "Facturado esperando cobro"].nlargest(top_n, "monto_total"),
           f"Mayores órdenes de taller facturadas sin cobrar (top {top_n})",
           lambda r: f"- {r.id_orden} · {r.cliente} · {r.servicio} · ingreso {r.fecha_ingreso:%d/%m/%Y} · {F.moneda(r.monto_total)}")
    listar(servicios[servicios["estado"] == "Trabajo realizado"].nlargest(top_n, "monto_total"),
           f"Trabajos terminados sin facturar (top {top_n})",
           lambda r: f"- {r.id_orden} · {r.cliente} · {r.servicio} · {F.moneda(r.monto_total)}")
    listar(ventas[~ventas["cobrado"]].nlargest(top_n, "comision_monto"),
           f"Mayores comisiones de autos sin cobrar (top {top_n})",
           lambda r: f"- {r.id_operacion} · {r.modelo} · vendedor {r.vendedor} · {r.fecha_venta:%d/%m/%Y} · "
                     f"comisión {F.moneda(r.comision_monto)}")
    return Bloque("cobranza", "Cobranza y pendientes", "\n".join(lineas), keywords=(
        "cobro", "cobrar", "cobrado", "cobranza", "pendiente", "deuda", "deben", "debe", "facturar",
        "facturado", "pagar", "saldo", "plata", "caja"))


def _bloque_evolucion(ventas: pd.DataFrame, servicios: pd.DataFrame, periodo: str) -> Bloque:
    s = servicios.assign(ordenes=1)
    v = ventas.assign(ops=1)
    serie = (D.serie_mensual(s, "fecha_ingreso", "monto_total").set_index("mes")
             .join(D.serie_mensual(s, "fecha_ingreso", "ordenes").set_index("mes"), how="outer")
             .join(D.serie_mensual(v, "fecha_venta", "comision_monto").set_index("mes"), how="outer")
             .join(D.serie_mensual(v, "fecha_venta", "ops").set_index("mes"), how="outer")
             .fillna(0).sort_index())
    lineas = [f"Período: {periodo}. Serie mensual (taller · comisiones de autos · total):"]
    for mes, r in serie.iterrows():
        lineas.append(f"- {F.etiqueta_mes(mes)}: taller {F.moneda(r.monto_total)} ({int(r.ordenes)} órd.) · "
                      f"comisiones {F.moneda(r.comision_monto)} ({int(r.ops)} ops.) · "
                      f"total {F.moneda(r.monto_total + r.comision_monto)}")
    cat = servicios.assign(cat=servicios["servicio"].map(D.CATEGORIA_SERVICIO), mes=servicios["fecha_ingreso"].dt.month)
    if not cat.empty:
        tabla = cat.pivot_table(index="mes", columns="cat", values="id_orden", aggfunc="count", fill_value=0)
        lineas.append("\nEstacionalidad: órdenes por mes calendario y categoría (todos los años):")
        for mes, fila in tabla.iterrows():
            lineas.append(f"- {F.MESES_ES[mes - 1]}: " + ", ".join(f"{c} {int(n)}" for c, n in fila.items()))
    return Bloque("evolucion", "Evolución mensual y estacionalidad", "\n".join(lineas), keywords=(
        "mes", "meses", "mensual", "evolucion", "tendencia", "crecimiento", "crecio", "estacional",
        "estacionalidad", "temporada", "verano", "invierno", "vacaciones", "ano", "anual", "historico",
        "comparar", "comparacion", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
        "agosto", "septiembre", "octubre", "noviembre", "diciembre"))


def _bloque_calidad(datos: D.DatosAutoPulse) -> Bloque:
    alertas = D.detectar_alertas(datos)
    lineas = ["Hallazgos de calidad sobre los archivos crudos (afectan la confiabilidad de algunos números):"]
    lineas += [f"- [{a.nivel}] {a.titulo} ({a.fuente}): {len(a.filas)} fila(s). {a.detalle}" for a in alertas]
    if not alertas:
        lineas.append("- Sin hallazgos.")
    return Bloque("calidad", "Calidad de datos", "\n".join(lineas), keywords=(
        "calidad", "error", "errores", "dato", "datos", "faltante", "inconsistencia", "confiable",
        "problema", "alerta", "vacio", "carga"))


def construir_bloques(datos: D.DatosAutoPulse, desde, hasta, estados_servicio: Sequence[str],
                      top_n: int = 5, hoy=None) -> list[Bloque]:
    """Resume los datos del período filtrado en bloques temáticos para el LLM.

    Usa los mismos filtros y KPIs que el dashboard, así las respuestas del
    asistente coinciden con lo que muestran las páginas.
    """
    desde, hasta = pd.Timestamp(desde), pd.Timestamp(hasta)
    hoy = pd.Timestamp(hoy) if hoy is not None else pd.Timestamp.today().normalize()
    servicios = D.filtrar_servicios(datos.servicios, desde, hasta, list(estados_servicio))
    ventas = D.filtrar_ventas(datos.ventas, desde, hasta, D.ESTADOS_VENTA_DEFAULT)
    unidades = datos.unidades[datos.unidades["id_operacion"].isin(ventas["id_operacion"])]
    mecanicos = datos.mecanicos[datos.mecanicos["id_orden"].isin(servicios["id_orden"])]
    k = D.calcular_kpis(ventas, servicios)
    periodo = f"{desde:%d/%m/%Y} al {hasta:%d/%m/%Y}"
    ultimo = datos.ingresos.loc[datos.ingresos["fecha_valida"], "fecha"].max()
    return [
        _bloque_diccionario(datos, hoy, ultimo),
        _bloque_resumen(k, ventas, servicios, periodo, estados_servicio),
        _bloque_servicios(servicios, periodo, top_n),
        _bloque_ventas(ventas, unidades, datos.referencia.precios, periodo),
        _bloque_vendedores(ventas, periodo),
        _bloque_mecanicos(mecanicos, servicios, periodo),
        _bloque_cobranza(ventas, servicios, k, periodo, top_n),
        _bloque_evolucion(ventas, servicios, periodo),
        _bloque_calidad(datos),
    ]


# ─────────────────────────── 2. recuperación ────────────────────────────────

_STOPWORDS = {
    "que", "cual", "cuales", "como", "cuanto", "cuanta", "cuantos", "cuantas", "tengo", "tenemos", "este",
    "esta", "estos", "para", "los", "las", "del", "con", "mas", "por", "una", "uno", "unos", "hay", "fue",
    "son", "sus", "donde", "cuando", "quien", "quienes", "dame", "decime", "mostrame", "sobre", "entre",
}


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tildes.lower()


def _tokens(texto: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _normalizar(texto)) if len(t) >= 3 and t not in _STOPWORDS}


def _coincide(token: str, keyword: str) -> bool:
    """Igualdad exacta o prefijo común de 4–6 letras (stemming liviano: mecánico ≈ mecánicos)."""
    if token == keyword:
        return True
    p = min(len(token), len(keyword), 6)
    return p >= 4 and token[:p] == keyword[:p]


def recuperar_por_keywords(pregunta: str, bloques: Sequence[Bloque], k: int = 3) -> list[Bloque]:
    """Fallback sin embeddings: puntúa por keywords del tema y solapamiento de términos.

    Si ninguna palabra coincide, devuelve todos los bloques (el contexto es chico).
    """
    tokens = _tokens(pregunta)
    puntajes = []
    for b in bloques:
        kws = {_normalizar(w) for w in b.keywords}
        hits_kw = sum(1 for t in tokens if any(_coincide(t, w) for w in kws))
        solapamiento = len(tokens & _tokens(b.texto))
        puntajes.append(3 * hits_kw + 0.5 * solapamiento)
    if not any(puntajes):
        return list(bloques)
    orden = sorted(range(len(bloques)), key=lambda i: puntajes[i], reverse=True)
    return [bloques[i] for i in orden[:k] if puntajes[i] > 0]


def similitud_coseno(consulta: np.ndarray, matriz: np.ndarray) -> np.ndarray:
    """Similitud coseno entre un vector y cada fila de ``matriz``."""
    normas = np.linalg.norm(matriz, axis=1) * np.linalg.norm(consulta)
    normas[normas == 0] = 1e-12
    return matriz @ consulta / normas


class Recuperador:
    """Selecciona bloques relevantes con embeddings de Cohere; fallback a keywords.

    Los embeddings se cachean por hash de (modelo, tipo, texto), así un rerun de
    Streamlit con los mismos filtros no vuelve a llamar a la API.
    """

    _cache: dict[str, np.ndarray] = {}
    MAX_CACHE = 512

    def __init__(self, cliente: Any | None = None, modelo_embed: str | None = None):
        self.cliente = cliente
        self.modelo_embed = modelo_embed or os.getenv("COHERE_EMBED_MODEL") or MODELO_EMBED_DEFAULT

    def _clave(self, texto: str, tipo: str) -> str:
        return hashlib.sha1(f"{self.modelo_embed}|{tipo}|{texto}".encode()).hexdigest()

    def _embeddings(self, textos: list[str], tipo: str) -> np.ndarray:
        faltan = [t for t in dict.fromkeys(textos) if self._clave(t, tipo) not in self._cache]
        if faltan:
            resp = self.cliente.embed(model=self.modelo_embed, input_type=tipo, texts=faltan,
                                      embedding_types=["float"])
            vectores = resp.embeddings.float_
            if not vectores or len(vectores) != len(faltan):
                raise ValueError("Respuesta de embeddings incompleta")
            if len(self._cache) > self.MAX_CACHE:
                self._cache.clear()
            for texto, vector in zip(faltan, vectores):
                self._cache[self._clave(texto, tipo)] = np.asarray(vector, dtype=float)
        return np.vstack([self._cache[self._clave(t, tipo)] for t in textos])

    def recuperar(self, pregunta: str, bloques: Sequence[Bloque], k: int = 3) -> tuple[list[Bloque], str]:
        """Devuelve ``(bloques, metodo)``; ``metodo`` es ``"embeddings"`` o ``"keywords"``.

        Los bloques fijos (diccionario y resumen) se incluyen siempre.
        """
        fijos = [b for b in bloques if b.id in BLOQUES_FIJOS]
        candidatos = [b for b in bloques if b.id not in BLOQUES_FIJOS]
        if self.cliente is not None and candidatos:
            try:
                docs = self._embeddings([b.como_contexto() for b in candidatos], "search_document")
                consulta = self._embeddings([pregunta], "search_query")[0]
                sims = similitud_coseno(consulta, docs)
                elegidos = [candidatos[i] for i in np.argsort(-sims)[:k]]
                return fijos + elegidos, "embeddings"
            except Exception:  # noqa: BLE001 — cualquier falla de red/API cae al fallback
                pass
        return fijos + recuperar_por_keywords(pregunta, candidatos, k), "keywords"


# ─────────────────────────── 3. generación ──────────────────────────────────

@dataclass
class Respuesta:
    """Resultado de una consulta al asistente."""

    texto: str
    fuentes: list[str] = field(default_factory=list)
    metodo: str = "keywords"
    error: str | None = None


def construir_mensajes(pregunta: str, bloques: Sequence[Bloque],
                       historial: Sequence[dict[str, str]] = ()) -> list[dict[str, str]]:
    """Mensajes para ``ClientV2.chat``: system + historial reciente + contexto con la pregunta.

    El contexto viaja sólo en el último mensaje: el historial conserva
    preguntas y respuestas, no contextos viejos, para no inflar tokens.
    """
    contexto = "\n\n".join(b.como_contexto() for b in bloques)
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    mensajes += [{"role": m["role"], "content": m["content"]}
                 for m in list(historial)[-MAX_MENSAJES_HISTORIAL:]
                 if m.get("role") in {"user", "assistant"} and m.get("content")]
    mensajes.append({"role": "user", "content": f"CONTEXTO (datos agregados de AutoPulse)\n\n{contexto}\n\n"
                                                f"---\nPREGUNTA: {pregunta}"})
    return mensajes


def extraer_texto(respuesta: Any) -> str:
    """Concatena los bloques de tipo texto de una respuesta ``V2ChatResponse``."""
    contenido = getattr(respuesta.message, "content", None) or []
    return "".join(c.text for c in contenido if getattr(c, "type", "text") == "text").strip()


class AsistenteRAG:
    """Orquesta recuperación + generación sobre Cohere ``ClientV2``."""

    def __init__(self, api_key: str | None = None, *, cliente: Any | None = None,
                 modelo_chat: str | None = None, modelo_embed: str | None = None, k: int = 3):
        if cliente is None:
            import cohere  # import diferido: la app funciona en modo demo sin llamar a la API
            cliente = cohere.ClientV2(api_key=api_key, timeout=60)
        self.cliente = cliente
        self.modelo_chat = modelo_chat or os.getenv("COHERE_CHAT_MODEL") or MODELO_CHAT_DEFAULT
        self.recuperador = Recuperador(cliente, modelo_embed)
        self.k = k

    def responder(self, pregunta: str, bloques: Sequence[Bloque],
                  historial: Sequence[dict[str, str]] = ()) -> Respuesta:
        """Recupera contexto relevante y genera la respuesta. Nunca lanza: los errores vuelven en ``error``."""
        usados, metodo = self.recuperador.recuperar(pregunta, bloques, self.k)
        fuentes = [b.titulo for b in usados]
        try:
            resp = self.cliente.chat(model=self.modelo_chat, messages=construir_mensajes(pregunta, usados, historial),
                                     temperature=0.2)
            texto = extraer_texto(resp)
        except Exception as e:  # noqa: BLE001
            return Respuesta("No pude consultar al modelo en este momento. Probá de nuevo en unos segundos.",
                             fuentes, metodo, error=f"{type(e).__name__}: {str(e)[:300]}")
        return Respuesta(texto or "El modelo devolvió una respuesta vacía.", fuentes, metodo)
