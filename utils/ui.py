"""Helpers de Streamlit compartidos por las páginas: caché de datos y filtros globales."""

from __future__ import annotations

from datetime import date

import streamlit as st

from utils import data as D


@st.cache_data(show_spinner="Cargando datos…")
def datos() -> D.DatosAutoPulse:
    """Datos procesados, cacheados entre reruns y páginas."""
    return D.cargar_datos()


def rango_disponible() -> tuple[date, date]:
    """Primera y última fecha válida entre ambas unidades de negocio."""
    fechas = datos().ingresos.query("fecha_valida")["fecha"]
    return fechas.min().date(), fechas.max().date()


def sidebar_filtros() -> None:
    """Dibuja los filtros globales. Se llama desde ``app.py`` en cada rerun,
    así el estado persiste al navegar entre páginas."""
    fmin, fmax = rango_disponible()
    with st.sidebar:
        st.markdown("### Filtros globales")
        st.date_input("Período", value=(fmin, fmax), min_value=fmin, max_value=fmax,
                      format="DD/MM/YYYY", key="filtro_periodo")
        st.multiselect("Estado del servicio", D.ESTADOS_SERVICIO,
                       default=D.ESTADOS_SERVICIO_DEFAULT, key="filtro_estados",
                       help="Aplica a las órdenes de taller. Por defecto sólo los estados que generan ingreso.")
        st.caption("Datos 100% sintéticos · montos en US$")


def filtros() -> tuple[date, date, list[str]]:
    """``(desde, hasta, estados_servicio)`` según los filtros globales.

    Si el usuario está a mitad de elegir el rango (una sola fecha), usa esa
    fecha como inicio y el máximo disponible como fin.
    """
    fmin, fmax = rango_disponible()
    rango = st.session_state.get("filtro_periodo", (fmin, fmax))
    if not isinstance(rango, (tuple, list)):
        rango = (rango,)
    desde = rango[0] if len(rango) > 0 else fmin
    hasta = rango[1] if len(rango) > 1 else fmax
    estados = st.session_state.get("filtro_estados", D.ESTADOS_SERVICIO_DEFAULT)
    return desde, hasta, estados


def descripcion_periodo(desde: date, hasta: date) -> str:
    """``'01/01/2024 – 31/08/2026'``."""
    return f"{desde:%d/%m/%Y} – {hasta:%d/%m/%Y}"
