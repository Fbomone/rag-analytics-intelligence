"""Calidad de Datos: perfilado de los CSV crudos y alertas de carga."""

import pandas as pd
import streamlit as st

from utils import data as D
from utils import format as F
from utils import ui

d = ui.datos()
alertas = D.detectar_alertas(d)

st.title("Calidad de Datos")
st.caption("Se analizan los archivos crudos, antes del ETL y sin filtros. "
           "Las alertas informan: la app sigue funcionando y aplica las correcciones indicadas.")

niveles = pd.Series([a.nivel for a in alertas])
c1, c2, c3 = st.columns(3)
c1.metric("🔴 Errores", int((niveles == "error").sum()), border=True,
          help="Afectan los números: ingresos que no se están contabilizando.")
c2.metric("🟠 Advertencias", int((niveles == "warning").sum()), border=True,
          help="Filas excluidas o que no se pueden analizar completas.")
c3.metric("🔵 Informativas", int((niveles == "info").sum()), border=True,
          help="Problemas de formato que el ETL corrige solo.")

st.subheader("Alertas")
if not alertas:
    st.success("Sin hallazgos.")
ICONO = {"error": "🔴", "warning": "🟠", "info": "🔵"}
for alerta in alertas:
    with st.expander(f"{ICONO[alerta.nivel]} {alerta.titulo} — {alerta.fuente} · {len(alerta.filas)} fila(s)",
                     expanded=alerta.nivel == "error"):
        st.markdown(alerta.detalle)
        st.dataframe(alerta.filas, hide_index=True)

st.subheader("Perfilado por archivo")
tab_v, tab_s, tab_p = st.tabs(["ventas_autos.csv", "servicios_taller.csv", "precios_referencia.json"])
for tab, crudo in ((tab_v, d.ventas_crudo), (tab_s, d.servicios_crudo)):
    with tab:
        st.caption(f"{len(crudo)} filas · {crudo.shape[1]} columnas. "
                   "Un dtype `str` en una columna de montos u horas indica números cargados como texto.")
        st.dataframe(
            D.perfilar(crudo), hide_index=True,
            column_config={"% nulos": st.column_config.ProgressColumn("% nulos", format="%.1f%%",
                                                                      min_value=0, max_value=100)},
        )

with tab_p:
    vendidas = d.unidades.groupby("modelo").size()
    modelos = sorted(set(d.referencia.catalogo) | set(vendidas.index))
    st.dataframe(pd.DataFrame({
        "Modelo": modelos,
        "Precio de referencia": [F.moneda(d.referencia.precios.get(m)) for m in modelos],
        "Unidades vendidas": [int(vendidas.get(m, 0)) for m in modelos],
        "Estado": ["OK" if m in d.referencia.precios else D.PENDIENTE_PRECIO for m in modelos],
    }), hide_index=True)
