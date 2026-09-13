"""Reporte General: ingreso consolidado de AutoPulse y su composición."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import data as D
from utils import format as F
from utils import ui

d = ui.datos()
desde, hasta, estados = ui.filtros()
servicios = D.filtrar_servicios(d.servicios, desde, hasta, estados)
ventas = D.filtrar_ventas(d.ventas, desde, hasta, D.ESTADOS_VENTA_DEFAULT)
k = D.calcular_kpis(ventas, servicios)

st.title("Reporte General")
st.caption(
    f"Período {ui.descripcion_periodo(desde, hasta)} · Taller: {len(estados)} estado(s) seleccionados · "
    "Autos: operaciones entregadas y reservadas"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Ingreso AutoPulse", F.moneda(k["ingreso_total"]), border=True,
    help="Lo que efectivamente gana la empresa: monto facturado de servicios de taller "
         "+ comisiones por venta de autos. No incluye el precio de los vehículos.",
)
c2.metric(
    "Servicios de taller", F.moneda(k["ingreso_taller"]), border=True,
    help="Suma de monto_total de las órdenes en los estados filtrados. "
         "En el taller, lo que se factura es ingreso.",
)
c3.metric(
    "Comisiones de autos", F.moneda(k["comisiones"]), border=True,
    help="Suma de comision_monto. Es el ingreso de la unidad de venta de autos.",
)
c4.metric(
    "Volumen intermediado", F.moneda(k["volumen_intermediado"]), border=True,
    help="Facturación total de los autos vendidos (factura_sin_iva). Pasa por AutoPulse, "
         "pero es plata del vehículo, NO ingreso: por eso no suma al Ingreso AutoPulse "
         "ni entra en la participación por unidad.",
)

with st.expander("¿Por qué el volumen intermediado no es ingreso?"):
    st.markdown(
        f"Cuando AutoPulse vende un auto de {F.moneda(32000)}, la factura pasa por la empresa pero "
        f"casi todo ese dinero es del vendedor del vehículo. Lo que AutoPulse gana es la comisión "
        f"(3–8 %, unos {F.moneda(1600)}). En el taller, en cambio, todo lo facturado es ingreso. "
        f"Sumar la facturación de autos al ingreso inflaría el negocio "
        f"~{F.numero(k['volumen_intermediado'] / k['ingreso_total'] if k['ingreso_total'] else 0, 0)}x "
        f"en este período y haría parecer que el taller es irrelevante."
    )

if k["ingreso_total"] == 0:
    st.info("No hay movimientos para los filtros elegidos.")
    st.stop()

# ── Participación por unidad ──
st.subheader("Participación sobre el ingreso")
fig = go.Figure()
for unidad, valor, part in [
    (D.UNIDAD_TALLER, k["ingreso_taller"], k["participacion_taller"]),
    (D.UNIDAD_AUTOS, k["comisiones"], k["participacion_autos"]),
]:
    fig.add_bar(
        y=["Ingreso"], x=[part], name=unidad, orientation="h",
        marker_color=F.COLOR_UNIDAD[unidad],
        text=[f"{F.porcentaje(part, 0)} · {F.moneda_corta(valor)}"], textposition="inside",
        insidetextanchor="middle",
        hovertext=[f"{unidad}<br>{F.moneda(valor)} ({F.porcentaje(part)})"], hoverinfo="text",
    )
fig.update_layout(barmode="stack", bargap=0.1)
fig.update_xaxes(visible=False, range=[0, 1])
fig.update_yaxes(visible=False)
st.plotly_chart(F.estilo_figura(fig, altura=130), config={"displayModeBar": False})

# ── Evolución mensual ──
col_evo, col_cobro = st.columns([3, 2])
with col_evo:
    st.subheader("Evolución mensual del ingreso")
    ingresos = D.unificar_ingresos(ventas, servicios)
    serie = D.serie_mensual(ingresos, "fecha", "ingreso", por="unidad")
    serie["hover"] = [f"{F.etiqueta_mes(m)} · {u}<br>{F.moneda(v)}"
                      for m, u, v in zip(serie["mes"], serie["unidad"], serie["ingreso"])]
    fig = px.bar(serie, x="mes", y="ingreso", color="unidad", color_discrete_map=F.COLOR_UNIDAD,
                 custom_data=["hover"], category_orders={"unidad": [D.UNIDAD_TALLER, D.UNIDAD_AUTOS]})
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.15)
    st.plotly_chart(F.eje_moneda(F.estilo_figura(fig, altura=360)), config={"displayModeBar": False})

# ── Estado de cobranza ──
with col_cobro:
    st.subheader("Estado de cobranza")
    cobro = pd.concat([
        servicios[["estado_cobro", "monto_total"]].rename(columns={"monto_total": "monto"})
        .assign(unidad=D.UNIDAD_TALLER),
        ventas[["estado_cobro", "comision_monto"]].rename(columns={"comision_monto": "monto"})
        .assign(unidad=D.UNIDAD_AUTOS),
    ]).dropna(subset=["estado_cobro"])
    resumen = cobro.groupby(["estado_cobro", "unidad"], as_index=False)["monto"].sum()
    resumen["hover"] = [f"{e} · {u}<br>{F.moneda(v)}"
                        for e, u, v in zip(resumen["estado_cobro"], resumen["unidad"], resumen["monto"])]
    fig = px.bar(resumen, x="estado_cobro", y="monto", color="unidad", color_discrete_map=F.COLOR_UNIDAD,
                 custom_data=["hover"],
                 category_orders={"estado_cobro": D.ESTADOS_COBRO, "unidad": [D.UNIDAD_TALLER, D.UNIDAD_AUTOS]})
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(barmode="stack")
    st.plotly_chart(F.eje_moneda(F.estilo_figura(fig, altura=300)), config={"displayModeBar": False})
    m1, m2, m3 = st.columns(3)
    m1.metric("Cobrado", F.moneda_corta(k["cobrado"]))
    m2.metric("Por cobrar", F.moneda_corta(k["por_cobrar"]), help="Facturado y todavía no cobrado.")
    m3.metric("Por facturar", F.moneda_corta(k["por_facturar"]),
              help="Trabajos de taller terminados que aún no se facturaron.")

excluidas = int((~d.servicios["fecha_valida"]).sum() + (~d.ventas["fecha_valida"]).sum())
if excluidas:
    st.caption(f"⚠️ {excluidas} fila(s) con fecha fuera de rango quedan fuera de este reporte. "
               "Ver la página Calidad de Datos.")
