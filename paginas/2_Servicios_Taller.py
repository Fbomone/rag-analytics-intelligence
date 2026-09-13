"""Servicios de Taller: ingresos por servicio, vehículo y cliente, y trabajo por mecánico."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import data as D
from utils import format as F
from utils import ui

d = ui.datos()
desde, hasta, estados = ui.filtros()
servicios = D.filtrar_servicios(d.servicios, desde, hasta, estados)

st.title("Servicios de Taller")
st.caption(f"Período {ui.descripcion_periodo(desde, hasta)} · Estados: {', '.join(estados) or '—'} "
           "(cambialos en la barra lateral)")

if servicios.empty:
    st.info("No hay órdenes para los filtros elegidos.")
    st.stop()

ingreso = servicios["monto_total"].sum()
horas = servicios["horas_trabajo"].sum()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ingreso del taller", F.moneda(ingreso), border=True,
          help="Suma de monto_total. En el taller todo lo facturado es ingreso.")
c2.metric("Órdenes", F.numero(len(servicios)), border=True)
c3.metric("Horas trabajadas", F.numero(horas, 1), border=True,
          help="Sale de horas_trabajo de cada orden, NO de la suma de horas por mecánico: "
               "varios mecánicos trabajan en simultáneo, así que sus horas individuales "
               "suman más que las horas reales de la orden.")
c4.metric("Ingreso por hora", F.moneda(ingreso / horas if horas else None), border=True,
          help="Ingreso del taller / horas trabajadas.")
c5.metric("Ticket promedio", F.moneda(ingreso / len(servicios)), border=True)

# ── Por servicio y por vehículo ──
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Ingreso por tipo de servicio")
    por_serv = servicios.groupby("servicio", as_index=False).agg(
        monto=("monto_total", "sum"), ordenes=("id_orden", "size"), horas=("horas_trabajo", "sum"))
    por_serv["texto"] = por_serv["monto"].map(F.moneda_corta)
    por_serv["hover"] = [
        f"<b>{s}</b><br>{F.moneda(m)} · {o} órdenes<br>{F.numero(h, 1)} h · {F.moneda(m / h if h else None)}/h"
        for s, m, o, h in zip(por_serv["servicio"], por_serv["monto"], por_serv["ordenes"], por_serv["horas"])
    ]
    st.plotly_chart(F.barras_h(por_serv, "servicio", "monto", texto="texto", hover="hover"),
                    config={"displayModeBar": False})

with col_b:
    st.subheader("Ingreso por vehículo atendido (top 10)")
    por_veh = (servicios.groupby("vehiculo", as_index=False)
               .agg(monto=("monto_total", "sum"), ordenes=("id_orden", "size"))
               .nlargest(10, "monto"))
    por_veh["texto"] = por_veh["monto"].map(F.moneda_corta)
    por_veh["hover"] = [f"<b>{v}</b><br>{F.moneda(m)} · {o} órdenes"
                        for v, m, o in zip(por_veh["vehiculo"], por_veh["monto"], por_veh["ordenes"])]
    st.plotly_chart(F.barras_h(por_veh, "vehiculo", "monto", texto="texto", hover="hover"),
                    config={"displayModeBar": False})

# ── Estacionalidad y clientes ──
col_c, col_d = st.columns([3, 2])
with col_c:
    st.subheader("Órdenes por mes y categoría")
    serv_cat = servicios.assign(categoria=servicios["servicio"].map(D.CATEGORIA_SERVICIO), ordenes=1)
    serie = D.serie_mensual(serv_cat, "fecha_ingreso", "ordenes", por="categoria")
    serie["hover"] = [f"{F.etiqueta_mes(m)} · {c}<br>{o} órdenes"
                      for m, c, o in zip(serie["mes"], serie["categoria"], serie["ordenes"])]
    orden = ["Estética", "Mecánica", "Chapa y pintura"]
    fig = px.line(serie, x="mes", y="ordenes", color="categoria", custom_data=["hover"], markers=True,
                  category_orders={"categoria": orden},
                  color_discrete_map=dict(zip(orden, [F.PALETA[2], F.PALETA[3], F.PALETA[4]])))
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>", marker_size=6)
    st.plotly_chart(F.estilo_figura(fig, altura=340), config={"displayModeBar": False})
    st.caption("Estética sube en verano (dic–feb); mecánica, al volver de las vacaciones (feb–mar y ago).")

with col_d:
    st.subheader("Top 10 clientes")
    top = (servicios.groupby("cliente", as_index=False)
           .agg(ordenes=("id_orden", "size"), monto=("monto_total", "sum"))
           .nlargest(10, "monto"))
    st.dataframe(
        top.assign(monto=top["monto"].map(F.moneda)).rename(
            columns={"cliente": "Cliente", "ordenes": "Órdenes", "monto": "Ingreso"}),
        hide_index=True,
    )

# ── Trabajo por mecánico ──
st.divider()
st.subheader("Trabajo por mecánico")
mec = d.mecanicos[d.mecanicos["id_orden"].isin(servicios["id_orden"])]
if mec.empty:
    st.info("Las órdenes filtradas no tienen mecánicos asignados (presupuestos o cancelaciones).")
    st.stop()

resumen = mec.groupby("mecanico").agg(
    horas=("horas", "sum"),
    ordenes=("id_orden", "nunique"),
    solo=("modalidad", lambda s: int((s == "Solo").sum())),
    acompanado=("modalidad", lambda s: int((s == "Acompañado").sum())),
).reset_index()

col_e, col_f = st.columns(2)
with col_e:
    resumen["texto"] = resumen["horas"].map(lambda h: f"{F.numero(h, 1)} h")
    resumen["hover"] = [f"<b>{m}</b><br>{F.numero(h, 1)} h en {o} órdenes"
                        for m, h, o in zip(resumen["mecanico"], resumen["horas"], resumen["ordenes"])]
    fig = F.barras_h(resumen, "mecanico", "horas", texto="texto", hover="hover", color_fijo=F.PALETA[0])
    fig.update_layout(title_text="Horas por mecánico", margin_t=40)
    st.plotly_chart(fig, config={"displayModeBar": False})

with col_f:
    orden_mec = resumen.sort_values("ordenes")["mecanico"]
    fig = go.Figure()
    for col, nombre, color in (("solo", "Solo", F.PALETA[2]), ("acompanado", "Acompañado", F.PALETA[6])):
        fila = resumen.set_index("mecanico").loc[orden_mec]
        fig.add_bar(y=orden_mec, x=fila[col], name=nombre, orientation="h", marker_color=color,
                    text=fila[col], textposition="inside",
                    hovertemplate=f"%{{y}}<br>{nombre}: %{{x}} órdenes<extra></extra>")
    fig.update_layout(barmode="stack", title_text="Órdenes solo vs. acompañado")
    fig.update_xaxes(visible=False)
    st.plotly_chart(F.estilo_figura(fig, altura=max(200, 34 * len(resumen) + 80)),
                    config={"displayModeBar": False})

st.caption(
    f"Las horas individuales suman {F.numero(mec['horas'].sum(), 1)} h contra "
    f"{F.numero(servicios.loc[servicios['id_orden'].isin(mec['id_orden']), 'horas_trabajo'].sum(), 1)} h "
    "reales de esas órdenes: cuando dos mecánicos trabajan juntos, ambos imputan horas."
)
with st.expander("Detalle por mecánico"):
    tabla = resumen.assign(pct=(resumen["acompanado"] / (resumen["solo"] + resumen["acompanado"])))
    st.dataframe(
        tabla.sort_values("horas", ascending=False).assign(
            horas=tabla["horas"].map(lambda h: F.numero(h, 1)), pct=tabla["pct"].map(F.porcentaje),
        )[["mecanico", "horas", "ordenes", "solo", "acompanado", "pct"]].rename(columns={
            "mecanico": "Mecánico", "horas": "Horas", "ordenes": "Órdenes", "solo": "Solo",
            "acompanado": "Acompañado", "pct": "% acompañado"}),
        hide_index=True,
    )
