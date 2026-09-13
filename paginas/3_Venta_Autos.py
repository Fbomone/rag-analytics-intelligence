"""Venta de Autos: unidades por modelo, prorrateo de facturación, comisiones y vendedores."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import data as D
from utils import format as F
from utils import ui

d = ui.datos()
desde, hasta, _ = ui.filtros()

st.title("Venta de Autos")
estados_venta = st.multiselect("Estado de la operación", D.ESTADOS_VENTA, default=D.ESTADOS_VENTA_DEFAULT,
                               key="filtro_estado_venta")
st.caption(f"Período {ui.descripcion_periodo(desde, hasta)}")

ventas = D.filtrar_ventas(d.ventas, desde, hasta, estados_venta)
if ventas.empty:
    st.info("No hay operaciones para los filtros elegidos.")
    st.stop()
unidades = d.unidades[d.unidades["id_operacion"].isin(ventas["id_operacion"])]

comisiones = ventas["comision_monto"].sum()
volumen = ventas["factura_sin_iva"].sum()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Comisiones (ingreso)", F.moneda(comisiones), border=True,
          help="Suma de comision_monto. Es lo que gana AutoPulse por intermediar.")
c2.metric("Volumen intermediado", F.moneda(volumen), border=True,
          help="Facturación total de los autos (factura_sin_iva). No es ingreso de AutoPulse.")
c3.metric("Operaciones", F.numero(len(ventas)), border=True)
c4.metric("Unidades", F.numero(len(unidades)), border=True,
          help="Cada modelo listado en la celda es una unidad: 'Hilux, Hilux' son dos.")
c5.metric("Comisión efectiva", F.porcentaje(comisiones / volumen if volumen else None), border=True,
          help="Comisiones / volumen intermediado.")

# ── Modelos ──
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Unidades por modelo")
    por_modelo = unidades.groupby(["modelo", "tiene_precio"], as_index=False).agg(unidades=("id_operacion", "size"))
    por_modelo["color"] = por_modelo["tiene_precio"].map({True: F.PALETA[1], False: F.COLOR_PENDIENTE})
    por_modelo["texto"] = [f"{u}" if p else f"{u} · pendiente de precio"
                           for u, p in zip(por_modelo["unidades"], por_modelo["tiene_precio"])]
    por_modelo["hover"] = [f"<b>{m}</b><br>{u} unidades" + ("" if p else "<br>Sin precio de referencia")
                           for m, u, p in zip(por_modelo["modelo"], por_modelo["unidades"], por_modelo["tiene_precio"])]
    st.plotly_chart(F.barras_h(por_modelo, "modelo", "unidades", texto="texto", hover="hover", color="color"),
                    config={"displayModeBar": False})

with col_b:
    st.subheader("Facturación prorrateada por modelo")
    con_precio = (unidades.dropna(subset=["factura_asignada"])
                  .groupby("modelo", as_index=False)
                  .agg(factura=("factura_asignada", "sum"), comision=("comision_asignada", "sum")))
    con_precio["color"] = F.PALETA[1]
    pendientes = ventas[ventas["pendiente_precio"]]
    if not pendientes.empty:
        con_precio = pd.concat([con_precio, pd.DataFrame([{
            "modelo": D.PENDIENTE_PRECIO, "factura": pendientes["factura_sin_iva"].sum(),
            "comision": pendientes["comision_monto"].sum(), "color": F.COLOR_PENDIENTE,
        }])], ignore_index=True)
    con_precio["texto"] = con_precio["factura"].map(F.moneda_corta)
    con_precio["hover"] = [f"<b>{m}</b><br>Factura: {F.moneda(f)}<br>Comisión: {F.moneda(c)}"
                           for m, f, c in zip(con_precio["modelo"], con_precio["factura"], con_precio["comision"])]
    st.plotly_chart(F.barras_h(con_precio, "modelo", "factura", texto="texto", hover="hover", color="color"),
                    config={"displayModeBar": False})
    st.caption("Si una factura incluye varios modelos, se reparte en proporción al precio de referencia. "
               "Si alguno no tiene precio, no hay base para repartir: la factura completa va a "
               "“Pendiente de precio” (suma al total, pero no a ningún modelo).")

# ── Comisiones y vendedores ──
st.divider()
st.subheader("Ranking de vendedores")
cobradas = ventas.loc[ventas["cobrado"], "comision_monto"].sum()
m1, m2, _ = st.columns([1, 1, 2])
m1.metric("Comisiones cobradas", F.moneda(cobradas))
m2.metric("Comisiones por cobrar", F.moneda(comisiones - cobradas))

vend = ventas.groupby("vendedor").agg(
    operaciones=("id_operacion", "size"), unidades=("unidades", "sum"),
    volumen=("factura_sin_iva", "sum"), comision=("comision_monto", "sum"),
    cobrado=("comision_monto", lambda s: s[ventas.loc[s.index, "cobrado"]].sum()),
).reset_index()
vend["por_cobrar"] = vend["comision"] - vend["cobrado"]
orden = vend.sort_values("comision")["vendedor"]
v_idx = vend.set_index("vendedor").loc[orden]

col_c, col_d = st.columns([3, 2])
with col_c:
    fig = go.Figure()
    for col, nombre in (("cobrado", "Cobrado"), ("por_cobrar", "Por cobrar")):
        fig.add_bar(y=orden, x=v_idx[col], name=nombre, orientation="h", marker_color=F.COLOR_COBRO[nombre],
                    hovertext=[f"<b>{v}</b><br>{nombre}: {F.moneda(x)}" for v, x in zip(orden, v_idx[col])],
                    hoverinfo="text")
    fig.add_scatter(y=orden, x=v_idx["comision"], mode="text", text=v_idx["comision"].map(F.moneda_corta),
                    textposition="middle right", showlegend=False, hoverinfo="skip")
    fig.update_layout(barmode="stack", title_text="Comisiones por vendedor")
    fig.update_xaxes(visible=False, range=[0, v_idx["comision"].max() * 1.2])
    st.plotly_chart(F.estilo_figura(fig, altura=max(220, 40 * len(vend) + 80)), config={"displayModeBar": False})
with col_d:
    tabla = vend.sort_values("comision", ascending=False)
    st.dataframe(
        tabla.assign(**{c: tabla[c].map(F.moneda_corta) for c in ("volumen", "comision", "por_cobrar")})
        [["vendedor", "operaciones", "unidades", "volumen", "comision", "por_cobrar"]]
        .rename(columns={"vendedor": "Vendedor", "operaciones": "Ops.", "unidades": "Unid.",
                         "volumen": "Volumen", "comision": "Comisión", "por_cobrar": "Por cobrar"}),
        hide_index=True,
    )

# ── Formas de pago y canales ──
st.divider()
col_e, col_f = st.columns(2)
for contenedor, columna, titulo in ((col_e, "forma_pago", "Formas de pago"), (col_f, "origen_lead", "Canales de lead")):
    with contenedor:
        st.subheader(titulo)
        agg = ventas.groupby(columna, as_index=False).agg(
            operaciones=("id_operacion", "size"), comision=("comision_monto", "sum"), volumen=("factura_sin_iva", "sum"))
        agg["texto"] = agg["operaciones"].map(lambda n: f"{n} ops.")
        agg["hover"] = [f"<b>{c}</b><br>{o} operaciones<br>Comisión: {F.moneda(m)}<br>Volumen: {F.moneda(v)}"
                        for c, o, m, v in zip(agg[columna], agg["operaciones"], agg["comision"], agg["volumen"])]
        st.plotly_chart(F.barras_h(agg, columna, "operaciones", texto="texto", hover="hover", color_fijo=F.PALETA[1]),
                        config={"displayModeBar": False}, key=f"chart_{columna}")

with st.expander(f"Operaciones con modelos pendientes de precio ({len(pendientes)})"):
    st.dataframe(
        pendientes.assign(factura_sin_iva=pendientes["factura_sin_iva"].map(F.moneda),
                          fecha_venta=pendientes["fecha_venta"].dt.strftime("%d/%m/%Y"))
        [["id_operacion", "fecha_venta", "modelo", "vendedor", "factura_sin_iva"]],
        hide_index=True,
    )
    st.caption("Cargá el precio en data/precios_referencia.json para que se repartan por modelo.")
