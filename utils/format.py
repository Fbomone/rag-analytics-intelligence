"""Formato de números en es-AR y estilo visual común de los gráficos.

La paleta categórica es la de referencia validada para daltonismo (orden fijo,
nunca ciclado). El color sigue a la entidad: "Servicios de taller" es siempre
azul y "Venta de autos" siempre naranja, en todas las páginas. La tinta
(textos, ejes, grilla) la pone el tema de Streamlit, así los gráficos se leen
bien en modo claro y oscuro.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

SIMBOLO_MONEDA = "US$"

MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

# Paleta categórica en orden fijo.
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

COLOR_UNIDAD = {"Servicios de taller": PALETA[0], "Venta de autos": PALETA[1]}
# Estados de cobro: paleta de estado (reservada), siempre acompañada de su etiqueta.
COLOR_COBRO = {"Cobrado": "#0ca30c", "Por cobrar": "#fab219", "Por facturar": "#ec835a"}
COLOR_PENDIENTE = "#898781"   # gris neutro: dato sin clasificar, no una serie más


def _separadores_ar(texto: str) -> str:
    """Intercambia separadores en-US ('1,234.5') por es-AR ('1.234,5')."""
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def numero(valor: float | None, decimales: int = 0) -> str:
    """``1234.5`` → ``'1.234,5'`` (con ``decimales=1``). Nulos → ``'—'``."""
    if valor is None or pd.isna(valor):
        return "—"
    return _separadores_ar(f"{valor:,.{decimales}f}")


def moneda(valor: float | None, decimales: int = 0) -> str:
    """``-32000`` → ``'-US$ 32.000'``."""
    if valor is None or pd.isna(valor):
        return "—"
    signo = "-" if valor < 0 else ""
    return f"{signo}{SIMBOLO_MONEDA} {numero(abs(valor), decimales)}"


def moneda_corta(valor: float | None) -> str:
    """Moneda abreviada para etiquetas: ``'US$ 1,2 M'``, ``'US$ 35 k'``, ``'US$ 950'``."""
    if valor is None or pd.isna(valor):
        return "—"
    absoluto = abs(valor)
    if absoluto >= 1_000_000:
        cuerpo = f"{numero(absoluto / 1_000_000, 1)} M"
    elif absoluto >= 10_000:
        cuerpo = f"{numero(absoluto / 1_000, 0)} k"
    else:
        cuerpo = numero(absoluto, 0)
    return f"{'-' if valor < 0 else ''}{SIMBOLO_MONEDA} {cuerpo}"


def porcentaje(fraccion: float | None, decimales: int = 1) -> str:
    """``0.125`` → ``'12,5 %'``."""
    if fraccion is None or pd.isna(fraccion):
        return "—"
    return f"{numero(fraccion * 100, decimales)} %"


def etiqueta_mes(fecha) -> str:
    """``Timestamp('2025-03-01')`` → ``'mar 2025'``."""
    fecha = pd.Timestamp(fecha)
    return f"{MESES_ES[fecha.month - 1]} {fecha.year}"


def estilo_figura(fig: go.Figure, altura: int = 340, leyenda: bool = True) -> go.Figure:
    """Estilo común: separadores es-AR, leyenda arriba, marcas finas y ejes recesivos."""
    fig.update_layout(
        height=altura,
        margin=dict(l=8, r=8, t=40 if fig.layout.title.text else 8, b=8),
        separators=",.",
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        bargap=0.3,
        barcornerradius=4,
    )
    fig.update_xaxes(title_text="", showgrid=False)
    fig.update_yaxes(title_text="")
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    fig.update_traces(selector=dict(type="scatter"), line_width=2, marker_size=8)
    return fig


def eje_moneda(fig: go.Figure, eje: str = "y") -> go.Figure:
    """Ticks de moneda con separador de miles es-AR en el eje indicado."""
    actualizar = fig.update_yaxes if eje == "y" else fig.update_xaxes
    actualizar(tickprefix=f"{SIMBOLO_MONEDA} ", tickformat=",.0f")
    return fig


def barras_h(df: pd.DataFrame, categoria: str, valor: str, *, texto: str | None = None,
             hover: str | None = None, color: str | None = None,
             color_fijo: str = PALETA[0]) -> go.Figure:
    """Barras horizontales ordenadas (la mayor arriba) con etiqueta directa.

    ``texto``, ``hover`` y ``color`` son nombres de columnas de ``df`` con la
    etiqueta ya formateada, el tooltip y el color de cada barra.
    """
    df = df.sort_values(valor)
    fig = go.Figure(go.Bar(
        x=df[valor],
        y=df[categoria],
        orientation="h",
        marker_color=df[color].tolist() if color else color_fijo,
        text=df[texto] if texto else None,
        textposition="outside",
        cliponaxis=False,
        hovertext=df[hover] if hover else None,
        hoverinfo="text" if hover else "x+y",
    ))
    fig.update_xaxes(visible=False)
    return estilo_figura(fig, altura=max(200, 34 * len(df) + 40), leyenda=False)
