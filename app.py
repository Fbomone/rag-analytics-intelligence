"""AutoPulse Analytics — punto de entrada de la app Streamlit.

Las páginas viven en ``paginas/`` (no ``pages/``) para que Streamlit no active
su navegación automática: la navegación se declara acá con ``st.navigation``,
lo que mantiene URLs directas estables y permite dibujar los filtros globales
una sola vez para todas las páginas.

Correr con:  streamlit run app.py
"""

import streamlit as st

from utils import ui

st.set_page_config(page_title="AutoPulse Analytics", page_icon="🚗", layout="wide")

paginas = [
    st.Page("paginas/1_Reporte_General.py", title="Reporte General", icon="📊",
            url_path="reporte-general", default=True),
    st.Page("paginas/2_Servicios_Taller.py", title="Servicios de Taller", icon="🔧",
            url_path="servicios-taller"),
    st.Page("paginas/3_Venta_Autos.py", title="Venta de Autos", icon="🚙",
            url_path="venta-autos"),
    st.Page("paginas/4_Asistente_IA.py", title="Asistente IA", icon="💬",
            url_path="asistente-ia"),
    st.Page("paginas/5_Calidad_Datos.py", title="Calidad de Datos", icon="🧪",
            url_path="calidad-datos"),
]

navegacion = st.navigation(paginas)
ui.sidebar_filtros()
navegacion.run()
