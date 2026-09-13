"""Asistente IA: chat en lenguaje natural sobre los datos, con RAG sobre Cohere."""

import streamlit as st

from utils import rag, ui


def _api_key() -> str | None:
    clave = rag.obtener_api_key()
    if clave:
        return clave
    try:
        return st.secrets.get("COHERE_API_KEY") or None
    except Exception:  # noqa: BLE001 — no hay secrets.toml
        return None


@st.cache_resource(show_spinner=False)
def _asistente(api_key: str) -> rag.AsistenteRAG:
    return rag.AsistenteRAG(api_key=api_key)


def _mostrar(mensaje: dict) -> None:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])
        if mensaje.get("fuentes"):
            st.caption(f"Fuentes: {' · '.join(mensaje['fuentes'])} — recuperación por {mensaje.get('metodo')}")
        if mensaje.get("error"):
            st.caption(f"Detalle técnico: {mensaje['error']}")


def _responder(pregunta: str, historial: list[dict]) -> dict:
    if api_key:
        r = _asistente(api_key).responder(pregunta, bloques, historial)
        return {"role": "assistant", "content": r.texto, "fuentes": r.fuentes, "metodo": r.metodo, "error": r.error}
    ejemplo = next((x for x in demo if x["pregunta"] == pregunta), None)
    if ejemplo:
        return {"role": "assistant", "content": ejemplo["respuesta"], "fuentes": ejemplo.get("fuentes", []),
                "metodo": "respuesta precalculada (modo demo)"}
    return {"role": "assistant", "content": (
        "Estoy en **modo demo**: sólo puedo mostrar las respuestas de ejemplo de las preguntas sugeridas. "
        "Configurá `COHERE_API_KEY` para hacer preguntas libres.")}


d = ui.datos()
desde, hasta, estados = ui.filtros()
api_key = _api_key()
demo = [] if api_key else rag.cargar_demo()
bloques = rag.construir_bloques(d, desde, hasta, estados)
st.session_state.setdefault("chat", [])

st.title("Asistente IA")
st.caption(f"Preguntá en lenguaje natural. Contexto: período {ui.descripcion_periodo(desde, hasta)} "
           "y estados de taller elegidos en la barra lateral.")

if not api_key:
    st.warning(
        "**Modo demo — no hay `COHERE_API_KEY` configurada.** Las preguntas sugeridas muestran respuestas "
        "reales precalculadas con el período completo y los filtros por defecto.\n\n"
        "Para chatear libremente:\n"
        "1. Creá una API key gratuita en [dashboard.cohere.com](https://dashboard.cohere.com/api-keys).\n"
        "2. Copiá `.env.example` como `.env` y completá `COHERE_API_KEY=...` "
        "(o agregala en `.streamlit/secrets.toml`).\n"
        "3. Reiniciá la app.",
        icon="🔑",
    )

with st.expander("¿Cómo funciona? Ver el contexto que puede recibir el modelo"):
    st.markdown(
        "El asistente **no recibe los CSV**. Los datos filtrados se resumen en bloques temáticos; ante cada "
        "pregunta se eligen los más relevantes con embeddings de Cohere (similitud coseno, con fallback a "
        "keywords) y se envían junto con el diccionario de datos y el resumen general."
    )
    for bloque in bloques:
        st.markdown(f"**{bloque.titulo}** · `{bloque.id}` · {len(bloque.texto):,} caracteres".replace(",", "."))
        st.code(bloque.texto, language=None, wrap_lines=True)

sugeridas = rag.PREGUNTAS_SUGERIDAS if api_key else [x["pregunta"] for x in demo]
pregunta = None
with st.container(horizontal=True):
    for i, sugerida in enumerate(sugeridas):
        if st.button(sugerida, key=f"sugerida_{i}"):
            pregunta = sugerida
    if st.session_state.chat and st.button("Limpiar conversación", icon=":material/delete:", type="tertiary"):
        st.session_state.chat = []
        st.rerun()

escrita = st.chat_input("Preguntá sobre ventas, taller, mecánicos o cobranza…")
pregunta = escrita or pregunta

for mensaje in st.session_state.chat:
    _mostrar(mensaje)

if pregunta:
    historial = list(st.session_state.chat)
    usuario = {"role": "user", "content": pregunta}
    st.session_state.chat.append(usuario)
    _mostrar(usuario)
    with st.spinner("Analizando los datos…"):
        respuesta = _responder(pregunta, historial)
    st.session_state.chat.append(respuesta)
    _mostrar(respuesta)
