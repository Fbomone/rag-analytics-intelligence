"""Tests del motor RAG sin red: cliente Cohere falso."""

from types import SimpleNamespace

import pandas as pd
import pytest

from utils import data as D
from utils import rag


@pytest.fixture(scope="module")
def bloques():
    datos = D.cargar_datos()
    return rag.construir_bloques(datos, "2024-01-01", "2026-08-31", D.ESTADOS_SERVICIO_DEFAULT,
                                 hoy=pd.Timestamp("2026-09-13"))


def _por_id(bloques):
    return {b.id: b for b in bloques}


def test_bloques_temas_y_regla_de_negocio(bloques):
    ids = [b.id for b in bloques]
    assert ids[:2] == ["diccionario", "resumen"]
    assert {"servicios", "ventas", "vendedores", "mecanicos", "cobranza", "evolucion", "calidad"} <= set(ids)
    assert "NO ingreso" in _por_id(bloques)["diccionario"].texto
    assert "No hay datos de costos" in _por_id(bloques)["diccionario"].texto


def test_contexto_es_resumen_no_csv_crudo(bloques):
    contexto = "\n".join(b.texto for b in bloques)
    assert "OT-150" not in contexto                     # no se vuelcan todas las filas
    assert len(contexto) < 40_000


@pytest.mark.parametrize("pregunta, tema", [
    ("¿Qué mecánico trabajó más horas?", "mecanicos"),
    ("¿Cuánto tengo pendiente de cobro?", "cobranza"),
    ("¿Qué vendedor tuvo mejor desempeño?", "vendedores"),
    ("¿Qué servicio dejó más margen este año?", "servicios"),
    ("¿Cómo evolucionó el ingreso mes a mes?", "evolucion"),
])
def test_recuperacion_por_keywords(bloques, pregunta, tema):
    candidatos = [b for b in bloques if b.id not in rag.BLOQUES_FIJOS]
    elegidos = rag.recuperar_por_keywords(pregunta, candidatos, k=3)
    assert elegidos[0].id == tema


class ClienteFalso:
    def __init__(self, falla_embed=False):
        self.falla_embed = falla_embed
        self.mensajes = None

    def embed(self, *, model, input_type, texts, embedding_types):
        if self.falla_embed:
            raise RuntimeError("sin red")
        # Vector "one-hot" por tema: la consulta sobre mecánicos apunta al bloque de mecánicos.
        vectores = [[1.0, 0.0] if "mecánico" in t.lower() else [0.0, 1.0] for t in texts]
        return SimpleNamespace(embeddings=SimpleNamespace(float_=vectores))

    def chat(self, *, model, messages, temperature):
        self.mensajes = messages
        return SimpleNamespace(message=SimpleNamespace(content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text="Respuesta con US$ 1.000"),
        ]))


def test_recuperador_usa_embeddings(bloques):
    rag.Recuperador._cache.clear()
    usados, metodo = rag.Recuperador(ClienteFalso(), "fake-embed").recuperar("¿Qué mecánico trabajó más?", bloques, k=1)
    assert metodo == "embeddings"
    assert [b.id for b in usados] == ["diccionario", "resumen", "mecanicos"]


def test_recuperador_fallback_a_keywords(bloques):
    usados, metodo = rag.Recuperador(ClienteFalso(falla_embed=True)).recuperar("pendiente de cobro", bloques)
    assert metodo == "keywords" and "cobranza" in [b.id for b in usados]


def test_asistente_arma_mensajes_y_extrae_texto(bloques):
    cliente = ClienteFalso(falla_embed=True)
    asistente = rag.AsistenteRAG(cliente=cliente, modelo_chat="fake-chat")
    historial = [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "¡Hola!"}]
    r = asistente.responder("¿Cuánto tengo pendiente de cobro?", bloques, historial)
    assert r.texto == "Respuesta con US$ 1.000" and r.error is None
    roles = [m["role"] for m in cliente.mensajes]
    assert roles == ["system", "user", "assistant", "user"]
    assert "PREGUNTA: ¿Cuánto tengo pendiente de cobro?" in cliente.mensajes[-1]["content"]
    assert "Cobranza y pendientes" in cliente.mensajes[-1]["content"]


def test_asistente_no_lanza_si_falla_el_chat(bloques):
    cliente = ClienteFalso(falla_embed=True)
    cliente.chat = lambda **_: (_ for _ in ()).throw(RuntimeError("503"))
    r = rag.AsistenteRAG(cliente=cliente).responder("hola", bloques)
    assert r.error and "503" in r.error
