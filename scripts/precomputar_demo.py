"""Precalcula las respuestas del modo demo del Asistente IA.

Corre el pipeline RAG real (período completo, filtros por defecto) para las
preguntas de ``rag.PREGUNTAS_DEMO`` y guarda el resultado en
``data/demo_respuestas.json``. Así el repo se puede recorrer sin API key.

Uso:
    python scripts/precomputar_demo.py      # requiere COHERE_API_KEY
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from utils import data as D  # noqa: E402
from utils import rag  # noqa: E402


def main() -> None:
    """Genera ``data/demo_respuestas.json``."""
    api_key = rag.obtener_api_key()
    if not api_key:
        sys.exit("Falta COHERE_API_KEY (en el entorno o en .env).")

    datos = D.cargar_datos()
    fechas = datos.ingresos.loc[datos.ingresos["fecha_valida"], "fecha"]
    desde, hasta = fechas.min().date(), fechas.max().date()
    bloques = rag.construir_bloques(datos, desde, hasta, D.ESTADOS_SERVICIO_DEFAULT)
    asistente = rag.AsistenteRAG(api_key=api_key)

    respuestas = []
    for pregunta in rag.PREGUNTAS_DEMO:
        r = asistente.responder(pregunta, bloques)
        if r.error:
            sys.exit(f"Error consultando '{pregunta}': {r.error}")
        print(f"\n> {pregunta}  [{r.metodo}: {', '.join(r.fuentes)}]\n{r.texto}")
        respuestas.append({"pregunta": pregunta, "respuesta": r.texto, "fuentes": r.fuentes})

    salida = {
        "descripcion": "Respuestas reales del asistente, precalculadas para el modo demo (sin API key).",
        "generado": date.today().isoformat(),
        "periodo": f"{desde:%d/%m/%Y} – {hasta:%d/%m/%Y}",
        "estados_servicio": D.ESTADOS_SERVICIO_DEFAULT,
        "modelo_chat": asistente.modelo_chat,
        "modelo_embed": asistente.recuperador.modelo_embed,
        "respuestas": respuestas,
    }
    with open(rag.ARCHIVO_DEMO, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado en {rag.ARCHIVO_DEMO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
