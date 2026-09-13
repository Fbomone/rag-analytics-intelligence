# AutoPulse — Notas

## Estado
Funcional. Dashboard + RAG andando en local.

## Pendiente
- [ ] Borrar codigo viejo: src/, config/, data/datasets/, data/documents/,
      notebooks/, test/
- [ ] Correr tests: pip install -r requirements-dev.txt && pytest
- [ ] Screenshots en docs/screenshots/
- [ ] Deploy a Streamlit Community Cloud (repo publico, sin secrets)

## Decisiones de diseño
- Ingreso = comision (autos) + facturacion (taller). El precio del auto
  es volumen intermediado, no ingreso
- Si una operacion mezcla modelos con y sin precio, la factura entera va
  a "pendiente de precio" (no hay base para prorratear parcialmente)
- Horas por mecanico pueden superar horas_trabajo: trabajan en simultaneo
- El RAG recibe resumenes estructurados, no los CSV crudos (~17k chars)

## Setup
- cohere==7.1.1 (la 5.0.0 no tiene ClientV2)
- .env con COHERE_API_KEY
- Correr: streamlit run app.py