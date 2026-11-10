#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Arranca un servidor Hindsight (memoria a largo plazo del agente) SIN Docker.
#
#   - Usa el paquete ligero `hindsight-api-slim` con PostgreSQL embebido (pg0),
#     ejecutado con `uvx` en un entorno aislado: NO se instala en el .venv del app.
#   - LLM de extracción/razonamiento: Ollama local (por defecto, sin cuotas) o Gemini.
#     Ojo: el plan gratuito de Gemini permite ~20 peticiones/día y Hindsight hace varias
#     por cada memoria (extracción + consolidación); úsalo solo con facturación activa.
#   - Embeddings: Gemini (`gemini-embedding-001`), así no se necesita PyTorch.
#   - Reranker: RRF (fusión por rango recíproco, sin modelo local).
#
# Uso:
#   ./scripts/start_hindsight.sh                    # Ollama (llama3.2) como LLM
#   HINDSIGHT_LLM_PROVIDER=gemini ./scripts/start_hindsight.sh
#
# Si prefieres Docker (incluye embeddings/reranker locales y la UI web en :9999):
#   docker compose -f docker-compose.hindsight.yml up -d
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

HINDSIGHT_VERSION="${HINDSIGHT_SERVER_VERSION:-1.2.0}"
PROVIDER="${HINDSIGHT_LLM_PROVIDER:-ollama}"

if [[ -z "${GOOGLE_API_KEY:-}" || "${GOOGLE_API_KEY}" == "tu_api_key_aqui" ]]; then
  echo "❌ GOOGLE_API_KEY no está configurada en .env (se usa para los embeddings de Hindsight)." >&2
  echo "   Alternativa sin Gemini: docker compose -f docker-compose.hindsight.yml up -d" >&2
  exit 1
fi

case "$PROVIDER" in
  gemini)
    export HINDSIGHT_API_LLM_PROVIDER=gemini
    export HINDSIGHT_API_LLM_API_KEY="$GOOGLE_API_KEY"
    export HINDSIGHT_API_LLM_MODEL="${HINDSIGHT_LLM_MODEL:-${GEMINI_MODEL:-gemini-3.5-flash}}"
    ;;
  ollama)
    export HINDSIGHT_API_LLM_PROVIDER=ollama
    export HINDSIGHT_API_LLM_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}/v1"
    export HINDSIGHT_API_LLM_MODEL="${HINDSIGHT_LLM_MODEL:-${OLLAMA_MODEL:-llama3.2}}"
    # Modelos locales pequeños (p. ej. llama3.2 3B) pueden entrar en bucles de generación:
    # el valor por defecto de Hindsight (64k tokens) bloquea Ollama hasta el timeout de 2 min
    # y, como Ollama atiende una petición a la vez, también bloquea las respuestas del chat.
    # Debe coincidir con el num_ctx del agente (OLLAMA_NUM_CTX): si difieren, Ollama recarga
    # el modelo cada vez que se alternan el chat y la extracción de memorias.
    export HINDSIGHT_API_LLM_OLLAMA_NUM_CTX="${HINDSIGHT_API_LLM_OLLAMA_NUM_CTX:-${OLLAMA_NUM_CTX:-8192}}"
    # (Hindsight exige RETAIN_MAX_COMPLETION_TOKENS > RETAIN_CHUNK_SIZE)
    export HINDSIGHT_API_RETAIN_CHUNK_SIZE="${HINDSIGHT_API_RETAIN_CHUNK_SIZE:-2000}"
    export HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS="${HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS:-2048}"
    export HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS="${HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS:-1024}"
    export HINDSIGHT_API_REFLECT_MAX_COMPLETION_TOKENS="${HINDSIGHT_API_REFLECT_MAX_COMPLETION_TOKENS:-1024}"
    export HINDSIGHT_API_RETAIN_MAX_CONCURRENT="${HINDSIGHT_API_RETAIN_MAX_CONCURRENT:-1}"
    export HINDSIGHT_API_LLM_TIMEOUT="${HINDSIGHT_API_LLM_TIMEOUT:-90}"
    ;;
  *)
    echo "❌ HINDSIGHT_LLM_PROVIDER='$PROVIDER' no soportado por este script (usa gemini u ollama)." >&2
    exit 1
    ;;
esac

export HINDSIGHT_API_EMBEDDINGS_PROVIDER=google
export HINDSIGHT_API_EMBEDDINGS_GEMINI_API_KEY="$GOOGLE_API_KEY"
export HINDSIGHT_API_RERANKER_PROVIDER=rrf
export HINDSIGHT_API_PORT="${HINDSIGHT_API_PORT:-8888}"

echo "🧠 Iniciando Hindsight ${HINDSIGHT_VERSION} en http://localhost:${HINDSIGHT_API_PORT}"
echo "   LLM: ${HINDSIGHT_API_LLM_PROVIDER} (${HINDSIGHT_API_LLM_MODEL}) | Embeddings: Gemini | DB: pg0 embebido"

exec uvx --python 3.12 --from "hindsight-api-slim[embedded-db]==${HINDSIGHT_VERSION}" hindsight-api
