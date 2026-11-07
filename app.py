import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Ensure src is in sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from langchain_app.main import (
    build_chat_graph,
    call_mcp_server,
    format_ai_content,
    get_langsmith_client,
    invoke_direct_mcp,
    search_web,
)
from langchain_app.voice_service import (
    clean_text_for_speech,
    is_phonon_available,
    speak_response,
    synthesize_speech,
    transcribe_audio,
)
from langchain_app.pdf_service import (
    extract_text_from_pdf,
    format_pdf_context_for_prompt,
    read_pdf_file,
)

load_dotenv(override=True)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LangChain & LangGraph AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge-container {
        display: flex;
        gap: 8px;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .status-badge {
        background-color: #f1f5f9;
        color: #334155;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"st-session-{os.urandom(4).hex()}"

if "graph" not in st.session_state:
    st.session_state.graph = None

if "current_provider" not in st.session_state:
    st.session_state.current_provider = "gemini"


# -----------------------------------------------------------------------------
# Sidebar: Settings & Observability
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuración")

    # Provider Selection
    selected_provider = st.radio(
        "Proveedor LLM:",
        options=["gemini", "ollama"],
        format_func=lambda x: "⚡ Google Gemini (Cloud API)" if x == "gemini" else "🦙 Ollama (Local)",
        index=0 if st.session_state.current_provider == "gemini" else 1,
    )

    if selected_provider != st.session_state.current_provider or st.session_state.graph is None:
        st.session_state.current_provider = selected_provider
        with st.spinner(f"Inicializando grafo para {selected_provider.upper()}..."):
            try:
                st.session_state.graph = build_chat_graph(selected_provider)
                st.toast(f"Conectado a {selected_provider.upper()}", icon="✅")
            except Exception as e:
                st.error(f"Error conectando a {selected_provider}: {e}")

    # Provider details
    if selected_provider == "gemini":
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        st.caption(f"**Modelo:** `{model_name}`")
        api_key = os.getenv("GOOGLE_API_KEY", "")
        has_key = bool(api_key and api_key != "tu_api_key_aqui")
        st.caption(f"**API Key:** {'🟢 Configurada' if has_key else '🔴 Falta en .env'}")
    else:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        st.caption(f"**URL:** `{ollama_url}` | **Modelo:** `{ollama_model}`")

    st.divider()

    # Voice (TTS / STT) Configuration
    st.subheader("🎙️ Servicio de Voz (TTS Ollama)")
    has_phonon = is_phonon_available()
    if has_phonon:
        st.success("🟢 phonon:40m-multilingual activo")
    else:
        st.warning("🟡 Modelo phonon:40m no detectado en Ollama")

    voice_enabled = st.checkbox("🔊 Responder con audio de voz", value=False)
    voice_options = ["Alba", "Borja", "Candela", "Guillermo", "Jimena", "Lucía-Sol", "default", "Alfie"]
    selected_voice = st.selectbox("Voz de síntesis:", options=voice_options, index=0)

    st.divider()

    # PDF Document Uploader & Context Service
    st.subheader("📄 Subida de Documentos PDF")
    uploaded_pdf = st.file_uploader(
        "Sube un PDF para analizar antes de la consulta:",
        type=["pdf"],
        key="pdf_uploader",
    )
    if uploaded_pdf is not None:
        if st.session_state.get("last_uploaded_pdf_name") != uploaded_pdf.name:
            with st.spinner(f"Leyendo y procesando '{uploaded_pdf.name}'..."):
                try:
                    pdf_bytes = uploaded_pdf.read()
                    pdf_info = extract_text_from_pdf(pdf_bytes)
                    pdf_info["filename"] = uploaded_pdf.name
                    st.session_state.active_pdf = pdf_info
                    st.session_state.last_uploaded_pdf_name = uploaded_pdf.name
                    st.toast(f"PDF cargado: {pdf_info['total_pages']} páginas", icon="📄")
                except Exception as exc:
                    st.error(f"Error procesando PDF: {exc}")

    if st.session_state.get("active_pdf"):
        pdf_info = st.session_state.active_pdf
        st.success(f"📎 **{pdf_info['filename']}**\n- Páginas: `{pdf_info['total_pages']}`\n- Caracteres: `{pdf_info['char_count']}`")
        if st.button("❌ Quitar PDF del contexto", use_container_width=True):
            st.session_state.active_pdf = None
            st.session_state.last_uploaded_pdf_name = None
            st.rerun()

    st.divider()

    # LangSmith Observability Status
    st.subheader("📊 LangSmith Observability")
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    project_name = os.getenv("LANGSMITH_PROJECT", "langchain-app")
    has_ls_key = bool(os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_API_KEY") != "lsv2_pt_tu_api_key_aqui")

    if tracing_enabled and has_ls_key:
        st.success(f"🟢 Monitoreo activo ({project_name})")
        st.link_button(
            "Ver Dashboard en LangSmith ↗",
            "https://smith.langchain.com",
            use_container_width=True,
        )
    else:
        st.warning("🟡 LangSmith tracing no habilitado o sin API Key")

    # Recent traces expander
    with st.expander("Últimas trazas registradas"):
        client = get_langsmith_client()
        if client:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    runs = list(client.list_runs(project_name=project_name, limit=5))
                if runs:
                    for r in runs:
                        ts = r.start_time.strftime("%H:%M:%S") if r.start_time else ""
                        st.markdown(f"- **{r.name}** (`{r.run_type}`) - {r.status} *{ts}*")
                else:
                    st.caption("No hay trazas registradas todavía.")
            except Exception as exc:
                st.caption(f"No se pudieron leer trazas: {exc}")
        else:
            st.caption("Cliente LangSmith no inicializado.")

    st.divider()

    # Quick Action Buttons
    st.subheader("⚡ Acciones Rápidas")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Estado MCP", use_container_width=True):
            with st.spinner("Consultando MCP Server..."):
                status_res = invoke_direct_mcp("system_status", "")
                st.session_state.messages.append({"role": "user", "content": "Verifica el estado del servidor MCP y sistema local."})
                st.session_state.messages.append({"role": "assistant", "content": status_res})
                st.rerun()

    with col2:
        if st.button("🧮 Test Math", use_container_width=True):
            expr = "125 * 48 + sqrt(144)"
            with st.spinner("Calculando en MCP Server..."):
                math_res = invoke_direct_mcp("math", expr)
                st.session_state.messages.append({"role": "user", "content": f"Calcula con MCP: {expr}"})
                st.session_state.messages.append({"role": "assistant", "content": math_res})
                st.rerun()

    if st.button("🔊 Probar Voz (phonon)", use_container_width=True):
        sample_text = f"Hola, soy el asistente de voz de LangGraph usando el modelo phonon de Ollama con la voz {selected_voice}."
        with st.spinner("Generando audio con phonon:40m..."):
            try:
                audio_bytes, _ = synthesize_speech(sample_text, voice=selected_voice, play_audio=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": sample_text,
                    "audio_data": audio_bytes,
                })
                st.rerun()
            except Exception as exc:
                st.error(f"Error generando voz: {exc}")

    st.divider()

    # Session Control
    if st.button("🧹 Limpiar Conversación", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.thread_id = f"st-session-{os.urandom(4).hex()}"
        st.toast("Historial y memoria limpiados", icon="🧹")
        st.rerun()


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🤖 LangChain & LangGraph AI Agent</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="badge-container">
        <span class="status-badge">⚡ LangGraph Orchestration</span>
        <span class="status-badge">🔌 FastMCP stdio Server</span>
        <span class="status-badge">🌐 Tavily Web Search On-Demand</span>
        <span class="status-badge">📄 PDF Ingestion Service</span>
        <span class="status-badge">🎙️ Ollama phonon:40m TTS</span>
        <span class="status-badge">📊 LangSmith Tracing</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Chat History Display
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "audio_data" in msg and msg["audio_data"]:
            st.audio(msg["audio_data"], format="audio/wav")
        if "tools_called" in msg and msg["tools_called"]:
            with st.expander("🛠️ Detalle de herramientas ejecutadas", expanded=False):
                for tool in msg["tools_called"]:
                    st.markdown(f"**Herramienta:** `{tool['name']}`")
                    if tool.get("output"):
                        st.code(tool["output"], language="text")


# -----------------------------------------------------------------------------
# Voice Input Expander
# -----------------------------------------------------------------------------
with st.expander("🎤 Entrada de voz (Grabar mensaje con micrófono)", expanded=False):
    st.caption("Graba tu voz. Al presionar finalizar, se transcribirá automáticamente y se enviará al agente.")
    audio_input_bytes = None
    try:
        audio_recorder = st.audio_input("Grabar audio:")
        if audio_recorder is not None:
            audio_input_bytes = audio_recorder.read()
    except Exception:
        # Fallback if audio_input is not supported in current environment
        st.info("Grabación de micrófono disponible a través del componente st.audio_input de Streamlit.")


# -----------------------------------------------------------------------------
# User Input & Agent Response Loop
# -----------------------------------------------------------------------------
text_prompt = st.chat_input("Escribe tu pregunta o petición para el agente...")

prompt = None
if audio_input_bytes:
    with st.spinner("🎙️ Transcribiendo audio con Gemini..."):
        try:
            transcribed_text = transcribe_audio(audio_input_bytes)
            if transcribed_text:
                st.success(f"🗣️ Transcripción: *'{transcribed_text}'*")
                prompt = transcribed_text
        except Exception as exc:
            st.error(f"Error transcribiendo audio: {exc}")

if text_prompt:
    prompt = text_prompt

if prompt:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with the LangGraph agent
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_container = st.status("🧠 Evaluando con LangGraph...", expanded=True)

        try:
            # Build conversation history for LangGraph
            graph_input_messages = [
                SystemMessage(
                    content=(
                        "You are a helpful, concise AI assistant equipped with four tools:\n"
                        "1) 'search_web' (Tavily): ONLY call this tool when you need live, real-time web information, latest news, or current facts. If your existing knowledge is sufficient, answer directly without calling search_web.\n"
                        "2) 'call_mcp_server': use this tool to query local MCP system/project status or calculate math expressions.\n"
                        "3) 'speak_response': use this tool to synthesize spoken voice audio with Ollama's phonon:40m-multilingual model whenever the user asks for voice, speech, audio output, or to read aloud.\n"
                        "4) 'read_pdf_file': use this tool to read and analyze local PDF documents."
                    )
                )
            ]

            # Inject active PDF context before the conversation if available
            if st.session_state.get("active_pdf"):
                pdf_context_block = format_pdf_context_for_prompt(st.session_state.active_pdf)
                graph_input_messages.append(HumanMessage(content=pdf_context_block))

            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    graph_input_messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    graph_input_messages.append(AIMessage(content=m["content"]))

            graph_input_messages.append(HumanMessage(content=prompt))

            config = {
                "configurable": {"thread_id": st.session_state.thread_id},
                "run_name": f"streamlit_turn_{st.session_state.current_provider}",
                "tags": ["streamlit-gui", st.session_state.current_provider],
                "metadata": {
                    "provider": st.session_state.current_provider,
                    "thread_id": st.session_state.thread_id,
                },
            }

            status_container.update(label="🤖 El agente está razonando...", state="running")
            result = st.session_state.graph.invoke({"messages": graph_input_messages}, config=config)

            # Extract any tool executions and final response
            tools_called = []
            final_content = ""

            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tools_called.append({
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                            "output": None,
                        })
                elif isinstance(msg, ToolMessage):
                    for tc in tools_called:
                        if tc["name"] == msg.name and tc["output"] is None:
                            tc["output"] = msg.content
                            break

            final_msg = result["messages"][-1]
            final_content = format_ai_content(final_msg.content)

            if tools_called:
                tool_names = ", ".join([f"`{t['name']}`" for t in tools_called])
                status_container.update(
                    label=f"✅ Herramientas utilizadas: {tool_names}",
                    state="complete",
                    expanded=False,
                )
            else:
                status_container.update(
                    label="✅ Respondido directamente (conocimiento del LLM suficiente)",
                    state="complete",
                    expanded=False,
                )

            # Display final assistant message
            response_placeholder.markdown(final_content)

            # Synthesize voice audio if requested or enabled
            generated_audio_bytes = None
            has_voice_tool = any(t["name"] == "speak_response" for t in tools_called)
            if voice_enabled or has_voice_tool:
                try:
                    with st.spinner("🎙️ Generando audio con Ollama phonon:40m..."):
                        generated_audio_bytes, _ = synthesize_speech(
                            final_content,
                            voice=selected_voice,
                            play_audio=False,
                        )
                        st.audio(generated_audio_bytes, format="audio/wav")
                except Exception as voice_exc:
                    st.caption(f"⚠️ No se pudo generar audio de voz: {voice_exc}")

            # Flush LangSmith trace
            ls_client = get_langsmith_client()
            if ls_client:
                try:
                    ls_client.flush()
                except Exception:
                    pass

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_content,
                "tools_called": tools_called,
                "audio_data": generated_audio_bytes,
            })

        except Exception as exc:
            status_container.update(label="❌ Error durante la ejecución", state="error")
            st.error(f"Error procesando la solicitud: {exc}")
