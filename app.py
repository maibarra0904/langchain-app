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
    .tool-box {
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.88rem;
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

    # LangSmith Observability Status
    st.subheader("📊 LangSmith Observability")
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    project_name = os.getenv("LANGSMITH_PROJECT", "langchain-app")
    has_ls_key = bool(os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_API_KEY") != "lsv2_pt_tu_api_key_aqui")

    if tracing_enabled and has_ls_key:
        st.success(f"🟢 Monitoreo activo ({project_name})")
        st.link_button(
            "Ver Dashboard en LangSmith ↗",
            f"https://smith.langchain.com",
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
        if "tools_called" in msg and msg["tools_called"]:
            with st.expander("🛠️ Detalle de herramientas ejecutadas", expanded=False):
                for tool in msg["tools_called"]:
                    st.markdown(f"**Herramienta:** `{tool['name']}`")
                    if tool.get("output"):
                        st.code(tool["output"], language="text")


# -----------------------------------------------------------------------------
# User Input & Agent Response Loop
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Escribe tu pregunta o petición para el agente..."):
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
                        "You are a helpful, concise AI assistant equipped with two tools:\n"
                        "1) 'search_web' (Tavily): ONLY call this tool when you need live, real-time web information, latest news, or current facts. If your existing knowledge is sufficient, answer directly without calling search_web.\n"
                        "2) 'call_mcp_server': use this tool to query local MCP system/project status or calculate math expressions."
                    )
                )
            ]
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
            })

        except Exception as exc:
            status_container.update(label="❌ Error durante la ejecución", state="error")
            st.error(f"Error procesando la solicitud: {exc}")
