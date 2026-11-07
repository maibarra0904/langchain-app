import argparse
import asyncio
import os
import re
import sys
import unicodedata
import uuid
from pathlib import Path
from typing import Annotated, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import Client as LangSmithClient
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tavily import TavilyClient

from langchain_app.voice_service import (
    is_phonon_available,
    speak_response,
    synthesize_speech,
)
from langchain_app.pdf_service import (
    extract_text_from_pdf,
    format_pdf_context_for_prompt,
    read_pdf_file,
)

load_dotenv(override=True)

SEARCH_KEYWORDS = (
    "latest",
    "news",
    "today",
    "current",
    "now",
    "2025",
    "2026",
    "search",
    "weather",
    "price",
    "who won",
    "update",
)

MCP_KEYWORDS = (
    "mcp",
    "system status",
    "system info",
    "project status",
    "calculate",
    "math",
    "compute",
    "sqrt",
)


class ChatState(TypedDict):
    """Conversation state managed by LangGraph."""
    messages: Annotated[list[BaseMessage], add_messages]


def get_langsmith_client() -> LangSmithClient | None:
    """Return an authenticated LangSmith client if LANGSMITH_API_KEY is configured."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key or api_key == "lsv2_pt_tu_api_key_aqui":
        return None
    return LangSmithClient(api_key=api_key)


@traceable(run_type="retriever", name="tavily_api_fetch", tags=["tavily", "web-search"])
def fetch_tavily_results(query: str, max_results: int = 3) -> list[dict]:
    """Low-level traced function that queries the Tavily API."""
    print(
        f"  📡 [EXECUTION: Tavily API + LangSmith] Fetching live web results for: '{query}' "
        "(LangSmith is recording this retriever span, not making decisions)."
    )
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "tvly-tu_api_key_aqui":
        raise ValueError("TAVILY_API_KEY is not configured in .env.")

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results)
    return response.get("results", [])


async def _call_mcp_stdio(tool_name: str, arguments: dict) -> str:
    """Connect to the local FastMCP server over stdio and execute a tool."""
    server_script = Path(__file__).resolve().parent / "mcp_server.py"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env=os.environ.copy(),
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_list = await session.list_tools()
            available_names = [t.name for t in tools_list.tools]
            print(
                f"  🔌 [EXECUTION: MCP Server via stdio] Connected! "
                f"Server exposed MCP tools: {available_names} -> Calling '{tool_name}'({arguments})"
            )
            result = await session.call_tool(tool_name, arguments=arguments)
            texts = [
                block.text for block in result.content if hasattr(block, "text")
            ]
            return "\n".join(texts) if texts else str(result)


@traceable(run_type="tool", name="mcp_server_stdio_call", tags=["mcp", "stdio-server"])
def invoke_mcp_server(tool_name: str, arguments: dict | None = None) -> str:
    """Synchronous wrapper traced by LangSmith that calls the MCP server over stdio."""
    return asyncio.run(_call_mcp_stdio(tool_name, arguments or {}))


@tool
def search_web(query: str) -> str:
    """Search the web in real time using Tavily for current information, news, or facts."""
    print(f"  🛠️  [EXECUTION: LangChain @tool 'search_web'] Running tool with query='{query}'...")
    try:
        results = fetch_tavily_results(query=query, max_results=3)
    except Exception as exc:
        return f"Error executing search: {exc}"

    if not results:
        return "No results found."

    formatted = []
    for idx, item in enumerate(results, 1):
        title = item.get("title", "No title")
        url = item.get("url", "")
        content = item.get("content", "")
        formatted.append(f"[{idx}] {title}\nURL: {url}\nSummary: {content}")
    return "\n\n".join(formatted)


@tool
def call_mcp_server(
    action: Literal["system_status", "math", "both"] = "system_status",
    expression: str = "",
) -> str:
    """Call the local Model Context Protocol (MCP) server to get system/project status, evaluate math expressions, or both."""
    print(
        f"  🛠️  [EXECUTION: LangChain @tool 'call_mcp_server'] "
        f"Action='{action}', Expression='{expression}'"
    )
    try:
        outputs = []
        if action in ("system_status", "both"):
            outputs.append(invoke_mcp_server("get_system_and_project_status", {}))
        if action in ("math", "both") and expression.strip():
            outputs.append(
                invoke_mcp_server("calculate_math", {"expression": expression.strip()})
            )
        return "\n\n".join(outputs) if outputs else invoke_mcp_server("get_system_and_project_status", {})
    except Exception as exc:
        return f"Error calling MCP server: {exc}"


def get_tools() -> list:
    """Return the list of @tool-decorated functions available to the agent."""
    return [search_web, call_mcp_server, speak_response, read_pdf_file]


def get_llm(provider: str):
    """Return the configured LLM instance for the selected provider."""
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "tu_api_key_aqui":
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please configure a valid key in your .env file."
            )
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            max_retries=2,
            timeout=20,
        )

    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.2)

    raise ValueError(f"Unsupported provider: {provider}")


SYSTEM_PROMPT = (
    "You are an intelligent, evidence-driven AI assistant.\n"
    "CRITICAL PREMISE - LIMITED MODEL KNOWLEDGE FOR WEB QUERIES:\n"
    "Your internal memory has a static cutoff and strictly limited knowledge for web queries, current events, "
    "real-time facts, software versions, and live documentation. "
    "For any web, technical, or factual inquiry, evidence is retrieved in real-time via Tavily Search, FastMCP, or document tools.\n"
    "Always formulate your response grounded directly on the verified retrieved evidence provided in the context."
)


def normalize_text(text: str) -> str:
    """Normalize text by removing accents and converting to lowercase for reliable intent parsing."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", " ", ascii_text).strip()


def classify_query_intent(query: str) -> tuple[str, any, str]:
    """
    Intelligently analyze query intent under the explicit premise that the LLM has limited knowledge for web queries.
    Returns:
        (intent, payload, reasoning)
    """
    cleaned = normalize_text(query)
    words = cleaned.split()

    # 1. Math calculation request
    math_expr = _extract_math_expression(query)
    if math_expr:
        return (
            "mcp_math",
            math_expr,
            f"Mathematical expression detected ('{math_expr}'). Routing to FastMCP AST evaluation.",
        )

    # 2. Local system status inquiry
    if any(k in cleaned for k in ["estado del sistema", "system status", "info del sistema", "servidor mcp", "mcp status"]):
        return (
            "mcp_status",
            None,
            "Local machine / FastMCP diagnostic status requested.",
        )

    # 3. Spoken voice audio request
    if any(k in cleaned for k in ["en voz alta", "lee con voz", "sintetiza voz", "reproduce audio", "habla", "read aloud", "speak"]):
        return (
            "voice",
            query,
            "Voice speech synthesis requested with Ollama phonon:40m.",
        )

    # 4. Local PDF inspection request
    if ".pdf" in query.lower() or "archivo pdf" in cleaned or "lee el pdf" in cleaned:
        pdf_match = re.search(r"([\w\-./\\]+\.pdf)", query, re.IGNORECASE)
        path = pdf_match.group(1) if pdf_match else query
        return (
            "pdf",
            path,
            f"PDF file inspection requested for '{path}'.",
        )

    # 5. Pure conversational greeting / small talk
    greeting_terms = {
        "hola", "buen", "buenos", "dias", "dia", "tardes", "noches", "que", "tal",
        "como", "estas", "bienvenido", "saludos", "gracias", "muchas", "adios", "hasta",
        "luego", "chau", "bye", "hello", "hi", "hey"
    }
    if len(words) <= 5 and all(w in greeting_terms for w in words):
        return (
            "conversational",
            None,
            "Conversational pleasantry or greeting. Internal knowledge sufficient, no external tools needed.",
        )

    # 6. DEFAULT FOR ALL FACTUAL, TECHNICAL, CURRENT, OR REAL-WORLD INQUIRIES:
    # Explicit premise: The LLM has limited knowledge for web queries.
    return (
        "search",
        query,
        "Factual, informational, or technical inquiry. Because the model has limited knowledge for web queries, routing to Tavily Search for real-time ground truth.",
    )


def _extract_math_expression(text: str) -> str | None:
    """Extract a mathematical expression from a natural language prompt if present."""
    token = r"(?:sqrt\([^)]+\)|sin\([^)]+\)|cos\([^)]+\)|tan\([^)]+\)|log\([^)]+\)|abs\([^)]+\)|[\d.]+)"
    pattern = rf"({token}(?:\s*[\+\-\*\/\%\^]\s*{token})+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).replace("^", "**")
    func_match = re.search(r"((?:sqrt|sin|cos|tan|log|abs)\([^)]+\))", text)
    if func_match:
        return func_match.group(1)
    return None


def build_chat_graph(provider: str):
    """
    Build an intelligible, multi-stage LangGraph workflow.
    
    Structure:
    1. 'analyzer': Evaluates user query under the premise that the LLM has limited web knowledge.
       Decides whether to invoke Tavily ('search_web'), FastMCP ('call_mcp_server'), Voice, or direct conversation.
    2. 'tools': ToolNode executing the selected external service (Tavily, FastMCP, etc.).
    3. 'chatbot': Synthesis node where the LLM (Ollama / Gemini) articulates the grounded final response.
    """
    tools = get_tools()
    base_llm = get_llm(provider)
    provider_label = provider.upper()

    def analyzer_node(state: ChatState):
        """Intelligible intent evaluation node: determines external tool requirements considering limited model knowledge."""
        user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        query = user_msg.content if user_msg else ""
        intent, payload, reasoning = classify_query_intent(query)

        print(
            f"\n🧭 [INTENT ANALYZER (Premisa: Conocimiento del modelo limitado para consultas web)]\n"
            f"   ↳ Consulta: '{query}'\n"
            f"   ↳ Decisión: {intent.upper()} -> {reasoning}"
        )

        call_id = f"call_{uuid.uuid4().hex[:8]}"
        if intent == "search":
            ai_msg = AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": payload}, "id": call_id}],
            )
            return {"messages": [ai_msg]}

        elif intent == "mcp_math":
            ai_msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": "call_mcp_server",
                    "args": {"action": "math", "expression": payload},
                    "id": call_id,
                }],
            )
            return {"messages": [ai_msg]}

        elif intent == "mcp_status":
            ai_msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": "call_mcp_server",
                    "args": {"action": "system_status", "expression": ""},
                    "id": call_id,
                }],
            )
            return {"messages": [ai_msg]}

        elif intent == "voice":
            ai_msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": "speak_response",
                    "args": {"text": payload},
                    "id": call_id,
                }],
            )
            return {"messages": [ai_msg]}

        elif intent == "pdf":
            ai_msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_pdf_file",
                    "args": {"file_path": payload},
                    "id": call_id,
                }],
            )
            return {"messages": [ai_msg]}

        # Conversational greeting
        return {"messages": []}

    def analyzer_router(state: ChatState) -> Literal["tools", "chatbot"]:
        last_msg = state["messages"][-1] if state["messages"] else None
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            print("🔀 [LangGraph Router] Herramienta requerida -> enrutando a 'tools' (ToolNode).")
            return "tools"
        print("🔀 [LangGraph Router] Respuesta directa sin herramientas -> enrutando a 'chatbot'.")
        return "chatbot"

    def chatbot_node(state: ChatState):
        """Synthesizer node: generates grounded answer using the retrieved tool outputs and conversation history."""
        # Isolate the current turn by finding the last HumanMessage
        last_human_idx = -1
        for idx in range(len(state["messages"]) - 1, -1, -1):
            if isinstance(state["messages"][idx], HumanMessage):
                last_human_idx = idx
                break

        current_turn_messages = (
            state["messages"][last_human_idx:]
            if last_human_idx != -1
            else state["messages"]
        )
        query = state["messages"][last_human_idx].content if last_human_idx != -1 else ""

        # Extract tool outputs ONLY from the current turn!
        tool_outputs = []
        math_results = []
        for m in current_turn_messages:
            if isinstance(m, ToolMessage):
                tool_outputs.append(f"[{m.name.upper()}]:\n{m.content}")
                if "math result" in m.content.lower() or "=" in m.content:
                    math_results.append(m.content)

        if math_results:
            math_text = "\n".join(math_results)
            grounded_prompt = (
                f"El servidor de herramientas FastMCP ha evaluado exitosamente la siguiente operación matemática:\n"
                f"{math_text}\n\n"
                f"Presenta de forma concisa y clara al usuario el resultado del cálculo en español:"
            )
            try:
                response = base_llm.invoke([HumanMessage(content=grounded_prompt)])
                resp_text = response.content if hasattr(response, "content") else str(response)
                # Ensure numbers/result are present in the response
                if not any(char.isdigit() for char in resp_text):
                    resp_text = f"🧮 **Resultado del cálculo (FastMCP):**\n\n{math_text}"
                final_text = resp_text
            except Exception:
                final_text = f"🧮 **Resultado del cálculo (FastMCP):**\n\n{math_text}"

        elif tool_outputs:
            context_block = "\n\n".join(tool_outputs)
            grounded_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"EVIDENCIA OBTENIDA EN TIEMPO REAL (Tavily Search / Herramientas):\n"
                f"{context_block}\n\n"
                f"CONSULTA DEL USUARIO:\n{query}\n\n"
                "INSTRUCCIÓN: Responde de manera clara, precisa y directamente fundamentada en la evidencia anterior."
            )
            try:
                response = base_llm.invoke([HumanMessage(content=grounded_prompt)])
                resp_text = response.content if hasattr(response, "content") else str(response)
                final_text = resp_text if resp_text.strip() else context_block
            except Exception as exc:
                if "does not support tools" in str(exc).lower():
                    response = base_llm.invoke([HumanMessage(content=grounded_prompt)])
                    resp_text = response.content if hasattr(response, "content") else str(response)
                    final_text = resp_text if resp_text.strip() else context_block
                else:
                    final_text = context_block
        else:
            # Direct response for conversational greetings
            try:
                response = base_llm.invoke(state["messages"])
            except Exception as exc:
                if "does not support tools" in str(exc).lower():
                    response = base_llm.invoke(state["messages"])
                else:
                    raise exc
            final_text = response.content if hasattr(response, "content") else str(response)

        print(f"💬 [SÍNTESIS FINAL ({provider_label})]: Respuesta fundamentada generada.")
        return {"messages": [AIMessage(content=final_text)]}

    builder = StateGraph(ChatState)
    builder.add_node("analyzer", analyzer_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("chatbot", chatbot_node)

    # Workflow edges:
    builder.add_edge(START, "analyzer")
    builder.add_conditional_edges("analyzer", analyzer_router, {"tools": "tools", "chatbot": "chatbot"})
    builder.add_edge("tools", "chatbot")
    builder.add_edge("chatbot", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def format_ai_content(content) -> str:
    """Extract readable text from string or structured content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


@traceable(
    run_type="chain",
    name="langsmith_decision_pipeline",
    tags=["langsmith-demo", "langgraph-mcp-tavily"],
)
def test_search(
    query: str = "What are the latest news about LangGraph in 2026?",
    provider: str | None = None,
) -> dict:
    """Run a LangSmith-traced test demonstrating LangGraph's multi-option routing (MCP, Tavily, LLM)."""
    project_name = os.getenv("LANGSMITH_PROJECT", "langchain-app")
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"

    print("==================================================================")
    print("  📊 LangGraph + MCP Server + Tavily + LangSmith Decision Pipeline")
    print("==================================================================")
    print(f"  • LangSmith Tracing : {tracing_enabled} (Observer only - records all steps)")
    print(f"  • LangSmith Project : {project_name}")
    print(f"  • User Query        : {query}")

    run_tree = get_current_run_tree()
    run_id = str(run_tree.id) if run_tree else None

    tool_output = None
    agent_answer = None

    if provider:
        graph = build_chat_graph(provider)
        config = {
            "configurable": {"thread_id": "langsmith-test-thread"},
            "run_name": f"langgraph_agent_{provider}",
            "tags": ["langgraph", "mcp", provider],
            "metadata": {"provider": provider, "thread_id": "langsmith-test-thread"},
        }
        result = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ]
            },
            config=config,
        )
        final_msg = result["messages"][-1]
        agent_answer = format_ai_content(final_msg.content)
        print(f"\n💬 Final Answer ({provider.upper()}):\n{agent_answer}\n")
    else:
        print(
            "\n🧭 [DECISION BY: Direct CLI Test Mode (--test-search without --provider)]\n"
            "   ↳ Calling @tool 'search_web' directly without an LLM."
        )
        tool_output = search_web.invoke(
            {"query": query},
            config={
                "run_name": "search_web_tool_call",
                "tags": ["tool-execution", "tavily"],
                "metadata": {"query": query, "source": "test_search"},
            },
        )
        print(f"\n📄 Search Results:\n{tool_output}\n")

    ls_client = get_langsmith_client()
    if ls_client and run_id:
        try:
            ls_client.flush()
            print(f"✅ [OBSERVER: LangSmith] Trace flushed! (Run ID: {run_id})")
            print(f"   Project Dashboard: https://smith.langchain.com (Project: '{project_name}')")
        except Exception as exc:
            print(f"⚠️ Could not flush LangSmith trace: {exc}")

    return {
        "query": query,
        "run_id": run_id,
        "project": project_name,
        "search_output": tool_output,
        "agent_answer": agent_answer,
    }


def select_provider() -> str:
    """Prompt the user to choose an LLM provider."""
    print("\n==============================================================")
    print("  🤖 LangChain + LangGraph + MCP + Tavily + LangSmith CLI")
    print("==============================================================")
    print("Select an LLM provider:")
    print("  [1] Ollama (Local model)")
    print("  [2] Google Gemini (Cloud API)")

    while True:
        choice = input("\nEnter choice (1 or 2, default 1): ").strip() or "1"
        if choice == "1":
            return "ollama"
        if choice == "2":
            return "gemini"
        print("Invalid option. Please type 1 or 2.")


@traceable(run_type="tool", name="direct_mcp_cli_call", tags=["mcp-cli"])
def invoke_direct_mcp(
    action: Literal["system_status", "math", "both"],
    expression: str,
) -> str:
    """Execute direct MCP command traced in LangSmith without Tavily."""
    mcp_result = call_mcp_server.invoke({"action": action, "expression": expression})
    ls_client = get_langsmith_client()
    if ls_client:
        try:
            ls_client.flush()
        except Exception:
            pass
    return mcp_result


@traceable(run_type="chain", name="interactive_chat_turn", tags=["cli-turn"])
def invoke_traced_turn(graph, messages: list[BaseMessage], config: dict, provider: str) -> dict:
    """Execute a single conversational turn traced as a parent chain in LangSmith."""
    print(
        "\n📊 [OBSERVER: LangSmith] Started parent trace 'interactive_chat_turn' "
        "(recording all decisions below)."
    )
    result = graph.invoke({"messages": messages}, config=config)
    run_tree = get_current_run_tree()
    if run_tree:
        print(f"✅ [OBSERVER: LangSmith] Trace flushed! (Run ID: {run_tree.id})")
    ls_client = get_langsmith_client()
    if ls_client:
        try:
            ls_client.flush()
        except Exception:
            pass
    return result


def check_langsmith_status() -> None:
    """Check LangSmith configuration and list recent traces from the cloud dashboard."""
    project_name = os.getenv("LANGSMITH_PROJECT", "langchain-app")
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    print("\n==============================================================")
    print("  📊 Verificación de Monitoreo en LangSmith")
    print("==============================================================")
    print(f"  • LANGSMITH_TRACING : {'ENABLED (true)' if tracing_enabled else 'DISABLED (false)'}")
    print(f"  • LANGSMITH_ENDPOINT: {endpoint}")
    print(f"  • LANGSMITH_PROJECT : {project_name}")
    print(f"  • LANGSMITH_API_KEY : {'Configurada (' + api_key[:12] + '...)' if api_key and api_key != 'lsv2_pt_tu_api_key_aqui' else 'NO CONFIGURADA'}")

    client = get_langsmith_client()
    if not client:
        print("\n❌ No se pudo conectar a LangSmith. Verifica tu LANGSMITH_API_KEY en .env")
        return

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runs = list(client.list_runs(project_name=project_name, limit=6))
        print(f"\n✅ Conexión exitosa a LangSmith. Últimas {len(runs)} trazas registradas:")
        for r in runs:
            ts = r.start_time.strftime("%Y-%m-%d %H:%M:%S") if r.start_time else "N/A"
            tokens = getattr(r, "total_tokens", None)
            tokens_str = f" | Tokens: {tokens}" if tokens else ""
            print(f"  • [{ts}] Run: {r.name:<28} | Tipo: {r.run_type:<10} | Estado: {r.status}{tokens_str}")
        print(f"\n🔗 Dashboard de monitoreo en vivo: https://smith.langchain.com (Proyecto: '{project_name}')\n")
    except Exception as exc:
        print(f"\n⚠️ Error al consultar el proyecto en LangSmith: {exc}\n")


def run_cli() -> None:
    """Interactive terminal loop to chat with the tool-enabled LangGraph agent."""
    parser = argparse.ArgumentParser(
        description="LangChain + LangGraph + MCP Server + Tavily @tool + LangSmith CLI"
    )
    parser.add_argument(
        "--test-search",
        nargs="?",
        const="What are the latest news about LangGraph in 2026?",
        help="Run a LangSmith-traced search test showing who makes each decision",
    )
    parser.add_argument(
        "--test-mcp",
        nargs="?",
        const="Check the MCP system status and calculate 125 * 48 + sqrt(144)",
        help="Run a LangSmith-traced MCP Server test through LangGraph",
    )
    parser.add_argument(
        "--verify-langsmith",
        action="store_true",
        help="Verify connection with LangSmith and display recent traces",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Streamlit graphical user interface (GUI)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to a PDF file to read and attach as context before processing query",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini"],
        default=None,
        help="LLM provider to use with --test-search or --test-mcp",
    )
    args = parser.parse_args()

    if args.ui:
        import subprocess
        app_path = Path(__file__).resolve().parents[2] / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
        return

    if args.verify_langsmith:
        check_langsmith_status()
        return

    # If --pdf is supplied with test commands
    pdf_context = ""
    if args.pdf:
        try:
            print(f"\n📄 [Servicio PDF] Procesando documento: '{args.pdf}'...")
            pdf_data = extract_text_from_pdf(args.pdf)
            pdf_context = format_pdf_context_for_prompt(pdf_data)
            print(f"✅ Documento cargado: {pdf_data['total_pages']} páginas ({pdf_data['char_count']} caracteres extraídos).\n")
        except Exception as exc:
            print(f"❌ Error leyendo PDF '{args.pdf}': {exc}\n")

    if args.test_mcp:
        query_text = f"{pdf_context}\n\n{args.test_mcp}" if pdf_context else args.test_mcp
        test_search(query=query_text, provider=args.provider or "ollama")
        return

    if args.test_search:
        query_text = f"{pdf_context}\n\n{args.test_search}" if pdf_context else args.test_search
        test_search(query=query_text, provider=args.provider)
        return

    provider = args.provider or select_provider()

    try:
        graph = build_chat_graph(provider)
    except Exception as exc:
        print(f"\n❌ Error initializing '{provider}': {exc}")
        sys.exit(1)

    thread_id = "default-session"
    project_name = os.getenv("LANGSMITH_PROJECT", "langchain-app")
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    initialized_system = False

    print(
        f"\n✅ Connected to provider: {provider.upper()} "
        "(with @tool 'search_web' & @tool 'call_mcp_server')"
    )
    print(f"📊 LangSmith Tracing: {'ENABLED' if tracing_enabled else 'DISABLED'} (Project: {project_name})")
    print("Commands: '/mcp' | '/search <query>' | '/switch' | '/clear' | '/exit'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower().rstrip("\\/")
        if cmd in {"/exit", "exit", "quit", "/quit"}:
            print("👋 Goodbye!")
            break

        if cmd == "/clear":
            thread_id = f"session-{os.urandom(4).hex()}"
            initialized_system = False
            print("🧹 Conversation memory cleared.\n")
            continue

        if cmd == "/switch":
            new_provider = "gemini" if provider == "ollama" else "ollama"
            try:
                graph = build_chat_graph(new_provider)
                provider = new_provider
                print(f"🔄 Switched provider to: {provider.upper()}\n")
            except Exception as exc:
                print(f"❌ Could not switch to '{new_provider}': {exc}\n")
            continue

        if cmd.startswith("/mcp"):
            expr = user_input[4:].strip()
            action: Literal["system_status", "math", "both"] = "math" if expr else "system_status"
            print(f"\n🔌 Direct MCP Server call (action='{action}'):")
            try:
                mcp_output = invoke_direct_mcp(action, expr)
                print(f"{mcp_output}\n")
            except Exception as exc:
                print(f"⚠️ Error ejecutando comando MCP: {exc}\n")
            continue

        if cmd.startswith("/speak"):
            speak_text = user_input[6:].strip() or "Hola, este es el modelo phonon de Ollama hablando."
            print(f"\n🎙️ Generando voz con Ollama phonon:40m...")
            try:
                _, audio_path = synthesize_speech(speak_text, voice="Alba", play_audio=True)
                print(f"🔊 Audio generado y reproducido: {audio_path}\n")
            except Exception as exc:
                print(f"❌ Error en síntesis de voz: {exc}\n")
            continue

        if cmd.startswith("/pdf"):
            pdf_path_str = user_input[4:].strip()
            if not pdf_path_str:
                print("⚠️ Especifica la ruta al archivo PDF. Ejemplo: /pdf documentos/reporte.pdf\n")
                continue
            print(f"\n📄 [Servicio PDF] Procesando documento: '{pdf_path_str}'...")
            try:
                pdf_data = extract_text_from_pdf(pdf_path_str)
                pdf_ctx_msg = format_pdf_context_for_prompt(pdf_data)
                messages.append(HumanMessage(content=pdf_ctx_msg))
                print(f"✅ Documento '{pdf_data['filename']}' ({pdf_data['total_pages']} páginas) inyectado al contexto del agente.\n")
            except Exception as exc:
                print(f"❌ Error leyendo PDF: {exc}\n")
            continue

        if cmd.startswith("/search"):
            search_query = user_input[7:].strip() or "Latest AI news"
            try:
                test_search(query=search_query, provider=None)
            except Exception as exc:
                print(f"❌ Error running search: {exc}\n")
            continue

        messages: list[BaseMessage] = []
        if not initialized_system:
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
            initialized_system = True
        messages.append(HumanMessage(content=user_input))

        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": f"chat_turn_{provider}",
            "tags": ["cli-chat", "mcp", provider],
            "metadata": {"provider": provider, "thread_id": thread_id},
        }

        try:
            result = invoke_traced_turn(
                graph=graph,
                messages=messages,
                config=config,
                provider=provider,
            )
            ai_message = result["messages"][-1]
            print(f"\nAI ({provider}): {format_ai_content(ai_message.content)}\n")
        except Exception as exc:
            print(f"\n❌ Error communicating with {provider}: {exc}\n")


if __name__ == "__main__":
    run_cli()
