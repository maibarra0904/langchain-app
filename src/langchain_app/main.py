import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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
    return [search_web, call_mcp_server]


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
    """Build a stateful LangGraph workflow with 3 entry paths: Forced Search, Forced MCP, or LLM Autonomy."""
    tools = get_tools()
    llm_with_tools = get_llm(provider).bind_tools(tools)
    provider_label = provider.upper()

    def langgraph_entry_router(
        state: ChatState,
    ) -> Literal["langgraph_forced_mcp", "langgraph_forced_search", "chatbot"]:
        """STEP 1 DECISION: LangGraph inspects the user message and selects from 3 options BEFORE the LLM runs."""
        last_user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_msg = str(msg.content)
                break

        lower_msg = last_user_msg.lower()
        print(
            "\n🧭 [STEP 1 - DECISION BY: LangGraph Entry Router]\n"
            "   Available options for LangGraph:\n"
            "     [Option A] 'langgraph_forced_mcp'    -> Call the local MCP Server directly\n"
            "     [Option B] 'langgraph_forced_search' -> Call the Tavily Web Search tool directly\n"
            f"     [Option C] 'chatbot'                 -> Delegate decision to {provider_label}"
        )

        matched_mcp = [kw for kw in MCP_KEYWORDS if kw in lower_msg]
        math_expr = _extract_math_expression(lower_msg)
        if matched_mcp or math_expr:
            reason = f"keywords {matched_mcp}" if matched_mcp else f"math expression '{math_expr}'"
            print(
                f"   ↳ Detected MCP trigger ({reason}).\n"
                f"   ↳ ✅ LangGraph SELECTED [Option A]: Run 'langgraph_forced_mcp' FIRST "
                f"(calling MCP Server before {provider_label})."
            )
            return "langgraph_forced_mcp"

        matched_search = [kw for kw in SEARCH_KEYWORDS if kw in lower_msg]
        if matched_search:
            print(
                f"   ↳ Detected real-time web keywords {matched_search}.\n"
                f"   ↳ ✅ LangGraph SELECTED [Option B]: Run 'langgraph_forced_search' FIRST "
                f"(calling Tavily before {provider_label})."
            )
            return "langgraph_forced_search"

        print(
            "   ↳ No forced MCP or Search triggers detected.\n"
            f"   ↳ ✅ LangGraph SELECTED [Option C]: Route directly to 'chatbot' "
            f"and let {provider_label} choose between answering or calling a tool."
        )
        return "chatbot"

    def langgraph_forced_mcp_node(state: ChatState):
        """Node executed when LangGraph selects Option A (MCP Server)."""
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        math_expr = _extract_math_expression(query)
        wants_status = any(k in query.lower() for k in ("status", "system", "project", "mcp"))
        if math_expr and wants_status:
            action: Literal["system_status", "math", "both"] = "both"
        elif math_expr:
            action = "math"
        else:
            action = "system_status"
        print(
            f"\n⚙️  [STEP 2 - EXECUTION BY: LangGraph 'langgraph_forced_mcp' Node]\n"
            f"   ↳ LangGraph is invoking @tool 'call_mcp_server' (action='{action}') and passing the MCP response to {provider_label}."
        )
        mcp_output = call_mcp_server.invoke(
            {"action": action, "expression": math_expr or ""}
        )
        context_msg = HumanMessage(
            content=(
                f"[LangGraph Pre-Fetched MCP Server Output for '{query}']:\n\n"
                f"{mcp_output}\n\n"
                f"Please use the MCP server output above to answer my request directly without calling call_mcp_server again."
            )
        )
        return {"messages": [context_msg]}

    def langgraph_forced_search_node(state: ChatState):
        """Node executed when LangGraph selects Option B (Tavily Web Search)."""
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        print(
            f"\n⚙️  [STEP 2 - EXECUTION BY: LangGraph 'langgraph_forced_search' Node]\n"
            f"   ↳ LangGraph is invoking @tool 'search_web' directly and injecting context for {provider_label}."
        )
        search_results = search_web.invoke({"query": query})
        context_msg = HumanMessage(
            content=(
                f"[LangGraph Pre-Fetched Web Search Results for '{query}']:\n\n"
                f"{search_results}\n\n"
                f"Please use the search results above to answer my question accurately without calling search_web again."
            )
        )
        return {"messages": [context_msg]}

    def chatbot_node(state: ChatState):
        """Node where the LLM (Gemini or Ollama) runs and makes its own decision."""
        print(
            f"\n🧠 [STEP - DECISION BY: LLM ({provider_label}) inside 'chatbot' Node]\n"
            f"   ↳ {provider_label} has access to tools: ['search_web', 'call_mcp_server']\n"
            f"   ↳ Sending conversation state ({len(state['messages'])} messages) to {provider_label}..."
        )
        response = llm_with_tools.invoke(state["messages"])

        if getattr(response, "tool_calls", None):
            tool_names = [tc["name"] for tc in response.tool_calls]
            print(
                f"   ↳ ✅ {provider_label} DECIDED it needs external tools and requested: {tool_names}"
            )
        else:
            print(
                f"   ↳ ✅ {provider_label} DECIDED it has enough information to write the final response directly."
            )
        return {"messages": [response]}

    def langgraph_post_llm_router(
        state: ChatState,
    ) -> Literal["tools", "__end__"]:
        """DECISION: LangGraph inspects the LLM's output message to route next step."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            print(
                f"\n🔀 [STEP - DECISION BY: LangGraph Post-LLM Router]\n"
                f"   ↳ Inspected {provider_label}'s output and found tool_calls={last_message.tool_calls}.\n"
                f"   ↳ ✅ LangGraph DECIDED to route to the 'tools' node (ToolNode) to execute the requested tool(s)."
            )
            return "tools"

        print(
            f"\n🏁 [STEP - DECISION BY: LangGraph Post-LLM Router]\n"
            f"   ↳ Inspected {provider_label}'s output: no tool_calls found.\n"
            f"   ↳ ✅ LangGraph DECIDED to route to END and return the final answer to the user."
        )
        return END

    builder = StateGraph(ChatState)
    builder.add_node("langgraph_forced_mcp", langgraph_forced_mcp_node)
    builder.add_node("langgraph_forced_search", langgraph_forced_search_node)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", ToolNode(tools))

    # 1. Entry routing decision made by LangGraph BEFORE the LLM runs (3 options!)
    builder.add_conditional_edges(
        START,
        langgraph_entry_router,
        {
            "langgraph_forced_mcp": "langgraph_forced_mcp",
            "langgraph_forced_search": "langgraph_forced_search",
            "chatbot": "chatbot",
        },
    )
    # 2. Both pre-execution nodes flow into chatbot so the LLM summarizes the result
    builder.add_edge("langgraph_forced_mcp", "chatbot")
    builder.add_edge("langgraph_forced_search", "chatbot")

    # 3. Post-LLM routing decision made by LangGraph based on whether LLM asked for tools
    builder.add_conditional_edges(
        "chatbot",
        langgraph_post_llm_router,
        {
            "tools": "tools",
            END: END,
        },
    )
    # 4. If ToolNode ran, loop back to chatbot
    builder.add_edge("tools", "chatbot")

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
                    SystemMessage(
                        content=(
                            "You are a helpful AI assistant equipped with two tools: "
                            "'search_web' (Tavily web search) and 'call_mcp_server' (local MCP server for system status and math)."
                        )
                    ),
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


@traceable(run_type="chain", name="interactive_chat_turn", tags=["cli-turn"])
def invoke_traced_turn(graph, messages: list[BaseMessage], config: dict, provider: str) -> dict:
    """Execute a single conversational turn traced as a parent chain in LangSmith."""
    print(
        "\n📊 [OBSERVER: LangSmith] Started parent trace 'interactive_chat_turn' "
        "(recording all decisions below)."
    )
    return graph.invoke({"messages": messages}, config=config)


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
        "--provider",
        choices=["ollama", "gemini"],
        default=None,
        help="LLM provider to use with --test-search or --test-mcp",
    )
    args = parser.parse_args()

    if args.test_mcp:
        test_search(query=args.test_mcp, provider=args.provider or "gemini")
        return

    if args.test_search:
        test_search(query=args.test_search, provider=args.provider)
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
            action = "math" if expr else "system_status"
            print(f"\n🔌 Direct MCP Server call (action='{action}'):")
            print(call_mcp_server.invoke({"action": action, "expression": expr}) + "\n")
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
            messages.append(
                SystemMessage(
                    content=(
                        "You are a helpful, concise AI assistant equipped with two tools: "
                        "1) 'search_web' for current events, facts, or live web information. "
                        "2) 'call_mcp_server' for querying the local MCP server's system/project status or calculating math."
                    )
                )
            )
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
