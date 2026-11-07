import pytest
from langchain_core.messages import HumanMessage
from langchain_app.main import (
    _extract_math_expression,
    format_ai_content,
    build_chat_graph,
    classify_query_intent,
    normalize_text,
)


def test_extract_math_expression_simple():
    expr = _extract_math_expression("Calculate 125 * 48 please")
    assert expr == "125 * 48"


def test_extract_math_expression_function():
    expr = _extract_math_expression("What is sqrt(144)?")
    assert expr == "sqrt(144)"


def test_extract_math_expression_none():
    expr = _extract_math_expression("Tell me a story about robots")
    assert expr is None


def test_format_ai_content_string():
    assert format_ai_content("Hello world") == "Hello world"


def test_format_ai_content_dict_blocks():
    content = [{"text": "First part"}, {"text": "Second part"}]
    assert format_ai_content(content) == "First part\nSecond part"


def test_build_chat_graph_structure():
    graph = build_chat_graph("ollama")
    node_names = set(graph.nodes.keys())
    assert "analyzer" in node_names
    assert "chatbot" in node_names
    assert "tools" in node_names


def test_classify_query_intent_knowledge_search():
    intent, payload, reasoning = classify_query_intent("¿Cuáles son las novedades de LangGraph en 2026?")
    assert intent == "search"
    assert "novedades de LangGraph" in payload
    assert "limited knowledge" in reasoning or "limitado" in reasoning


def test_classify_query_intent_conversational_greeting():
    intent, payload, reasoning = classify_query_intent("Hola, buenos días")
    assert intent == "conversational"
    assert payload is None


def test_classify_query_intent_math():
    intent, payload, _ = classify_query_intent("Calcula 125 * 48 + sqrt(144)")
    assert intent == "mcp_math"
    assert payload is not None


def test_classify_query_intent_system_status():
    intent, payload, _ = classify_query_intent("Verifica el estado del sistema con MCP")
    assert intent == "mcp_status"


def test_classify_query_intent_voice():
    intent, payload, _ = classify_query_intent("Lee en voz alta este resumen")
    assert intent == "voice"


def test_classify_query_intent_pdf():
    intent, payload, _ = classify_query_intent("Analiza el archivo documento_trimestral.pdf")
    assert intent == "pdf"
    assert "documento_trimestral.pdf" in payload


def test_graph_executes_math_via_mcp():
    graph = build_chat_graph("ollama")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Calcula 125 * 48 + sqrt(144)")]},
        config={"configurable": {"thread_id": "test-math-exec"}},
    )
    final_message = result["messages"][-1]
    assert "6012" in final_message.content


def test_get_tools_contains_all_services():
    from langchain_app.main import get_tools
    tool_names = [t.name for t in get_tools()]
    assert "search_web" in tool_names
    assert "call_mcp_server" in tool_names
    assert "speak_response" in tool_names
    assert "read_pdf_file" in tool_names
