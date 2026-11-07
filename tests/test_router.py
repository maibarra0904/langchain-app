import pytest
from langchain_core.messages import HumanMessage
from langchain_app.main import (
    _extract_math_expression,
    format_ai_content,
    build_chat_graph,
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


def test_build_chat_graph_has_all_tavily_nodes():
    graph = build_chat_graph("gemini")
    node_names = set(graph.nodes.keys())
    assert "langgraph_forced_mcp" in node_names
    assert "langgraph_forced_search" in node_names
    assert "langgraph_general_search" in node_names
    assert "chatbot" in node_names
    assert "tools" in node_names
