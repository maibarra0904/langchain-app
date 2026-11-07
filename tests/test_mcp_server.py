import pytest
from langchain_app.mcp_server import calculate_math, get_system_and_project_status


def test_calculate_math_basic_arithmetic():
    result = calculate_math("125 * 48")
    assert "[MCP Server Math Result]" in result
    assert "6000" in result


def test_calculate_math_functions():
    result = calculate_math("sqrt(144) + abs(-10)")
    assert "[MCP Server Math Result]" in result
    assert "22.0" in result


def test_calculate_math_invalid_expression():
    result = calculate_math("invalid_syntax &&&")
    assert "[MCP Server Math Error]" in result


def test_get_system_and_project_status():
    status = get_system_and_project_status()
    assert "[MCP Server: LangChainApp-MCP-Server]" in status
    assert "OS Platform" in status
    assert "Python Version" in status
    assert "Active Stack" in status

