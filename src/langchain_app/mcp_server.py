import ast
import math
import operator
import platform
import sys
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer

# Initialize an official Model Context Protocol (MCP) Server using MCPServer (mcp v2)
mcp = MCPServer("LangChainApp-MCP-Server")

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "abs": abs,
    "round": round,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in SAFE_FUNCTIONS:
            args = [_eval_node(arg) for arg in node.args]
            return SAFE_FUNCTIONS[func_name](*args)
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


@mcp.tool()
def get_system_and_project_status() -> str:
    """Return local system specifications, Python runtime, and project status from the MCP server."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    project_root = Path(__file__).resolve().parents[2]
    return (
        f"[MCP Server: LangChainApp-MCP-Server]\n"
        f"- Timestamp      : {now}\n"
        f"- OS Platform    : {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"- Python Version : {sys.version.split()[0]}\n"
        f"- Project Root   : {project_root}\n"
        f"- Active Stack   : LangChain + LangGraph + Tavily + LangSmith + MCP (stdio)"
    )


@mcp.tool()
def calculate_math(expression: str) -> str:
    """Evaluate a mathematical expression safely inside the MCP server (e.g. '125 * 48 + sqrt(144)')."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_node(parsed.body)
        return f"[MCP Server Math Result] {expression} = {result}"
    except Exception as exc:
        return f"[MCP Server Math Error] Could not evaluate '{expression}': {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")

