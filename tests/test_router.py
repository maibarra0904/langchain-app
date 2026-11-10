import pytest
from unittest.mock import patch
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
    assert "summarizer" in node_names


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


def test_multiturn_no_tool_state_leak():
    graph = build_chat_graph("ollama")
    config = {"configurable": {"thread_id": "test-leak-isolation"}}

    # Turn 1: Math calculation
    res1 = graph.invoke(
        {"messages": [HumanMessage(content="Calcula 50 * 2")]},
        config=config,
    )
    assert "100" in res1["messages"][-1].content

    # Turn 2: Conversational greeting
    res2 = graph.invoke(
        {"messages": [HumanMessage(content="Hola, buen día")]},
        config=config,
    )
    turn2_answer = res2["messages"][-1].content
    assert "100" not in turn2_answer
    assert "MCP" not in turn2_answer


def test_get_tools_contains_all_services():
    from langchain_app.main import get_tools
    tool_names = [t.name for t in get_tools()]
    assert "search_web" in tool_names
    assert "call_mcp_server" in tool_names
    assert "speak_response" in tool_names
    assert "read_pdf_file" in tool_names
    assert "upload_pdf_to_cloud" in tool_names
    assert "list_cloud_documents" in tool_names
    assert "search_pdf_knowledge" in tool_names
    assert "read_cloud_document" in tool_names


def test_classify_query_intent_cloud_document():
    with patch("langchain_app.cloud_context_service.detect_cloud_document_allusion") as mock_detect:
        mock_detect.return_value = (True, {"file_name": "resultados.pdf"})
        intent, payload, reasoning = classify_query_intent("Cuáles son los resultados de mis exámenes?")
        assert intent == "cloud_doc"
        assert "Cloudinary" in reasoning


# ---------------------------------------------------------------------------
# Hindsight long-term memory integration
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage


def test_classify_query_intent_memory_store():
    for text in [
        "Recuerda que mi color favorito es el azul",
        "No olvides que tengo una reunión el lunes",
        "Remember that I prefer dark mode",
    ]:
        intent, payload, reasoning = classify_query_intent(text)
        assert intent == "memory_store", text
        assert payload == text
        assert "Hindsight" in reasoning


def test_classify_query_intent_memory_reflect():
    for text in [
        "¿Qué sabes de mí?",
        "sabes algo sobre mi",  # real user question that previously went to Tavily
        "¿Recuerdas cuál es mi color favorito?",
        "What do you know about me?",
    ]:
        intent, _, _ = classify_query_intent(text)
        assert intent == "memory_reflect", text


def test_classify_personal_questions_go_to_long_term_memory():
    """Questions about the user must be answered from memory, not from a web search."""
    for text in [
        "¿En qué aplicación estoy trabajando?",
        "¿Cómo me llamo?",
        "¿Qué me recomiendas para mi proyecto?",
        "Me llamo Mario y estoy construyendo un agente con LangGraph",
        "Hola, soy Mario",
        "Tengo un perro llamado Toby",
        "¿Dónde vivo?",
        "¿En qué ciudad vivo?",
        "¿Cuántos años tengo?",
    ]:
        intent, payload, reasoning = classify_query_intent(text)
        assert intent == "user_memory", text
        assert payload == text
        assert "memoria" in reasoning.lower()


def test_classify_personal_questions_needing_live_data_use_memory_and_web():
    for text in ["¿Cuál es el clima en mi ciudad hoy?", "¿Cómo puedo mejorar mi app con las últimas novedades?"]:
        intent, _, reasoning = classify_query_intent(text)
        assert intent == "personal_search", text
        assert "memoria" in reasoning.lower() and "web" in reasoning.lower()


def test_memory_detection_does_not_hijack_other_intents():
    assert classify_query_intent("¿Qué sabes de LangGraph 2026?")[0] == "search"
    assert classify_query_intent("¿Me puedes decir quién ganó el mundial?")[0] == "search"
    assert classify_query_intent("¿Cuáles son las novedades de LangGraph en 2026?")[0] == "search"
    assert classify_query_intent("Calcula 125 * 48")[0] == "mcp_math"
    assert classify_query_intent("Hola, buenos días")[0] == "conversational"
    assert classify_query_intent("Estoy bien, gracias, ¿y tú?")[0] == "conversational"
    assert classify_query_intent("¿Dónde ver el partido en vivo?")[0] == "search"
    assert classify_query_intent("¿Cuál es el mejor trabajo?")[0] == "search"
    assert classify_query_intent("Lee en voz alta mi resumen")[0] == "voice"
    assert classify_query_intent("Analiza el archivo mi_reporte.pdf")[0] == "pdf"


def test_build_chat_graph_has_memory_nodes():
    graph = build_chat_graph("ollama")
    assert {"memory_recall", "memory_retain"} <= set(graph.nodes.keys())


def test_get_tools_contains_memory_tools():
    from langchain_app.main import get_tools
    names = {t.name for t in get_tools()}
    assert {"remember_information", "search_long_term_memory", "reflect_on_user_memory"} <= names


def _fake_llm(*answers: str) -> MagicMock:
    llm = MagicMock()
    replies = [AIMessage(content=a) for a in (answers or ("Respuesta del modelo",))]
    llm.invoke.side_effect = replies + [replies[-1]] * 5
    return llm


MEMORIES = [{"text": "El usuario se llama Mario y vive en Guayaquil", "type": "world", "occurred": None, "entities": []}]


def _tool_names(result) -> list[str]:
    return [m.name for m in result["messages"] if isinstance(m, ToolMessage)]


def test_personal_question_answers_from_memory_not_web():
    """Root cause reproduction: '¿En qué aplicación estoy trabajando?' used to hit Tavily."""
    llm = _fake_llm("Estás desarrollando una aplicación de medicina.")
    with patch("langchain_app.main.get_llm", return_value=llm), \
         patch("langchain_app.main.fetch_tavily_results") as mock_web, \
         patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch("langchain_app.memory_service.recall_memories", return_value=[
             {"text": "El usuario desarrolla una aplicación de medicina", "type": "world", "occurred": None, "entities": []},
         ]) as mock_recall, \
         patch("langchain_app.main.retain_conversation_turn") as mock_retain:
        graph = build_chat_graph("ollama")
        result = graph.invoke(
            {"messages": [HumanMessage(content="¿En qué aplicación estoy trabajando?")]},
            config={"configurable": {"thread_id": "mem-personal", "memory_bank_id": "bank-mario"}},
        )

    assert _tool_names(result) == ["search_long_term_memory"]
    mock_web.assert_not_called()
    assert mock_recall.call_args.kwargs["bank_id"] == "bank-mario"
    prompt = llm.invoke.call_args.args[0][0].content
    assert "aplicación de medicina" in prompt
    assert "fuente de verdad" in prompt.lower()
    assert "primera frase" in prompt.lower()  # answer directly, don't echo the question back
    # Memory + rules go in the system message and the question is a real user turn:
    # a single packed HumanMessage made llama3.2 echo the question (2/9 vs 9/9 correct).
    sent = llm.invoke.call_args.args[0]
    assert isinstance(sent[0], SystemMessage)
    assert isinstance(sent[-1], HumanMessage) and sent[-1].content == "¿En qué aplicación estoy trabajando?"
    assert result["messages"][-1].content == "Estás desarrollando una aplicación de medicina."
    mock_retain.assert_not_called()  # questions are not stored


def test_personal_statement_is_retained_as_instant_note():
    with patch("langchain_app.main.get_llm", return_value=_fake_llm("¡Encantado, Mario!")), \
         patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch("langchain_app.memory_service.recall_memories", return_value=[]), \
         patch("langchain_app.main.retain_conversation_turn", return_value=True) as mock_retain:
        graph = build_chat_graph("ollama")
        graph.invoke(
            {"messages": [HumanMessage(content="Me llamo Mario y estoy construyendo un agente con LangGraph")]},
            config={"configurable": {"thread_id": "mem-statement", "memory_bank_id": "bank-mario"}},
        )

    mock_retain.assert_called_once()
    args, kwargs = mock_retain.call_args
    assert args[0] == "Me llamo Mario y estoy construyendo un agente con LangGraph"
    assert kwargs["bank_id"] == "bank-mario"
    assert kwargs["save_note"] is True
    assert kwargs["include_assistant"] is False
    assert kwargs["intent"] == "user_memory"


def test_generic_web_questions_are_not_retained():
    with patch("langchain_app.main.get_llm", return_value=_fake_llm("LangGraph es un framework.")), \
         patch("langchain_app.main.fetch_tavily_results", return_value=[{"title": "T", "url": "u", "content": "c"}]), \
         patch("langchain_app.main.is_hindsight_available", return_value=True), \
         patch("langchain_app.main.recall_memories", return_value=MEMORIES), \
         patch("langchain_app.main.retain_conversation_turn") as mock_retain:
        graph = build_chat_graph("ollama")
        graph.invoke(
            {"messages": [HumanMessage(content="¿Qué es LangGraph?")]},
            config={"configurable": {"thread_id": "mem-generic", "memory_bank_id": "bank-mario"}},
        )
    mock_retain.assert_not_called()


def test_web_answers_treat_memory_as_source_of_truth_for_user_facts():
    llm = _fake_llm("LangGraph es un framework de agentes.")
    with patch("langchain_app.main.get_llm", return_value=llm), \
         patch("langchain_app.main.fetch_tavily_results", return_value=[{"title": "T", "url": "u", "content": "c"}]), \
         patch("langchain_app.main.is_hindsight_available", return_value=True), \
         patch("langchain_app.main.recall_memories", return_value=MEMORIES):
        graph = build_chat_graph("ollama")
        graph.invoke(
            {"messages": [HumanMessage(content="¿Qué es LangGraph?")]},
            config={"configurable": {"thread_id": "mem-web", "memory_bank_id": "bank-mario"}},
        )

    prompt = llm.invoke.call_args.args[0][0].content
    assert prompt.index("MEMORIA A LARGO PLAZO") < prompt.index("EVIDENCIA OBTENIDA EN TIEMPO REAL")
    assert "fuente de verdad" in prompt.lower()


def test_personal_search_rewrites_web_query_with_memory():
    llm = _fake_llm("clima en Guayaquil hoy", "Hoy en Guayaquil hace 30 °C.")
    with patch("langchain_app.main.get_llm", return_value=llm), \
         patch("langchain_app.main.fetch_tavily_results", return_value=[{"title": "T", "url": "u", "content": "30 °C"}]) as mock_web, \
         patch("langchain_app.main.is_hindsight_available", return_value=True), \
         patch("langchain_app.main.recall_memories", return_value=MEMORIES), \
         patch("langchain_app.main.retain_conversation_turn"):
        graph = build_chat_graph("ollama")
        result = graph.invoke(
            {"messages": [HumanMessage(content="¿Cuál es el clima en mi ciudad hoy?")]},
            config={"configurable": {"thread_id": "mem-hybrid", "memory_bank_id": "bank-mario"}},
        )

    assert mock_web.call_args.kwargs["query"] == "clima en Guayaquil hoy"
    assert result["messages"][-1].content == "Hoy en Guayaquil hace 30 °C."


def test_personal_search_keeps_original_query_without_memories():
    llm = _fake_llm("No sé tu ciudad, pero puedo ayudarte.")
    with patch("langchain_app.main.get_llm", return_value=llm), \
         patch("langchain_app.main.fetch_tavily_results", return_value=[]) as mock_web, \
         patch("langchain_app.main.is_hindsight_available", return_value=True), \
         patch("langchain_app.main.recall_memories", return_value=[]), \
         patch("langchain_app.main.retain_conversation_turn"):
        graph = build_chat_graph("ollama")
        graph.invoke(
            {"messages": [HumanMessage(content="¿Cuál es el clima en mi ciudad hoy?")]},
            config={"configurable": {"thread_id": "mem-hybrid-empty", "memory_bank_id": "bank-mario"}},
        )
    assert mock_web.call_args.kwargs["query"] == "¿Cuál es el clima en mi ciudad hoy?"


def test_graph_injects_recalled_memories_into_conversation():
    llm = _fake_llm("¡Buenos días, Mario!")
    with patch("langchain_app.main.get_llm", return_value=llm), \
         patch("langchain_app.main.is_hindsight_available", return_value=True), \
         patch("langchain_app.main.recall_memories", return_value=MEMORIES) as mock_recall, \
         patch("langchain_app.main.retain_conversation_turn") as mock_retain:
        graph = build_chat_graph("ollama")
        result = graph.invoke(
            {"messages": [SystemMessage(content="SYS"), HumanMessage(content="Hola, buenos días")]},
            config={"configurable": {"thread_id": "mem-conv", "memory_bank_id": "bank-mario"}},
        )

    mock_recall.assert_called_once_with("Hola, buenos días", bank_id="bank-mario")
    sent_messages = llm.invoke.call_args.args[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert "El usuario se llama Mario" in sent_messages[0].content
    assert sum(isinstance(m, SystemMessage) for m in sent_messages) == 1
    assert "por su nombre" in sent_messages[0].content
    assert result["recalled_memories"] == MEMORIES
    mock_retain.assert_not_called()


def test_graph_retains_document_turns_with_bank_and_thread():
    with patch("langchain_app.main.get_llm", return_value=_fake_llm("El PDF trata de X.")), \
         patch("langchain_app.main.is_hindsight_available", return_value=False), \
         patch("langchain_app.main.retain_conversation_turn", return_value=True) as mock_retain:
        graph = build_chat_graph("ollama")
        graph.invoke(
            {"messages": [HumanMessage(content="Analiza el archivo informe.pdf")]},
            config={"configurable": {"thread_id": "mem-pdf", "memory_bank_id": "bank-mario"}},
        )

    mock_retain.assert_called_once()
    kwargs = mock_retain.call_args.kwargs
    assert kwargs["bank_id"] == "bank-mario"
    assert kwargs["thread_id"] == "mem-pdf"
    assert kwargs["intent"] == "pdf"
    assert kwargs["include_assistant"] is True
    assert kwargs["save_note"] is False


def test_graph_respects_auto_retain_opt_out():
    with patch("langchain_app.main.get_llm", return_value=_fake_llm()), \
         patch("langchain_app.memory_service.is_hindsight_available", return_value=False), \
         patch("langchain_app.main.retain_conversation_turn") as mock_retain:
        graph = build_chat_graph("ollama")
        graph.invoke(
            {"messages": [HumanMessage(content="Me llamo Mario")]},
            config={"configurable": {"thread_id": "mem-optout", "memory_auto_retain": False}},
        )
    mock_retain.assert_not_called()


def test_graph_memory_reflect_synthesizes_from_user_memories():
    llm = _fake_llm("Eres Mario y vives en Guayaquil.")
    with patch("langchain_app.main.get_llm", return_value=llm), \
         patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch("langchain_app.memory_service.recall_memories", return_value=MEMORIES), \
         patch("langchain_app.main.retain_conversation_turn") as mock_retain:
        graph = build_chat_graph("ollama")
        result = graph.invoke(
            {"messages": [HumanMessage(content="sabes algo sobre mi")]},
            config={"configurable": {"thread_id": "mem-reflect", "memory_bank_id": "bank-mario"}},
        )

    assert _tool_names(result) == ["reflect_on_user_memory"]
    prompt = llm.invoke.call_args.args[0][0].content
    assert "El usuario se llama Mario y vive en Guayaquil" in prompt
    # Vague questions ("sabes algo sobre mi") must get an explicit summary, not a generic greeting
    assert "enumera" in prompt.lower() and "todo lo que sabes" in prompt.lower()
    assert result["messages"][-1].content == "Eres Mario y vives en Guayaquil."
    mock_retain.assert_not_called()


def test_graph_memory_store_calls_remember_tool():
    with patch("langchain_app.main.get_llm", return_value=_fake_llm("Lo recordaré.")), \
         patch("langchain_app.memory_service.remember", return_value=(True, "Guardado")) as mock_remember:
        graph = build_chat_graph("ollama")
        result = graph.invoke(
            {"messages": [HumanMessage(content="Recuerda que soy vegetariano")]},
            config={"configurable": {"thread_id": "mem-store", "memory_bank_id": "bank-mario"}},
        )

    mock_remember.assert_called_once_with("Recuerda que soy vegetariano", bank_id="bank-mario")
    assert result["messages"][-1].content == "Lo recordaré."


def test_ollama_context_matches_hindsight_to_avoid_model_reloads(monkeypatch):
    """Ollama reloads llama3.2 whenever two clients request different num_ctx values.

    Hindsight runs with OLLAMA_NUM_CTX (default 8192); the agent must use the same value or
    every alternation between chat and background memory extraction triggers a reload.
    """
    from langchain_app.main import get_llm

    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    assert get_llm("ollama").num_ctx == 8192
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    assert get_llm("ollama").num_ctx == 4096


def test_ollama_responses_are_capped_to_stop_runaway_generations(monkeypatch):
    """Without num_predict a looping llama3.2 generates forever (observed: 37k+ tokens, 13 min)."""
    from langchain_app.main import get_llm

    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
    assert get_llm("ollama").num_predict == 2048
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "512")
    assert get_llm("ollama").num_predict == 512
