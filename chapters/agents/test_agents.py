from solutions import (answer_population_question, mcp_calculate,
                       observation_as_data, required_argument_names)
from tool_loop import MCPClient, MCPServer, ScriptedModel, TOOLS, call_tool, react_loop


def test_schema_validation_accepts_required_arguments_and_rejects_bad_ones():
    schema = TOOLS["calculate"]["inputSchema"]
    assert required_argument_names(schema) == ("expression",)
    assert call_tool("calculate", {"expression": "2 + 3 * 4"}) == {
        "ok": True,
        "content": "14.0",
    }
    assert call_tool("calculate", {})["ok"] is False
    assert call_tool("calculate", {"expression": 3})["ok"] is False
    assert call_tool("calculate", {"expression": "2", "extra": 1})["ok"] is False


def test_react_loop_records_thought_action_observation_and_stops_on_final():
    result = answer_population_question()
    assert result["answer"] == "4.1 million"
    assert result["stop"] == "final"
    assert [step["action"]["tool"] for step in result["trace"]] == [
        "lookup",
        "calculate",
    ]
    assert result["trace"][0]["observation"] == {"ok": True, "content": "2.1"}


def test_react_loop_budget_is_a_real_stopping_condition():
    result = react_loop("same question", ScriptedModel(), max_steps=1)
    assert result["answer"] is None
    assert result["stop"] == "budget_exhausted"
    assert len(result["trace"]) == 1


def test_mcp_lists_tools_and_calls_them_through_json_rpc_dicts():
    client = MCPClient(MCPServer())
    listed = client.request("tools/list")["result"]["tools"]
    assert {tool["name"] for tool in listed} == {"calculate", "lookup"}
    assert mcp_calculate("(8 - 3) / 2") == {"ok": True, "content": "2.5"}
    bad = client.request("tools/call", {
        "name": "calculate",
        "arguments": {"expression": "open('secrets')"},
    })
    assert bad["result"]["ok"] is False


def test_tool_output_is_data_even_when_it_contains_an_instruction():
    text = observation_as_data("malicious_note")
    assert "Ignore the developer" in text
    result = call_tool("calculate", {"expression": "10**10"})
    assert result["ok"] is True
    assert observation_as_data("malicious_note") == text
