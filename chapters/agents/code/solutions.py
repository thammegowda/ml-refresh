"""Worked solutions to the Chapter 42 exercises."""
from tool_loop import MCPClient, MCPServer, ScriptedModel, call_tool, react_loop


# tag::schema[]
def required_argument_names(schema):
    return tuple(schema.get("required", ()))
# end::schema[]


# tag::trace[]
def answer_population_question():
    return react_loop("What is the Paris population plus two?", ScriptedModel())
# end::trace[]


# tag::mcp-call[]
def mcp_calculate(expression):
    client = MCPClient(MCPServer())
    return client.request("tools/call", {
        "name": "calculate",
        "arguments": {"expression": expression},
    })["result"]
# end::mcp-call[]


# tag::injection[]
def observation_as_data(key):
    """Return lookup text without treating it as a developer instruction."""
    result = call_tool("lookup", {"key": key})
    if not result["ok"]:
        return "missing"
    return result["content"]
# end::injection[]
