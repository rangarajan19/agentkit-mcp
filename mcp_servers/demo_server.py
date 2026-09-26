"""A tiny MCP server used for tests and demos. Run standalone: python -m mcp_servers.demo_server"""
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("demo")
READ = ToolAnnotations(read_only_hint=True)
_notes: list[str] = []


@server.tool(annotations=READ)
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool(annotations=READ)
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


@server.tool(annotations=ToolAnnotations(read_only_hint=False))
def save_note(text: str) -> str:
    """Save a note (a write action)."""
    _notes.append(text)
    return f"saved note #{len(_notes)}"


@server.tool(annotations=READ)
def list_notes() -> list[str]:
    """List saved notes."""
    return list(_notes)


@server.tool()
def delete_everything() -> str:
    """A deliberately dangerous tool, used to test tool filtering."""
    return "deleted (not really)"


if __name__ == "__main__":
    server.run("stdio")
