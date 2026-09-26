import os

from dotenv import load_dotenv

load_dotenv()

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() != "false"

# The only tools the agent may see from the (45-tool) GitHub MCP server.
GITHUB_TOOLS_ALLOWED = ["issue_read", "add_issue_comment", "get_file_contents"]
MAX_COMMENT_CHARS = 2000
