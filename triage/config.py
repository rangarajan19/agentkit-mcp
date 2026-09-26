import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() != "false"

# Labels the agent may apply. Anything else is rejected (limits prompt-injection damage).
ALLOWED_LABELS = {
    "bug", "enhancement", "question", "documentation", "duplicate",
    "needs-info", "priority:high", "priority:medium", "priority:low",
}
SIMILARITY_CUTOFF = 0.6
