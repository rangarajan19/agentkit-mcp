import argparse
import json
import os
import sys

from . import agent, config


def issue_number_from_event() -> int | None:
    path = os.getenv("GITHUB_EVENT_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("issue", {}).get("number")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="AI issue triage agent (MCP tools)")
    p.add_argument("--issue", type=int, help="issue number (defaults to the Actions event)")
    p.add_argument("--repo", default=config.GITHUB_REPOSITORY, help="owner/repo")
    p.add_argument("--approve", action="store_true",
                   help="ask for confirmation before every write (labels, comments)")
    args = p.parse_args()

    number = args.issue or issue_number_from_event()
    if not number or "/" not in args.repo:
        print("Need --issue and --repo owner/repo (or GITHUB_REPOSITORY).", file=sys.stderr)
        return 1
    if not config.GITHUB_TOKEN or not config.GEMINI_API_KEY:
        print("Set GITHUB_TOKEN and GEMINI_API_KEY (see .env.example).", file=sys.stderr)
        return 1

    print(agent.triage_issue(args.repo, number, mode="approve" if args.approve else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
