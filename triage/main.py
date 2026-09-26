import argparse
import json
import os
import sys

from . import agent


def issue_number_from_event() -> int | None:
    path = os.getenv("GITHUB_EVENT_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("issue", {}).get("number")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="AI issue triage agent")
    p.add_argument("--issue", type=int, help="issue number (defaults to the Actions event)")
    args = p.parse_args()
    number = args.issue or issue_number_from_event()
    if not number:
        print("No issue number given.", file=sys.stderr)
        return 1
    print(agent.triage_issue(number))
    return 0


if __name__ == "__main__":
    sys.exit(main())
