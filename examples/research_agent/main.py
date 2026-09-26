import argparse
import sys

from . import agent


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # models emit non-ASCII; Windows consoles default to cp1252
        stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Research agent (MCP web tools)")
    p.add_argument("topic")
    p.add_argument("--out", help="save the report to this markdown file")
    args = p.parse_args()

    result = agent.research(args.topic)
    for s in result.steps:
        if s.kind != "model":
            print(f"[{s.kind}] {s.name}({s.args}) -> {s.output[:100]!r}", file=sys.stderr)
    if result.stopped_by_limit:
        print("warning: stopped at the step limit; the report may be incomplete", file=sys.stderr)
    print(result.text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
