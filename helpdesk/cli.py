"""The terminal program.

python -m helpdesk part1 "My Wi-Fi password does not work."                         # Part 1, no model
python -m helpdesk part1 "My Wi-Fi password does not work." --variant with_clarification
python -m helpdesk tools                                                              # call each tool directly
python -m helpdesk chat --user demo-user-a --thread t1 --status normal               # chat (type /quit)
python -m helpdesk chat --user demo-user-a --thread t1 --status normal -m "..." -m "..."
python -m helpdesk graphs                                                             # draw the graphs
python -m helpdesk reset                                                              # delete all memory
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import data, memory, part1, tools


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_part1(args):
    result = part1.run(args.message, args.variant)
    print("route: START -> " + " -> ".join(result["route"]) + " -> END")
    for step in result["trace"]:
        print(f"  {step['node']}: {json.dumps(step['update'], ensure_ascii=False)[:120]}")
    print("reply: " + result["final_state"]["reply"])
    if args.log:
        save_json(args.log, result)


def cmd_tools(args):
    """Call the three tools directly, with no model and no graph."""
    folder = args.ticket_dir
    calls = [
        ("find_instructions('wifi')", lambda: tools.find_instructions("wifi")),
        ("find_instructions('printer')", lambda: tools.find_instructions("printer")),
        ("check_service_status('wifi', 'wifi_outage')",
         lambda: tools.check_service_status("wifi", status_profile="wifi_outage")),
        ("check_service_status('wifi', 'status_unavailable')",
         lambda: tools.check_service_status("wifi", status_profile="status_unavailable")),
        ("prepare_ticket(full details)", lambda: tools.prepare_ticket(
            category="wifi", summary="Campus-Demo will not connect.",
            steps_tried=["Enabled Wi-Fi", "Forgot the saved network and reconnected"],
            user_id="demo-user-a", thread_id="direct-call", status_profile="normal",
            service_status="operational", reference_ids=["GUIDE-WIFI"], ticket_dir=folder)),
        ("prepare_ticket(no details)", lambda: tools.prepare_ticket(
            category="wifi", summary="", steps_tried=[], user_id="demo-user-a",
            thread_id="direct-call", status_profile="normal", ticket_dir=folder)),
    ]
    out = []
    for label, fn in calls:
        result = fn()
        out.append({"call": label, "result": result})
        print(f"{label}\n  -> {json.dumps(result, ensure_ascii=False)}")
    if args.log:
        save_json(args.log, out)


def print_turn(turn, short):
    if not short:
        print("  nodes: START -> " + " -> ".join(turn["nodes"]) + " -> END")
        for req, res in zip(turn["tool_requests"], turn["tool_results"]):
            print(f"  model asked: {req['name']}({json.dumps(req['args'], ensure_ascii=False)})")
            print(f"  tool result: {json.dumps(res['result'], ensure_ascii=False)[:200]}")
        for line in turn["read_from"]:
            print(f"  read: {line}")
        if turn["loop_limit_reached"]:
            print("  LOOP LIMIT REACHED")
    print(f"Helpdesk: {turn['reply']}\n")


def cmd_chat(args):
    from .agent import HelpdeskSession
    from .models import EchoFakeModel, real_model

    model = EchoFakeModel() if args.fake_model else real_model()
    session = HelpdeskSession(model, args.user, args.thread, args.status,
                              ticket_dir=os.getenv("HELPDESK_TICKET_DIR", "local-storage/tickets"))
    process = {"pid": os.getpid(), "python": sys.version.split()[0],
               "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    print(f"user={args.user} thread={args.thread} status={args.status} "
          f"model={'FAKE echo model' if args.fake_model else os.getenv('MODEL_NAME')} process={process['pid']}")
    saved = len(session.history())
    if saved:
        print(f"(this thread already has {saved} saved messages)")
    turns = []
    try:
        if args.message:
            for text in args.message:
                print(f"You: {text}")
                turns.append(session.send(text))
                print_turn(turns[-1], args.short)
        else:
            print("Type a message. Type /quit to exit.")
            while True:
                try:
                    text = input("You: ").strip()
                except EOFError:
                    break
                if text == "/quit":
                    break
                if text:
                    turns.append(session.send(text))
                    print_turn(turns[-1], args.short)
    finally:
        session.close()
    if args.log:  # one JSON line per process, so a restart shows as a second line
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"process": process, "turns": turns}, ensure_ascii=False) + "\n")


def cmd_graphs(args):
    from .agent import build_graph
    from .models import EchoFakeModel

    text = ["# Graphs drawn by LangGraph (`python -m helpdesk graphs`)\n"]
    for variant in part1.VARIANTS:
        text += [f"## Part 1 - {variant}\n", "```mermaid", part1.mermaid(variant).strip(), "```\n"]
    text += ["## Parts 2-3 - model and tools loop\n", "```mermaid",
             build_graph(EchoFakeModel()).get_graph().draw_mermaid().strip(), "```\n"]
    Path(args.out).write_text("\n".join(text), encoding="utf-8")
    print(f"wrote {args.out}")


def cmd_reset(args):
    print(f"deleted {memory.reset_storage()} (all conversations and user preferences)")


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(data.ROOT / ".env")

    p = argparse.ArgumentParser(prog="python -m helpdesk")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("part1", help="Part 1 graph, no model")
    a.add_argument("message")
    a.add_argument("--variant", choices=part1.VARIANTS, default="original")
    a.add_argument("--log")
    a.set_defaults(func=cmd_part1)

    t = sub.add_parser("tools", help="call the three tools directly")
    t.add_argument("--ticket-dir", default="outputs/tickets/direct")
    t.add_argument("--log")
    t.set_defaults(func=cmd_tools)

    c = sub.add_parser("chat", help="model + tools + memory")
    c.add_argument("--user", required=True, help="invented user id, e.g. demo-user-a")
    c.add_argument("--thread", required=True, help="conversation id")
    c.add_argument("--status", default="normal", choices=data.status_profiles())
    c.add_argument("-m", "--message", action="append", help="a user message (repeat for more)")
    c.add_argument("--log", help="add this run's trace to a JSON-lines file")
    c.add_argument("--short", action="store_true", help="print only the replies")
    c.add_argument("--fake-model", action="store_true", help="FAKE echo model, for tests only")
    c.set_defaults(func=cmd_chat)

    g = sub.add_parser("graphs", help="write Mermaid drawings of the graphs")
    g.add_argument("--out", default="outputs/graphs.md")
    g.set_defaults(func=cmd_graphs)

    r = sub.add_parser("reset", help="delete all saved memory")
    r.set_defaults(func=cmd_reset)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
