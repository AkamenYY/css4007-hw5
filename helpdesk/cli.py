"""Terminal program.

python -m helpdesk part1 "My Wi-Fi password does not work." [--variant with_clarification]
python -m helpdesk tools                                   # call the three tools directly
python -m helpdesk chat --user demo-user-a --thread t1 --status normal          # interactive
python -m helpdesk chat --user demo-user-a --thread t1 --status normal -m "..." -m "..."   # scripted
python -m helpdesk show --thread t1                        # print a saved conversation
python -m helpdesk graphs                                  # write outputs/graphs.md
python -m helpdesk reset                                   # delete local-storage/ (all memory)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import data, memory, part1, tools


def _print_part1(result):
    print(f"[{result['variant']}] route: START -> " + " -> ".join(result["route"]) + " -> END")
    for step in result["trace"]:
        print(f"  {step['node']:<16} update={json.dumps(step['update'], ensure_ascii=False)[:120]}")
    print("reply:\n" + result["final_state"]["reply"])


def cmd_part1(args):
    result = part1.run(args.message, args.variant)
    _print_part1(result)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_tools(args):
    demo_dir = args.ticket_dir
    calls = [
        ("find_instructions", lambda: tools.find_instructions("wifi")),
        ("find_instructions", lambda: tools.find_instructions("printer")),
        ("check_service_status", lambda: tools.check_service_status("wifi", status_profile="wifi_outage")),
        ("check_service_status", lambda: tools.check_service_status("account_recovery", status_profile="status_unavailable")),
        ("prepare_ticket", lambda: tools.prepare_ticket(
            category="wifi", summary="Campus-Demo will not connect.",
            steps_tried=["Enabled Wi-Fi", "Forgot the saved network and reconnected"],
            user_id="demo-user-a", thread_id="direct-call", status_profile="normal",
            service_status="operational", reference_ids=["GUIDE-WIFI"], ticket_dir=demo_dir)),
        ("prepare_ticket", lambda: tools.prepare_ticket(
            category="wifi", summary="", steps_tried=[], user_id="demo-user-a",
            thread_id="direct-call", status_profile="normal", ticket_dir=demo_dir)),
    ]
    out = []
    for name, fn in calls:
        result = fn()
        out.append({"tool": name, "result": result})
        print(f"{name}: {json.dumps(result, ensure_ascii=False)}")
    if args.log:
        Path(args.log).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _model(fake: bool):
    from .models import RuleBasedFakeModel, real_model
    return RuleBasedFakeModel() if fake else real_model()


def _print_turn(turn, verbose):
    if verbose:
        print("  nodes: START -> " + " -> ".join(turn["nodes"]) + " -> END")
        for req, res in zip(turn["tool_requests"], turn["tool_results"]):
            print(f"  tool  {req['name']}({json.dumps(req['args'], ensure_ascii=False)})")
            print(f"     -> {json.dumps(res['result'], ensure_ascii=False)[:200]}")
        for s in turn["sources"]:
            print(f"  read  {s}")
        if turn["loop_limit_reached"]:
            print("  LOOP LIMIT REACHED")
    print(f"Helpdesk: {turn['reply']}\n")


def cmd_chat(args):
    from .agent import HelpdeskSession

    session = HelpdeskSession(_model(args.fake_model), args.user, args.thread, args.status)
    turns = []
    process = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "python": sys.version.split()[0]}
    label = "FAKE rule-based model" if args.fake_model else os.getenv("MODEL_NAME", "")
    print(f"user={args.user} thread={args.thread} status={args.status} model={label} pid={process['pid']}")
    earlier = len(session.history())
    if earlier:
        print(f"(resumed thread '{args.thread}' with {earlier} saved messages)")
    try:
        messages = args.message or []
        if messages:
            for text in messages:
                print(f"You: {text}")
                turn = session.send(text)
                turns.append(turn)
                _print_turn(turn, args.verbose)
        else:
            print("Type a message. Commands: /quit, /history")
            while True:
                try:
                    text = input("You: ").strip()
                except EOFError:
                    break
                if text in ("/quit", "/exit"):
                    break
                if text == "/history":
                    for m in session.history():
                        print(f"  {m.type}: {str(m.content)[:100]}")
                    continue
                if text:
                    turn = session.send(text)
                    turns.append(turn)
                    _print_turn(turn, args.verbose)
    finally:
        session.close()
    if args.log:
        path = Path(args.log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:  # one JSON line per process run
            fh.write(json.dumps({"process": process, "turns": turns}, ensure_ascii=False) + "\n")


def cmd_show(args):
    saver = memory.open_checkpointer()
    try:
        tup = saver.get_tuple({"configurable": {"thread_id": args.thread}})
        msgs = tup.checkpoint["channel_values"].get("messages", []) if tup else []
        for m in msgs:
            print(f"{m.type}: {m.content}")
        if not msgs:
            print(f"no saved conversation for thread '{args.thread}'")
    finally:
        memory.close(saver)


def cmd_graphs(args):
    from .agent import build_graph
    from .models import ScriptedFakeModel

    parts = ["# Graphs drawn by LangGraph (`python -m helpdesk graphs`)\n"]
    for variant in part1.VARIANTS:
        parts += [f"## Part 1 - {variant}\n", "```mermaid", part1.mermaid(variant).strip(), "```\n"]
    agent_graph = build_graph(ScriptedFakeModel()).get_graph().draw_mermaid()
    parts += ["## Parts 2-3 - model and tools loop\n", "```mermaid", agent_graph.strip(), "```\n"]
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out}")


def cmd_reset(args):
    path = memory.reset_storage()
    print(f"deleted {path} (conversation checkpoints and user preferences)")


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(data.ROOT / ".env")  # storage dir, ticket dir and model settings
    p = argparse.ArgumentParser(prog="python -m helpdesk", description="HW5 practice IT helpdesk")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("part1", help="Part 1 graph, no model")
    a.add_argument("message")
    a.add_argument("--variant", choices=part1.VARIANTS, default="original")
    a.add_argument("--log")
    a.set_defaults(func=cmd_part1)

    t = sub.add_parser("tools", help="call the three tools directly, without the agent")
    t.add_argument("--ticket-dir", default="outputs/tickets")
    t.add_argument("--log")
    t.set_defaults(func=cmd_tools)

    c = sub.add_parser("chat", help="model + tools + memory")
    c.add_argument("--user", required=True, help="invented user id, e.g. demo-user-a")
    c.add_argument("--thread", required=True, help="conversation id")
    c.add_argument("--status", default="normal", choices=data.status_profiles())
    c.add_argument("-m", "--message", action="append", help="send this message (repeatable); omit for interactive")
    c.add_argument("--fake-model", action="store_true", help="FAKE rule-based model, no API key (dry run only)")
    c.add_argument("--log", help="append this process's trace as one JSON line")
    c.add_argument("-q", "--quiet", dest="verbose", action="store_false")
    c.set_defaults(func=cmd_chat)

    s = sub.add_parser("show", help="print a saved conversation")
    s.add_argument("--thread", required=True)
    s.set_defaults(func=cmd_show)

    gr = sub.add_parser("graphs", help="write Mermaid drawings of the graphs")
    gr.add_argument("--out", default="outputs/graphs.md")
    gr.set_defaults(func=cmd_graphs)

    r = sub.add_parser("reset", help="delete all saved memory")
    r.set_defaults(func=cmd_reset)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
