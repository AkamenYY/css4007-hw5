"""Run the 14 supplied scenarios (data/scenarios.json) and save one log per scenario in outputs/.

    python run_scenarios.py               # all of them (Parts 2-3 need the API key in .env)
    python run_scenarios.py --only S11    # just one

How it follows the scenario file:
- reset_before: true    -> delete local-storage/ (all memory) before the scenario.
- Part 1 scenarios      -> the Part 1 graph (no model). S04 uses "original", S05 "with_clarification".
- Parts 2-3 scenarios   -> messages are sent by a NEW `python -m helpdesk chat` process.
- restart_process       -> that process ends and a new one starts, using the same local-storage/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from helpdesk import data, memory, part1  # noqa: E402


def run_part1(sc):
    variant = sc.get("graph_variant", "original")
    return {"graph_variant": variant,
            "runs": [dict(part1.run(ev["message"], variant), user_id=ev["user_id"], thread_id=ev["thread_id"])
                     for ev in sc["events"]]}


def split_into_processes(events):
    """Messages for the same user and thread go to one process. A restart (or another user) starts a new one."""
    groups, restart = [], False
    for ev in events:
        if ev["action"] == "restart_process":
            groups.append(None)      # marks the end of a process
            restart = True
            continue
        key = (ev["user_id"], ev["thread_id"])
        if not groups or groups[-1] is None or groups[-1]["key"] != key:
            groups.append({"key": key, "messages": [], "after_restart": restart})
            restart = False
        groups[-1]["messages"].append(ev["message"])
    return [g for g in groups if g]


def run_agent(sc, ticket_dir):
    env = dict(os.environ, HELPDESK_TICKET_DIR=ticket_dir, PYTHONIOENCODING="utf-8")
    runs = []
    for group in split_into_processes(sc["events"]):
        user, thread = group["key"]
        log_file = Path(tempfile.mkdtemp()) / "run.jsonl"
        args = ["chat", "--user", user, "--thread", thread, "--status", sc["status_profile"]]
        for m in group["messages"]:
            args += ["-m", m]
        proc = subprocess.run([sys.executable, "-m", "helpdesk", *args, "--log", str(log_file)],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        print(proc.stdout)
        if proc.returncode:
            print(proc.stderr, file=sys.stderr)
        record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1]) if log_file.exists() else {}
        runs.append({"command": "python -m helpdesk " + " ".join(f'"{a}"' if " " in a else a for a in args),
                     "new_process_after_restart": group["after_restart"], "exit_code": proc.returncode,
                     "process": record.get("process"), "turns": record.get("turns", []),
                     "errors": proc.stderr[-1500:] if proc.returncode else ""})
    return {"runs": runs}


def summary_rows(sc, log):
    if sc["part"] == 1:
        return [("START -> " + " -> ".join(r["route"]) + " -> END", "-", r["final_state"]["reply"])
                for r in log["runs"]]
    rows = []
    for run in log["runs"]:
        for t in run["turns"]:
            note = " (new process after restart)" if run["new_process_after_restart"] else ""
            rows.append((" -> ".join(t["nodes"]) + note,
                         ", ".join(q["name"] for q in t["tool_requests"]) or "-", t["reply"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated ids, e.g. S06,S07")
    wanted = [s for s in ap.parse_args().only.split(",") if s]
    out = ROOT / "outputs"
    table = ["| Scenario | Snapshot | Nodes | Tools the model asked for | Reply (start) |",
             "|---|---|---|---|---|"]

    for sc in data.scenarios():
        if wanted and sc["id"] not in wanted:
            continue
        print(f"===== {sc['id']} {sc['title']} (snapshot: {sc['status_profile']})")
        if sc.get("reset_before"):
            memory.reset_storage()
        ticket_dir = f"outputs/tickets/{sc['id']}"
        shutil.rmtree(ROOT / ticket_dir, ignore_errors=True)
        body = run_part1(sc) if sc["part"] == 1 else run_agent(sc, ticket_dir)
        log = {"scenario": sc["id"], "title": sc["title"], "part": sc["part"],
               "status_profile": sc["status_profile"], "reset_before": sc.get("reset_before"),
               "model": "none (Part 1 has no model)" if sc["part"] == 1 else f"openai:{os.getenv('MODEL_NAME')}",
               "memory_storage": "local-storage/ (not in git)", **body}
        (out / f"{sc['id']}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for nodes, used, reply in summary_rows(sc, log):
            short = " ".join(str(reply).split())[:150].replace("|", "/")
            table.append(f"| {sc['id']} | {sc['status_profile']} | {nodes} | {used} | {short} |")
        print(f"saved outputs/{sc['id']}.json\n")

    if not wanted:
        (out / "scenarios_summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
        print("saved outputs/scenarios_summary.md")


if __name__ == "__main__":
    main()
