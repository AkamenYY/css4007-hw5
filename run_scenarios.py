"""Run the supplied scenarios S01-S14 from data/scenarios.json and save one log per scenario.

    python run_scenarios.py                 # all scenarios, real model from .env  -> outputs/S01.json ...
    python run_scenarios.py --only S06,S07  # some scenarios
    python run_scenarios.py --fake-model    # FAKE rule-based model, no API key   -> outputs/fake/...

Part 1 scenarios run the Part 1 graph (no model). For Parts 2-3, every group of messages is sent by a
separate `python -m helpdesk chat ...` process. A restart_process event therefore really ends one
process and starts a new one against the same local-storage/ files. reset_before deletes local-storage/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from helpdesk import data, memory, part1  # noqa: E402


def run_part1_scenario(sc):
    variant = sc.get("graph_variant", "original")
    results = []
    for ev in sc["events"]:
        r = part1.run(ev["message"], variant)
        r.update(user_id=ev["user_id"], thread_id=ev["thread_id"])
        results.append(r)
    return {"graph_variant": variant, "runs": results}


def group_events(events):
    """Consecutive messages for the same user+thread go to one process; restart_process splits."""
    groups, current, restart_next = [], None, False
    for ev in events:
        if ev["action"] == "restart_process":
            current, restart_next = None, True
            continue
        key = (ev["user_id"], ev["thread_id"])
        if current is None or current["key"] != key:
            current = {"key": key, "messages": [], "after_restart": restart_next}
            groups.append(current)
            restart_next = False
        current["messages"].append(ev["message"])
    return groups


def run_agent_scenario(sc, fake, ticket_dir, env):
    runs = []
    for group in group_events(sc["events"]):
        user, thread = group["key"]
        with tempfile.NamedTemporaryFile("r", suffix=".jsonl", delete=False) as tmp:
            log_path = tmp.name
        cmd = [sys.executable, "-m", "helpdesk", "chat", "--user", user, "--thread", thread,
               "--status", sc["status_profile"], "--log", log_path]
        if fake:
            cmd.append("--fake-model")
        for m in group["messages"]:
            cmd += ["-m", m]
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        lines = Path(log_path).read_text(encoding="utf-8").splitlines() if Path(log_path).exists() else []
        os.unlink(log_path)
        record = json.loads(lines[-1]) if lines else {"process": None, "turns": []}
        shown = ["python", "-m", "helpdesk", *cmd[3:]]
        record.update(command=" ".join(_quote(c) for c in shown if c not in (log_path, "--log")),
                      after_restart=group["after_restart"], exit_code=proc.returncode,
                      stdout=proc.stdout, stderr=proc.stderr[-2000:])
        runs.append(record)
        print(proc.stdout)
        if proc.returncode:
            print(proc.stderr, file=sys.stderr)
    return {"runs": runs, "ticket_dir": ticket_dir}


def _quote(s):
    return f'"{s}"' if " " in s else s


def summarise(sc, log):
    rows = []
    if sc["part"] == 1:
        for r in log["runs"]:
            rows.append(("START -> " + " -> ".join(r["route"]) + " -> END", "-", r["final_state"]["reply"]))
    else:
        for run in log["runs"]:
            for t in run["turns"]:
                tools_used = ", ".join(f"{q['name']}" for q in t["tool_requests"]) or "-"
                tag = " (after restart)" if run["after_restart"] else ""
                rows.append((" -> ".join(t["nodes"]) + tag, tools_used, t["reply"]))
    return rows


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")  # same storage dir and model settings as the child processes
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--fake-model", action="store_true", help="FAKE model; logs go to outputs/fake/")
    args = ap.parse_args()

    out = ROOT / "outputs" / ("fake" if args.fake_model else "")
    out.mkdir(parents=True, exist_ok=True)
    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    summary = ["| Scenario | Profile | Nodes | Tools requested | Reply (start) |", "|---|---|---|---|---|"]

    for sc in data.scenarios():
        if wanted and sc["id"] not in wanted:
            continue
        print(f"===== {sc['id']} {sc['title']} (profile={sc['status_profile']})")
        if sc.get("reset_before"):
            memory.reset_storage()
        ticket_rel = (Path("outputs") / ("fake" if args.fake_model else "") / "tickets" / sc["id"]).as_posix()
        shutil.rmtree(ROOT / ticket_rel, ignore_errors=True)
        env = dict(os.environ, HELPDESK_TICKET_DIR=ticket_rel, PYTHONIOENCODING="utf-8")

        if sc["part"] == 1:
            log = run_part1_scenario(sc)
        else:
            log = run_agent_scenario(sc, args.fake_model, ticket_rel, env)
        log = {"scenario": sc["id"], "title": sc["title"], "part": sc["part"],
               "status_profile": sc["status_profile"], "reset_before": sc.get("reset_before"),
               "model": "none (Part 1 graph)" if sc["part"] == 1 else
                        ("FAKE rule-based model - not a result" if args.fake_model
                         else f"openai:{os.getenv('MODEL_NAME') or _env_model()}"),
               "storage_dir": "local-storage/ (git-ignored)", **log}
        path = out / f"{sc['id']}.json"
        path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for nodes, tools_used, reply in summarise(sc, log):
            reply = " ".join(str(reply).split())[:140].replace("|", "/")
            summary.append(f"| {sc['id']} | {sc['status_profile']} | {nodes} | {tools_used} | {reply} |")
        print(f"saved {path.relative_to(ROOT)}\n")

    (out / "scenarios_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"summary: {(out / 'scenarios_summary.md').relative_to(ROOT)}")


def _env_model():
    from dotenv import dotenv_values
    return dotenv_values(ROOT / ".env").get("MODEL_NAME", "")


if __name__ == "__main__":
    main()
