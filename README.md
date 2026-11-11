# HW5 — Practice university IT helpdesk with LangGraph

A terminal helpdesk built in four stages with LangGraph:

1. **Part 1** — a fixed-route graph with no model (`helpdesk/part1.py`).
2. **Part 2** — a model that chooses three tools in a loop with a limit (`helpdesk/tools.py`, `helpdesk/agent.py`).
3. **Part 3** — conversation memory (LangGraph SQLite checkpointer) and user memory (LangGraph SQLite store) that survive a restart (`helpdesk/memory.py`).
4. **Part 4** — pinned dependencies, tests with a fake model, and GitHub Actions.

All data is fictional (`data/`). Nothing connects to real university systems, and tickets are local draft files only.
The original task text is in [ASSIGNMENT.md](ASSIGNMENT.md); my explanations and results are in [SUBMISSION.md](SUBMISSION.md).

## 1. Install

Tested with **Python 3.13** (`langchain-core` needs Python 3.11 or newer; the macOS system `python3` is 3.9 and is too old).

macOS / Linux:

```bash
git clone https://github.com/AkamenYY/css4007-hw5.git
cd css4007-hw5
python3.13 -m venv .venv            # or: uv venv --python 3.13 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell: `py -3.13 -m venv .venv`, then `.venv\Scripts\Activate.ps1`, then the same `pip install`.

No Python 3.13? The quickest way is [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`, open a new terminal, then `uv venv --python 3.13 .venv` (uv downloads Python for you).

`requirements.txt` pins the exact versions used (LangGraph 1.2.20, langgraph-checkpoint-sqlite 3.1.3, langchain-openai 1.7.4).

## 2. Configure the model (only for Parts 2–3 with a real model)

```bash
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. The demos used provider **OpenAI** with `MODEL_NAME=gpt-5.6-luna` through `langchain-openai` (Chat Completions). That model only accepts tool calls there with `MODEL_REASONING_EFFORT=none`; leave that variable empty for a model that does not support the parameter. `.env` is git-ignored — never commit a key.

Part 1, `python -m helpdesk tools`, the tests and `--fake-model` runs need no key.

## 3. Run

```bash
# Part 1: fixed routing, no model
python -m helpdesk part1 "My Wi-Fi password does not work."
python -m helpdesk part1 "My Wi-Fi password does not work." --variant with_clarification

# Part 2: call each tool directly, without the agent
python -m helpdesk tools

# Parts 2-3: the helpdesk (interactive; type /quit to exit, /history to see saved messages)
python -m helpdesk chat --user demo-user-a --thread t1 --status normal

# The same, sending messages from the command line (each -m is one user message)
python -m helpdesk chat --user demo-user-a --thread s06 --status normal \
  -m "Show me the Wi-Fi troubleshooting steps." \
  -m "I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem."

# Print a saved conversation, draw the graphs, delete all memory
python -m helpdesk show --thread s06
python -m helpdesk graphs
python -m helpdesk reset
```

Options for `chat`: `--user` (invented id such as `demo-user-a`), `--thread` (conversation id), `--status` (`normal`, `wifi_outage`, `recovery_outage`, `status_unavailable`), `--log FILE` (append this run's trace as JSON), `-q` (shorter output), `--fake-model` (a **fake** rule-based model for a dry run without a key — not a result).

## 4. Run the supplied scenarios S01–S14

```bash
python run_scenarios.py              # real model; writes outputs/S01.json ... outputs/S14.json
python run_scenarios.py --only S11   # one scenario
python run_scenarios.py --fake-model # dry run with the FAKE model; writes outputs/fake/ (git-ignored)
```

The runner follows `data/scenarios.json`: `reset_before` deletes the memory folder; every group of messages is sent by a **separate** `python -m helpdesk chat` process, so `restart_process` really ends one process and starts another one on the same storage. Each log records the command, process id, snapshot, user, thread, nodes, tool requests and results, what was read from conversation state / user memory / reference files, the reply, ticket files and whether the loop limit was reached. `outputs/scenarios_summary.md` is a one-line-per-turn overview. Ticket drafts go to `outputs/tickets/<scenario>/`.

## 5. Where memory is saved and how to clear it

| What | File | Key |
| --- | --- | --- |
| Conversation memory (messages, problem, steps tried) | `local-storage/conversations.sqlite` (LangGraph `SqliteSaver`) | `thread_id` |
| User memory (reply language) | `local-storage/user_memory.sqlite` (LangGraph `SqliteStore`) | namespace `("users", user_id)` |
| Ticket drafts from `chat` | `local-storage/tickets/` (from `.env`) | ticket id |

The folder is set by `HELPDESK_STORAGE_DIR` and is git-ignored. Start again with `python -m helpdesk reset` (or delete the folder). `data/demo_users.json` is never used as memory.

## 6. Tests and CI

```bash
python -m pytest -v
```

The tests use a clearly labelled **fake model** (`ScriptedFakeModel` in `helpdesk/models.py`) and temporary storage folders, so they need no API key and never touch `local-storage/`. GitHub Actions (`.github/workflows/tests.yml`) runs them on every push with Python 3.13. The real-model demonstration is separate (`run_scenarios.py`). What each test catches is listed in SUBMISSION.md, Part 4.

## Project layout

```
helpdesk/data.py      load data/*.json, detect topics
helpdesk/part1.py     Part 1 graph (original and with_clarification)
helpdesk/tools.py     find_instructions, check_service_status, prepare_ticket
helpdesk/agent.py     Parts 2-3 graph, loop limit, HelpdeskSession
helpdesk/memory.py    checkpointer, user store, reset, preference parsing
helpdesk/models.py    real OpenAI model + FAKE models for tests / dry runs
helpdesk/cli.py       terminal program (python -m helpdesk ...)
run_scenarios.py      runs S01-S14 and saves logs to outputs/
tests/                pytest tests (no API key)
outputs/              saved logs, graph drawings, demo ticket drafts
```
