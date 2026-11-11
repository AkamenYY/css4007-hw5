# HW5 — My LangGraph helpdesk

How to install and run everything: [README.md](README.md). LangGraph's own drawings of all graphs: [outputs/graphs.md](outputs/graphs.md).

**AI assistance:** I used Claude (Anthropic) to help write the code, tests and this report. I ran the programs myself; every result below comes from the logs in `outputs/`.

---

## Part 1 — How the graph works

### My graph and state

```text
START → read_request ──(choose_next_node)──┬─ wifi_help     → END
                                           ├─ password_help → END
                                           └─ other_help    → END
```

Code: `helpdesk/part1.py`. There is no model in this part.

The state has three fields:

| Field | What it holds | Who sets it |
| --- | --- | --- |
| `message` | the user's original message | the input at START; nobody changes it |
| `category` | `wifi`, `password` or `other` | `read_request` |
| `reply` | the answer text | the one help node that runs |

| Node | Reads | Changes |
| --- | --- | --- |
| `read_request` | `message` | `category`. It looks for the aliases from `data/helpdesk.json` (`wifi`, `wi-fi`, `password`) in the lowercase message. Wi-Fi is checked first. |
| `wifi_help` | nothing from the state; it reads `data/helpdesk.json` | `reply` = the GUIDE-WIFI steps |
| `password_help` | nothing from the state; it reads `data/helpdesk.json` | `reply` = the GUIDE-PASSWORD steps |
| `other_help` | nothing | `reply` = "Is your problem about Wi-Fi or about a password?" |
| `choose_next_node` (the conditional edge, not a node) | `category` | nothing; it only returns the name of the next node |

**1. What does each node read and change?** See the table above.

**2. Node update vs conditional edge.** A node returns a small dictionary, for example `{"category": "wifi"}`. LangGraph merges it into the state. A conditional edge changes nothing. It reads the state and returns a node name, for example `"wifi_help"`, so LangGraph knows where to go next.

**3. What does the next node receive?** The whole current state, not only the last update. In my S01 log, `wifi_help` runs after `read_request` and the state it sees still has `message` plus the new `category`.

**4. START, END and compiling.** `START` is where the input enters: the first node is `read_request`. `END` means "stop and return the state". `compile()` checks the graph (every node exists, every edge points somewhere) and turns it into a program I can run with `invoke` or `stream`.

### Predictions before running

My rule order is: Wi-Fi first, then password, then other.

| Message | Predicted route | Actual route | What I learned |
| --- | --- | --- | --- |
| I cannot connect to Wi-Fi. | read_request → wifi_help | read_request → wifi_help ([S01](outputs/S01.json)) | `wi-fi` is found after lowercasing. |
| I forgot my PASSWORD. | read_request → password_help | read_request → password_help ([S02](outputs/S02.json)) | Upper case does not matter because I lowercase first. |
| My laptop screen is broken. | read_request → other_help | read_request → other_help ([S03](outputs/S03.json)) | No topic word, so the fallback node asks a question. |
| My Wi-Fi password does not work. | read_request → wifi_help | read_request → wifi_help ([S04](outputs/S04.json)) | Both topics appear, but the **Wi-Fi rule is checked first**, so the password part is ignored. |

Every log saves the order of nodes and the full state after each node, for example S01:

```text
read_request  update {"category": "wifi"}  state: message + category
wifi_help     update {"reply": "..."}      state: message + category + reply
```

**Two topics in one message.** "My Wi-Fi password does not work." goes to `wifi_help` only. The cause is the rule order inside `read_request`: `if "wifi" ... elif "password"`. The user may really have a password problem, but they never get password help.

### My change to the graph

I added a `clarify_problem` node. `read_request` now sets `category = "both"` when both topics appear, and the edge sends `"both"` to `clarify_problem`:

```text
START → read_request ──(choose_next_node)──┬─ wifi_help       → END
                                           ├─ password_help   → END
                                           ├─ other_help      → END
                                           └─ clarify_problem → END   (new)
```

New trace ([S05](outputs/S05.json), variant `with_clarification`):

```text
read_request     update {"category": "both"}
clarify_problem  update {"reply": "Your message mentions both Wi-Fi and a password. Which should we solve
                  first: connecting to Wi-Fi, or recovering your account password?"}
```

The difference is not only a different sentence: a **different node runs** (`clarify_problem` instead of `wifi_help`). The other three messages still take their old routes (checked in `tests/test_part1_routing.py`).

**What LangGraph controls:** the order of execution (START → node → edge → node → END), merging each node's update into the state, passing the full state to the next node, and calling my edge function to pick the branch.

**What my own Python functions decide:** everything about meaning. `read_request` decides the category with my keyword rules, `choose_next_node` maps a category to a node name, and the help nodes choose the reply text. No model decides anything in Part 1.

---

## Part 2 — Model and tools

**Model and provider:** OpenAI `gpt-5.6-luna` through `langchain-openai` (Chat Completions), with `reasoning_effort="none"`. This model only accepts tools there with that setting. Code: `helpdesk/agent.py`, `helpdesk/tools.py`.

### The three tools (they work without the agent)

They are plain Python functions in `helpdesk/tools.py`. `python -m helpdesk tools` calls them directly, with no model and no graph. Output: [outputs/tools_direct.json](outputs/tools_direct.json).

| Direct call | Result |
| --- | --- |
| `find_instructions("wifi")` | GUIDE-WIFI, 4 steps, source `data/helpdesk.json` |
| `find_instructions("printer")` | `error`: only wifi and password exist |
| `check_service_status("wifi", status_profile="wifi_outage")` | `outage`, repair_time `null` |
| `check_service_status("wifi", status_profile="status_unavailable")` | `unknown` (not working, not an outage) |
| `prepare_ticket(...)` with a problem and steps | draft file `outputs/tickets/direct/TKT-DEMO-….json`, `sent: false` |
| `prepare_ticket(...)` with no problem and no steps | `error: missing_information`; no file is written |

Rules inside the tools:
- `check_service_status` reads the snapshot the **program** chose (`--status`). The model can only choose the service.
- `prepare_ticket` follows `data/ticket_template.json`, makes its own id (`TKT-DEMO-<time>-<random>`), refuses to work without a problem and steps, and refuses text that looks like a password or code.
- When the model asks for a ticket, my program fills `user_id`, `thread_id`, the status profile, the **service status from the earlier status check** (or `not_checked`) and the reference ids. The model only gives the category, a summary and the steps the user said.

### The loop

```text
START → load_memory → agent ──(route_after_agent)──┬─ model asked for a tool, rounds < 4 → tools ─┐
                        ▲                          ├─ model asked for a tool, rounds = 4 → loop_limit → END
                        │                          └─ model wrote an answer               → END
                        └──────────────── tool results go back to the model ───────────────┘
```

1. `agent`: the model sees the system prompt and the whole conversation. It returns either **tool requests** (`tool_calls`) or a normal answer.
2. `route_after_agent` (conditional edge): tool requests → `tools`; an answer → `END`.
3. `tools`: **my program** checks each request (known tool name?) and runs the Python function.
4. The results are added to `messages` as `ToolMessage`s, and the graph goes back to `agent`.
5. The model reads the results and decides again.

The model asks for **one tool at a time** (`parallel_tool_calls=False`). That way it always sees, for example, the status result before it decides about a ticket.

**Loop limit:** at most `MAX_TOOL_ROUNDS = 4` tool rounds per user message. It is checked in `route_after_agent`: if the model asks for a 5th round, the graph goes to `loop_limit`, which does not run the tools, writes "not run: loop limit reached" as their result, and gives a final answer. `load_memory` sets the counter back to 0 for each new user message. As a second safety net, LangGraph's `recursion_limit` is 30 steps. The test `test_loop_limit_stops_an_endless_model` uses a fake model that never stops asking for tools.

### The four questions

**1. How does the graph decide between running a tool and ending?** The conditional edge `route_after_agent` looks at the newest message. If the model's message has `tool_calls`, it goes to `tools` (or `loop_limit` at the limit). If not, the message is the answer and it goes to `END`.

**2. Who requests a tool, and who executes it?** The **model** only requests: it writes a tool name and arguments. **My program** (the `tools` node, function `run_one_tool`) checks the request and executes the Python function. The model never runs code.

**3. What changes in the state after a tool result returns?** `messages` gets one new `ToolMessage` per request (with the same `tool_call_id`), `tool_rounds` goes up by 1, and `tickets` gets the file name if a ticket was saved.

**4. What is different from Part 1?** In Part 1, my keyword rules choose the route, so the same input always takes the same path. In Part 2, the edge still decides, but it decides from the **model's output**: the model chooses which tools, in what order, and when to stop, based on the conversation and on the tool results. The same request can take different paths when the data changes (see S06, S07, S08 below).

### An actual run with several tools (S06, real model)

Command: `python run_scenarios.py` → `python -m helpdesk chat --user demo-user-a --thread s06 --status normal -m "Show me the Wi-Fi troubleshooting steps." -m "I enabled Wi-Fi and forgot the saved network, ... If it is operating, prepare a ticket about this problem."` Log: [outputs/S06.json](outputs/S06.json).

Turn 1: the model asked for `find_instructions({"topic": "wifi"})` and answered with the 4 GUIDE-WIFI steps.

Turn 2 (from the log, in order):

```text
load_memory  tool_rounds = 0; conversation memory has 4 earlier messages; no user memory
agent        MODEL ASKS   check_service_status({"service": "wifi"})
tools        RESULT       {"status": "operational", "message": "The demo Wi-Fi service has no reported outage."}
agent        MODEL ASKS   prepare_ticket({"category": "wifi",
                            "summary": "Campus-Demo will not connect even though Wi-Fi is enabled.",
                            "steps_tried": ["Enabled Wi-Fi.", "Forgot the saved network."]})
tools        RESULT       {"ticket_id": "TKT-DEMO-20261111200057-71b9",
                           "file": "outputs/tickets/S06/TKT-DEMO-20261111200057-71b9.json", "sent": false}
agent        ANSWER       "Wi-Fi is operational with no reported outage. I prepared a local ticket draft
                           using the steps you reported ... It was not sent to the helpdesk."
```

The model asked for the ticket **only after** it saw `operational`. The ticket ([file](outputs/tickets/S06/TKT-DEMO-20261111200057-71b9.json)) has only the two steps the user said, `service_status: "operational"` (taken by my program from the status result) and `reference_ids: ["GUIDE-WIFI"]` (from turn 1). 2 tool rounds; the limit (4) was not reached.

### Same request, three snapshots

The request is the same in all three: "check whether Wi-Fi is down; if it is operating, prepare a ticket". Only `--status` changes.

| Scenario | Snapshot | Tools the model asked for | Status result | What it did |
| --- | --- | --- | --- | --- |
| [S06](outputs/S06.json) | `normal` | check_service_status → prepare_ticket | operational | made the ticket draft |
| [S07](outputs/S07.json) | `wifi_outage` | find_instructions → check_service_status | outage | "Wi-Fi is currently experiencing a campus-wide outage. I did not prepare a ticket because the service is not operating. No repair time is available." |
| [S08](outputs/S08.json) | `status_unavailable` | check_service_status | unknown | "Wi-Fi status could not be checked ... I did not prepare a ticket. Would you like a local ticket draft stating that the status could not be checked?" |

So the action depends on the tool result: one ticket, no ticket, or a question. No repair time was invented. In S14 (`recovery_outage`) the model checked `account_recovery`, not Wi-Fi, and reported the outage with "No repair time is available".

---

## Part 3 — Memory

### Two kinds of memory

| | Conversation memory | User memory |
| --- | --- | --- |
| What | the messages of one conversation: the problem, what was said, steps already tried, tool results | the user's saved reply language |
| Where | `local-storage/conversations.sqlite` | `local-storage/user_memory.sqlite` |
| How | LangGraph **checkpointer** `SqliteSaver` | LangGraph **store** `SqliteStore` |
| Key | `thread_id` | namespace `("users", user_id)`, key `preferences` |
| Shared by | one thread only | all threads of the same user |

- **`thread_id`** identifies **one conversation**. The checkpointer saves the state after every step under this id, and the next message in the same thread starts from that saved state.
- **`user_id`** identifies **a person**. One user can have many threads. The language preference is saved under the user id, so every thread of that user can read it, and other users cannot.
- The program (command line `--user` and `--thread`) chooses both ids. They are passed to the graph as `context` and in the config, so the model cannot pick which user's data is loaded.
- **Saving the preference:** the `load_memory` node checks each new user message for a sentence such as "I prefer replies in Kazakh" and saves the language to the store. Then it reads this user's saved language and puts it into the system prompt. `data/demo_users.json` is never used as memory.
- **Surviving a restart:** both are SQLite files on disk. Each `python -m helpdesk chat ...` process opens them, and closing the program closes them. A new process opens the same files again.

**Retrieval vs memory.** Looking up Wi-Fi instructions is **retrieval**: `find_instructions` reads the fixed reference file `data/helpdesk.json`, and every user gets the same answer. Recalling the user's language is **memory**: it was learned from this user's own message, saved by my program, and only this user gets it back later.

### Demonstrations (real model, `python run_scenarios.py`)

Every message group is a separate `python -m helpdesk chat ...` process. The process ids in the logs prove the restarts.

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message | one process (pid 8937), thread `s10`: `-m "I cannot connect to Wi-Fi. I enabled it and forgot the saved network, but reconnecting failed." -m "Which steps did I already try?"` | [S10](outputs/S10.json) | Second answer: "You already tried: Enabling Wi-Fi, Forgetting the saved network, Reconnecting, which failed". **Read from:** conversation memory (6 earlier messages of thread `s10`). No tools were needed. |
| Close, restart and resume | process 8940, then a **new** process 8943, both `--thread restart-a` | [S11](outputs/S11.json) | The new process started with "this thread already has 4 saved messages". Answer (in Kazakh): problem "Wi‑Fi желісіне қосыла алмау" (cannot connect to Wi-Fi); steps tried: enabled Wi-Fi, tried reconnecting. It even separated the steps it had *suggested* from the steps the user *did*. **Read from:** conversation memory (4 messages) + user memory (Kazakh). The old text was not typed again. |
| Same user, new conversation | process 8946 thread `preference-a1`, then new process 8949 thread `preference-a2` | [S12](outputs/S12.json) | In the new thread: "Сіз жауаптарды қазақ тілінде беруді сұрадыңыз" (you asked for replies in Kazakh) + the list of what it can help with. **Read from:** user memory `{'language': 'Kazakh'}`; conversation memory: "new thread, nothing earlier" (no replay of the old thread). |
| Different user | `--user demo-user-a --thread isolation-a`, then `--user demo-user-b --thread isolation-b` | [S13](outputs/S13.json) | demo-user-b got (in English): "No. In this conversation, you haven't told me a preferred language or a previous problem." **Read from:** user memory: nothing saved for `demo-user-b`; new thread. demo-user-a's Kazakh preference did not appear. |

In the first message of S11, S12 and S13 the trace shows `user memory: SAVED {'language': 'Kazakh'}`: this is where `load_memory` wrote the preference.

---

## Part 4 — Running and checking my work

- **Python used for all saved runs:** 3.13.18 (recorded in every log under `process.python`). Tests: `45 passed`.

- **Setup:** README sections 1–3 (Python 3.13, `pip install -r requirements.txt`, `.env`).
- **Versions:** `requirements.txt` pins `langgraph==1.2.20`, `langgraph-checkpoint-sqlite==3.1.3`, `langchain-core==1.8.0`, `langchain-openai==1.7.4` and the others.
- **Memory files and reset:** README section 5. Memory is in `local-storage/` (git-ignored); `python -m helpdesk reset` deletes it.
- **CI:** `.github/workflows/tests.yml` runs `python -m pytest -v` on every push with Python 3.13 and **no API key**.
- **Fake models:** the tests use `ScriptedFakeModel` (returns replies written in the test) and `EchoFakeModel` (repeats the user messages it can see). Both are labelled FAKE in `helpdesk/models.py`. Only `run_scenarios.py` uses the real model.

| Test file | What it checks | Mistake it would catch |
| --- | --- | --- |
| `test_part1_routing.py` | the 4 routes; clarify variant; upper/lower case; when `category` and `reply` are set; replies from `helpdesk.json` | wrong rule order, missing `wi-fi` alias, no lowercasing, a broken conditional edge, a node overwriting `message` |
| `test_tools.py` | each tool called directly: both guides, all 4 snapshots × 2 services, `unknown` ≠ operational, ticket file matches the template, missing details → no file, secrets refused | the profile being ignored, mixed-up services, invented ticket details, a ticket that claims it was sent |
| `test_agent_loop.py` | a tool result goes back to the model; the **same request** makes a ticket with `normal` but not with `wifi_outage` or `status_unavailable`; ticket ids and status come from the program; the loop limit; unknown tool names | results not added to the state, the model choosing the user or status, an endless loop, a crash on a bad request |
| `test_memory.py` | conversation saved after closing and reopening storage; **two real processes** (different process ids) share a thread; preference shared by threads of one user without replaying the old messages; no leak to another user; reset | an in-memory checkpointer, memory keyed by the wrong id, the preference stored per thread, user data shared between users |

---

## Choices and limitations

- **SQLite for both memories.** It is one local file each, needs no server, survives a restart, and the package was supplied. I keep conversations (checkpointer) and preferences (store) in separate files, so it is clear which memory is which and both can be cleared together.
- **The preference is found by a simple rule** (a regular expression for "prefer replies in / reply in" + Kazakh, Russian or English). It is predictable and testable, but it misses other wordings (for example "қазақша жаз").
- **The model decides the conditions** ("only if operating"). My program makes it safer (one tool at a time, the ticket status comes from the real status result, tickets need the user's steps), but a model could still make a wrong choice. A stricter version would check the condition in code.
- **The model writes the ticket summary and steps** from the conversation, so the wording differs a little from the user's message (for example "Enabled Wi-Fi." instead of "I enabled Wi-Fi").
- **The model sometimes calls an extra tool.** In S07 it looked up the instructions before checking the status. This is harmless, but it costs one more round.
- **The model can do more than it was asked.** In my first full run (S10) it made a ticket without being asked. A prompt rule fixed it, but a rule in a prompt is not a guarantee; a stricter version would allow `prepare_ticket` only when the user's message asks for a ticket.
- **The answer to demo-user-b (S13) talks about "this conversation" only.** It is correct (nothing is saved for that user), but it could also say clearly that no user memory exists.
- The real-model wording changes between runs; the logs show my final run (after commit f0c9cb3).

---

## Supplied scenarios — my actual results

All logs: `outputs/S01.json` … `outputs/S14.json`. Overview: `outputs/scenarios_summary.md`. Command for all: `python run_scenarios.py`.

| Scenario | Command or manual steps | Log file | What happened / what I changed |
| --- | --- | --- | --- |
| S01 — Simple Wi-Fi route | `python run_scenarios.py --only S01` (Part 1 graph) | [S01](outputs/S01.json) | read_request → wifi_help; GUIDE-WIFI steps. As predicted. |
| S02 — Case-insensitive password route | `--only S02` | [S02](outputs/S02.json) | read_request → password_help; "PASSWORD" matched after lowercasing. |
| S03 — Unsupported request | `--only S03` | [S03](outputs/S03.json) | read_request → other_help; asks Wi-Fi or password. |
| S04 — Two topics before the graph change | `--only S04` (variant original) | [S04](outputs/S04.json) | read_request → wifi_help; the password part is lost because the Wi-Fi rule is first. |
| S05 — Two topics after adding clarification | `--only S05` (variant with_clarification) | [S05](outputs/S05.json) | read_request (category both) → clarify_problem; asks which problem first. |
| S06 — Instructions then local ticket | `python run_scenarios.py` (thread s06, `normal`, one process, 2 messages) | [S06](outputs/S06.json) | Turn 1: find_instructions → 4 steps. Turn 2: check_service_status (operational) → prepare_ticket → local draft [TKT-DEMO-20261111200057-71b9](outputs/tickets/S06/TKT-DEMO-20261111200057-71b9.json) with only the user's 2 steps; "not sent". |
| S07 — The same request during a Wi-Fi outage | same request, `--status wifi_outage` | [S07](outputs/S07.json) | find_instructions → check_service_status (outage) → no ticket: "I did not prepare a ticket because the service is not operating. No repair time is available." |
| S08 — Status source unavailable | `--status status_unavailable` | [S08](outputs/S08.json) | check_service_status (unknown) → no ticket; asks "Would you like a local ticket draft stating that the status could not be checked?" |
| S09 — Missing ticket details | thread s09, 2 messages | [S09](outputs/S09.json) | Turn 1: no tool; asks for the problem and steps tried. Turn 2: prepare_ticket with "Enabled Wi-Fi", "Tried reconnecting" → [draft](outputs/tickets/S09/TKT-DEMO-20261111200110-d4d5.json) with `service_status: not_checked` (status was not checked, and the ticket says so). |
| S10 — Use the current conversation | thread s10, 2 messages, one process | [S10](outputs/S10.json) | Lists the 3 steps tried from the earlier message. **Change I made:** in my first run the model made a ticket in turn 1 without being asked; I added the rule "Prepare a ticket only when the user asks for one" (commit f0c9cb3) and re-ran all scenarios. Now it only offers a ticket. |
| S11 — Persist and recall after restart | 2 processes (8940, 8943), thread restart-a | [S11](outputs/S11.json) | Preference saved; after restart the thread had 4 saved messages; answer in Kazakh with the problem and the user's steps. |
| S12 — Preference across threads without replay | 2 processes, threads preference-a1 → preference-a2 | [S12](outputs/S12.json) | New thread, no earlier messages, but user memory gave Kazakh; answered in Kazakh and listed what it can help with. |
| S13 — Do not share memory between users | demo-user-a/isolation-a, then demo-user-b/isolation-b | [S13](outputs/S13.json) | demo-user-b: nothing in user memory, new thread; answered in English that no preference or problem was told. |
| S14 — Password recovery outage | `--status recovery_outage` | [S14](outputs/S14.json) | check_service_status(**account_recovery**) → outage; "there is a known problem. No repair time is available." No invented repair time. |
