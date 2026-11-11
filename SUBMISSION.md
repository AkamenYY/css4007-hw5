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

### An actual run with several tools

TODO (after the real run): S06 turn 2 — model requests, tool results and final answer, copied from `outputs/S06.json`.

### Same request, three snapshots

TODO (after the real run): S06 (`normal`) vs S07 (`wifi_outage`) vs S08 (`status_unavailable`).

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

### Demonstrations

TODO (after the real run): fill from `outputs/S10.json`–`outputs/S13.json`.

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message | | outputs/S10.json | |
| Close, restart and resume | | outputs/S11.json | |
| Same user, new conversation | | outputs/S12.json | |
| Different user | | outputs/S13.json | |

---

## Part 4 — Running and checking my work

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
- **The model writes the ticket summary and steps** from the conversation. They can be worded differently from the user's message.
- The real-model wording changes between runs; the logs show one run.

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
| S06 — Instructions then local ticket | TODO | | |
| S07 — The same request during a Wi-Fi outage | TODO | | |
| S08 — Status source unavailable | TODO | | |
| S09 — Missing ticket details | TODO | | |
| S10 — Use the current conversation | TODO | | |
| S11 — Persist and recall after restart | TODO | | |
| S12 — Preference across threads without replay | TODO | | |
| S13 — Do not share memory between users | TODO | | |
| S14 — Password recovery outage | TODO | | |
