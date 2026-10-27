"""
Autonomous Detective Agent (Groq / MCP)

An LLM agent that investigates a fictional murder case. It decides on its own
which suspect to interrogate, which evidence to search and which alibi to
verify, then names the culprit with an evidence chain.

Layers (top to bottom):
  1. Case data layer   - CASE dictionary (fictional)
  2. MCP server layer  - CaseMCPServer (tool definitions + tool execution)
  3. Agent layer       - run_detective_agent() (ReAct-style loop)
  4. Presentation      - Streamlit UI
"""

import json
import os
import time

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_STEPS = 12
MAX_RETRIES = 3  # retries when the model produces a malformed tool call

# ---------------------------------------------------------------------------
# 1. CASE DATA LAYER  (all characters and events are fictional)
# ---------------------------------------------------------------------------
CASE = {
    "title": "The Midnight Server Room",
    "victim": "Dr. Rohan Kapoor, CEO of Nexora Software",
    "location": "His 14th-floor office at Nexora Software HQ",
    "time_of_death": "Between 9:00 PM and 10:00 PM",
    "suspects": {
        "Meera Iyer": {
            "role": "CFO",
            "claim": "Left the building at about 8:30 PM.",
            "statement": (
                "I argued with Rohan about the budget in the afternoon, I admit "
                "that. But I went home at around 8:30 PM and was never in the "
                "building after that."
            ),
        },
        "Vikram Sethi": {
            "role": "CTO",
            "claim": "In the 3rd-floor server room from 9:00 PM to 10:00 PM, never went upstairs.",
            "statement": (
                "I was in the server room on the 3rd floor from 9 to 10 PM, "
                "fixing a system fault. I never went up to the 14th floor. My "
                "server session will prove I was working the whole time."
            ),
        },
        "Anjali Rao": {
            "role": "Executive assistant",
            "claim": "Having dinner at a restaurant with a friend around 9 PM.",
            "statement": (
                "I finished work early and went to dinner with a friend. I was "
                "at the restaurant around 9 PM. I paid the bill myself."
            ),
        },
        "Karan Desai": {
            "role": "Security guard",
            "claim": "At the lobby desk the whole evening.",
            "statement": (
                "I was at the lobby desk the whole evening. I did not leave my "
                "post, and I saw nobody suspicious go up."
            ),
        },
    },
    # "effects" holds how each record relates to each suspect's claim.
    # It is used only by check_alibi and is never shown to the model directly.
    "evidence": [
        {"id": "E1", "source": "Forensic report",
         "detail": "Time of death estimated between 9:00 PM and 10:00 PM.",
         "effects": {}},
        {"id": "E2", "source": "Elevator access log",
         "detail": "Badge of V. Sethi used to reach the 14th floor at 9:20 PM.",
         "effects": {"Vikram Sethi": "contradicts"}},
        {"id": "E3", "source": "Server login record",
         "detail": "V. Sethi's session started at 9:05 PM, but no commands were run between 9:12 PM and 9:58 PM.",
         "effects": {"Vikram Sethi": "contradicts"}},
        {"id": "E4", "source": "Restaurant receipt",
         "detail": "Table of A. Rao billed at 9:10 PM at a restaurant across town.",
         "effects": {"Anjali Rao": "supports"}},
        {"id": "E5", "source": "Building exit log",
         "detail": "Badge of M. Iyer exited the building at 8:32 PM; no re-entry recorded.",
         "effects": {"Meera Iyer": "supports"}},
        {"id": "E6", "source": "Lobby CCTV log",
         "detail": "K. Desai visible at the lobby desk continuously from 9:00 PM to 9:45 PM.",
         "effects": {"Karan Desai": "supports"}},
        {"id": "E7", "source": "Torn note",
         "detail": "Note on the victim's desk reads: 'Your access ends tomorrow.'",
         "effects": {"Vikram Sethi": "motive"}},
        {"id": "E8", "source": "HR draft",
         "detail": "Unsent draft dated the same day to revoke V. Sethi's system access.",
         "effects": {"Vikram Sethi": "motive"}},
    ],
}

SYSTEM_PROMPT = """You are a careful, methodical detective agent.
Your job is to solve the case using ONLY the tools provided.

Workflow:
1. Call get_case_file to learn the victim, place, time of death and suspects.
2. Call interrogate_suspect for EVERY suspect.
3. Call search_evidence with useful keywords to collect records.
4. Call check_alibi for EVERY suspect to compare their claim with the records.
5. Accuse a suspect only when check_alibi shows their claim is CONTRADICTED
   by a record that falls inside the time-of-death window.

Final answer (Markdown) must contain:
- **Culprit**
- **Motive** (with evidence IDs)
- **Evidence chain** (with evidence IDs)
- **Suspects ruled out** (one line each, with the reason)

Never invent evidence. Cite evidence IDs such as E2."""


# ---------------------------------------------------------------------------
# 2. MCP-STYLE SERVER LAYER
# ---------------------------------------------------------------------------
def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class CaseMCPServer:
    """In-process, MCP-style server: publishes tools and executes tool calls."""

    def get_tool_definitions(self):
        return [
            _tool(
                "get_case_file",
                "Get the basic facts of the case: victim, location, time of death and suspect names.",
                {},
                [],
            ),
            _tool(
                "interrogate_suspect",
                "Question one suspect and get their statement.",
                {"suspect_name": {"type": "string"}},
                ["suspect_name"],
            ),
            _tool(
                "search_evidence",
                "Search the evidence table by keyword (for example 'log', 'receipt', 'note').",
                {"keyword": {"type": "string"}},
                ["keyword"],
            ),
            _tool(
                "check_alibi",
                "Compare a suspect's claim with the recorded evidence for a time window. "
                "Returns CONSISTENT, CONTRADICTED or UNVERIFIED.",
                {"suspect_name": {"type": "string"}, "time_window": {"type": "string"}},
                ["suspect_name", "time_window"],
            ),
        ]

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _find_suspect(name):
        name = (name or "").strip().lower()
        if not name:
            return None
        for suspect in CASE["suspects"]:
            if name == suspect.lower() or name in suspect.lower():
                return suspect
        return None

    # -- tools -------------------------------------------------------------
    def get_case_file(self):
        return {
            "case": CASE["title"],
            "victim": CASE["victim"],
            "location": CASE["location"],
            "time_of_death": CASE["time_of_death"],
            "suspects": [
                {"name": n, "role": s["role"]} for n, s in CASE["suspects"].items()
            ],
        }

    def interrogate_suspect(self, suspect_name):
        suspect = self._find_suspect(suspect_name)
        if not suspect:
            return {"error": f"Suspect '{suspect_name}' not found.",
                    "known_suspects": list(CASE["suspects"])}
        info = CASE["suspects"][suspect]
        return {"suspect": suspect, "role": info["role"], "statement": info["statement"]}

    def search_evidence(self, keyword):
        kw = (keyword or "").strip().lower()
        if not kw:
            return {"error": "Please provide a keyword."}
        hits = [
            {"id": e["id"], "source": e["source"], "detail": e["detail"]}
            for e in CASE["evidence"]
            if kw in e["id"].lower() or kw in e["source"].lower() or kw in e["detail"].lower()
        ]
        return {"keyword": keyword, "count": len(hits), "results": hits}

    def check_alibi(self, suspect_name, time_window):
        suspect = self._find_suspect(suspect_name)
        if not suspect:
            return {"error": f"Suspect '{suspect_name}' not found.",
                    "known_suspects": list(CASE["suspects"])}
        claim = CASE["suspects"][suspect]["claim"]
        related = [e for e in CASE["evidence"] if suspect in e["effects"]]
        contradicting = [e for e in related if e["effects"][suspect] == "contradicts"]
        supporting = [e for e in related if e["effects"][suspect] == "supports"]
        motive = [e for e in related if e["effects"][suspect] == "motive"]

        if contradicting:
            status, used = "CONTRADICTED", contradicting
        elif supporting:
            status, used = "CONSISTENT", supporting
        else:
            status, used = "UNVERIFIED", []

        return {
            "suspect": suspect,
            "time_window": time_window,
            "claim": claim,
            "status": status,
            "evidence": [{"id": e["id"], "detail": e["detail"]} for e in used],
            "motive_evidence": [{"id": e["id"], "detail": e["detail"]} for e in motive],
        }

    def execute_tool(self, name, args):
        tools = {
            "get_case_file": self.get_case_file,
            "interrogate_suspect": self.interrogate_suspect,
            "search_evidence": self.search_evidence,
            "check_alibi": self.check_alibi,
        }
        if name not in tools:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tools[name](**(args or {}))
        except TypeError as err:
            return {"error": f"Bad arguments for {name}: {err}"}
        except Exception as err:  # keep the loop alive on any tool failure
            return {"error": str(err)}


# ---------------------------------------------------------------------------
# 3. AGENT LAYER (ReAct-style loop)
# ---------------------------------------------------------------------------
def call_model(client, messages, tools):
    """Call Groq, retrying when the model emits a malformed tool call.

    Small models occasionally generate broken JSON for a tool call, and Groq
    rejects it with a 400 'tool_use_failed' error. The failure is random, so a
    simple retry normally fixes it.
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as err:
            text = str(err)
            if "tool_use_failed" in text or "Failed to parse tool call" in text:
                last_err = err
                time.sleep(1)
                continue
            raise
    raise RuntimeError(
        f"The model produced a malformed tool call {MAX_RETRIES} times in a row. "
        "Click Run Agent again, or set GROQ_MODEL in .env to a different model."
    ) from last_err


def run_detective_agent(goal, api_key, on_event=None):
    """Run the agent loop and return the final case report (Markdown)."""
    emit = on_event or (lambda *_: None)
    client = Groq(api_key=api_key)
    server = CaseMCPServer()
    tools = server.get_tool_definitions()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    for step in range(1, MAX_STEPS + 1):
        response = call_model(client, messages, tools)
        msg = response.choices[0].message

        # No tool calls -> this reply is the final report.
        if not msg.tool_calls:
            return msg.content or "The agent returned an empty report."

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments or "{}"},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = server.execute_tool(name, args)
            emit(step, name, args, result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result),
            })

    return "Max execution steps reached before the agent could finish the case."


# ---------------------------------------------------------------------------
# 4. PRESENTATION LAYER (Streamlit)
# ---------------------------------------------------------------------------
PRESETS = {
    "Solve the case": "Investigate the death of Dr. Rohan Kapoor and name the culprit with evidence.",
    "Verify every alibi": "Check every suspect's alibi against the records and tell me who is lying.",
    "Full case report": "Run a complete investigation and write a full case report, including why each innocent suspect was ruled out.",
}

TOOL_INFO = [
    ("📋", "get_case_file", "Reads the basic facts: victim, place, time of death."),
    ("🗣️", "interrogate_suspect", "Questions one suspect and gets their statement."),
    ("🔎", "search_evidence", "Searches logs, receipts and records by keyword."),
    ("⚖️", "check_alibi", "Compares a suspect's claim with the records."),
]


def describe_step(name, args, result):
    """Turn a raw tool call into a plain-English title and summary."""
    if name == "get_case_file":
        n = len(result.get("suspects", []))
        return "📋 Reading the case file", f"Victim: {result.get('victim', '?')}. {n} suspects listed."
    if name == "interrogate_suspect":
        who = result.get("suspect", args.get("suspect_name", "?"))
        if "error" in result:
            return f"🗣️ Interrogating {who}", f"⚠️ {result['error']}"
        return f"🗣️ Interrogating {who}", f'They said: "{result.get("statement", "")}"'
    if name == "search_evidence":
        ids = ", ".join(r["id"] for r in result.get("results", [])) or "nothing"
        kw = args.get("keyword", "")
        return f'🔎 Searching evidence for "{kw}"', f"Found {result.get('count', 0)} item(s): {ids}"
    if name == "check_alibi":
        who = result.get("suspect", args.get("suspect_name", "?"))
        status = result.get("status", "ERROR")
        icon = {"CONSISTENT": "✅", "CONTRADICTED": "❌", "UNVERIFIED": "❓"}.get(status, "⚠️")
        ev = ", ".join(e["id"] for e in result.get("evidence", []))
        meaning = {
            "CONSISTENT": "the records support their story",
            "CONTRADICTED": "the records prove their story wrong",
            "UNVERIFIED": "no records either way",
        }.get(status, result.get("error", ""))
        text = f"{icon} **{status}**: {meaning}"
        if ev:
            text += f" (evidence {ev})"
        return f"⚖️ Checking {who}'s alibi", text
    return name, json.dumps(result)[:200]


def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def main():
    st.set_page_config(page_title="AI Detective Agent", page_icon="🕵️", layout="wide")

    # ---- Sidebar: what is this project? ----
    with st.sidebar:
        st.header("About this project")
        st.write(
            "An **AI agent** is an AI that does not just answer a question. "
            "It decides what to do next, uses tools, looks at the results and "
            "repeats until the job is done."
        )
        st.subheader("The agent's 4 tools")
        for icon, name, desc in TOOL_INFO:
            st.markdown(f"{icon} **{name}**  \n{desc}")
        st.divider()
        st.caption(f"Model: {MODEL} (via Groq)")
        st.caption("All characters and events are fictional.")

    # ---- Header ----
    st.title("🕵️ AI Detective Agent")
    st.markdown(
        "**An AI that solves a murder mystery on its own.** "
        "Press **Run Agent** at the bottom and watch it question suspects, "
        "dig through evidence and catch the liar."
    )

    with st.expander("How does it work?", expanded=True):
        st.markdown(
            "1. **You give the AI a goal**, for example *solve the case*.\n"
            "2. **The AI chooses a tool**, such as questioning a suspect or searching records.\n"
            "3. **The tool returns real data** from the case file.\n"
            "4. **The AI reads the result and decides the next step.** "
            "This repeats until it is sure.\n"
            "5. **It names the culprit and explains the evidence.**\n\n"
            "*The order of the steps is chosen by the AI, not programmed. "
            "That is what makes it an agent and not just a script.*"
        )

    # ---- The case ----
    st.subheader(f"📁 The case: {CASE['title']}")
    st.info(
        f"**Victim:** {CASE['victim']}  \n"
        f"**Where:** {CASE['location']}  \n"
        f"**Time of death:** {CASE['time_of_death']}"
    )

    st.markdown("**The four suspects** (what each one claims):")
    cols = st.columns(len(CASE["suspects"]))
    for col, (name, info) in zip(cols, CASE["suspects"].items()):
        with col.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(info["role"])
            st.write(info["claim"])

    # ---- Evidence file (what the agent can search) ----
    st.subheader("🗂️ The evidence file")
    st.markdown(
        "These are all the records the AI can search. It has no other information."
    )
    st.table(
        [
            {"ID": e["id"], "Source": e["source"], "What it shows": e["detail"]}
            for e in CASE["evidence"]
        ]
    )
    st.warning(
        "**Your turn first:** compare each suspect's claim with the evidence. "
        "Can you spot who is lying? Then run the AI and see if it agrees."
    )

    # ---- Run ----
    st.subheader("▶️ Run the investigation")
    preset = st.selectbox("Choose a goal for the AI", list(PRESETS))
    goal = st.text_area("Or write your own goal", value=PRESETS[preset], height=90)

    if st.button("Run Agent", type="primary"):
        api_key = get_api_key()
        if not api_key:
            st.error("GROQ_API_KEY not found. Add it to a .env file or Streamlit Secrets.")
            return

        report = None
        calls = [0]
        with st.status("🕵️ The detective is investigating...", expanded=True) as status:
            st.caption("Every step below was chosen by the AI, one at a time.")

            def log(step, name, args, result):
                calls[0] += 1
                title, summary = describe_step(name, args, result)
                st.markdown(f"**Step {calls[0]}: {title}**")
                st.markdown(summary)
                raw = json.dumps(result)
                st.caption("Raw tool output: " + raw[:300] + ("..." if len(raw) > 300 else ""))

            try:
                report = run_detective_agent(goal, api_key, on_event=log)
                status.update(label="Investigation complete", state="complete")
            except Exception as err:
                status.update(label="Agent failed", state="error")
                st.error(f"Error while running the agent: {err}")

        if report:
            st.subheader("🧾 Final Case Report")
            with st.container(border=True):
                st.markdown(report)
            st.caption(f"The agent made {calls[0]} tool calls to reach this conclusion.")


if __name__ == "__main__":
    main()