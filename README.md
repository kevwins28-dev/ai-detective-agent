# Autonomous Detective Agent (Groq / MCP)

An autonomous detective agent built with **Streamlit**, powered by **Groq** (`openai/gpt-oss-20b`), and architected around the **Model Context Protocol (MCP)** tool-calling pattern.

Given a fictional murder case, the agent decides on its own which suspect to interrogate, which evidence to search and which alibi to verify. It then names the culprit and produces a case report with the motive, the evidence chain and the reasons the other suspects were ruled out.

All characters and events in the case are fictional.

---

## Architecture & Workflow

```
User Request / Preset
         │
         ▼
Streamlit Interface (app.py)
         │
         ▼
Groq LLM (openai/gpt-oss-20b)
   │  ▲
   │  │ Tool Calls / Responses (MCP Protocol)
   ▼  │
CaseMCPServer
   ├── 1. get_case_file          ──► Victim, time of death, suspects
   ├── 2. interrogate_suspect    ──► Suspect's statement
   ├── 3. search_evidence        ──► Keyword search over evidence
   └── 4. check_alibi            ──► Claim vs. record comparison
         │
         ▼
Final Case Report (Culprit, Motive, Evidence Chain, Suspects Ruled Out)
```

## The Case: "The Midnight Server Room"

Dr. Rohan Kapoor, CEO of a software company, is found dead in his 14th-floor office. Time of death is between 9:00 PM and 10:00 PM. Four people are suspects. One of them has an alibi that looks true on the surface but breaks down when it is cross-checked with the records. The agent has to find it.

## How the Agent Loop Works

1. The goal and the system prompt are placed in a message list.
2. The messages and tool definitions are sent to the Groq model.
3. If the model asks for tools, the app runs them through `CaseMCPServer` and appends the results.
4. The history is sent back so the model can reason with real data.
5. When the model replies without tool calls, that reply is the final report.
6. A limit of 12 steps prevents endless loops.

The order of tool calls is chosen by the model at run time and is not hard-coded.

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Case data, MCP-style server, agent loop and Streamlit UI |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview |
| `.gitignore` | Keeps `.env` and local files out of version control |

## Setup

```bash
git clone <your-repository-link>
cd <your-repository-folder>
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key from https://console.groq.com. On Streamlit Cloud, add the same key under **Secrets** instead.

## Run

```bash
streamlit run app.py
```

## Limitations

- One small fictional case; this is a demonstration of agent architecture, not a forensic tool.
- `check_alibi` uses relationship flags stored in the case data, so part of the comparison is done by the tool and not by the model.
- Model output can vary slightly between runs.
- Needs an internet connection and a Groq API key.
