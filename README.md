---
title: Multi-Agent Research
emoji: 🔎
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: LangGraph vs AutoGen multi-agent research, side-by-side
---

# Multi-Agent Research

Production-grade research assistant that runs the **same task on two different agent frameworks side-by-side**: [LangGraph](https://github.com/langchain-ai/langgraph) (explicit state-machine routing) vs [AutoGen v0.4](https://github.com/microsoft/autogen) (LLM-routed conversation).

A four-agent team — Planner → Researcher → Writer → Critic — takes a research query, searches the web (Tavily), drafts a cited report, and iterates with a critic until approved. Toggle the framework in the sidebar and watch the agent trace in real time.

The portfolio centerpiece is the **head-to-head comparison** in [docs/COMPARISON.md](docs/COMPARISON.md).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Streamlit UI (port 7860)                                    │
│                                                             │
│  Sidebar: framework toggle (LangGraph / AutoGen)            │
│  Main:    live agent trace + cited report                   │
│           │                                                 │
│           ▼ direct async-iterator import                    │
│                                                             │
│  ┌─ LangGraph 1.x StateGraph ─────────────────┐             │
│  │ planner → researcher → writer → critic     │             │
│  │              ▲             │                │            │
│  │              └── REVISE loop (capped) ──────┤            │
│  └────────────────────────────────────────────┘             │
│                                                             │
│  ┌─ AutoGen 0.7.x SelectorGroupChat ──────────┐             │
│  │ LLM picks next speaker from transcript     │             │
│  │ termination: "APPROVE" OR max-messages     │             │
│  └────────────────────────────────────────────┘             │
│                                                             │
│  Shared tools:                                              │
│    • Tavily web_search                                      │
│    • httpx + readability web_fetch                          │
│                                                             │
│  LLM: Groq + openai/gpt-oss-120b                            │
│       (OpenAI fallback via LLM_PROVIDER=openai)             │
└─────────────────────────────────────────────────────────────┘
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Agent framework A | **LangGraph 1.2** | Explicit graph — shows routing/loops clearly |
| Agent framework B | **AutoGen 0.7** | LLM-routed group chat — opposite philosophy |
| Agent factory | `langchain.agents.create_agent` | v1 successor to deprecated `create_react_agent` |
| LLM | **Groq · `openai/gpt-oss-120b`** | OpenAI's open-weights model, reliable tool-call format |
| Web search | **Tavily** | LLM-optimized, free 1000 req/mo |
| Web fetch | httpx + readability-lxml | Clean article text, no JS rendering needed |
| UI | **Streamlit 1.41** | One process, native streaming via `st.status` |
| Container | Single Dockerfile, HF Spaces conventions | Deploy by `git push` |
| Testing | pytest + pytest-asyncio + respx + FakeListChatModel | Fully offline, mocks LLMs and HTTP |

## Why `openai/gpt-oss-120b` instead of llama-3.3?

The original setup used `llama-3.3-70b-versatile`. In practice it emits malformed tool calls like `<function=name {...}>` on multi-tool prompts, and Groq's server rejects them. `openai/gpt-oss-120b` is OpenAI's open-weights model hosted on Groq, trained specifically for the OpenAI tool-call format — it just works.

## Quick start (local)

### 1. Get API keys

* **Groq** — https://console.groq.com (free tier is enough for the demo)
* **Tavily** — https://app.tavily.com (1000 free searches/mo)
* **OpenAI** — only needed if you set `LLM_PROVIDER=openai`

### 2. Configure env

```bash
cp .env.example .env
# fill in GROQ_API_KEY and TAVILY_API_KEY
```

### 3. Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
# → http://localhost:8501
```

Or with Docker (matches HF Spaces exactly):

```bash
docker build -t multi-agent-research .
docker run --rm -p 7860:7860 --env-file .env multi-agent-research
# → http://localhost:7860
```

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space.
2. Pick **Docker** SDK.
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/multi-agent-research
   git push space main
   ```
4. Under Space → Settings → Variables and secrets, add:
   * `GROQ_API_KEY` (secret)
   * `TAVILY_API_KEY` (secret)
   * `OPENAI_API_KEY` (secret, optional)
5. The Space rebuilds and starts on port 7860 automatically.

Free CPU Spaces sleep after 48h of idle and take ~30s to wake. Fine for portfolio demos.

## Try it

In the UI:
1. Pick **LangGraph** or **AutoGen** in the sidebar.
2. Enter a query, e.g. *"What are the most cited LLM evaluation benchmarks in 2026?"*.
3. Watch the agent trace pane light up as each agent takes its turn and calls tools. The report pane fills in once the Critic approves.

Run the same query with each framework to feel the difference described in [docs/COMPARISON.md](docs/COMPARISON.md).

## Tests

```bash
pytest
```

Fully mocked LLM (`FakeListChatModel`) and HTTP (`respx`). Runs in <2s offline. Covers:
* `app/schemas.py` — StreamEvent invariants
* `app/tools/` — search and fetch with mocked responses
* `app/agents/langgraph_team.py` — end-to-end graph runs, including the revision loop and the cap

## Project layout

```
.
├── streamlit_app.py             ← Single entrypoint. Imports the teams directly.
├── app/
│   ├── config.py                ← pydantic-settings (reads .env or HF env vars)
│   ├── schemas.py               ← StreamEvent + Citation
│   ├── llm.py                   ← Groq/OpenAI factory for both frameworks
│   ├── agents/
│   │   ├── langgraph_team.py    ← StateGraph implementation
│   │   └── autogen_team.py      ← SelectorGroupChat implementation
│   └── tools/
│       ├── web_search.py        ← Tavily wrapper
│       └── web_fetch.py         ← httpx + readability
├── tests/
├── docs/
│   └── COMPARISON.md            ← LangGraph vs AutoGen writeup
├── Dockerfile                   ← HF Spaces conventions (port 7860, uid 1000)
├── requirements.txt
└── .env.example
```
