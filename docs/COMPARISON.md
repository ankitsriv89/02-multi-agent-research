# LangGraph vs AutoGen — head-to-head writeup

Both frameworks solve the same problem — coordinating multiple LLM agents — but from opposite philosophies. This project runs the **exact same four-agent research task** through both so the differences are directly observable. Below is the comparison from building and operating both.

## TL;DR

| Dimension | LangGraph 1.x | AutoGen v0.4 (0.7.x) |
|---|---|---|
| Routing model | **Explicit edges** in a `StateGraph` | **LLM-selected speaker** in a group chat |
| State | Typed `TypedDict`, reducer-merged | Implicit in the message transcript |
| Determinism | High — same inputs → same path | Lower — selector LLM may pick differently |
| Debuggability | Excellent (you wrote the graph) | OK (you read the transcript) |
| Code per agent | More boilerplate (node fn + edge) | Less (just an `AssistantAgent`) |
| Streaming primitives | `astream(stream_mode="updates"/"values"/"messages")` | `team.run_stream()` async iter of typed events |
| Termination | You hardcode it as a node | First-class `TerminationCondition` objects (composable) |
| Tool calls | Through `create_agent` middleware | `AssistantAgent(tools=[...])` + `reflect_on_tool_use` |
| When it shines | Branching workflows, retries, supervised loops | Open-ended brainstorming, dynamic team composition |
| When it hurts | Boilerplate for a linear pipeline | Hard to enforce a strict order |

## Concrete example: this project's critic-revision loop

The Critic either says `APPROVE` (done) or `REVISE` (back to Writer, up to N times). Implementing this loop highlights the philosophical split.

**LangGraph** (from [`backend/app/agents/langgraph_team.py`](../backend/app/agents/langgraph_team.py)):

```python
def route_after_critic(state):
    if state["critique"].upper().startswith("APPROVE"):
        return "finalize"
    if state["revisions"] >= settings.critic_max_revisions:
        return "finalize"
    return "writer"

g.add_conditional_edges("critic", route_after_critic,
                        {"writer": "writer", "finalize": "finalize"})
```

You can read this code and predict every possible execution path. The state schema (`ResearchState` TypedDict) is the contract.

**AutoGen** (from [`backend/app/agents/autogen_team.py`](../backend/app/agents/autogen_team.py)):

```python
termination = TextMentionTermination("APPROVE") | MaxMessageTermination(
    max_messages=10 + 2 * settings.critic_max_revisions
)
team = SelectorGroupChat(
    participants=[planner, researcher, writer, critic],
    model_client=model_client,
    termination_condition=termination,
)
```

The "loop until approved" is *emergent* — the selector LLM keeps choosing whoever should speak next based on the transcript, and termination fires when the right text appears. Composable termination conditions (`|`, `&`) are genuinely nice.

## What I learned building both

1. **For a known workflow, LangGraph is strictly better.** Our research task has a known shape (plan → research → write → critique). LangGraph encodes that shape; AutoGen has to discover it every run via the selector. About 1 in 10 AutoGen runs picked a weird speaker order (e.g. critic before writer had drafted anything). LangGraph never does this.

2. **For ad-hoc agent teams, AutoGen is faster to prototype.** Adding a fifth agent (e.g. a Fact-checker) to the AutoGen team is a 5-line change — instantiate the agent, add to `participants`, the selector handles routing. The LangGraph equivalent needs new node, new edge, new routing logic.

3. **Streaming UX is comparable.** Both expose async iterators of typed events. LangGraph's `stream_mode="updates"` returns one dict per node-completion, which maps cleanly to the agent-trace UI. AutoGen's `run_stream` yields fine-grained events (`TextMessage`, `ToolCallRequestEvent`, `ToolCallExecutionEvent`, terminal `TaskResult`) — slightly more work to flatten but you get tool-call boundaries for free.

4. **Cost & latency parity.** With the same Groq model on both, average latency was within 10% (LangGraph slightly faster because the selector LLM call is skipped). Per-request token cost was within 15% (AutoGen burns extra tokens on selector decisions).

5. **Deprecations matter.** `langgraph.prebuilt.create_react_agent` is deprecated in LangGraph 1.x; the v1 path is `langchain.agents.create_agent` (from the `langchain` package). AutoGen has had a major rewrite (0.2 → 0.4 line) — make sure tutorials you read match the version on PyPI.

## When I'd reach for which

| Need | Pick |
|---|---|
| Approval workflows, supervised pipelines, anything you'd draw on a whiteboard as a DAG | LangGraph |
| Brainstorming bots, dynamic teams, prototype-an-idea-in-an-hour | AutoGen |
| Mission-critical production agent with audit trail | LangGraph (the graph IS the audit trail) |
| Long-running agent that needs human-in-the-loop interrupts | LangGraph (built-in `interrupt_before`/`interrupt_after`) |
| Cross-platform chatbots (Slack/Teams/Discord) | Neither — use Vercel Chat SDK with AI SDK underneath |
