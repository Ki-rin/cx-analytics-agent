# CX Analytics Agent

A **LangGraph ReAct agent** that answers natural-language questions about the
[Bitext Customer Service](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
dataset. Supports structured queries, open-ended summaries, and out-of-scope filtering,
with persistent memory, a FastMCP server, and a Streamlit UI.

---

## Architecture

### Graph

```
                      ┌─────────────┐
          User ──────▶│ router_node │
                      └──────┬──────┘
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
   out_of_scope_node      call_model        summarize_node
   (polite refusal)    (structured ReAct)  (unstructured ReAct)
           │                  │  ▲               │  ▲
           │            tools?│  │         tools?│  │
           │                  ▼  │               ▼  │
           │              call_tools ◀───────call_tools
           │                  │                  │
           └──────────────────▼──────────────────┘
                       update_profile_node
                              │
                             END
```

The **router** (small model, one-word output) classifies every message as:
- `structured` — data query (counts, filters, examples) → `call_model` ReAct loop
- `unstructured` — open-ended narrative / summary → `summarize_node` ReAct loop
- `out_of_scope` — unrelated to the dataset → polite refusal

Both ReAct loops share the same `call_tools` node but use different system prompts.
A hard cap of **12 iterations** prevents infinite loops; if hit, the agent returns
a graceful fallback message.

### Models

| Role | Model | Why |
|------|-------|-----|
| Query routing | `google/gemma-2-9b-it-fast` | Single-word classification only; fastest available small model on Nebius |
| Reasoning & tool use | `Qwen/Qwen3-30B-A3B-Instruct` | Best-in-class instruction-following and multi-step tool chaining among Nebius Token Factory models; MoE architecture keeps latency reasonable despite large parameter count |
| Profile extraction | `google/gemma-2-9b-it-fast` | Lightweight JSON extraction from recent messages; speed matters more than reasoning depth here |

All models are served via the **Nebius Token Factory** OpenAI-compatible endpoint.

### Tools (7)

| Tool | When to use |
|------|-------------|
| `list_categories` | Discover available top-level categories |
| `list_intents` | List intents within a category |
| `count_records` | Count rows with optional category/intent filters |
| `get_examples` | Fetch formatted example rows (category + intent + keyword filters) |
| `get_intent_distribution` | Intent frequency table for a category |
| `get_category_summary_data` | Pull a raw sample for LLM-driven narrative summaries |
| `recommend_next_query` | Suggest a follow-up query without executing it (Bonus B) |

### Memory

- **Episodic / conversation**: `SqliteSaver` writes the full message history to
  `memory.db`. The same `--session` ID restores the conversation across restarts,
  enabling follow-up queries like *"show 3 more"* (the agent sees prior tool calls
  in its context and reuses the same filters).
- **User profile**: After each turn the small LLM extracts distilled facts
  (preferred categories, focus areas, etc.) and writes them to
  `profiles/<session_id>.json`. These facts are injected into the system prompt
  on the next turn. Answers *"What do you remember about me?"*.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Ki-rin/cx-analytics-agent.git
cd cx-analytics-agent

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. API key  ← required
export NEBIUS_API_KEY=your_nebius_token_factory_key
# Or add it to a .env file:
cp .env.example .env   # then edit .env
```

The dataset (~25 MB) is downloaded automatically from HuggingFace on first run.

---

## Running the CLI

```bash
python main.py                        # default session
python main.py --session alice        # named session — persists across restarts
```

The CLI prints every tool call and observation so you can follow the reasoning:

```
You: How many refund requests did we get?

[Tool Call] count_records({'intent': 'get_refund'})
[Observation] Records matching intent='get_refund': 210
Assistant: There were **210 refund requests** in the dataset.
```

Type `quit` or `exit` to leave the session.

### Session continuity example

```
# Session 1
python main.py --session alice
You: Show me 3 examples from the REFUND category
...
You: Show me 3 more
# Agent reuses category=REFUND filter automatically

# Restart — same session
python main.py --session alice
You: What about refunds vs complaints total?
# Full history is restored from memory.db
```

---

## MCP Server

```bash
python mcp_server.py
```

The server speaks standard MCP over **stdio** and exposes 7 tools:
`categories`, `intents`, `count`, `examples`, `intent_distribution`,
`category_summary`, `suggest_next_query`.

### Connecting a Python client

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # List tools
            tools = await session.list_tools()
            print([t.name for t in tools.tools])
            # Call a tool
            result = await session.call_tool("categories", {})
            print(result.content[0].text)

asyncio.run(main())
```

### Claude Desktop / MCP client config

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cx-analytics": {
      "command": "python",
      "args": ["/absolute/path/to/cx-analytics-agent/mcp_server.py"],
      "env": {
        "NEBIUS_API_KEY": "your_key_here"
      }
    }
  }
}
```

---

## Streamlit UI

```bash
streamlit run app_streamlit.py
```

Open `http://localhost:8501`. Features:
- Chat interface with streaming-style agent responses
- Expandable **Reasoning steps** panel (tool calls + observations) per turn
- Session ID input in the sidebar to switch between or resume conversations
- Live user profile display

---

## Example Queries

| Query | Type |
|-------|------|
| What categories exist in the dataset? | structured |
| How many refund requests did we get? | structured |
| Show me 5 examples of the SHIPPING category | structured |
| Show me 3 more | structured (uses last_filters) |
| What is the intent distribution in ACCOUNT? | structured |
| Summarise how agents respond to complaints | unstructured |
| Show examples of people wanting their money back | structured |
| What should I query next? | structured (recommender) |
| Who won the 2024 Champions League? | out-of-scope |
| Write me a poem about customer service | out-of-scope |
