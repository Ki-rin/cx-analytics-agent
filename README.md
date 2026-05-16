# CX Analytics Agent

A **LangGraph ReAct agent** that answers questions about the [Bitext Customer Service](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) dataset.  Structured queries, open-ended summaries, and out-of-scope filtering — all in one CLI, Streamlit UI, and MCP server.

---

## 1. Overview

The agent lets you explore the Bitext customer support LLM chatbot training dataset using natural language. It can:

- List categories and intents in the dataset
- Count records with optional filters
- Show example customer queries and agent responses
- Display intent distributions within a category
- Generate qualitative summaries of a category
- Suggest the next interesting query to run (query recommender)

---

## 2. Architecture

### Models

| Role | Model | Why |
|------|-------|-----|
| Query routing | `google/gemma-2-9b-it-fast` | Fast, low-latency classification; only needs to output one word |
| Reasoning & generation | `Qwen/Qwen3-30B-A3B-Instruct` | Strong instruction-following and tool-use for multi-step analysis |
| Profile extraction | `google/gemma-2-9b-it-fast` | Lightweight JSON extraction from recent messages |

All models are served through the **Nebius Token Factory** OpenAI-compatible endpoint (`https://api.tokenfactory.nebius.com/v1/`).

### Graph (LangGraph ReAct)

```
START → router_node → [dispatch]
              ├─ out_of_scope → out_of_scope_node → END
              └─ structured / unstructured → call_model ⇄ call_tools (ReAct loop, max 12 iters)
                                                    └─ update_profile_node → END
```

**Router** uses the small model to classify every incoming message as one of:
- `structured` — dataset query (counts, filters, examples)
- `unstructured` — open-ended analysis or narrative question
- `out_of_scope` — nothing to do with the dataset

**ReAct loop** alternates between calling the LARGE_MODEL (with tools bound) and executing tool calls via LangGraph's `ToolNode`. The iteration counter prevents infinite loops (max 12 iterations).

### Tools (7 total)

| Tool | Description |
|------|-------------|
| `list_categories` | List all unique top-level categories |
| `list_intents` | List intents within a category |
| `count_records` | Count records with optional category/intent filters |
| `get_examples` | Return formatted example rows (supports category, intent, keyword filters) |
| `get_intent_distribution` | Intent frequency table for a category |
| `get_category_summary_data` | Fetch a sample for LLM-driven narrative summarisation |
| `recommend_next_query` | Suggest a follow-up query without executing it (query recommender) |

### Memory

- **Episodic (conversation)**: `SqliteSaver` writes to `memory.db`; restored by `--session` ID across restarts.
- **User profile**: Per-session JSON files in `profiles/`; the small LLM extracts facts (preferred categories, interests) after each turn and merges them with the existing profile.

---

## 3. Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd cx-analytics-agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) set your API key via env var
cp .env.example .env
# Edit .env and set NEBIUS_API_KEY=your_key
# The key is also hard-coded as a fallback, so this step is optional.
```

The dataset is downloaded automatically from HuggingFace on first run (~25 MB).

---

## 4. Running the CLI

```bash
python main.py                       # default session
python main.py --session alice       # named session (persists across restarts)
```

The CLI prints every tool call and observation so you can follow the agent's reasoning:

```
You: How many refund requests did we get?

[Tool Call] count_records({'intent': 'get_refund'})
[Observation] Records matching intent='get_refund': 210
Assistant: There were **210 refund requests** in the dataset.
```

Type `quit` to exit the loop.

---

## 5. MCP Server

```bash
python mcp_server.py
```

The server exposes 6 tools over stdio (standard MCP transport). To connect a client:

### Using the MCP Python client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

async def main():
    params = StdioServerParameters(command="python", args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # List available tools
            tools = await session.list_tools()
            print([t.name for t in tools.tools])
            # Call a tool
            result = await session.call_tool("categories", {})
            print(result.content[0].text)

asyncio.run(main())
```

Available MCP tools: `categories`, `intents`, `count`, `examples`, `intent_distribution`, `category_summary`, `suggest_next_query`.

---

## 6. Streamlit UI

```bash
streamlit run app_streamlit.py
```

Open your browser at `http://localhost:8501`. Features:
- Chat interface with agent responses
- Expandable "Reasoning steps" sections showing tool calls and observations
- Session ID input in the sidebar to resume past conversations

---

## 7. Example Queries

| Query | Type |
|-------|------|
| What categories exist in the dataset? | structured |
| How many refund requests did we get? | structured |
| Show me 5 examples of the SHIPPING category | structured |
| What is the intent distribution in ACCOUNT? | structured |
| Summarise how agents respond to complaints | unstructured |
| Show examples of people wanting their money back | structured |
| What should I query next? | structured (recommender) |
| Who won the 2024 Champions League? | out-of-scope |
| Write me a poem about customer service | out-of-scope |
