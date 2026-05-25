import os
import urllib
from pathlib import Path

import yaml
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

# Load API key from config file and set environment variables
_config_path = Path(__file__).resolve().parents[3] / "src" / "config" / "application.yaml"
with open(_config_path) as f:
    _config = yaml.safe_load(f)
for _key, _value in _config.items():
    if _key.endswith("API_KEY"):
        _api_key = _value.strip()
        # Set the named key (e.g. OPENAI_API_KEY)
        os.environ[_key.split()[-1]] = _api_key
        # Also set provider-specific aliases so init_chat_model works
        os.environ.setdefault("DEEPSEEK_API_KEY", _api_key)

_fetched_text: str | None = None

@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL and store it for later processing.
    Returns a summary (byte count, encoding, first 500 chars) instead of the full text
    to avoid flooding the conversation. Use count_lines_containing or
    find_first_line_containing to query the stored text.
    """
    global _fetched_text
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    _fetched_text = text
    n_lines = text.count("\n")
    return (
        f"Stored {len(raw)} bytes ({n_lines} lines, encoding={resp.headers.get_content_charset() or 'utf-8'}).\n"
        f"First 500 chars:\n{text[:500]}"
    )

@tool
def count_lines_containing(substring: str) -> int:
    """Return how many lines of the fetched text contain `substring` (counts lines,
    not occurrences within a line). Call fetch_text_from_url first.
    """
    global _fetched_text
    if _fetched_text is None:
        raise ValueError("No text stored. Call fetch_text_from_url first.")
    return sum(1 for line in _fetched_text.split("\n") if substring in line)

@tool
def find_first_line_containing(substring: str) -> str:
    """Return the 1-based line number of the first line that contains `substring`,
    and the line text. Call fetch_text_from_url first.
    """
    global _fetched_text
    if _fetched_text is None:
        raise ValueError("No text stored. Call fetch_text_from_url first.")
    for i, line in enumerate(_fetched_text.split("\n"), 1):
        if substring in line:
            return f"Line {i}: {line[:200]}"
    return "Not found."

SYSTEM_PROMPT = """You are a literary data assistant.
## Capabilities
- `fetch_text_from_url`: loads document text from a URL. Returns a summary (size, first 500 chars) — the full text is stored internally.
- `count_lines_containing`: given a substring, returns how many lines of the fetched text contain it.
- `find_first_line_containing`: given a substring, returns the 1-based line number and text of the first line that contains it.
Do not guess line counts or positions — use the counting tools. Call fetch_text_from_url first, then query with the other tools."""


model = init_chat_model(
    "deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)
checkpointer = InMemorySaver()

if __name__ == '__main__':
    agent = create_agent(
        model=model,
        tools=[fetch_text_from_url, count_lines_containing, find_first_line_containing],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    deep_agent = create_deep_agent(
        model=model,
        tools=[fetch_text_from_url, count_lines_containing, find_first_line_containing],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    content = f"""Project Gutenberg hosts a full plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
    URL: https://www.gutenberg.org/files/64317/64317-0.txt
    
    Answer as much as you can:
    
    1) How many lines in the complete Gutenberg file contain the substring `Gatsby` (count lines, not occurrences within a line, each line ends with a line break).
    2) The 1-based line number of the first line in the file that contains `Daisy`.
    3) A two-sentence neutral synopsis.
    
    Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
    your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
    the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was."""

    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={
            "configurable": {"thread_id": "great-gatsby-lc"},
            "recursion_limit": 25,
        },
    )
    deep_agent_result = deep_agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={
            "configurable": {"thread_id": "great-gatsby-da"},
            "recursion_limit": 25,
        },
    )
    print(agent_result["messages"][-1].content_blocks)
    print("\n")
    print(deep_agent_result["messages"][-1].content_blocks)
