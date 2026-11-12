#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Skills usage example.

This example shows how to integrate the seven tools exposed by the skills/
layer into an LLM agent. Two usage modes are included:

  Mode A: simulated agent tool-call loop (no API key required, runnable directly)
  Mode B: real agent loop using the Anthropic API (requires ANTHROPIC_API_KEY)

Run:
    cd /path/to/DataForge

    # Mode A (no API key)
    python examples/curator/agent_skills_example.py

    # Mode B (API key required)
    ANTHROPIC_API_KEY=sk-ant-... python examples/curator/agent_skills_example.py --api
"""

import json
import os
import sys

# Ensure the project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Windows encoding patch (must run before importing Ray-related modules)
from curator.utils.windows_compat import patch_windows_unicode_logging
patch_windows_unicode_logging()

# ─── Import the Skills layer (all tool functions and tool JSON definitions) ──
from skills import (
    list_operations,
    describe_operation,
    validate_pipeline_config,
    execute_pipeline,
    get_tracking_report,
    get_data_stats,
    FUNCTION_TOOLS,
    get_tools_json,
)

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "token_count_sample.json")


# =============================================================================
# Tool dispatcher: call the function selected by the LLM's tool_name and tool_input
# =============================================================================

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    Dispatch a tool call returned by the LLM to the corresponding skill handler.

    Return a JSON string to send back to the LLM as the tool_result.
    """
    handlers = {
        "list_operations":          lambda i: list_operations(**i),
        "describe_operation":       lambda i: describe_operation(**i),
        "validate_pipeline_config": lambda i: validate_pipeline_config(**i),
        "execute_pipeline":         lambda i: execute_pipeline(**i),
        "resume_pipeline":          lambda i: __import__("skills").resume_pipeline(**i),
        "get_tracking_report":      lambda i: get_tracking_report(**i),
        "get_data_stats":           lambda i: get_data_stats(**i),
    }

    if tool_name not in handlers:
        result = {"error": f"Unknown tool: '{tool_name}'"}
    else:
        try:
            result = handlers[tool_name](tool_input)
        except Exception as exc:
            result = {"error": str(exc)}

    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


# =============================================================================
# Mode A: simulated agent tool-call loop (no API, for quick verification)
# =============================================================================

def run_simulated_agent():
    """
    Simulate the tool-call sequence for an agent completing one task:
      1. Inspect the data -> get_data_stats
      2. Discover available operations -> list_operations
      3. Inspect parameter details -> describe_operation
      4. Validate the configuration -> validate_pipeline_config
      5. Execute the pipeline -> execute_pipeline
      6. View the result -> get_tracking_report
    """
    print("=" * 65)
    print("  Mode A: Simulated agent tool-calling loop")
    print("=" * 65)
    print("Task: count tokens for each record in data/token_count_sample.json")
    print()

    # ── Step 1: Inspect the data file ─────────────────────────────────────────
    _agent_call(
        step=1,
        thought="First inspect the data file structure, then choose an appropriate operation",
        tool="get_data_stats",
        inputs={"data_path": DATA_FILE, "sample_size": 2},
    )

    # ── Step 2: List mapping operations ───────────────────────────────────────
    _agent_call(
        step=2,
        thought="The data has a content field and I need to count tokens. Let me inspect the available mapping operations.",
        tool="list_operations",
        inputs={"category": "mapping"},
    )

    # ── Step 3: Inspect token_count_mapping parameters ────────────────────────
    _agent_call(
        step=3,
        thought="Found token_count_mapping; inspect its required parameters",
        tool="describe_operation",
        inputs={"name": "token_count_mapping"},
    )

    # ── Step 4: Build and validate the pipeline configuration ─────────────────
    config = {
        "pipeline": {
            "operations": [
                {
                    "name": "text_loading",
                    "params": {"data_paths": [DATA_FILE]},
                },
                {
                    "name": "token_count_mapping",
                    "params": {
                        "text_field": "content",
                        "token_count_field": "token_count",
                        "char_count_field": "char_count",
                        "tokenizer": "char",   # Count by character (suitable for Chinese)
                        "print_stats": False,
                    },
                },
            ]
        }
    }

    _agent_call(
        step=4,
        thought="Configuration assembled; validate it before execution",
        tool="validate_pipeline_config",
        inputs={"config": config},
    )

    # ── Step 5: Execute the pipeline ──────────────────────────────────────────
    raw_result = _agent_call(
        step=5,
        thought="Configuration is valid; starting execution",
        tool="execute_pipeline",
        inputs={"config": config},
    )
    session_id = json.loads(raw_result).get("session_id")

    # ── Step 6: View the tracking report ──────────────────────────────────────
    if session_id:
        _agent_call(
            step=6,
            thought=f"Pipeline execution complete (session_id={session_id[:8]}...); see the detailed trace report",
            tool="get_tracking_report",
            inputs={"session_id": session_id},
        )

    print("\n✅  Simulated agent task complete!\n")


def _agent_call(step: int, thought: str, tool: str, inputs: dict) -> str:
    """Print the agent's reasoning and tool-call result."""
    print(f"── Step {step}: {tool} " + "─" * max(0, 55 - len(tool)))
    print(f"  💭 Agent thought: {thought}")
    print(f"  🔧 Calling tool: {tool}({json.dumps(inputs, ensure_ascii=False)})")
    result_str = dispatch_tool(tool, inputs)
    result = json.loads(result_str)

    # Compact output (truncate long fields)
    print(f"  📋 Tool result: ")
    _pretty_print(result, indent=4)
    print()
    return result_str


def _pretty_print(obj, indent=0):
    """Recursively print dict/list values while truncating long strings."""
    prefix = " " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}:")
                _pretty_print(v, indent + 2)
            else:
                v_str = str(v)
                if len(v_str) > 120:
                    v_str = v_str[:117] + "..."
                print(f"{prefix}{k}: {v_str}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                _pretty_print(item, indent)
                print()
            else:
                print(f"{prefix}- {item}")
    else:
        print(f"{prefix}{obj}")


# =============================================================================
# Mode B: real agent loop using the Anthropic API
# =============================================================================

def run_api_agent(user_task: str = None):
    """
    Run a real agent loop with the Anthropic API.

    The agent decides which tools to call and how many times to call them
    until the task is complete.

    Requires the environment variable: ANTHROPIC_API_KEY
    """
    try:
        import anthropic
    except ImportError:
        print("❌  Install the Anthropic SDK first: pip install anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌  Set the ANTHROPIC_API_KEY environment variable")
        return

    client = anthropic.Anthropic(api_key=api_key)

    if user_task is None:
        user_task = (
            f"Please analyze the data file {DATA_FILE},"
            "After inspecting the data structure, count tokens per record's content field using char tokenization,"
            "and summarize the results (total records, average tokens, etc.)."
        )

    print("=" * 65)
    print("  Mode B: Real Anthropic API agent loop")
    print("=" * 65)
    print(f"User task: {user_task}")
    print()

    messages = [{"role": "user", "content": user_task}]
    turn = 0

    # ── Agent loop ────────────────────────────────────────────────────────────
    while True:
        turn += 1
        print(f"── Agent Turn {turn} " + "─" * 45)

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=FUNCTION_TOOLS,       # Pass all seven skills
            messages=messages,
        )

        print(f"  stop_reason: {response.stop_reason}")

        # Collect all tool calls and text responses
        tool_uses = []
        text_blocks = []

        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append(block)
                print(f"  🔧 Calling tool: {block.name}({json.dumps(block.input, ensure_ascii=False)[:100]}...)")
            elif block.type == "text" and block.text:
                text_blocks.append(block.text)

        if text_blocks:
            print(f"  💬 Agent reply: {text_blocks[0][:200]}{'...' if len(text_blocks[0]) > 200 else ''}")

        # Append the assistant message to the conversation history
        messages.append({"role": "assistant", "content": response.content})

        # If there are no tool calls (stop_reason=end_turn), the agent is done
        if response.stop_reason == "end_turn" or not tool_uses:
            print("\n✅  Agent task complete!")
            if text_blocks:
                print("\nFinal response:")
                print(text_blocks[-1])
            break

        # Execute all tool calls and build the tool_result list
        tool_results = []
        for tool_use_block in tool_uses:
            result_content = dispatch_tool(tool_use_block.name, tool_use_block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": result_content,
            })
            print(f"  📋 {tool_use_block.name} returned: {result_content[:150]}...")

        # Send tool results back as a user message
        messages.append({"role": "user", "content": tool_results})
        print()


# =============================================================================
# Helper: print a summary of all available tools for quick Skills-layer onboarding
# =============================================================================

def print_tools_overview():
    """Print all seven tool names and descriptions for quick integration reference."""
    print("=" * 65)
    print("  Skills-layer tool overview")
    print("=" * 65)
    for tool in FUNCTION_TOOLS:
        required = tool["input_schema"].get("required", [])
        print(f"  [{tool['name']}]")
        # Take the first sentence of the description
        desc_first = tool["description"].split(".")[0] + "."
        print(f"    {desc_first}")
        if required:
            print(f"    Required parameters: {required}")
        print()

    print(f"Full JSON definitions: skills/function_tools.json (Anthropic format)")
    print(f"                    skills/function_tools_openai.json (OpenAI format)")
    print()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    use_api = "--api" in sys.argv

    if not os.path.exists(DATA_FILE):
        print(f"[error] Data file does not exist: {DATA_FILE}")
        sys.exit(1)

    # Print the tool overview first
    print_tools_overview()

    if use_api:
        # Mode B: real API agent loop
        run_api_agent()
    else:
        # Mode A: simulated agent loop (no API key required)
        run_simulated_agent()
