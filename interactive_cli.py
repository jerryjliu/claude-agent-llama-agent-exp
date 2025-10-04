import asyncio
import argparse
import json
from pathlib import Path
from typing import Any

from claude_code_sdk import (
    ClaudeSDKClient,
    ClaudeCodeOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
    HookMatcher,
    HookContext,
)


HUMAN_GATED_TOOLS = {"bash", "edit", "multiedit", "write", "notebookedit"}

# Optional advanced line editing via prompt_toolkit
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.enums import EditingMode

    _PTK_AVAILABLE = True
except Exception:
    _PTK_AVAILABLE = False


SYSTEM_PROMPT = (
    "You are an expert agentic workflow builder working within a reference template repository."
    " Your goal is to understand the user's high-level task and extend the Python workflow"
    " components to accomplish it, using best practices for modular, multi-step, LLM-powered flows."
    "\n\nRepository & Interaction Principles:"
    "\n- Modify the existing workflow files instead of creating new endpoints."
    "\n- That said, make sure by editing existing workflow files that you obey the general input/output interface (since this affects the interface with the frontend)."
    "\n- Modify Python files only; do not modify TypeScript files."
    "\n- Avoid changing shared variable shapes like extracted_data_collection to prevent frontend drift."
    "\n- Keep schema field types simple (primitives or lists). Nested lists are fine; avoid dict values."
    "\n- Prefer explicit failures over silent failures."
    "\n- Split logic into multiple workflow steps with clear inputs/outputs, rather than a single monolith."
    "\n- Favor LLM-powered flows over heavy heuristics for generalizable behavior. Use LlamaIndex OpenAI"
    " abstractions and PromptTemplates; use structured prediction utilities where helpful."
    "\n- For document parsing, parse text directly first; consider LlamaParse when needed."
    "\n- For web validation or enrichment, leverage appropriate web search integrations (e.g., LlamaIndex + Tavily)."
    "\n\nAssistant Behavior:"
    "\n- Ask clarifying questions only when essential; otherwise propose a concrete plan and implement edits."
    "\n- Keep outputs readable and summarize progress at the end."
    "\n- Do not make unrelated changes; preserve existing style and constraints."
)

async def read_user_input(prompt: str) -> str:
    """Read a single line of user input (async), using prompt_toolkit when available.

    - Provides Emacs-like keybindings (Alt word-jump, Home/End, etc.)
    - Persists history to a file
    - Falls back to input() if prompt_toolkit is unavailable
    """
    if _PTK_AVAILABLE:
        history_file = Path.home() / ".claude_code_cli_history"
        session = PromptSession(
            history=FileHistory(str(history_file)),
            editing_mode=EditingMode.EMACS,
        )
        try:
            # Use native asyncio integration
            return await session.prompt_async(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""

    # Fallback: standard input in a thread
    def _ask() -> str:
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""

    return await asyncio.to_thread(_ask)


async def prompt_yes_no(message: str) -> bool:
    """Prompt the user for a yes/no decision without blocking the event loop."""
    # Reuse prompt_toolkit if available for consistent input feel
    answer_raw = await read_user_input(message)
    answer = answer_raw.strip().lower()
    return answer in {"y", "yes"}


async def can_use_tool_gate(
    tool_name: str,
    tool_input: dict[str, Any],
    context: ToolPermissionContext,
) -> PermissionResultAllow | PermissionResultDeny:
    """Human-in-the-loop gate for risky tools.

    Returns allow/deny based on a simple yes/no confirmation for Bash/Edit/MultiEdit/Write.
    """
    tool_name_lc = (tool_name or "").lower()
    # Gate by exact name or substring to catch variations
    if (
        tool_name_lc in HUMAN_GATED_TOOLS
        or any(key in tool_name_lc for key in ("bash", "edit", "write", "notebook"))
    ):
        pretty_input = json.dumps(tool_input, indent=2, ensure_ascii=False)
        allow = await prompt_yes_no(
            f"\nPermission required: {tool_name} intends to run with input:\n{pretty_input}\nProceed? [y/N]: "
        )
        if allow:
            return PermissionResultAllow()
        return PermissionResultDeny(message=f"User denied {tool_name}")

    # Default allow for other tools
    return PermissionResultAllow()


async def pre_tool_hitl(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Hook: confirm before executing risky tools.

    Returns an object with decision='block' to cancel execution when denied.
    """
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    name_lc = str(tool_name).lower()
    if name_lc in HUMAN_GATED_TOOLS or any(k in name_lc for k in ("bash", "edit", "write", "notebook")):
        pretty_input = json.dumps(tool_input, indent=2, ensure_ascii=False)
        allow = await prompt_yes_no(
            f"\nHook permission required: {tool_name} intends to run with input:\n{pretty_input}\nProceed? [y/N]: "
        )
        if not allow:
            return {"decision": "block"}

    return {}


def print_assistant_message(message: AssistantMessage) -> None:
    for block in message.content:
        if isinstance(block, TextBlock):
            print(f"Claude: {block.text}")
        elif isinstance(block, ToolUseBlock):
            summarized_input = json.dumps(block.input, ensure_ascii=False)[:500]
            print(f"→ Using tool: {block.name} | input: {summarized_input}")
        elif isinstance(block, ToolResultBlock):
            # Content might be text or structured
            result_preview = (
                block.content if isinstance(block.content, str) else json.dumps(block.content, ensure_ascii=False)
            )
            preview = str(result_preview)[:500]
            status = "error" if block.is_error else "ok"
            print(f"✓ Tool result ({status}): {preview}")


async def interactive_session(skip_hitl: bool = False) -> None:
    # Anchor the working directory to the target project to give Claude Code project context
    project_root = (
        Path(__file__).resolve().parents[1] / "extraction-review-exp1-cc"
    )

    options_kwargs: dict[str, Any] = {
        "allowed_tools": [
            "Read",
            "Write",
            "Edit",
            "MultiEdit",
            "Bash",
            "Glob",
            "Grep",
        ],
        # "can_use_tool": can_use_tool_gate,  # optional alternate gate
        "include_partial_messages": True,
        "system_prompt": SYSTEM_PROMPT,
        "cwd": str(project_root),
    }

    if not skip_hitl:
        options_kwargs["hooks"] = {
            "PreToolUse": [
                HookMatcher(
                    matcher="Bash|Edit|MultiEdit|Write|NotebookEdit",
                    hooks=[pre_tool_hitl],
                )
            ]
        }

    options = ClaudeCodeOptions(**options_kwargs)

    session_id = "interactive-cli"

    async with ClaudeSDKClient(options=options) as client:
        status = (
            "HITL disabled — tools will run without confirmation."
            if skip_hitl
            else "HITL gate active for Bash/Edit/Write tools."
        )
        print(f"Connected. {status} Type your requests. Press Ctrl+C or send empty line to exit.\n")
        while True:
            try:
                user_prompt = (await read_user_input("You: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting…")
                break

            if not user_prompt:
                print("Goodbye.")
                break

            # Send prompt and stream response including tool activity
            await client.query(user_prompt, session_id=session_id)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    print_assistant_message(msg)
                elif isinstance(msg, ResultMessage):
                    duration = f"{msg.duration_ms} ms"
                    cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd is not None else "n/a"
                    print(f"--- Result (session={msg.session_id}) • time={duration} • cost={cost} ---\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Claude Code CLI")
    parser.add_argument(
        "--no-hitl",
        action="store_true",
        help="Disable human-in-the-loop prompts for tool execution",
    )
    args = parser.parse_args()

    try:
        asyncio.run(interactive_session(skip_hitl=bool(args.no_hitl)))
    except KeyboardInterrupt:
        # Graceful shutdown
        print("\nInterrupted. Bye.")


if __name__ == "__main__":
    main()


