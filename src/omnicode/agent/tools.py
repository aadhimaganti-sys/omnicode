"""Agent Tool definitions (JSON Schema format)."""

from __future__ import annotations

from omnicode.providers import ToolSpec

read_file = ToolSpec(
    "read_file",
    "Read a file from the workspace. Use slice notation (start:end) for lines.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to file"},
            "start": {"type": "integer", "description": "Start line (1-indexed)"},
            "end": {"type": "integer", "description": "End line (inclusive)"},
        },
        "required": ["path"],
    },
)

list_dir = ToolSpec(
    "list_dir",
    "List contents of a directory.",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Relative path ('.' for root)"}},
        "required": ["path"],
    },
)

search_code = ToolSpec(
    "search_code",
    "Fuzzy search the codebase for keywords or identifiers.",
    {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for"}},
        "required": ["query"],
    },
)

run_command = ToolSpec(
    "run_command",
    "Execute a shell command. Use for build, test, git, or checking the environment.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "background": {"type": "boolean", "description": "Run in background (e.g. dev server)"},
        },
        "required": ["command"],
    },
)

edit_file = ToolSpec(
    "edit_file",
    "Edit a file using a unified diff format. Replace matched blocks with new text.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "diff": {
                "type": "string",
                "description": "Unified diff text (start with --- a/... \\n +++ b/...)",
            },
        },
        "required": ["path", "diff"],
    },
)

create_file = ToolSpec(
    "create_file",
    "Create a new file with the given content.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to create"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "content"],
    },
)

ask_user = ToolSpec(
    "ask_user",
    "Ask the user a question when you cannot proceed or need a decision.",
    {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "Question to ask"}},
        "required": ["question"],
    },
)

ALL_TOOLS = [
    read_file,
    list_dir,
    search_code,
    run_command,
    edit_file,
    create_file,
    ask_user,
]
