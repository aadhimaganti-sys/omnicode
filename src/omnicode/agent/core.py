"""The core Agent loop: handles tool execution, permissions, index, and state."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator

from omnicode.agent.diffing import apply_hunks, parse_unified_diff
from omnicode.agent.indexer import ProjectIndex
from omnicode.agent.permissions import ApprovalRequest, PermissionManager
from omnicode.agent.tools import ALL_TOOLS
from omnicode.paths import PathError, resolve_in_root
from omnicode.providers import LLMEngine, Message, StreamEvent


class ToolError(Exception):
    pass


class AgentState:
    def __init__(
        self,
        engine: LLMEngine,
        root: Path,
        system_prompt: str = "",
        permissions: PermissionManager | None = None,
    ):
        self.engine = engine
        self.root = root.resolve()
        self.messages: list[Message] = []
        self.system_prompt = system_prompt
        self.permissions = permissions or PermissionManager(None)
        self.index = ProjectIndex(self.root)
        self.max_steps = 25
        self._background_tasks: dict[int, subprocess.Popen] = {}

    async def step(self) -> AsyncIterator[StreamEvent]:
        """Execute one LLM turn, yielding text/tool_call events.
        If a tool is called, we process it and yield a special "tool_result" event
        (mapped to a notice) before returning."""
        self.index.ensure()
        sys = self._build_system()

        async for ev in self.engine.stream(self.messages, tools=ALL_TOOLS, system=sys):
            yield ev
            if ev.kind == "text" and ev.text:
                if not self.messages or self.messages[-1].role != "assistant":
                    self.messages.append(Message.assistant(ev.text))
                else:
                    self.messages[-1].content += ev.text
            elif ev.kind == "tool_call" and ev.tool_call:
                if not self.messages or self.messages[-1].role != "assistant":
                    self.messages.append(Message.assistant())
                self.messages[-1].tool_calls.append(ev.tool_call)

        # Execute any tools requested
        if self.messages and self.messages[-1].role == "assistant" and self.messages[-1].tool_calls:
            for tc in self.messages[-1].tool_calls:
                yield StreamEvent("notice", text=f"Executing tool {tc.name}...")
                try:
                    result = await self._execute_tool(tc.name, tc.arguments)
                    msg = Message.tool_result(tc.id, tc.name, result)
                except Exception as exc:
                    msg = Message.tool_result(tc.id, tc.name, str(exc), is_error=True)
                self.messages.append(msg)
                yield StreamEvent("notice", text=f"Result: {msg.content[:100]}...")

    async def run(self, prompt: str) -> AsyncIterator[StreamEvent]:
        """Run until the model stops or needs user input."""
        self.messages.append(Message.user(prompt))
        for _ in range(self.max_steps):
            stop = None
            async for ev in self.step():
                yield ev
                if ev.kind == "done":
                    stop = ev.stop_reason
            if stop != "tool_use":
                break
        else:
            yield StreamEvent("notice", text=f"Agent paused (hit {self.max_steps} step limit).")

    # ------------------------------------------------------------- tools
    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "read_file":
            p = resolve_in_root(self.root, args["path"])
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines(keepends=True)
            s = max(1, args.get("start", 1)) - 1
            e = min(len(lines), args.get("end", len(lines)))
            return "".join(lines[s:e])
        elif name == "list_dir":
            p = resolve_in_root(self.root, args["path"])
            if not p.is_dir():
                return f"Error: {args['path']} is not a directory."
            items = []
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
                if item.name.startswith(".git"): continue
                items.append(f"{item.name}/" if item.is_dir() else item.name)
            return "\n".join(items)
        elif name == "search_code":
            hits = self.index.search(args["query"])
            if not hits:
                return "No matches found."
            return "\n\n".join(f"{h.path}:{h.start}-{h.end} (score: {h.score})\n{h.snippet}" for h in hits)
        elif name == "run_command":
            cmd = args["command"]
            is_bg = args.get("background", False)
            if not await self.permissions.check(
                ApprovalRequest("command", cmd, cmd, command=cmd)
            ):
                raise ToolError(f"User denied permission to run: {cmd}")
            if is_bg:
                proc = subprocess.Popen(
                    cmd, shell=True, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                self._background_tasks[proc.pid] = proc
                return f"Started in background (PID {proc.pid})"
            else:
                proc = await asyncio.create_subprocess_shell(
                    cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                stdout, _ = await proc.communicate()
                out = stdout.decode("utf-8", errors="replace")
                return f"Exit {proc.returncode}:\n{out[:4000]}"
        elif name == "create_file":
            path = args["path"]
            content = args["content"]
            p = resolve_in_root(self.root, path)
            if p.exists():
                raise ToolError(f"{path} already exists. Use edit_file.")
            if not await self.permissions.check(ApprovalRequest("create", f"Create {path}", content)):
                raise ToolError(f"User denied creation of {path}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            self.index.invalidate()
            return f"Created {path}"
        elif name == "edit_file":
            diff = args["diff"]
            patches = parse_unified_diff(diff)
            out = []
            for patch in patches:
                if patch.path != args["path"]:
                    continue # Try to apply anyway if model got the header wrong but we know the target
                p = resolve_in_root(self.root, args["path"])
                if not p.exists():
                    raise ToolError(f"{args['path']} does not exist.")
                original = p.read_text(encoding="utf-8")
                new_text = apply_hunks(original, patch.hunks, patch.path)
                if not await self.permissions.check(ApprovalRequest("edit", f"Edit {args['path']}", diff)):
                    raise ToolError(f"User denied edits to {args['path']}")
                p.write_text(new_text, encoding="utf-8")
                self.index.invalidate()
                out.append(f"Applied patch to {args['path']}")
            if not out:
                raise ToolError("Diff did not contain hunks matching the target file.")
            return "\n".join(out)
        elif name == "ask_user":
            return f"Question asked: {args['question']}"
        raise ToolError(f"Unknown tool: {name}")

    def _build_system(self) -> str:
        prompt = [
            "You are OmniCode, an autonomous AI coding agent designed to act as a Principal Software Engineer.",
            "You have been assigned to work exclusively within the following workspace directory:",
            f"Workspace Root: {self.root}",
            "",
            "Rules & Persona:",
            "1. You have read/write access to this directory. Do not attempt to access files outside this workspace.",
            "2. Be proactive. If the user asks you to build a feature, use your tools to inspect the codebase, plan a solution, create/edit files, and run commands to test it.",
            "3. If a tool fails (e.g. command errors, patch fails to apply), read the error, correct your approach, and try again autonomously.",
            "4. Do not output raw code blocks and ask the user to copy-paste them. Use your `edit_file` and `create_file` tools to make the changes directly.",
            "",
            "Repository Map (Current state of your workspace):",
            self.index.repo_map(),
        ]
        if self.index.is_git_repo:
            status = self.index.git_status()
            if status:
                prompt.extend(["", "Git Status:", status])
        if self.system_prompt:
            prompt.extend(["", "Custom Instructions:", self.system_prompt])
        return "\n".join(prompt)
