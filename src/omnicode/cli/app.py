"""Interactive Terminal / CLI App (Typer + Rich + asyncio)."""

import asyncio
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from omnicode.agent.core import AgentState
from omnicode.agent.permissions import ApprovalRequest, PermissionManager
from omnicode.config import Config
from omnicode.providers.engine import LLMEngine, NotConfiguredError

console = Console()
app = typer.Typer(help="OmniCode - AI Coding Assistant")

async def cli_approver(req: ApprovalRequest) -> bool | str:
    print()
    console.print(Panel(req.detail, title=f"[bold red]Requires Approval: {req.title}[/]", border_style="red"))
    ans = console.input("Allow? [y/n/always/a] (default: n): ").lower()
    if ans in ("always", "a"):
        return "always"
    return ans == "y"


async def async_repl(cfg: Config, workspace: Path, initial_prompt: str | None = None):
    try:
        engine = LLMEngine.from_config(cfg)
    except NotConfiguredError as exc:
        console.print(f"[bold red]Configuration Error:[/bold red] {exc}")
        sys.exit(1)

    if not workspace.exists():
        workspace.mkdir(parents=True)
    
    perms = PermissionManager(cli_approver, mode=cfg.permissions.mode, allowed_commands=cfg.permissions.allowed_commands)
    state = AgentState(engine, workspace, system_prompt=cfg.system_prompt, permissions=perms)

    console.print(f"[green]OmniCode initialized in workspace: {workspace}[/green]")
    if engine.primary.name == "coder" or getattr(engine.primary.provider, "type_name", "") == "coder":
        console.print("[bold cyan]⚡ OmniCode Coder Active (Cloud Neural Engine | Zero Local CPU / No Overheating)[/bold cyan]")
    console.print(f"[dim]Provider: {engine.label} | Mode: {perms.mode}[/dim]")
    console.print("Type 'exit' to quit.\n")

    session = PromptSession()
    prompt = initial_prompt
    
    try:
        while True:
            if not prompt:
                with patch_stdout():
                    try:
                        prompt = await session.prompt_async("\nomnicode> ")
                    except (EOFError, KeyboardInterrupt):
                        break
                        
            if prompt.lower() in ("exit", "quit"):
                break
            if not prompt.strip():
                prompt = None
                continue

            with Live(console=console, refresh_per_second=10) as live:
                current_text = ""
                try:
                    async for ev in state.run(prompt):
                        if ev.kind == "text":
                            current_text += ev.text
                            live.update(Markdown(current_text))
                        elif ev.kind == "notice":
                            live.console.print(f"[dim italic]{ev.text}[/dim italic]")
                        elif ev.kind == "tool_call":
                            if ev.tool_call:
                                live.console.print(f"[cyan]Calling {ev.tool_call.name}...[/cyan]")
                except Exception as e:
                    live.console.print(f"[red]Error:[/red] {e}")
            prompt = None
    finally:
        await engine.aclose()


@app.command()
def repl(
    prompt_args: list[str] = typer.Argument(None, help="Initial prompt"),
    workspace: Path = typer.Option(Path.cwd(), "-w", "--workspace", help="Path to the workspace folder"),
):
    """Start the interactive REPL session (Default)."""
    cfg = Config.load()
    prompt_text = " ".join(prompt_args) if prompt_args else None
    asyncio.run(async_repl(cfg, workspace, initial_prompt=prompt_text))

@app.command()
def coder(
    prompt_args: list[str] = typer.Argument(None, help="Initial coding prompt for Coder model"),
    workspace: Path = typer.Option(Path.cwd(), "-w", "--workspace", help="Path to the workspace folder"),
):
    """Run OmniCode using the specialized Coder model (Cloud-accelerated, zero Pi heat)."""
    cfg = Config.load()
    cfg.active = "coder"
    prompt_text = " ".join(prompt_args) if prompt_args else None
    asyncio.run(async_repl(cfg, workspace, initial_prompt=prompt_text))

@app.command()
def config():
    """View current configuration."""
    cfg = Config.load()
    console.print_json(data=cfg.to_dict())

@app.command()
def key(provider: str, api_key: str):
    """Set the API key for a provider."""
    cfg = Config.load()
    cfg.set_api_key(provider, api_key)
    cfg.save()
    console.print(f"[green]Saved API key for {provider}.[/green]")

@app.command()
def use(provider: str):
    """Switch active provider."""
    cfg = Config.load()
    cfg.active = provider
    cfg.save()
    console.print(f"[green]Switched to {provider}.[/green]")

@app.command()
def model(provider: str, model_name: str):
    """Set the model name for a provider (e.g. `omnicode model ollama qwen2.5-coder:1.5b`)."""
    cfg = Config.load()
    cfg.set_model(provider, model_name)
    cfg.save()
    console.print(f"[green]Set model for {provider} to '{model_name}'.[/green]")

@app.command()
def endpoint(provider: str, base_url: str):
    """Set the API endpoint / base URL for a provider (e.g. `omnicode endpoint ollama http://localhost:11434/v1`)."""
    cfg = Config.load()
    cfg.set_base_url(provider, base_url)
    cfg.save()
    console.print(f"[green]Set endpoint for {provider} to '{base_url}'.[/green]")

@app.command()
def ollama(
    prompt_args: list[str] = typer.Argument(None, help="Initial prompt for Ollama"),
    workspace: Path = typer.Option(Path.cwd(), "-w", "--workspace", help="Path to the workspace folder"),
):
    """Run OmniCode using Ollama (local or networked)."""
    cfg = Config.load()
    cfg.active = "ollama"
    prompt_text = " ".join(prompt_args) if prompt_args else None
    asyncio.run(async_repl(cfg, workspace, initial_prompt=prompt_text))

@app.command()
def deepseek(
    prompt_args: list[str] = typer.Argument(None, help="Initial prompt for DeepSeek"),
    workspace: Path = typer.Option(Path.cwd(), "-w", "--workspace", help="Path to the workspace folder"),
):
    """Run OmniCode using DeepSeek (deepseek-chat or deepseek-reasoner)."""
    cfg = Config.load()
    cfg.active = "deepseek"
    prompt_text = " ".join(prompt_args) if prompt_args else None
    asyncio.run(async_repl(cfg, workspace, initial_prompt=prompt_text))

@app.command()
def gui(
    workspace: Path = typer.Option(Path.cwd(), "-w", "--workspace", help="Path to the workspace folder"),
    port: int = typer.Option(8000, "-p", "--port", help="Port to host the Web IDE"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host address"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not automatically open the browser"),
):
    """Launch the OmniCode Studio Web IDE / GUI."""
    import uvicorn
    import webbrowser
    from omnicode.web.api import app as fastapi_app, set_workspace

    set_workspace(workspace)
    console.print(f"[bold cyan]⚡ Launching OmniCode Studio IDE[/bold cyan]")
    console.print(f"[dim]Workspace:[/] {workspace.resolve()}")
    console.print(f"[dim]URL:[/] [link=http://{host}:{port}]http://{host}:{port}[/link]\n")

    if not no_browser:
        try:
            webbrowser.open(f"http://{host}:{port}")
        except Exception:
            pass

    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, workspace: Path = typer.Option(Path.cwd(), "-w", "--workspace", help="Path to the workspace folder")):
    """OmniCode CLI"""
    if ctx.invoked_subcommand is None:
        cfg = Config.load()
        asyncio.run(async_repl(cfg, workspace))

def run() -> None:
    known = {"repl", "coder", "ollama", "deepseek", "config", "key", "use", "model", "endpoint", "gui", "--help", "-h", "-w", "--workspace"}
    args = sys.argv[1:]
    if args and args[0] not in known and not args[0].startswith("-"):
        sys.argv.insert(1, "repl")
    app()


def run_coder() -> None:
    """Standalone entrypoint for `coder` executable command."""
    sys.argv.insert(1, "coder")
    app()

if __name__ == "__main__":
    run()
