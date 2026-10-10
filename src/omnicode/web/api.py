"""OmniCode Web IDE Backend (FastAPI + WebSockets + File Management)."""

from __future__ import annotations

import asyncio
import os
import json
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from omnicode.agent.core import AgentState
from omnicode.agent.permissions import ApprovalRequest, PermissionManager
from omnicode.config import Config
from omnicode.paths import PathError, resolve_in_root
from omnicode.providers.engine import LLMEngine, NotConfiguredError

app = FastAPI(title="OmniCode Studio")

# Global workspace reference for web server
WORKSPACE_DIR: Path = Path.cwd()


def set_workspace(path: Path) -> None:
    global WORKSPACE_DIR
    WORKSPACE_DIR = path.resolve()
    if not WORKSPACE_DIR.exists():
        WORKSPACE_DIR.mkdir(parents=True)


class SaveFileRequest(BaseModel):
    path: str
    content: str


class SetKeyRequest(BaseModel):
    provider: str
    api_key: str


class SetProviderRequest(BaseModel):
    provider: str


@app.get("/api/workspace")
async def get_workspace_tree():
    """Returns a directory tree of the workspace."""
    def build_tree(current: Path, max_depth: int = 4, depth: int = 0) -> list[dict[str, Any]]:
        if depth >= max_depth:
            return []
        items = []
        try:
            for entry in sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if entry.name.startswith((".", "__pycache__", "venv", "node_modules")):
                    continue
                rel_path = str(entry.relative_to(WORKSPACE_DIR))
                if entry.is_dir():
                    items.append({
                        "name": entry.name,
                        "path": rel_path,
                        "type": "directory",
                        "children": build_tree(entry, max_depth, depth + 1)
                    })
                else:
                    items.append({
                        "name": entry.name,
                        "path": rel_path,
                        "type": "file",
                        "size": entry.stat().st_size
                    })
        except Exception:
            pass
        return items

    return {
        "root": str(WORKSPACE_DIR),
        "tree": build_tree(WORKSPACE_DIR)
    }


@app.get("/api/file")
async def get_file_content(path: str = Query(...)):
    """Read a file safely from the workspace."""
    try:
        target = resolve_in_root(WORKSPACE_DIR, path)
        if not target.is_file():
            raise HTTPException(status_code=400, detail="Target is not a file")
        content = target.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "content": content}
    except PathError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/file")
async def save_file_content(req: SaveFileRequest):
    """Save a file safely in the workspace."""
    try:
        target = resolve_in_root(WORKSPACE_DIR, req.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(req.content, encoding="utf-8")
        return {"status": "ok", "path": req.path}
    except PathError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_config():
    cfg = Config.load()
    return {
        "active": cfg.active,
        "available_providers": list(cfg.providers.keys()),
        "has_keys": {k: bool(v.api_key) for k, v in cfg.providers.items()}
    }


@app.post("/api/config/key")
async def set_api_key(req: SetKeyRequest):
    cfg = Config.load()
    cfg.set_api_key(req.provider, req.api_key)
    cfg.save()
    return {"status": "ok", "provider": req.provider}


@app.post("/api/config/provider")
async def switch_provider(req: SetProviderRequest):
    cfg = Config.load()
    cfg.active = req.provider
    cfg.save()
    return {"status": "ok", "active": cfg.active}


@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    await websocket.accept()
    cfg = Config.load()

    try:
        engine = LLMEngine.from_config(cfg)
    except NotConfiguredError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    # Pending approval requests
    pending_approvals: dict[str, asyncio.Future[bool | str]] = {}

    async def web_approver(req: ApprovalRequest) -> bool | str:
        req_id = f"req_{id(req)}"
        fut = asyncio.get_event_loop().create_future()
        pending_approvals[req_id] = fut

        await websocket.send_json({
            "type": "approval_request",
            "id": req_id,
            "title": req.title,
            "detail": req.detail,
            "kind": req.kind
        })

        try:
            res = await fut
            return res
        finally:
            pending_approvals.pop(req_id, None)

    perms = PermissionManager(web_approver, mode=cfg.permissions.mode, allowed_commands=cfg.permissions.allowed_commands)
    state = AgentState(engine, WORKSPACE_DIR, system_prompt=cfg.system_prompt, permissions=perms)

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            action = data.get("type")
            if action == "prompt":
                user_prompt = data.get("prompt", "")
                await websocket.send_json({"type": "status", "status": "running"})

                try:
                    async for ev in state.run(user_prompt):
                        if ev.kind == "text":
                            await websocket.send_json({"type": "text", "text": ev.text})
                        elif ev.kind == "notice":
                            await websocket.send_json({"type": "notice", "text": ev.text})
                        elif ev.kind == "tool_call" and ev.tool_call:
                            await websocket.send_json({
                                "type": "tool_call",
                                "name": ev.tool_call.name,
                                "arguments": ev.tool_call.arguments
                            })
                    await websocket.send_json({"type": "done"})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif action == "approval_response":
                req_id = data.get("id")
                response_val = data.get("response") # "yes", "no", "always"
                if req_id in pending_approvals and not pending_approvals[req_id].done():
                    if response_val == "always":
                        pending_approvals[req_id].set_result("always")
                    elif response_val == "yes":
                        pending_approvals[req_id].set_result(True)
                    else:
                        pending_approvals[req_id].set_result(False)

    except WebSocketDisconnect:
        pass
    finally:
        await engine.aclose()


from fastapi.responses import FileResponse

@app.get("/api/preview")
async def preview_file(path: str = "index.html"):
    """Serve HTML or assets from workspace for live preview in the browser."""
    try:
        target = resolve_in_root(WORKSPACE_DIR, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(target)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_gui():
    """Serves the OmniCode Studio IDE Frontend."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>OmniCode GUI index.html not found</h1>", status_code=404)

