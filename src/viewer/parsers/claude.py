"""Parse Claude Code stream-json transcripts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.viewer.events import Event, Session, TokenUsage, ToolCall, ToolResult


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def parse(path: Path) -> Session:
    events: list[Event] = []
    model = None
    session_id = None
    cwd = None
    final_text = None
    total = TokenUsage()

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = obj.get("type")
            if t == "system":
                model = obj.get("model") or model
                session_id = obj.get("session_id") or obj.get("sessionId") or session_id
                cwd = obj.get("cwd") or cwd
                if obj.get("subtype") == "init":
                    events.append(Event(
                        role="system",
                        text=f"session start  cwd={cwd}  model={model}  permissions={obj.get('permissionMode','?')}",
                        subtype="init", raw=obj,
                    ))
                elif obj.get("subtype") == "compact_boundary":
                    summary = obj.get("message", {}).get("summary") or obj.get("summary") or ""
                    events.append(Event(
                        role="compaction",
                        text=str(summary),
                        subtype="compact_boundary",
                        raw=obj,
                    ))
                else:
                    # other system events (file-history-snapshot etc.) — skip rendering by default
                    pass
                continue

            if t == "user":
                msg = obj.get("message") or {}
                content = msg.get("content")
                # User content can be a string (initial prompt) or a list of blocks
                # (tool_results sent back to the model on subsequent turns).
                if isinstance(content, str):
                    events.append(Event(
                        role="user", text=content,
                        timestamp=_parse_ts(obj.get("timestamp")),
                        is_sidechain=bool(obj.get("isSidechain")),
                        parent_uuid=obj.get("parentUuid"),
                        raw=obj,
                    ))
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tr_content = block.get("content")
                            if isinstance(tr_content, list):
                                tr_text = "\n".join(
                                    b.get("text", "") for b in tr_content if isinstance(b, dict)
                                )
                            else:
                                tr_text = str(tr_content) if tr_content is not None else ""
                            events.append(Event(
                                role="tool",
                                tool_result=ToolResult(
                                    content=tr_text,
                                    call_id=block.get("tool_use_id"),
                                    is_error=bool(block.get("is_error")),
                                ),
                                timestamp=_parse_ts(obj.get("timestamp")),
                                raw=obj,
                            ))
                        elif isinstance(block, dict) and block.get("type") == "text":
                            events.append(Event(role="user", text=block.get("text", ""), raw=obj))
                continue

            if t == "assistant":
                msg = obj.get("message") or {}
                model = msg.get("model") or model
                content = msg.get("content") or []
                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        bt = block.get("type")
                        if bt == "text":
                            text_parts.append(block.get("text", ""))
                        elif bt in {"thinking", "reasoning"}:
                            reasoning_parts.append(block.get("thinking") or block.get("text") or "")
                        elif bt == "tool_use":
                            tool_calls.append(ToolCall(
                                name=block.get("name", "?"),
                                args=block.get("input") or {},
                                call_id=block.get("id"),
                            ))

                usage_obj = msg.get("usage") or {}
                usage = TokenUsage(
                    input_tokens=usage_obj.get("input_tokens", 0),
                    output_tokens=usage_obj.get("output_tokens", 0),
                    cache_read_tokens=usage_obj.get("cache_read_input_tokens", 0),
                    cache_write_tokens=usage_obj.get("cache_creation_input_tokens", 0),
                )
                total.input_tokens += usage.input_tokens
                total.output_tokens += usage.output_tokens
                total.cache_read_tokens += usage.cache_read_tokens
                total.cache_write_tokens += usage.cache_write_tokens

                events.append(Event(
                    role="assistant",
                    text="\n".join(p for p in text_parts if p) or None,
                    reasoning="\n".join(reasoning_parts) or None,
                    tool_calls=tool_calls,
                    usage=usage,
                    model=model,
                    timestamp=_parse_ts(obj.get("timestamp")),
                    is_sidechain=bool(obj.get("isSidechain")),
                    parent_uuid=obj.get("parentUuid"),
                    raw=obj,
                ))
                if text_parts:
                    final_text = "\n".join(text_parts)
                continue

            if t == "result":
                final_text = obj.get("result") or final_text
                continue

    return Session(
        harness="claude",
        model=model,
        session_id=session_id,
        cwd=cwd,
        events=events,
        final_text=final_text,
        total_usage=total,
    )
