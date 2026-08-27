"""Pluggable LLM backend for every agentic component (the Category D
shopping-agent sandbox, the red-team mutation loop, the Identify Agent's
proposal drafting).

Three providers behind one interface, selected by JANUS_LLM_BACKEND or
auto-detected:

    "openai"    real OpenAI chat-completions + native tool calling. Used
                whenever OPENAI_API_KEY is set and JANUS_LLM_BACKEND isn't
                forced to something else. Ported near-unchanged from the
                predecessor project's aegis/common/llm.py.
    "local"     a small instruct model pulled from the HuggingFace Hub
                (no account, no token, public weights over anonymous
                HTTPS) and run on CPU. Real local inference, not a stub.
    "scripted"  deterministic, no model at all. Default when neither of
                the above is available or explicitly requested; this is
                what tests, CI, and a fresh credential-less clone run on.

All three implement the same ChatBackend.chat() surface so callers never
branch on which provider is active; the difference is entirely inside
get_backend()'s selection logic.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
RED_TEAM_MODEL = "gpt-5.5"
SHOPPING_AGENT_MODEL = "gpt-5-mini"
LOCAL_MODEL_ID = os.environ.get("JANUS_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@dataclass
class LLMCallLog:
    provider: str
    model: str
    role_tag: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


CALL_LOG: list[LLMCallLog] = []


class ChatBackend(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict],
        *,
        role_tag: str,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> "ChatResult": ...


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


def call_log_summary() -> dict:
    by_role: dict[str, dict] = {}
    for entry in CALL_LOG:
        bucket = by_role.setdefault(
            entry.role_tag, {"calls": 0, "total_latency_ms": 0.0, "providers": set()}
        )
        bucket["calls"] += 1
        bucket["total_latency_ms"] += entry.latency_ms
        bucket["providers"].add(entry.provider)
    for bucket in by_role.values():
        bucket["avg_latency_ms"] = bucket["total_latency_ms"] / bucket["calls"]
        bucket["providers"] = sorted(bucket["providers"])
    return by_role


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class OpenAIBackend:
    name = "openai"

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set, copy .env.example to .env and fill it in.")
        from openai import OpenAI

        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def chat(
        self,
        messages: list[dict],
        *,
        role_tag: str,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> ChatResult:
        model = RED_TEAM_MODEL if role_tag == "red_team" else SHOPPING_AGENT_MODEL
        started = time.monotonic()
        kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        CALL_LOG.append(
            LLMCallLog(provider=self.name, model=model, role_tag=role_tag, latency_ms=(time.monotonic() - started) * 1000)
        )
        choice = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments or "{}"))
            for tc in (choice.tool_calls or [])
        ]
        return ChatResult(content=choice.content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Local HuggingFace provider: real inference, no credentials
# ---------------------------------------------------------------------------


def _flatten_for_local_template(messages: list[dict]) -> list[dict]:
    """Collapses OpenAI-shaped tool-call/tool-result messages into plain
    role/content pairs so a small local model's chat template, which may
    not understand the "tool_calls" assistant field or the "tool" role --
    still gets a coherent transcript instead of a template error."""

    flattened: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "tool":
            flattened.append({"role": "user", "content": f"[tool result] {m.get('content', '')}"})
        elif m.get("tool_calls"):
            calls_desc = "; ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})" for tc in m["tool_calls"]
            )
            content = (m.get("content") or "") + f"\n[called: {calls_desc}]"
            flattened.append({"role": "assistant", "content": content})
        else:
            flattened.append({"role": role, "content": m.get("content") or ""})
    return flattened


class LocalBackend:
    """Runs a small instruct model on CPU. Tool calling isn't native at this
    model scale, so tool selection is emulated: the prompt instructs the
    model to respond with a single JSON object naming the tool and
    arguments, which is then parsed. This is slower and less reliable than
    OpenAI's native tool calling, documented as a real limitation, not
    hidden, but it is genuine local model inference, not a scripted
    stand-in."""

    name = "local"

    def __init__(self, model_id: str = LOCAL_MODEL_ID) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        self._model.eval()

    def chat(
        self,
        messages: list[dict],
        *,
        role_tag: str,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> ChatResult:
        prompt_messages = list(messages)
        if tools:
            tool_desc = json.dumps(
                [{"name": t["function"]["name"], "description": t["function"].get("description", "")} for t in tools],
                indent=2,
            )
            prompt_messages = prompt_messages + [
                {
                    "role": "system",
                    "content": (
                        "You must respond with exactly one JSON object of the form "
                        '{"tool": "<name>", "arguments": {...}} choosing one of these tools:\n'
                        f"{tool_desc}\nRespond with ONLY the JSON object, nothing else."
                    ),
                }
            ]
        elif response_format == "json":
            prompt_messages = prompt_messages + [
                {"role": "system", "content": "Respond with ONLY a single valid JSON object, nothing else."}
            ]

        sanitized = _flatten_for_local_template(prompt_messages)
        try:
            text = self._tokenizer.apply_chat_template(sanitized, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Small instruct-model chat templates vary in which roles/keys
            # they accept; fall back to a plain role-labeled transcript
            # rather than crashing the whole session on a template quirk.
            text = "\n".join(f"[{m['role']}] {m.get('content') or ''}" for m in sanitized) + "\n[assistant]"
        inputs = self._tokenizer(text, return_tensors="pt")
        started = time.monotonic()
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        CALL_LOG.append(
            LLMCallLog(provider=self.name, model=LOCAL_MODEL_ID, role_tag=role_tag, latency_ms=(time.monotonic() - started) * 1000)
        )
        generated = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        if tools:
            match = re.search(r"\{.*\}", generated, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    return ChatResult(
                        content=None,
                        tool_calls=[ToolCall(id="local-0", name=parsed["tool"], arguments=parsed.get("arguments", {}))],
                    )
                except (json.JSONDecodeError, KeyError):
                    pass
        elif response_format == "json":
            match = re.search(r"\{.*\}", generated, re.DOTALL)
            if match:
                try:
                    json.loads(match.group(0))  # validate only; caller does its own parsing
                    return ChatResult(content=match.group(0))
                except json.JSONDecodeError:
                    pass
        return ChatResult(content=generated)


# ---------------------------------------------------------------------------
# Scripted provider: deterministic, no model, no download
# ---------------------------------------------------------------------------


class ScriptedBackend:
    """No model at all. Callers that need agentic behavior under this
    backend should prefer the dedicated scripted policies in
    janus/generate/agentic/policies.py over calling .chat() directly --
    this class exists mainly so the ChatBackend interface has a uniform
    no-op implementation for code paths that fall back to it."""

    name = "scripted"

    def chat(
        self,
        messages: list[dict],
        *,
        role_tag: str,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        temperature: float = 1.0,
        response_format: str | None = None,
    ) -> ChatResult:
        started = time.monotonic()
        CALL_LOG.append(LLMCallLog(provider=self.name, model="none", role_tag=role_tag, latency_ms=(time.monotonic() - started) * 1000))
        return ChatResult(content="[scripted backend: no free-text generation available]")


_backend_cache: ChatBackend | None = None


def get_backend(force: str | None = None) -> ChatBackend:
    global _backend_cache
    choice = force or os.environ.get("JANUS_LLM_BACKEND")
    if choice is None:
        choice = "openai" if OPENAI_API_KEY else "scripted"

    if _backend_cache is not None and _backend_cache.name == choice:
        return _backend_cache

    if choice == "openai":
        _backend_cache = OpenAIBackend()
    elif choice == "local":
        _backend_cache = LocalBackend()
    elif choice == "scripted":
        _backend_cache = ScriptedBackend()
    else:
        raise ValueError(f"unknown JANUS_LLM_BACKEND {choice!r} (expected openai/local/scripted)")
    return _backend_cache
