"""Colab-only fixed-provider transport retaining numeric usage on failures."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def checked_import(name, directory):
    expected = (directory / (name + ".py")).resolve()
    if name in sys.modules and Path(sys.modules[name].__file__).resolve() != expected:
        raise RuntimeError("Foreign cached module rejected; use a fresh Colab subprocess")
    sys.path.insert(0, str(directory))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError("Imported module origin differs from selected checkout")
    return module


def numeric_usage(value):
    if not isinstance(value, dict):
        return {}
    return {k: value[k] for k in TOKEN_FIELDS if type(value.get(k)) is int and value[k] >= 0}


class ResponseFailure(RuntimeError):
    def __init__(self, kind, *, usage=None, latency=None, fatal=False):
        super().__init__(kind)
        self.failure_kind = kind
        self.reported_usage = numeric_usage(usage)
        self.latency_seconds = latency
        self.fatal = fatal


def parse_received(payload, *, allowed_model, latency):
    """Pure response validation, also usable with tiny invented test payloads."""
    usage = numeric_usage(payload.get("usage")) if isinstance(payload, dict) else {}
    if not isinstance(payload, dict):
        raise ResponseFailure("invalid_response_shape", usage=usage, latency=latency, fatal=True)
    if payload.get("model") != allowed_model:
        raise ResponseFailure("model_identity_failure", usage=usage, latency=latency, fatal=True)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ResponseFailure("invalid_response_shape", usage=usage, latency=latency, fatal=True)
    choice = choices[0]
    content = choice.get("message", {}).get("content") if isinstance(choice.get("message"), dict) else None
    if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
        raise ResponseFailure("incomplete_response", usage=usage, latency=latency)
    return {"hypothesis": content, "provider_model": allowed_model, "finish_reason": "stop", "usage": usage, "latency_seconds": latency}


def generate_response(messages, *, model, allowed_model, api_base, key, max_response_tokens):
    guard = checked_import("colab_provider", ROOT / "app")
    guard.require_colab()
    base = checked_import("run_three_arm", ROOT / "evaluation/end-to-end-dialogue")
    endpoint = base.validated_api_base(api_base)
    if endpoint != "https://tokendance.space/gateway/v1" or model != "deepseek-v4.1-flash" or allowed_model != "deepseek-v4-1-flash-260910":
        raise RuntimeError("Fixed provider policy changed before any request")
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": max_response_tokens, "thinking": {"type": "disabled"}}
    request = Request(endpoint + "/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
    started = time.monotonic()
    try:
        with build_opener(base.NoCredentialRedirects()).open(request, timeout=180) as response:
            raw = response.read(2000001)
        if len(raw) > 2000000:
            raise ResponseFailure("response_size_limit", latency=time.monotonic() - started, fatal=True)
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ResponseFailure("invalid_response_json", latency=time.monotonic() - started, fatal=True) from None
        return parse_received(payload, allowed_model=allowed_model, latency=time.monotonic() - started)
    except HTTPError as error:
        usage = {}
        try:
            payload = json.loads(error.read(2000000))
            usage = numeric_usage(payload.get("usage")) if isinstance(payload, dict) else {}
        except Exception:
            pass
        raise ResponseFailure("provider_http_error", usage=usage, latency=time.monotonic() - started) from None
