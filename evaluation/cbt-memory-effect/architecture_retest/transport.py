"""Verbatim hosted gateway transport with distinct requested and returned models."""
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


def require_hosted():
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir() and
            importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("genuine hosted Colab required")


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise RuntimeError("credential-bearing redirects are disabled")


class HostedGatewayTransport:
    def __init__(self, identity, key):
        require_hosted()
        if (set(identity)!={"provider","model","returned_model","model_family"} or
                identity["provider"]!="https://tokendance.space/gateway/v1" or
                not all(isinstance(v,str) and v.strip() for v in identity.values()) or
                not isinstance(key,str) or not key):
            raise ValueError("fixed gateway identity and credential required")
        self.identity = dict(identity)
        self._key = key

    def __call__(self, payload):
        require_hosted()
        if (payload.get("model")!=self.identity["model"] or payload.get("temperature")!=0 or
                payload.get("thinking")!={"type":"disabled"}):
            raise ValueError("payload differs from fixed model/sampling policy")
        wire = json.dumps(payload, allow_nan=False).encode()
        request = Request(self.identity["provider"]+"/chat/completions",data=wire,
                          headers={"Content-Type":"application/json","Authorization":"Bearer "+self._key},
                          method="POST")
        try:
            with build_opener(NoRedirects()).open(request, timeout=180) as response:
                raw = response.read(2_000_001)
        except HTTPError:
            raise RuntimeError("provider_http_error") from None
        if len(raw)>2_000_000:
            raise ValueError("provider response exceeds frozen size bound")
        value = json.loads(raw)
        choice = value["choices"][0]
        if value.get("model")!=self.identity["returned_model"]:
            raise ValueError("returned provider model differs from frozen alias")
        return {"text":choice.get("message",{}).get("content"),
                "provider_model":value["model"],"finish_reason":choice.get("finish_reason"),
                "response_id":value.get("id"),"usage":value.get("usage",{}),"wire_sha256":sha256(wire).hexdigest()}
