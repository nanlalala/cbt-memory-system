"""Shared exact-wire journal, no credential access and no automatic retries."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import time
from . import projections as projection

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    if path.is_symlink() or temp.exists():
        raise ValueError("unexpected symlink or interrupted atomic write")
    with temp.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def rows(path):
    if not path.exists():
        return []
    if path.is_symlink():
        raise ValueError("symlinked journal")
    raw = path.read_text()
    if raw and not raw.endswith("\n"):
        raise ValueError("incomplete journal retained for inspection")
    return [projection.strict_json(line) for line in raw.splitlines() if line]


@contextmanager
def lock(folder):
    with (folder / ".run.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class UnknownAttempt(RuntimeError):
    pass


class CallLimit(RuntimeError):
    pass


class InvalidReceipt(RuntimeError):
    pass


class Journal:
    def __init__(self, folder, protocol, transport, max_new_calls):
        self.folder, self.protocol, self.transport = folder, protocol, transport
        self.max_new_calls, self.new_calls = max_new_calls, 0
        self.mutex, self.dense_mutex, self.stop = threading.RLock(), threading.Lock(), threading.Event()
        self.intents, self.outcomes = {}, {}
        self.pending = {}
        if (protocol.get("protocol_sha256") != digest({k:v for k,v in protocol.items() if k!="protocol_sha256"}) or
                type(protocol.get("maximum_attempts")) is not int or protocol["maximum_attempts"]<0):
            raise ValueError("frozen protocol checksum or cap differs")
        if getattr(transport,"identity",None) != protocol["provider"]:
            raise ValueError("transport identity differs from the frozen provider")
        for name, target in (("intents.jsonl",self.intents),("outcomes.jsonl",self.outcomes)):
            for row in rows(folder/name):
                key = row["key"]
                if key in target or row["protocol_sha256"] != protocol["protocol_sha256"]:
                    raise ValueError("duplicate or foreign journal record")
                if name=="intents.jsonl":
                    if (digest({"purpose":row["purpose"],"payload":row["payload"]})!=key or
                        sha256(row["serialized_request"].encode()).hexdigest()!=row["wire_sha256"] or
                        json.loads(row["serialized_request"])!=row["payload"]):
                        raise ValueError("retained request identity differs")
                target[key] = row
        if set(self.outcomes)-set(self.intents):
            raise ValueError("outcome without intent")
        if len(self.intents)>protocol["maximum_attempts"]:
            raise ValueError("retained intents exceed frozen cap")
        for key,row in self.outcomes.items():
            if row["status"]=="received" and (row.get("provider_model")!=protocol["provider"]["returned_model"] or
                    row.get("finish_reason")!="stop" or row.get("wire_sha256")!=self.intents[key]["wire_sha256"]):
                raise ValueError("retained successful receipt identity differs")
        if set(self.intents)-set(self.outcomes) or any(r["status"]=="unknown" for r in self.outcomes.values()):
            raise UnknownAttempt("unreconciled send retained; no automatic resend")

    def append(self, name, row):
        with (self.folder/name).open("a", encoding="utf-8") as stream:
            stream.write(canonical(row)+"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def call(self, case, purpose, request):
        cap = self.protocol["request_limits"]["answer" if purpose=="answer" else "preparation"]
        wire = json.dumps(request, allow_nan=False)
        if len(wire)>cap:
            raise ValueError("complete outbound request exceeds frozen character cap")
        key = digest({"purpose":purpose,"payload":request})
        with self.mutex:
            if key in self.outcomes:
                result=self.outcomes[key]
                if result["status"]!="received":
                    raise InvalidReceipt("previous rejected request preserved")
                return result
            event = self.pending.get(key)
            if event is None:
                if self.stop.is_set():
                    raise UnknownAttempt("another request stopped admission")
                if self.new_calls>=self.max_new_calls or len(self.intents)>=self.protocol["maximum_attempts"]:
                    raise CallLimit("invocation or frozen run cap reached")
                if digest(json.loads((self.folder/self.protocol.get("source_manifest_name","cases.json")).read_text()))!=self.protocol["cases_sha256"]:
                    raise ValueError("frozen source roster changed before send")
                intent={"key":key,"purpose":purpose,"case_id":case["case_id"],"payload":request,
                    "serialized_request":wire,
                    "wire_sha256":sha256(wire.encode()).hexdigest(),"wire_chars":len(wire),
                    "protocol_sha256":self.protocol["protocol_sha256"],"at_utc":utc()}
                self.append("intents.jsonl",intent)
                self.intents[key]=intent
                self.new_calls+=1
                event=threading.Event()
                self.pending[key]=event
                owner=True
            else:
                owner=False
        if not owner:
            event.wait()
            return self.call(case,purpose,request)
        start=time.monotonic()
        result={"key":key,"protocol_sha256":self.protocol["protocol_sha256"],"purpose":purpose,
                "case_id":case["case_id"],"status":"unknown","text":None,"usage":{},"provider_model":None}
        try:
            received=self.transport(request)
            if type(received) is not dict:
                raise ValueError("provider receipt object required")
            usage=received.get("usage") if type(received.get("usage")) is dict else {}
            usage={k:v for k,v in usage.items() if type(v) is int and v>=0}
            result.update(text=received.get("text"),usage=usage,
                          provider_model=received.get("provider_model"),status="invalid",
                          finish_reason=received.get("finish_reason"),response_id=received.get("response_id"),
                          wire_sha256=received.get("wire_sha256"))
            if (received.get("provider_model")!=self.protocol["provider"]["returned_model"] or received.get("finish_reason")!="stop" or
                received.get("wire_sha256")!=self.intents[key]["wire_sha256"] or
                type(received.get("text")) is not str or not received["text"].strip()):
                raise InvalidReceipt("incomplete or model/wire-mismatched response")
            result["status"]="received"
        except Exception as exc:
            result["error_type"]=type(exc).__name__
            self.stop.set()
        finally:
            result.update(observed_seconds=time.monotonic()-start,at_utc=utc())
            with self.mutex:
                self.append("outcomes.jsonl",result)
                self.outcomes[key]=result
                event.set()
        if result["status"]!="received":
            raise UnknownAttempt("provider send retained; admission stopped")
        return result

