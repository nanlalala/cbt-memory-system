"""Contract-boundary tests, independent of model outputs or benchmark gold."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "modules/long-term-memory/schema"
sys.path.insert(0, str(SCHEMA))
sys.path.insert(0, str(ROOT / "modules/long-term-memory/extraction"))

from memory_contract import (  # noqa: E402
    MemoryRecordV2, construct_memory, validate_memory_data, validate_json_memory,
    validate_schema_instance, legacy_metadata, split_legacy_memory, downgrade_memory,
    _semantic_rules,
)
from memory_models import MemoryRecord, SourceEvidence  # noqa: E402
from extraction_contract import ConversationTurn, parse_extraction_output  # noqa: E402
from validate_memory_data import validate_examples  # noqa: E402

try:
    import jsonschema
except ImportError:
    jsonschema = None


def example(version="2.0"):
    data = {
        "schema_version": version, "memory_id": "m-1", "user_id": "u-1",
        "memory_type": "explicit_fact", "content": "This week I paused the thought record.",
        "origin": "explicit_user_statement", "confirmation_state": "unconfirmed",
        "lifecycle_status": "candidate", "sensitivity": "standard",
        "confidence": 0.8, "importance": 0.5,
        "created_at": "2026-11-08T13:00:00Z", "updated_at": "2026-11-08T13:00:00Z",
        "observed_at": "2026-11-08T12:00:00Z",
        "source_evidence": {"session_id": "s-1", "turn_ids": ["t-1"],
                            "supporting_text": "This week I paused the thought record."},
        "structured_payload": {}, "tags": ["homework"], "related_memory_ids": [],
    }
    if version == "2.0":
        data.update(memory_class="mutable_state", category="V4", classification_status="assigned",
                    source_role="user", source_time="2026-11-08T12:00:00Z",
                    permissions={"save": True, "long_term_use": True,
                                 "response_external": False, "extraction_external": False})
    return data


class MemoryContractV2Tests(unittest.TestCase):
    def assert_both_reject(self, data, message=None):
        for constructor in (construct_memory, validate_json_memory, MemoryRecord.from_dict):
            with self.subTest(constructor=constructor.__qualname__):
                with self.assertRaises(ValueError, msg=message):
                    constructor(deepcopy(data))

    def test_v2_roundtrip_uses_canonical_constructor(self):
        record = construct_memory(example())
        self.assertIsInstance(record, MemoryRecordV2)
        self.assertEqual(record.source_evidence.turn_ids, ["t-1"])
        self.assertEqual(record.semantic_support["status"], "unknown")
        self.assertEqual(construct_memory(record.to_dict()).to_dict(), record.to_dict())
        self.assertIsInstance(MemoryRecord.from_dict(example()), MemoryRecordV2)

    def test_numbers_ids_booleans_and_unknown_keys_are_not_coerced(self):
        for key, value in [("confidence", "0.8"), ("importance", True), ("memory_id", 7),
                           ("user_id", 8), ("surprise", "ignored before"), ("tags", "homework")]:
            data = example(); data[key] = value
            self.assert_both_reject(data)
        for version in ("1.0", "2.0"):
            data=example(version); data["confidence"]="0.8"
            self.assert_both_reject(data)

    def test_direct_dataclass_construction_also_checks_types(self):
        data=example();data["source_evidence"]=SourceEvidence(**data["source_evidence"])
        data["confidence"]="0.8"
        with self.assertRaises(ValueError): MemoryRecordV2(**data)
        valid=construct_memory(example())
        with self.assertRaises(ValueError): replace(valid, memory_id=12)

    def test_legacy_direct_dataclass_validate_does_not_bypass_contract(self):
        data=example("1.0");data["source_evidence"]=SourceEvidence(**data["source_evidence"])
        data["confidence"]="0.8"
        record=MemoryRecord(**data)
        with self.assertRaises(ValueError): record.validate()

    def test_business_payload_rules_are_shared(self):
        for version in ("1.0", "2.0"):
            data=example(version);data["memory_type"]="goal"
            self.assert_both_reject(data)
            data["structured_payload"]={"description":"Sleep earlier", "status":""}
            self.assert_both_reject(data)
            data["structured_payload"]={"description":"Sleep earlier", "status":"proposed"}
            self.assertEqual(construct_memory(data).memory_type,"goal")
            if version == "2.0":
                data["structured_payload"]={"description":"Sleep earlier"}
                self.assertNotIn("status",construct_memory(data).structured_payload)
        data=example();data.update(memory_type="cognitive_pattern",origin="agent_inference",category="V5")
        for n in ("2",0,-1,True):
            data["structured_payload"]={"evidence_count":n}
            self.assert_both_reject(data)

    def test_classification_combination_and_pending_are_not_a_third_class(self):
        data=example();data["memory_class"]="fixed_historical"
        self.assert_both_reject(data)
        data.update(memory_class=None,category=None,classification_status="pending")
        self.assertIsNone(construct_memory(data).memory_class)
        data["memory_class"]="unknown"
        self.assert_both_reject(data)

    def test_every_supported_subclass_matches_its_parent(self):
        for cls, cats in [("fixed_historical",["F1","F2","F3","F4"]),
                          ("mutable_state",["V1","V2","V3","V4","V5"])]:
            for cat in cats:
                data=example();data.update(memory_class=cls,category=cat)
                self.assertEqual(construct_memory(data).category,cat)

    def test_formats_and_cross_field_time_order(self):
        for field,value in [("created_at","2026-02-30T00:00:00Z"),
                            ("created_at","2026-11-08"),
                            ("updated_at","2026-11-07T00:00:00Z"),
                            ("source_time","2026-11-09T00:00:00Z")]:
            data=example();data[field]=value;self.assert_both_reject(data)
        data=example();data.update(valid_from="2026-11-09T00:00:00Z",valid_to="2026-11-08T00:00:00Z")
        self.assert_both_reject(data)
        data=example();data.update(memory_class="fixed_historical",category="F1",event_time="2026-11-09T00:00:00Z")
        self.assert_both_reject(data)

    def test_duplicate_source_ids_tags_and_related_ids(self):
        for field in ("tags","related_memory_ids"):
            data=example();data[field]=["x","x"];self.assert_both_reject(data)
        data=example();data["source_evidence"]["turn_ids"]=["t-1","t-1"]
        self.assert_both_reject(data)
        with self.assertRaises(ValueError):
            SourceEvidence.from_dict({"session_id":1,"turn_ids":["t"],"supporting_text":"x"})

    def test_non_json_values_and_nonfinite_values_are_rejected(self):
        for value in (float("nan"),float("inf"),object(),("tuple",)):
            data=example();data["structured_payload"]={"extra":value}
            self.assert_both_reject(data)
        data=example();data["structured_payload"]={1:"numeric key"}
        self.assert_both_reject(data)

    def test_permissions_are_complete_bool_or_null_without_coercion(self):
        for value in ("allow","unknown",1,0):
            data=example();data["permissions"]["save"]=value;self.assert_both_reject(data)
        data=example();data["permissions"]["save"]=None
        self.assertIsNone(construct_memory(data).permissions["save"])
        data=example();del data["permissions"]["save"];self.assert_both_reject(data)
        data=example();del data["permissions"];self.assert_both_reject(data)

    def test_roles_semantic_support_and_confirmation_remain_separate(self):
        data=example();data["source_role"]="assistant";self.assert_both_reject(data)
        data=example();data["semantic_support"]={"status":"rule_verified","method":None,"version":None}
        self.assert_both_reject(data)
        data=example();data.update(memory_type="cognitive_pattern",origin="agent_inference",category="V5",
                                  lifecycle_status="active",structured_payload={"evidence_count":2})
        self.assert_both_reject(data)
        data["lifecycle_status"]="candidate"
        self.assertEqual(construct_memory(data).confirmation_state,"unconfirmed")

    def test_claims_have_unique_ids_and_real_unicode_span_shapes(self):
        data=example();data["claim_bindings"]=[{"claim_id":"c","field":"engagement","value":"paused",
            "turn_id":"t-1","start":0,"end":2,"quote":"暂停"}]
        self.assertEqual(construct_memory(data).claim_bindings[0]["end"],2)
        data["claim_bindings"].append(deepcopy(data["claim_bindings"][0]));self.assert_both_reject(data)
        data=example();data["claim_bindings"]=[{"claim_id":"c","field":"engagement","value":"paused",
            "turn_id":"t-1","start":0,"end":1,"quote":"暂停"}]
        self.assert_both_reject(data)

    def test_cli_runs_same_semantics_for_v1_and_v2_and_duplicate_memories(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"memories.json"
            for version in ("1.0","2.0"):
                data=example(version);data["memory_type"]="goal"
                p.write_text(json.dumps([data]))
                with self.assertRaises(ValueError): validate_examples(p)
            p.write_text(json.dumps([example(),example()]))
            with self.assertRaisesRegex(ValueError,"unique"): validate_examples(p)

    def test_legacy_read_preserves_version_and_leaves_unknown_metadata(self):
        data=example("1.0");record=construct_memory(data)
        self.assertEqual(record.schema_version,"1.0")
        self.assertNotIsInstance(record,MemoryRecordV2)
        meta=legacy_metadata(record)
        self.assertEqual(meta["source_role"],"unknown")
        self.assertIsNone(meta["source_time"])
        self.assertTrue(all(v is None for v in meta["permissions"].values()))
        with self.assertRaises(ValueError):construct_memory(data,allow_legacy=False)

    def test_explicit_split_retains_mapping_and_does_not_invent_confirmation(self):
        data=example("1.0");data.update(confirmation_state="user_confirmed",lifecycle_status="active")
        record=construct_memory(data)
        out=split_legacy_memory(record,[{"memory_id":"m-a","content":"A past attempt","memory_class":"fixed_historical","category":"F3"},
                                        {"memory_id":"m-b","content":"Currently paused","memory_class":"mutable_state","category":"V4"}])
        self.assertEqual([x.memory_id for x in out],["m-a","m-b"])
        self.assertTrue(all(x.confirmation_state=="unconfirmed" for x in out))
        self.assertTrue(all(x.source_role=="unknown" and x.source_time is None for x in out))
        self.assertEqual({m["target_memory_id"] for m in out[0].migration_map},{"m-a","m-b"})
        self.assertEqual(record.schema_version,"1.0")
        with self.assertRaisesRegex(ValueError,"lossy"):downgrade_memory(out[0])

    def test_nonlossy_default_only_downgrade_is_explicit(self):
        data=example();data.update(memory_class=None,category=None,classification_status="pending",
                                  source_role="unknown",source_time=None,
                                  permissions={k:None for k in data["permissions"]})
        old=downgrade_memory(construct_memory(data))
        self.assertEqual(old.schema_version,"1.0")
        data["permissions"]["save"]=True
        with self.assertRaisesRegex(ValueError,"lossy"):downgrade_memory(construct_memory(data))

    def test_extraction_v2_uses_normative_constructor_and_never_confirms(self):
        text="这周暂停记录。"
        candidate={"memory_type":"explicit_fact","content":text,"source_turn_ids":["t"],"supporting_text":text,
                   "memory_class":"mutable_state","category":"V4","classification_status":"assigned",
                   "claim_bindings":[{"claim_id":"c","field":"engagement","value":"paused","turn_id":"t","start":0,"end":len(text),"quote":text}]}
        kwargs=dict(user_id="u",session_id="s",turns=[ConversationTurn("t","user",text,"2026-11-08T12:00:00Z")],
                    extracted_at="2026-11-08T13:00:00Z",schema_version="2.0")
        record=parse_extraction_output(json.dumps({"candidates":[candidate]}),**kwargs)[0]
        self.assertIsInstance(record,MemoryRecordV2)
        self.assertEqual((record.confirmation_state,record.lifecycle_status),("unconfirmed","candidate"))
        self.assertEqual(record.semantic_support["status"],"not_checked")
        self.assertEqual(record.source_time,"2026-11-08T12:00:00Z")
        for key,value in [("confidence","0.8"),("content",42),("surprise",True)]:
            bad=deepcopy(candidate);bad[key]=value
            with self.assertRaises(ValueError):parse_extraction_output(json.dumps({"candidates":[bad]}),**kwargs)
        bad=deepcopy(candidate);bad["claim_bindings"][0]["quote"]="并未暂停记录"
        with self.assertRaises(ValueError):parse_extraction_output(json.dumps({"candidates":[bad]}),**kwargs)

    def test_schema_subset_rejects_unknown_keywords_including_nested_ones(self):
        with self.assertRaisesRegex(ValueError,"unsupported JSON Schema"):
            validate_schema_instance({}, {"type":"object","unevaluatedProperties":False})
        with self.assertRaisesRegex(ValueError,"unsupported JSON Schema"):
            validate_schema_instance({}, {"properties":{"x":{"formatMinimum":"2026-01-01"}}})

    def test_json_decoder_rejects_duplicate_keys_and_nonjson_constants(self):
        raw=json.dumps(example())
        self.assertIsInstance(validate_json_memory(raw),MemoryRecordV2)
        duplicate=raw.replace('"confidence": 0.8','"confidence": 0.8, "confidence": 0.2')
        with self.assertRaisesRegex(ValueError,"duplicate JSON"):
            validate_json_memory(duplicate)
        with self.assertRaisesRegex(ValueError,"non-JSON"):
            validate_json_memory(raw.replace('"confidence": 0.8','"confidence": NaN'))

    @unittest.skipIf(jsonschema is None,"optional reference jsonschema library not installed")
    def test_schema_and_common_semantics_match_reference_jsonschema(self):
        checker=jsonschema.FormatChecker()
        if "date-time" not in checker.checkers:
            self.skipTest("reference RFC3339 format checker not installed")
        corpus=[]
        for version in ("1.0","2.0"):
            base=example(version);corpus.append(base)
            for key,value in [("confidence","0.8"),("memory_id",9),("extra",True),("updated_at","not-a-date"),
                               ("created_at","2026-02-30T00:00:00Z"),("updated_at","2026-01-01T00:00:00Z")]:
                data=deepcopy(base);data[key]=value;corpus.append(data)
            data=deepcopy(base);data["memory_type"]="goal";corpus.append(data)
            data=deepcopy(base);data["source_evidence"]["turn_ids"]*=2;corpus.append(data)
        for category in ("F1","V4",None,"third_category"):
            data=example();data["category"]=category;corpus.append(data)
        data=example();data["state_version"]=1.0;corpus.append(data)
        for data in corpus:
            name="memory_record_v2.schema.json" if data["schema_version"]=="2.0" else "memory_record.schema.json"
            schema=json.loads((SCHEMA/name).read_text())
            try:
                jsonschema.Draft202012Validator(schema,format_checker=checker).validate(data)
                _semantic_rules(data)
                reference=True
            except (ValueError,jsonschema.ValidationError):reference=False
            try:validate_memory_data(data);actual=True
            except ValueError:actual=False
            with self.subTest(data=data):self.assertEqual(actual,reference)


if __name__=="__main__":unittest.main()
