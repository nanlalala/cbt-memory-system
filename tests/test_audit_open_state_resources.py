"""Pure tiny accounting fixtures only; no actual model/data/runtime audit."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('open_state_resources',ROOT/'evaluation/cbt-memory-effect/audit_open_state_resources.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class OpenStateResourceTests(unittest.TestCase):
 def test_shared_canonical_answer_is_counted_once_and_allocated_without_recount(self):
  intents={'k':{'output':'before'}}
  outcomes={'k':{'status':'published','usage':{'prompt_tokens':8,'completion_tokens':2,'total_tokens':10},'generation_wall_seconds':3.}}
  physical=M.costs(['k','k'],intents,outcomes)
  self.assertEqual(physical['unique_admitted_attempts'],1)
  self.assertEqual(physical['observed_tokens']['total_tokens'],10)
  rows=[{'arm':'raw','canonical_key':'k'},{'arm':'raw','canonical_key':'k'},{'arm':'managed','canonical_key':'k'}]
  allocated=M.arm_allocations(rows,{'k'},outcomes)['arms']
  self.assertEqual(allocated['raw']['allocated_attempt_equivalents'],.5)
  self.assertEqual(allocated['managed']['allocated_observed_tokens']['total_tokens'],5.)
  self.assertEqual(sum(x['allocated_observed_tokens']['total_tokens'] for x in allocated.values()),10.)
  self.assertEqual(sum(x['allocated_observed_answer_path_wall_seconds'] for x in allocated.values()),3.)
  # A later run references the receipt, but owns no additional generation.
  later=M.arm_allocations(rows,set(),outcomes)['arms']
  self.assertEqual(sum(x['allocated_attempt_equivalents'] for x in later.values()),0.)
 def test_failed_or_unknown_usage_stays_unknown_not_zero(self):
  intents={'failed':{},'unknown':{}}
  outcomes={'failed':{'status':'generation_failed_or_unknown','usage':None,'observed_wall_seconds':1.5}}
  result=M.costs(intents,intents,outcomes)
  self.assertEqual(result['unique_admitted_attempts'],2)
  self.assertEqual(result['status_counts'],{'generation_failed_or_unknown':1,'intent_without_outcome':1})
  self.assertIsNone(result['observed_tokens']['total_tokens'])
  self.assertEqual(result['usage_missing_attempts']['total_tokens'],2)
  self.assertEqual(result['answer_path_wall_seconds_sum'],1.5)
  self.assertEqual(result['missing_wall_time_attempts'],1)
  empty=M.costs(set(),{},{});self.assertEqual(empty['observed_tokens']['total_tokens'],0)
  self.assertEqual(empty['answer_path_wall_seconds_sum'],0)

if __name__=='__main__':unittest.main()
