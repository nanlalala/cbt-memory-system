#!/usr/bin/env python3
"""One bounded phase in a fresh genuine-Colab subprocess; no Secret reads."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

def main():
    from persona_management_runner import require_colab
    require_colab()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','extract','answer','status','score'));parser.add_argument('--run',required=True)
    parser.add_argument('--original-suite');parser.add_argument('--executable-commit');parser.add_argument('--max-new-calls',type=int,default=0)
    parser.add_argument('--maximum-provider-attempts',type=int,default=2500)
    args=parser.parse_args()
    command=[sys.executable,'-u',str(Path(__file__).with_name('persona_management_runner.py')),args.phase,'--run',args.run,
             '--max-new-calls',str(args.max_new_calls),'--maximum-provider-attempts',str(args.maximum_provider_attempts)]
    if args.original_suite:command.extend(['--original-suite',args.original_suite])
    if args.executable_commit:command.extend(['--executable-commit',args.executable_commit])
    # Credential is inherited from the already-authorized Colab process only.
    environment=os.environ.copy();environment['PYTHONUNBUFFERED']='1'
    return subprocess.call(command,env=environment,cwd=Path(__file__).resolve().parents[2])

if __name__=='__main__':raise SystemExit(main())
