"""Bounded source-first persistence and withdrawal fences for the v2 prototype.

This module stores provenance and orchestration facts, not semantic truth. It
does not grant a candidate current-state eligibility and never reads scoring
labels. A file must be explicitly placed under an approved root; a Colab VM or
an in-memory connection alone is not a cross-runtime durability guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

from sqlite_store import MemoryStorageConflict, SQLiteMemoryStore, StorageControlFailure


PERMISSIONS = ("save", "long_term_use", "response_external", "extraction_external")


DurableConflict = MemoryStorageConflict


class AttemptUnknown(DurableConflict):
    """A request may already have been charged; automatic resend is forbidden."""


class BudgetExhausted(DurableConflict):
    pass


class SafetyPaused(DurableConflict):
    """The persistent user-level normal-processing gate is closed."""


@dataclass(frozen=True)
class DurableLimits:
    max_run_calls: int = 12
    max_session_calls: int = 12
    max_queue_length: int = 32
    max_batch_size: int = 4

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("durable limits must be positive integers")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage(value: dict | None) -> str | None:
    """Usage is numeric metadata, never an alternate body-retention channel."""
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError('usage must be numeric metadata or unknown None')
    for key, item in value.items():
        if type(key) is not str or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',key):
            raise ValueError('invalid usage metric name')
        if type(item) is dict:
            _usage(item)
        elif item is not None and (type(item) not in {int,float} or item<0):
            raise ValueError('usage values must be nonnegative numeric metadata or None')
    return _json(value)


class DurableMemoryStore:
    """Use one existing store/connection per worker, never across threads.

    File stores require their position binding plus the latest checkpoint in
    ``approved_root/.cbt-memory-control``. Keep that control namespace separately
    from database backups; never restore it with an older DB/sidecar. A deployment
    that rolls back the entire DB and independent anchor together needs an
    externally retained latest checkpoint: local files cannot prove freshness
    after a whole-domain rollback or guarantee cross-Colab-runtime persistence.

    ``begin_attempt`` is the serialized send-admission boundary. Withdrawal
    after admission cannot retract a request already handed to a provider;
    ``finish_attempt`` fences publication and persistence of its result.
    """

    def __init__(self, store: SQLiteMemoryStore, *, approved_root: str | Path | None = None,
                 allow_memory_for_tests: bool = False, run_limits: DurableLimits | None = None):
        self.store = store
        self.connection = store.connection
        self.limits = run_limits or DurableLimits()
        self._file_identity=None
        self._binding_path=None
        self._anchor_path=None
        had_v2=bool(self.connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name LIKE 'dm_%'").fetchone())
        if store.database == ":memory:":
            if not allow_memory_for_tests:
                raise ValueError("prototype requires a file-backed database under approved_root")
        else:
            if approved_root is None:
                raise ValueError("approved_root is required for file-backed durable storage")
            root = Path(approved_root).resolve(strict=True)
            database = Path(store.database)
            if not database.is_absolute() or database.is_symlink():
                raise ValueError("database must be an explicit nonsymlink absolute path")
            if not database.resolve().is_relative_to(root):
                raise PermissionError("database is outside the approved storage root")
            self._binding_path=Path(str(database.resolve())+'.control.json')
            self._anchor_root=root/'.cbt-memory-control'
            if self._anchor_root.is_symlink() or (self._anchor_root.exists() and not self._anchor_root.is_dir()):
                raise PermissionError('independent control namespace must be a real directory under approved_root')
            if had_v2 and not self._binding_path.is_file():
                raise DurableConflict('existing v2 database requires its current independent control binding')
        self._create_schema()
        if store.database!=':memory:':self._initialize_identity(had_v2)
        self.store._before_commit=self._checkpoint_before_commit
        self.store._before_read=self._verify_control_anchor

    @classmethod
    def open(cls, database: str | Path, *, approved_root: str | Path,
             run_limits: DurableLimits | None = None) -> "DurableMemoryStore":
        # Validate before sqlite creates a file anywhere outside the allowlist.
        root, path = Path(approved_root).resolve(strict=True), Path(database)
        if not path.is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise PermissionError("database must be an explicit path under approved_root")
        store=SQLiteMemoryStore(path)
        try:return cls(store, approved_root=root, run_limits=run_limits)
        except BaseException:
            store.close()
            raise

    @classmethod
    def open_restored(cls,database: str | Path,*,approved_root: str | Path,
                      latest_control_ledger: dict | None,run_limits: DurableLimits | None=None) -> 'DurableMemoryStore':
        if latest_control_ledger is None:
            raise DurableConflict('restore requires the latest external control ledger')
        durable=cls.open(database,approved_root=approved_root,run_limits=run_limits)
        try:
            flag=durable.connection.execute("SELECT value FROM dm_meta WHERE key='maintenance'").fetchone()[0]
            if not flag:durable.begin_maintenance()
            durable.finish_maintenance(latest_control_ledger=latest_control_ledger)
            return durable
        except BaseException:
            durable.close()
            raise

    def close(self) -> None:
        self.store.close()

    def _create_schema(self):
        # DDL is outside a source/effect transaction; no model work holds locks.
        tables={row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'dm_%'")}
        complete={'dm_meta','dm_storage_identity','dm_users','dm_runs','dm_inputs','dm_sources','dm_jobs','dm_objects','dm_object_versions','dm_effects','dm_representations','dm_dependencies','dm_revocations','dm_attempts'}
        if complete.issubset(tables):
            revocations={row[1] for row in self.connection.execute('PRAGMA table_info(dm_revocations)')}
            attempts={row[1] for row in self.connection.execute('PRAGMA table_info(dm_attempts)')}
            revision=self.connection.execute("SELECT 1 FROM dm_meta WHERE key='storage_revision'").fetchone()
            if revision and 'target_kind' in revocations and {'current_source_ids_json','current_input_id','can_retain_response'}.issubset(attempts):
                return
        schema="""
        CREATE TABLE IF NOT EXISTS dm_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
        INSERT OR IGNORE INTO dm_meta VALUES ('maintenance',0);
        INSERT OR IGNORE INTO dm_meta VALUES ('storage_revision',0);
        CREATE TABLE IF NOT EXISTS dm_storage_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), storage_id TEXT NOT NULL, database_path TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS dm_users (user_id TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS dm_runs (run_id TEXT PRIMARY KEY, limits_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS dm_inputs (
          user_id TEXT NOT NULL, input_id TEXT NOT NULL, fingerprint TEXT,
          receipt_json TEXT NOT NULL, PRIMARY KEY(user_id,input_id));
        CREATE TABLE IF NOT EXISTS dm_sources (
          user_id TEXT NOT NULL, source_id TEXT NOT NULL, input_id TEXT NOT NULL,
          text TEXT NOT NULL, role TEXT, source_time TEXT, permissions_json TEXT NOT NULL,
          active INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(user_id,source_id), UNIQUE(user_id,input_id));
        CREATE TABLE IF NOT EXISTS dm_jobs (
          job_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, source_id TEXT NOT NULL,
          run_id TEXT NOT NULL, session_id TEXT NOT NULL, status TEXT NOT NULL,
          generation INTEGER NOT NULL, claim_token TEXT, created_at TEXT NOT NULL,
          UNIQUE(user_id,source_id,run_id));
        CREATE INDEX IF NOT EXISTS dm_jobs_pending ON dm_jobs(status,created_at,job_id);
        CREATE TABLE IF NOT EXISTS dm_objects (
          user_id TEXT NOT NULL, object_id TEXT NOT NULL, version INTEGER NOT NULL,
          payload_json TEXT, blocked INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(user_id,object_id));
        CREATE TABLE IF NOT EXISTS dm_object_versions (
          user_id TEXT NOT NULL, object_id TEXT NOT NULL, version INTEGER NOT NULL,
          payload_json TEXT, PRIMARY KEY(user_id,object_id,version));
        CREATE TABLE IF NOT EXISTS dm_effects (
          user_id TEXT NOT NULL, effect_key TEXT NOT NULL, object_id TEXT NOT NULL,
          fingerprint TEXT NOT NULL, version INTEGER NOT NULL, PRIMARY KEY(user_id,effect_key));
        CREATE TABLE IF NOT EXISTS dm_representations (
          user_id TEXT NOT NULL, representation_id TEXT NOT NULL, object_id TEXT NOT NULL,
          body_json TEXT, active INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(user_id,representation_id));
        CREATE TABLE IF NOT EXISTS dm_dependencies (
          user_id TEXT NOT NULL, owner_kind TEXT NOT NULL, owner_id TEXT NOT NULL,
          source_id TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
          PRIMARY KEY(user_id,owner_kind,owner_id,source_id,start,end));
        CREATE TABLE IF NOT EXISTS dm_revocations (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
          scope TEXT NOT NULL, target_kind TEXT NOT NULL DEFAULT 'source', target_id TEXT NOT NULL, start INTEGER, end INTEGER,
          generation INTEGER NOT NULL, occurred_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS dm_attempts (
          attempt_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, request_key TEXT NOT NULL,
          purpose TEXT NOT NULL, source_ids_json TEXT NOT NULL, current_source_ids_json TEXT NOT NULL DEFAULT '[]', run_id TEXT NOT NULL,
          session_id TEXT NOT NULL, generation INTEGER NOT NULL, status TEXT NOT NULL,
          response_json TEXT, usage_json TEXT, job_id TEXT, created_at TEXT NOT NULL,
          current_input_id TEXT, can_retain_response INTEGER NOT NULL DEFAULT 0,
          UNIQUE(user_id,request_key));
        """
        with self.store.transaction(immediate=True):
            for statement in schema.split(';'):
                if statement.strip():self.connection.execute(statement)
            if 'target_kind' not in {row[1] for row in self.connection.execute('PRAGMA table_info(dm_revocations)')}:
                self.connection.execute("ALTER TABLE dm_revocations ADD COLUMN target_kind TEXT NOT NULL DEFAULT 'source'")
            if 'current_source_ids_json' not in {row[1] for row in self.connection.execute('PRAGMA table_info(dm_attempts)')}:
                self.connection.execute("ALTER TABLE dm_attempts ADD COLUMN current_source_ids_json TEXT NOT NULL DEFAULT '[]'")
            for column,declaration in [('current_input_id','TEXT'),('can_retain_response','INTEGER NOT NULL DEFAULT 0')]:
                if column not in {row[1] for row in self.connection.execute('PRAGMA table_info(dm_attempts)')}:
                    self.connection.execute(f'ALTER TABLE dm_attempts ADD COLUMN {column} {declaration}')

    @staticmethod
    def _atomic_control_file(path: Path,payload: dict) -> None:
        path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if path.is_symlink():raise DurableConflict('control anchor must not be a symlink')
        name=None
        try:
            with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.control-',delete=False,encoding='utf-8') as stream:
                name=stream.name;stream.write(_json(payload));stream.flush();os.fsync(stream.fileno())
            os.replace(name,path);name=None
            descriptor=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        finally:
            if name is not None:Path(name).unlink(missing_ok=True)

    def _initialize_identity(self,had_v2: bool) -> None:
        row=self.connection.execute('SELECT storage_id,database_path FROM dm_storage_identity WHERE singleton=1').fetchone()
        database_path=str(Path(self.store.database).resolve())
        if row is None:
            if had_v2 or self._binding_path.exists():raise DurableConflict('unregistered existing store cannot recreate a missing latest ledger')
            with self.store.transaction(immediate=True):
                self.connection.execute('INSERT INTO dm_storage_identity VALUES (1,?,?)',(uuid.uuid4().hex,database_path))
            row=self.connection.execute('SELECT storage_id,database_path FROM dm_storage_identity WHERE singleton=1').fetchone()
            self._file_identity={'storage_id':row[0],'database_path':row[1]}
            self._anchor_path=self._anchor_root/(row[0]+'.json')
            self._persist_control_anchor()
        else:
            if row['database_path']!=database_path:raise DurableConflict('database copied to another location; approved migration and latest ledger required')
            self._file_identity={'storage_id':row[0],'database_path':row[1]}
            self._anchor_path=self._anchor_root/(row[0]+'.json')
            self._verify_control_anchor(allow_maintenance=True)

    def _control_checkpoint(self) -> dict:
        revision=self.connection.execute("SELECT value FROM dm_meta WHERE key='storage_revision'").fetchone()[0]
        return {**self._file_identity,**self.control_ledger(),'storage_revision':revision}

    def _checkpoint_before_commit(self) -> None:
        self._verify_control_anchor(allow_maintenance=True)
        self.connection.execute("UPDATE dm_meta SET value=value+1 WHERE key='storage_revision'")
        self._persist_control_anchor()

    def _verify_control_anchor(self,*,allow_maintenance: bool=False) -> None:
        if getattr(self.store,'_control_fault',False):raise StorageControlFailure('control-anchor failure requires maintenance')
        flag=self.connection.execute("SELECT value FROM dm_meta WHERE key='maintenance'").fetchone()
        if flag and flag[0] and not allow_maintenance:raise DurableConflict('maintenance blocks memory use until latest ledger verification')
        if self._file_identity is None:return
        expected=self._control_checkpoint()
        try:
            if self._binding_path.is_symlink() or self._anchor_path.is_symlink():raise ValueError('symlink control binding')
            binding=json.loads(self._binding_path.read_text(encoding='utf-8'))
            anchor=json.loads(self._anchor_path.read_text(encoding='utf-8'))
        except (OSError,ValueError,TypeError) as exc:
            raise DurableConflict('latest independent control anchor is required; no automatic reconstruction') from exc
        if binding!=expected or anchor!=expected:
            raise DurableConflict('control watermark or storage identity differs; restored/stale database is blocked')

    def _persist_control_anchor(self) -> None:
        if self._file_identity is None:return
        checkpoint=self._control_checkpoint()
        # Anchor and local binding are durable before the DB control commit.
        # A crash/rollback between these writes fails closed on the next check.
        try:
            self._atomic_control_file(self._anchor_path,checkpoint)
            self._atomic_control_file(self._binding_path,checkpoint)
        except BaseException as exc:
            self.store._control_fault=True
            raise StorageControlFailure('control anchor could not be persisted; memory use is blocked') from exc

    def _writable(self):
        self._verify_control_anchor()
        if self.connection.execute("SELECT value FROM dm_meta WHERE key='maintenance'").fetchone()[0]:
            raise DurableConflict("maintenance window: close old connections before migration")

    def _user(self, user_id):
        _id(user_id, "user_id")
        self.connection.execute("INSERT OR IGNORE INTO dm_users(user_id) VALUES (?)", (user_id,))

    def current_generation(self, user_id: str) -> int:
        _id(user_id, "user_id")
        row = self.connection.execute("SELECT generation FROM dm_users WHERE user_id=?", (user_id,)).fetchone()
        return row[0] if row else 0

    def _paused_locked(self,user_id: str) -> bool:
        return self.connection.execute("SELECT 1 FROM dm_revocations WHERE user_id=? AND scope='safety_pause' LIMIT 1",(user_id,)).fetchone() is not None

    def is_paused(self,user_id: str) -> bool:
        _id(user_id,'user_id');self._verify_control_anchor()
        return self._paused_locked(user_id)

    def _normal_allowed(self,user_id: str) -> None:
        if self._paused_locked(user_id):raise SafetyPaused('normal source, queue and external processing are persistently paused for this user')

    def pause_user(self,user_id: str) -> dict:
        """Persist a body-free safety latch; no automatic resume API exists.

        Local withdrawal and body-free receipts remain available. Previously
        admitted provider requests cannot be unsent, but their results become
        stale and cannot be published or cached.
        """
        _id(user_id,'user_id')
        with self.store.transaction(immediate=True):
            self._writable();self._user(user_id)
            if self._paused_locked(user_id):return {'paused':True,'generation':self.current_generation(user_id),'replayed':True}
            self.connection.execute('UPDATE dm_users SET generation=generation+1 WHERE user_id=?',(user_id,))
            generation=self.current_generation(user_id)
            self.connection.execute("INSERT INTO dm_revocations(user_id,scope,target_kind,target_id,start,end,generation,occurred_at) VALUES (?,'safety_pause','user',?,NULL,NULL,?,?)",(user_id,user_id,generation,_now()))
            self.connection.execute("UPDATE dm_jobs SET status='safety_paused' WHERE user_id=? AND status IN ('queued','claimed','sending')",(user_id,))
            self.connection.execute("UPDATE dm_attempts SET response_json=NULL,status=CASE WHEN status='succeeded' THEN 'stale' ELSE status END WHERE user_id=?",(user_id,))
            self._persist_control_anchor()
            return {'paused':True,'generation':generation,'replayed':False}

    def configure_run(self, run_id: str, limits: DurableLimits | None = None) -> dict:
        _id(run_id, "run_id")
        payload = _json(asdict(limits or self.limits))
        with self.store.transaction(immediate=True):
            self._writable()
            old = self.connection.execute("SELECT limits_json FROM dm_runs WHERE run_id=?", (run_id,)).fetchone()
            if old and old[0] != payload:
                raise DurableConflict("run limits are frozen; use a new run_id")
            self.connection.execute("INSERT OR IGNORE INTO dm_runs VALUES (?,?)", (run_id, payload))
        return json.loads(payload)

    def _run(self, run_id):
        row = self.connection.execute("SELECT limits_json FROM dm_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            self.connection.execute("INSERT INTO dm_runs VALUES (?,?)", (run_id, _json(asdict(self.limits))))
            return asdict(self.limits)
        return json.loads(row[0])

    def limits_for_run(self,run_id: str) -> dict:
        self._verify_control_anchor();_id(run_id,'run_id')
        row=self.connection.execute('SELECT limits_json FROM dm_runs WHERE run_id=?',(run_id,)).fetchone()
        if row is None:raise DurableConflict('run limits must be configured before worker execution')
        return json.loads(row[0])

    def record_input(self, user_id: str, input_id: str, receipt: dict, *, fingerprint: str | None = None) -> dict:
        """Persist body-free control metadata even when saving text is refused.

        Caller must not put text, summaries or semantic payloads in this receipt.
        Only this whitelist is persisted; no-save records omit fingerprints too.
        """
        _id(user_id, "user_id"); _id(input_id, "input_id")
        allowed = {"source_status", "source_id", "job_id", "processing_status", "control_status", "generation", "response_status", "run_id", "session_id"}
        if type(receipt) is not dict or set(receipt) - allowed:
            raise ValueError("input receipt only accepts body-free control fields")
        if any(type(v) not in {str, int, bool, type(None)} for v in receipt.values()):
            raise ValueError("input receipt values must be scalar metadata")
        if receipt.get('source_status')=='not_saved' and fingerprint is not None:
            raise ValueError('no-save inputs do not retain body fingerprints')
        with self.store.transaction(immediate=True):
            self._writable(); self._user(user_id)
            old = self.connection.execute("SELECT fingerprint,receipt_json FROM dm_inputs WHERE user_id=? AND input_id=?", (user_id,input_id)).fetchone()
            if old:
                if old[0] != fingerprint:
                    raise DurableConflict("input_id already identifies another input")
                return {**json.loads(old[1]), "replayed": True}
            self.connection.execute("INSERT INTO dm_inputs VALUES (?,?,?,?)", (user_id,input_id,fingerprint,_json(receipt)))
        return {**receipt, "replayed": False}

    def get_input(self, user_id: str, input_id: str) -> dict | None:
        """Read only the durable body-free receipt, with current job status."""
        self._verify_control_anchor()
        row=self.connection.execute('SELECT receipt_json FROM dm_inputs WHERE user_id=? AND input_id=?',(user_id,input_id)).fetchone()
        if row is None:return None
        receipt=json.loads(row[0])
        if receipt.get('job_id'):
            job=self.connection.execute('SELECT status FROM dm_jobs WHERE user_id=? AND job_id=?',(user_id,receipt['job_id'])).fetchone()
            if job:receipt['processing_status']=job[0]
        return receipt

    def submit_source(self, source: dict, *, enqueue: bool = True, run_id: str = "default", session_id: str = "default") -> dict:
        required = {"user_id", "source_id", "input_id", "text", "role", "source_time", "permissions"}
        if type(source) is not dict or set(source) != required:
            raise ValueError("source fields must exactly match the source-first contract")
        user, sid, iid = (_id(source[k],k) for k in ("user_id","source_id","input_id"))
        _id(run_id,"run_id"); _id(session_id,"session_id")
        if type(source["text"]) is not str or not source["text"]:
            raise ValueError("source text must be nonempty")
        if source["role"] not in {"user","assistant",None,"unknown"}:
            raise ValueError("unknown source role must remain unknown")
        if source["source_time"] is not None:
            if type(source["source_time"]) is not str:
                raise ValueError("source_time must be an aware timestamp or None")
            stamp = datetime.fromisoformat(source["source_time"].replace("Z","+00:00"))
            if stamp.tzinfo is None: raise ValueError("source_time requires timezone")
        permissions = source["permissions"]
        if type(permissions) is not dict or set(permissions) - set(PERMISSIONS):
            raise ValueError("invalid source permissions")
        permissions = {k:permissions.get(k) for k in PERMISSIONS}
        if any(v is not None and type(v) is not bool for v in permissions.values()):
            raise ValueError("permissions are True, False or unknown None")
        canonical = {**source,"permissions":permissions}
        fingerprint = hashlib.sha256(_json(canonical).encode()).hexdigest() if permissions['save'] is True else None
        with self.store.transaction(immediate=True):
            self._writable(); self._user(user);self._normal_allowed(user)
            if permissions['save'] is not True:
                return self.record_input(user,iid,{'source_status':'not_saved','processing_status':'not_queued','generation':self.current_generation(user)})
            limits=self._run(run_id)
            old = self.connection.execute("SELECT fingerprint,receipt_json FROM dm_inputs WHERE user_id=? AND input_id=?",(user,iid)).fetchone()
            if old:
                if old[0] != fingerprint: raise DurableConflict("input_id already identifies another input")
                return {**self.get_input(user,iid),"replayed":True}
            if self.connection.execute("SELECT 1 FROM dm_sources WHERE user_id=? AND source_id=?",(user,sid)).fetchone():
                raise DurableConflict("source_id already exists, including withdrawn sources")
            self.connection.execute("INSERT INTO dm_sources VALUES (?,?,?,?,?,?,?,1)",(user,sid,iid,source['text'],source['role'],source['source_time'],_json(permissions)))
            generation=self.current_generation(user)
            allowed = enqueue and permissions['long_term_use'] is True and permissions['extraction_external'] is True
            queued = self.connection.execute("SELECT COUNT(*) FROM dm_jobs WHERE run_id=? AND status IN ('queued','claimed','sending')",(run_id,)).fetchone()[0]
            job_id=None
            status='not_queued_permission' if not allowed else 'unprocessed_queue_limit'
            if allowed and queued < limits['max_queue_length']:
                job_id='job-'+uuid.uuid4().hex;status='queued'
                self.connection.execute("INSERT INTO dm_jobs VALUES (?,?,?,?,?,'queued',?,NULL,?)",(job_id,user,sid,run_id,session_id,generation,_now()))
            receipt={'source_status':'committed','source_id':sid,'job_id':job_id,'processing_status':status,'generation':generation,'run_id':run_id,'session_id':session_id}
            self.connection.execute("INSERT INTO dm_inputs VALUES (?,?,?,?)",(user,iid,fingerprint,_json(receipt)))
        return {**receipt,'replayed':False}

    def get_source(self, user_id: str, source_id: str, *, as_of: str | None = None, require_use: bool = True,
                   _internal_control_read: bool=False) -> dict | None:
        if not _internal_control_read:self._verify_control_anchor()
        row=self.connection.execute("SELECT * FROM dm_sources WHERE user_id=? AND source_id=?",(user_id,source_id)).fetchone()
        if row is None or not row['active']:return None
        if require_use and self._paused_locked(user_id):return None
        permissions=json.loads(row['permissions_json'])
        if require_use and permissions['long_term_use'] is not True:return None
        if as_of is not None:
            cutoff=datetime.fromisoformat(as_of.replace('Z','+00:00'))
            if cutoff.tzinfo is None:raise ValueError('source cutoff requires timezone')
            if row['source_time'] is None or datetime.fromisoformat(row['source_time'].replace('Z','+00:00')) > cutoff:return None
        text=row['text']
        if require_use:
            for withdrawn in self.connection.execute("SELECT start,end FROM dm_revocations WHERE user_id=? AND target_kind='source' AND target_id=? AND scope='span_stop_use'",(user_id,source_id)):
                text=text[:withdrawn['start']]+'█'*(withdrawn['end']-withdrawn['start'])+text[withdrawn['end']:]
        return {'user_id':row['user_id'],'source_id':row['source_id'],'input_id':row['input_id'],'text':text,'role':row['role'],'source_time':row['source_time'],'permissions':permissions}

    def list_sources(self, user_id: str, *, as_of: str | None = None) -> list[dict]:
        return [source for row in self.connection.execute("SELECT source_id FROM dm_sources WHERE user_id=? ORDER BY source_time,source_id",(user_id,)).fetchall()
                if (source:=self.get_source(user_id,row[0],as_of=as_of)) is not None]

    def claim_job(self, *, user_id: str | None = None,run_id: str | None=None) -> dict | None:
        with self.store.transaction(immediate=True):
            self._writable()
            sql="SELECT * FROM dm_jobs WHERE status='queued' AND NOT EXISTS(SELECT 1 FROM dm_revocations p WHERE p.user_id=dm_jobs.user_id AND p.scope='safety_pause')";args=[]
            if user_id is not None:sql+=' AND user_id=?';args.append(user_id)
            if run_id is not None:
                _id(run_id,'run_id');sql+=' AND run_id=?';args.append(run_id)
            row=self.connection.execute(sql+' ORDER BY created_at,job_id LIMIT 1',args).fetchone()
            if row is None:return None
            token=uuid.uuid4().hex
            if row['generation'] != self.current_generation(row['user_id']) or self.get_source(row['user_id'],row['source_id']) is None:
                self.connection.execute("UPDATE dm_jobs SET status='stale' WHERE job_id=?",(row['job_id'],));return None
            self.connection.execute("UPDATE dm_jobs SET status='claimed',claim_token=? WHERE job_id=? AND status='queued'",(token,row['job_id']))
            return {**dict(row),'status':'claimed','claim_token':token}

    def complete_job(self, job_id: str, claim_token: str, *, status: str = 'completed') -> dict:
        if status not in {'completed','failed','unprocessed','attempt_unknown'}:raise ValueError('invalid job terminal status')
        with self.store.transaction(immediate=True):
            self._writable()
            row=self.connection.execute('SELECT * FROM dm_jobs WHERE job_id=?',(job_id,)).fetchone()
            if row is None or row['claim_token'] != claim_token:raise DurableConflict('job claim token differs')
            if self._paused_locked(row['user_id']):return {'status':'safety_paused'}
            if row['status'] in {'stale','completed','attempt_unknown'}:return {'status':row['status']}
            if row['generation'] != self.current_generation(row['user_id']):status='stale'
            self.connection.execute('UPDATE dm_jobs SET status=? WHERE job_id=?',(status,job_id))
            return {'status':status}

    def get_attempt(self,user_id: str,request_key: str) -> dict | None:
        self._verify_control_anchor()
        row=self.connection.execute('SELECT * FROM dm_attempts WHERE user_id=? AND request_key=?',(user_id,request_key)).fetchone()
        if row is None:return None
        live=row['generation']==self.current_generation(user_id)
        return {'attempt_id':row['attempt_id'],'status':row['status'],'purpose':row['purpose'],
                'generation':row['generation'],'run_id':row['run_id'],'session_id':row['session_id'],
                'publishable':live and row['status']=='succeeded' and row['response_json'] is not None,
                'response':json.loads(row['response_json']) if live and row['status']=='succeeded' and row['response_json'] is not None else None,
                'usage':json.loads(row['usage_json']) if row['usage_json'] is not None else None}

    def begin_attempt(self, user_id: str, request_key: str, purpose: str, source_ids: list[str], *,
                      run_id: str = 'default', session_id: str = 'default', expected_generation: int | None = None,
                      job_id: str | None = None, claim_token: str | None = None,
                      current_source_ids: list[str] | None = None,current_input_id: str | None=None) -> dict:
        _id(user_id,'user_id');_id(request_key,'request_key');_id(run_id,'run_id');_id(session_id,'session_id')
        if purpose not in {'response','extraction','verification'}:raise ValueError('invalid external purpose')
        if type(source_ids) is not list or any(type(s) is not str for s in source_ids) or len(set(source_ids))!=len(source_ids):raise ValueError('source_ids must be unique strings')
        if purpose!='response' and not source_ids:raise PermissionError('background external work requires persisted permitted sources')
        if expected_generation is not None and (type(expected_generation) is not int or expected_generation<0):raise ValueError('expected_generation must be a nonnegative integer')
        current_source_ids=[] if current_source_ids is None else current_source_ids
        if type(current_source_ids) is not list or len(set(current_source_ids))!=len(current_source_ids) or any(s not in source_ids for s in current_source_ids):raise ValueError('current_source_ids must be a unique subset of source_ids')
        if purpose!='response' and current_source_ids:raise PermissionError('background processing cannot bypass long-term permission')
        if current_source_ids and current_input_id is None:raise PermissionError('current source exception requires its bound input_id')
        if current_input_id is not None:_id(current_input_id,'current_input_id')
        if purpose=='response' and current_input_id is not None and request_key!='response:'+current_input_id:
            raise PermissionError('bound response request must use its canonical input identity')
        with self.store.transaction(immediate=True):
            self._writable();self._user(user_id);self._normal_allowed(user_id);limits=self._run(run_id)
            generation=self.current_generation(user_id)
            if expected_generation is not None and expected_generation != generation:raise DurableConflict('control generation changed before send admission')
            existing=self.connection.execute('SELECT * FROM dm_attempts WHERE user_id=? AND request_key=?',(user_id,request_key)).fetchone()
            if existing:
                if (existing['purpose'],existing['source_ids_json'],existing['current_source_ids_json'],existing['current_input_id'],existing['run_id'],existing['session_id']) != (purpose,_json(source_ids),_json(current_source_ids),current_input_id,run_id,session_id):raise DurableConflict('canonical request identity changed')
                if existing['generation']!=generation:return {'status':'stale','publishable':False,'response':None,'clear_payload':True,'replayed':True}
                if existing['status']=='succeeded':return {'status':'succeeded','attempt_id':existing['attempt_id'],'publishable':existing['response_json'] is not None,'response':json.loads(existing['response_json']) if existing['response_json'] is not None else None,'response_retained':existing['response_json'] is not None,'replayed':True}
                if existing['status'] in {'sending','attempt_unknown'}:raise AttemptUnknown('external outcome unknown; never resend on a lease timeout')
                raise DurableConflict('canonical request already attempted; explicit new attempt identity required')
            permission='response_external' if purpose=='response' else 'extraction_external'
            bound_input=self.get_input(user_id,current_input_id) if current_input_id is not None else None
            retain_eligible=bool(source_ids) and (purpose!='response' or bool(bound_input and bound_input.get('source_status')=='committed'))
            for sid in source_ids:
                source=self.get_source(user_id,sid,require_use=sid not in current_source_ids)
                if source is None or source['permissions'][permission] is not True:raise PermissionError('source cannot be sent for this purpose')
                retain_eligible=retain_eligible and source['permissions']['save'] is True
                if sid in current_source_ids:
                    if source['input_id']!=current_input_id or bound_input is None or bound_input.get('source_id')!=sid or bound_input.get('run_id')!=run_id or bound_input.get('session_id')!=session_id:
                        raise PermissionError('current source must match its actual input/run/session binding')
                    withdrawn=self.connection.execute("SELECT 1 FROM dm_revocations WHERE user_id=? AND target_kind='source' AND target_id=? AND scope IN ('source','span','span_stop_use','stop_use')",(user_id,sid)).fetchone()
                    if withdrawn:raise PermissionError('explicit withdrawal overrides the current-input exception')
            if job_id is not None:
                job=self.connection.execute('SELECT * FROM dm_jobs WHERE job_id=? AND user_id=?',(job_id,user_id)).fetchone()
                if job is None or job['claim_token']!=claim_token or job['status']!='claimed' or job['generation']!=generation:raise DurableConflict('send requires current persistent job claim')
                if job['source_id'] not in source_ids:raise DurableConflict('job source missing from request')
                if job['run_id']!=run_id or job['session_id']!=session_id:raise DurableConflict('job cannot escape frozen run/session budgets')
            run_count=self.connection.execute('SELECT COUNT(*) FROM dm_attempts WHERE run_id=?',(run_id,)).fetchone()[0]
            session_count=self.connection.execute('SELECT COUNT(*) FROM dm_attempts WHERE run_id=? AND user_id=? AND session_id=?',(run_id,user_id,session_id)).fetchone()[0]
            if run_count>=limits['max_run_calls'] or session_count>=limits['max_session_calls']:raise BudgetExhausted('frozen run/session call limit reached')
            attempt_id='attempt-'+uuid.uuid4().hex
            self.connection.execute("INSERT INTO dm_attempts(attempt_id,user_id,request_key,purpose,source_ids_json,current_source_ids_json,run_id,session_id,generation,status,response_json,usage_json,job_id,created_at,current_input_id,can_retain_response) VALUES (?,?,?,?,?,?,?,?,?,'sending',NULL,NULL,?,?,?,?)",(attempt_id,user_id,request_key,purpose,_json(source_ids),_json(current_source_ids),run_id,session_id,generation,job_id,_now(),current_input_id,int(retain_eligible)))
            if job_id:self.connection.execute("UPDATE dm_jobs SET status='sending' WHERE job_id=?",(job_id,))
            return {'attempt_id':attempt_id,'status':'sending','generation':generation,'replayed':False,'publishable':False}

    def finish_attempt(self, attempt_id: str, *, outcome: str, response: Any = None, usage: dict | None = None,
                       retain_response: bool = True) -> dict:
        if outcome not in {'success','failed','unknown'}:raise ValueError('invalid external outcome')
        if type(retain_response) is not bool:raise ValueError('retain_response must be boolean')
        response_json=_json(response) if response is not None else None
        usage_json=_usage(usage)
        with self.store.transaction(immediate=True):
            self._writable()
            row=self.connection.execute('SELECT * FROM dm_attempts WHERE attempt_id=?',(attempt_id,)).fetchone()
            if row is None:raise DurableConflict('unknown attempt')
            if row['status'] not in {'sending','attempt_unknown'}:raise DurableConflict('attempt already finalized')
            stale=row['generation']!=self.current_generation(row['user_id']) or self._paused_locked(row['user_id'])
            permission='response_external' if row['purpose']=='response' else 'extraction_external'
            current_source_ids=json.loads(row['current_source_ids_json'])
            for sid in json.loads(row['source_ids_json']):
                source=self.get_source(row['user_id'],sid,require_use=sid not in current_source_ids)
                stale=stale or source is None or source['permissions'][permission] is not True
            status='stale' if stale else {'success':'succeeded','failed':'failed','unknown':'attempt_unknown'}[outcome]
            stored=response_json if status=='succeeded' and retain_response and row['can_retain_response'] else None
            self.connection.execute('UPDATE dm_attempts SET status=?,response_json=?,usage_json=? WHERE attempt_id=?',(status,stored,usage_json,attempt_id))
            if row['job_id'] and status in {'stale','failed','attempt_unknown'}:self.connection.execute('UPDATE dm_jobs SET status=? WHERE job_id=?',(status,row['job_id']))
            return {'attempt_id':attempt_id,'status':status,'publishable':status=='succeeded','clear_payload':status!='succeeded','response_retained':stored is not None,'response':response if status=='succeeded' else None}

    def _dependencies(self,user_id: str,dependencies: list[dict]) -> list[tuple[str,int,int]]:
        if type(dependencies) is not list or not dependencies:raise ValueError('effects require source/span dependencies')
        spans=[]
        for dependency in dependencies:
            if type(dependency) is not dict or set(dependency)-{'source_id','start','end'}:raise ValueError('invalid dependency')
            sid=_id(dependency.get('source_id'),'source_id');source=self.get_source(user_id,sid)
            if source is None:raise DurableConflict('dependency source unavailable')
            start,end=dependency.get('start',0),dependency.get('end',len(source['text']))
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(source['text']):raise ValueError('invalid Unicode dependency span')
            revoked=self.connection.execute("SELECT 1 FROM dm_revocations WHERE user_id=? AND scope IN ('span','span_stop_use') AND target_kind='source' AND target_id=? AND start<? AND end>?",(user_id,sid,end,start)).fetchone()
            if revoked:raise DurableConflict('dependency span withdrawn')
            spans.append((sid,start,end))
        if len(set(spans))!=len(spans):raise ValueError('duplicate dependency spans')
        return spans

    def put_effect(self,user_id: str,effect_key: str,object_id: str,*,expected_version: int,payload: dict,
                   dependencies: list[dict],expected_generation: int | None=None,job_id: str | None=None,claim_token: str | None=None) -> dict:
        _id(user_id,'user_id');_id(effect_key,'effect_key');_id(object_id,'object_id')
        if type(expected_version) is not int or expected_version<0:raise ValueError('expected_version must be a nonnegative integer')
        if type(payload) is not dict:raise ValueError('effect payload must be an object')
        fingerprint=hashlib.sha256(_json({'object_id':object_id,'payload':payload,'dependencies':dependencies}).encode()).hexdigest()
        with self.store.transaction(immediate=True):
            self._writable();self._user(user_id);self._normal_allowed(user_id)
            if expected_generation is not None and expected_generation!=self.current_generation(user_id):raise DurableConflict('withdrawal generation changed')
            old=self.connection.execute('SELECT * FROM dm_objects WHERE user_id=? AND object_id=?',(user_id,object_id)).fetchone()
            if old and old['blocked']:raise DurableConflict('object blocked; old values must not revive')
            previous=self.connection.execute('SELECT * FROM dm_effects WHERE user_id=? AND effect_key=?',(user_id,effect_key)).fetchone()
            if previous:
                if previous['fingerprint']!=fingerprint:raise DurableConflict('effect key changed payload')
                return {'version':previous['version'],'replayed':True}
            spans=self._dependencies(user_id,dependencies)
            if job_id is not None:
                job=self.connection.execute('SELECT * FROM dm_jobs WHERE job_id=? AND user_id=?',(job_id,user_id)).fetchone()
                if job is None or job['claim_token']!=claim_token or job['status'] not in {'claimed','sending'} or job['generation']!=self.current_generation(user_id):raise DurableConflict('effect job claim is stale')
            version=old['version'] if old else 0
            if version!=expected_version:raise DurableConflict('expected object version differs')
            next_version=version+1;body=_json(payload)
            if old:
                result=self.connection.execute('UPDATE dm_objects SET version=?,payload_json=? WHERE user_id=? AND object_id=? AND version=? AND blocked=0',(next_version,body,user_id,object_id,expected_version))
                if result.rowcount!=1:raise DurableConflict('concurrent state update')
            else:self.connection.execute('INSERT INTO dm_objects VALUES (?,?,?,?,0)',(user_id,object_id,next_version,body))
            self.connection.execute('INSERT INTO dm_object_versions VALUES (?,?,?,?)',(user_id,object_id,next_version,body))
            self.connection.execute('INSERT INTO dm_effects VALUES (?,?,?,?,?)',(user_id,effect_key,object_id,fingerprint,next_version))
            # Retain dependencies across object versions: deleting the corrected
            # current value must not uncover an older source-backed value.
            self.connection.executemany("INSERT OR IGNORE INTO dm_dependencies VALUES (?,'object',?,?,?,?)",[(user_id,object_id,sid,start,end) for sid,start,end in spans])
            return {'version':next_version,'replayed':False}

    def get_object_state(self,user_id: str,object_id: str) -> dict:
        self._verify_control_anchor()
        row=self.connection.execute('SELECT * FROM dm_objects WHERE user_id=? AND object_id=?',(user_id,object_id)).fetchone()
        if self._paused_locked(user_id):return {'status':'blocked','version':row['version'] if row else 0,'payload':None,'reason':'safety_paused'}
        if row is None:return {'status':'unknown','version':0,'payload':None}
        return {'status':'blocked' if row['blocked'] else 'available','version':row['version'],'payload':None if row['blocked'] else json.loads(row['payload_json'])}

    def put_representation(self,user_id: str,representation_id: str,object_id: str,body: dict,*,dependencies: list[dict]) -> None:
        _id(representation_id,'representation_id');_id(object_id,'object_id')
        with self.store.transaction(immediate=True):
            self._writable();self._normal_allowed(user_id);state=self.get_object_state(user_id,object_id)
            if state['status']!='available':raise DurableConflict('representation object unavailable')
            spans=self._dependencies(user_id,dependencies)
            existing=self.connection.execute('SELECT object_id FROM dm_representations WHERE user_id=? AND representation_id=?',(user_id,representation_id)).fetchone()
            if existing and existing[0]!=object_id:raise DurableConflict('representation identity belongs to another object')
            if self.connection.execute("SELECT 1 FROM dm_revocations WHERE user_id=? AND scope='representation' AND target_id=?",(user_id,representation_id)).fetchone():raise DurableConflict('representation withdrawn')
            self.connection.execute('INSERT INTO dm_representations VALUES (?,?,?,?,1) ON CONFLICT(user_id,representation_id) DO UPDATE SET body_json=excluded.body_json,active=1',(user_id,representation_id,object_id,_json(body)))
            self.connection.execute("DELETE FROM dm_dependencies WHERE user_id=? AND owner_kind='representation' AND owner_id=?",(user_id,representation_id))
            self.connection.executemany("INSERT INTO dm_dependencies VALUES (?,'representation',?,?,?,?)",[(user_id,representation_id,sid,start,end) for sid,start,end in spans])

    def revoke(self,user_id: str,scope: str,target_id: str,*,start: int | None=None,end: int | None=None,
               target_kind: str='source') -> dict:
        if scope not in {'representation','source','span','object','stop_use'}:raise ValueError('invalid withdrawal scope')
        _id(user_id,'user_id');_id(target_id,'target_id')
        if target_kind not in {'source','object'}:raise ValueError('invalid stop-use target kind')
        if scope=='object':target_kind='object'
        if scope=='representation':target_kind='representation'
        with self.store.transaction(immediate=True):
            self._writable();self._user(user_id)
            if scope=='span':
                source=self.get_source(user_id,target_id,require_use=False)
                if source is None or type(start) is not int or type(end) is not int or not 0<=start<end<=len(source['text']):raise ValueError('withdrawal span must exist')
            elif start is not None or end is not None:raise ValueError('only span withdrawal takes offsets')
            self.connection.execute('UPDATE dm_users SET generation=generation+1 WHERE user_id=?',(user_id,))
            generation=self.current_generation(user_id)
            self.connection.execute('INSERT INTO dm_revocations(user_id,scope,target_kind,target_id,start,end,generation,occurred_at) VALUES (?,?,?,?,?,?,?,?)',(user_id,scope,target_kind,target_id,start,end,generation,_now()))
            objects=set()
            if target_kind=='object':
                objects.add(target_id)
                dependencies=self.connection.execute("SELECT DISTINCT d.source_id,d.start,d.end FROM dm_dependencies d WHERE d.user_id=? AND ((d.owner_kind='object' AND d.owner_id=?) OR (d.owner_kind='representation' AND d.owner_id IN (SELECT representation_id FROM dm_representations WHERE user_id=? AND object_id=?)))",(user_id,target_id,user_id,target_id)).fetchall()
                for dep in dependencies:
                    span_scope='span_stop_use' if scope=='stop_use' else 'span'
                    self.connection.execute('INSERT INTO dm_revocations(user_id,scope,target_kind,target_id,start,end,generation,occurred_at) VALUES (?,?,?,?,?,?,?,?)',(user_id,span_scope,'source',dep['source_id'],dep['start'],dep['end'],generation,_now()))
                    if span_scope=='span':
                        original=self.get_source(user_id,dep['source_id'],require_use=False,_internal_control_read=True)
                        if original:
                            body=original['text'];body=body[:dep['start']]+'█'*(dep['end']-dep['start'])+body[dep['end']:]
                            self.connection.execute('UPDATE dm_sources SET text=? WHERE user_id=? AND source_id=?',(body,user_id,dep['source_id']))
                    for dependent in self.connection.execute('SELECT owner_kind,owner_id FROM dm_dependencies WHERE user_id=? AND source_id=? AND start<? AND end>?',(user_id,dep['source_id'],dep['end'],dep['start'])).fetchall():
                        if dependent['owner_kind']=='object':objects.add(dependent['owner_id'])
                        else:self.connection.execute('UPDATE dm_representations SET active=0,body_json=NULL WHERE user_id=? AND representation_id=?',(user_id,dependent['owner_id']))
            elif scope in {'source','span','stop_use'}:
                sql='SELECT owner_kind,owner_id FROM dm_dependencies WHERE user_id=? AND source_id=?';params=[user_id,target_id]
                if scope=='span':sql+=' AND start<? AND end>?';params += [end,start]
                for dep in self.connection.execute(sql,params).fetchall():
                    if dep['owner_kind']=='object':objects.add(dep['owner_id'])
                    else:self.connection.execute('UPDATE dm_representations SET active=0,body_json=NULL WHERE user_id=? AND representation_id=?',(user_id,dep['owner_id']))
                if scope=='source':self.connection.execute("UPDATE dm_sources SET active=0,text='' WHERE user_id=? AND source_id=?",(user_id,target_id))
                elif scope=='stop_use':
                    row=self.connection.execute('SELECT permissions_json FROM dm_sources WHERE user_id=? AND source_id=?',(user_id,target_id)).fetchone()
                    if row:
                        permissions=json.loads(row[0]);permissions['long_term_use']=False
                        self.connection.execute('UPDATE dm_sources SET permissions_json=? WHERE user_id=? AND source_id=?',(_json(permissions),user_id,target_id))
                else:
                    replacement=source['text'][:start]+'█'*(end-start)+source['text'][end:]
                    self.connection.execute('UPDATE dm_sources SET text=? WHERE user_id=? AND source_id=?',(replacement,user_id,target_id))
            else:self.connection.execute('UPDATE dm_representations SET active=0,body_json=NULL WHERE user_id=? AND representation_id=?',(user_id,target_id))
            for oid in objects:
                self.connection.execute('INSERT INTO dm_objects VALUES (?,?,1,NULL,1) ON CONFLICT(user_id,object_id) DO UPDATE SET version=version+1,payload_json=NULL,blocked=1',(user_id,oid))
                self.connection.execute('UPDATE dm_object_versions SET payload_json=NULL WHERE user_id=? AND object_id=?',(user_id,oid))
                self.connection.execute('UPDATE dm_representations SET active=0,body_json=NULL WHERE user_id=? AND object_id=?',(user_id,oid))
            # Conservative per-user fencing prevents stale results/caches from
            # containing a withdrawn span even when a provider supplied extras.
            self.connection.execute("UPDATE dm_attempts SET response_json=NULL,status=CASE WHEN status='succeeded' THEN 'stale' ELSE status END WHERE user_id=?",(user_id,))
            self.connection.execute("UPDATE dm_jobs SET status='stale' WHERE user_id=? AND status IN ('queued','claimed','sending')",(user_id,))
            watermark=self.connection.execute('SELECT MAX(sequence) FROM dm_revocations').fetchone()[0]
            self._persist_control_anchor()
            return {'logical_blocked':True,'primary_payloads_cleared':scope!='stop_use','source_payloads_erased':scope in {'source','span','object'},'control_status':'stopped_use' if scope=='stop_use' else 'withdrawn','replica_cleanup':'not_claimed_external_copies_not_controlled','generation':generation,'control_watermark':watermark,'blocked_objects':sorted(objects),'scope':scope,'target_kind':target_kind}

    def stop_use_object(self,user_id: str,object_id: str) -> dict:
        return self.revoke(user_id,'stop_use',object_id,target_kind='object')

    def control_ledger(self) -> dict:
        rows=[dict(r) for r in self.connection.execute('SELECT * FROM dm_revocations ORDER BY sequence')]
        watermark=rows[-1]['sequence'] if rows else 0
        return {'watermark':watermark,'sha256':hashlib.sha256(_json(rows).encode()).hexdigest()}

    def begin_maintenance(self) -> dict:
        with self.store.transaction(immediate=True):
            self.connection.execute("UPDATE dm_meta SET value=1 WHERE key='maintenance'")
            return self.control_ledger()

    def close_for_maintenance(self) -> dict:
        ledger=self.begin_maintenance();self.close();return ledger

    def finish_maintenance(self, *, latest_control_ledger: dict | None) -> dict:
        """Refuse stale backups rather than replaying an incomplete control log.

        The coordinator must close old connections and provide its latest
        external checkpoint. This module does not pretend that a VM-local file
        or an old backup is a current control ledger.
        """
        with self.store.transaction(immediate=True,_maintenance=True):
            if self.connection.execute("SELECT value FROM dm_meta WHERE key='maintenance'").fetchone()[0]!=1:raise DurableConflict('explicit maintenance window required')
            current=self.control_ledger()
            if latest_control_ledger is None or latest_control_ledger!=current:raise DurableConflict('latest complete control ledger required; stale restore refused')
            self.connection.execute("UPDATE dm_meta SET value=0 WHERE key='maintenance'")
            return {'status':'ready','control_watermark':current['watermark'],'replica_cleanup':'not_claimed'}
