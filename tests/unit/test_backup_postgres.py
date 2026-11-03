"""Unit tests for scripts/backup_postgres.py's local rotation.

The rotation deletes database dumps, so every branch is pinned: the count
cap applies only with B2 configured and only after a HEAD on the dump's own
B2 key shows the same byte size; a definite 404 (a failed or skipped upload)
is healed by uploading that dump and verifying it before the delete, so one
failed upload never makes later runs exit 3; a size mismatch, a 403, an
unreachable B2, a failed re-upload or a client that cannot be built keeps the
file (ERROR, exit 3); without B2 only the age rule runs; the dump the run just
wrote is never deleted, whatever its stamp; nothing but <db>-<stamp>.dump is
ever a candidate; a bad setting exits 4 before any dump. A FakeS3 stands in
for boto3 (not installed for these tests).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DB = "betting_system"
NOW = datetime(2026, 11, 3, 2, 0, 1, tzinfo=timezone.utc)


def _load():
    spec = importlib.util.spec_from_file_location("backup_postgres_under_test", REPO / "scripts" / "backup_postgres.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bp = _load()


class FakeClientError(Exception):
    """Shaped like botocore's ClientError. A HEAD has no body, so the code is
    the bare HTTP status: prod B2 answered a HEAD on a missing key with
    Code '404' / HTTPStatusCode 404 (botocore 1.35.99, 2026-11-03)."""

    def __init__(self, status: int, code: str, message: str = "Not Found"):
        super().__init__(f"An error occurred ({code}) when calling the HeadObject operation: {message}")
        self.response = {"Error": {"Code": code, "Message": message}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeS3:
    def __init__(self, objects: dict[str, int] | None = None, head_error: Exception | None = None):
        self.objects = dict(objects or {})
        self.head_error = head_error
        self.forbidden: set[str] = set()  # keys whose HEAD answers 403
        self.heads: list[str] = []
        self.uploads: list[str] = []
        self.upload_error: Exception | None = None
        self.upload_size_delta = 0  # != 0: the upload lands with the wrong size

    def head_object(self, Bucket, Key):
        assert Bucket == "auspex-backups"
        self.heads.append(Key)
        if self.head_error is not None:
            raise self.head_error
        if Key in self.forbidden:
            raise FakeClientError(403, "403", "Forbidden")
        if Key not in self.objects:
            raise FakeClientError(404, "404")
        return {"ContentLength": self.objects[Key]}

    def upload_file(self, path, bucket, key):
        assert bucket == "auspex-backups"
        if self.upload_error is not None:
            raise self.upload_error
        self.uploads.append(key)
        self.objects[key] = Path(path).stat().st_size + self.upload_size_delta


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H%M%SZ")


def make_dump(d: Path, dt: datetime, size: int = 100, db: str = DB) -> Path:
    f = d / f"{db}-{stamp(dt)}.dump"
    f.write_bytes(b"x" * size)
    return f


def in_b2(*files: Path, size_delta: int = 0) -> dict[str, int]:
    return {f"postgres/{f.name}": f.stat().st_size + size_delta for f in files}


def names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


def rotate(d: Path, current: Path, s3: FakeS3 | None, **kw):
    kw.setdefault("retention_days", 7)
    kw.setdefault("keep", 3)
    if s3 is not None:
        kw.setdefault("s3_bucket", "auspex-backups")
        kw.setdefault("s3_client_factory", lambda: s3)
    return bp.rotate_local_backups(d, DB, current=current, now=NOW, **kw)


# ---------------------------------------------------------------- candidates


def test_only_this_databases_stamped_dumps_are_candidates(tmp_path):
    keep = {
        "settle_mma_held_rows_apply_20261102T224301Z.csv",
        "fixture_orphans_tennis_20261031T004915Z.jsonl.gz",
        "other_db-2026-10-01T020001Z.dump",
        f"{DB}-latest.dump",
        f"{DB}-2026-10-01T020001Z.dump.partial",
        f"x{DB}-2026-10-01T020001Z.dump",
    }
    for n in keep:
        (tmp_path / n).write_text("audit")
    (tmp_path / f"{DB}-2026-09-01T020001Z.dump").mkdir()  # a directory named like a dump
    (tmp_path / "odds_hist_cache").mkdir()
    old = make_dump(tmp_path, NOW - timedelta(days=40))
    cur = make_dump(tmp_path, NOW)

    assert [p for _, p in bp.local_dumps(tmp_path, DB)] == [cur, old]
    deleted, kept = rotate(tmp_path, cur, None)  # no B2: the age rule alone
    assert deleted == [old] and kept == []
    assert names(tmp_path) == keep | {cur.name, f"{DB}-2026-09-01T020001Z.dump", "odds_hist_cache"}


def test_plan_orders_by_the_stamp_in_the_name_not_mtime(tmp_path):
    newer = make_dump(tmp_path, NOW - timedelta(hours=1))
    older = make_dump(tmp_path, NOW - timedelta(hours=5))
    os.utime(older, (NOW.timestamp(), NOW.timestamp()))  # touched later than `newer`
    cur = make_dump(tmp_path, NOW)
    plan = bp.plan_rotation(bp.local_dumps(tmp_path, DB), current=cur, now=NOW, retention_days=7, keep=2)
    assert plan == [(older, "beyond the newest 2")]
    assert newer.exists()


# ---------------------------------------------------------------- with B2


def test_b2_verified_dumps_beyond_the_cap_are_deleted_newest_kept(tmp_path):
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h), size=300 + h) for h in (1, 2, 3, 4, 5)]
    cur = make_dump(tmp_path, NOW, size=999)
    s3 = FakeS3(in_b2(*olds))

    deleted, kept = rotate(tmp_path, cur, s3)

    assert kept == []
    assert sorted(deleted) == sorted(olds[2:])
    assert names(tmp_path) == {cur.name, olds[0].name, olds[1].name}
    # Only the dumps it deleted were checked, each at its own upload key.
    assert sorted(s3.heads) == sorted(f"postgres/{f.name}" for f in olds[2:])


def test_one_days_ops_pile_of_21_dumps_rotates_to_the_newest_three(tmp_path):
    day = datetime(2026, 11, 2, 2, 0, 1, tzinfo=timezone.utc)
    olds = [make_dump(tmp_path, day + timedelta(minutes=40 * i)) for i in range(21)]
    cur = make_dump(tmp_path, NOW)
    s3 = FakeS3(in_b2(*olds))
    deleted, kept = rotate(tmp_path, cur, s3)
    assert kept == [] and len(deleted) == 19
    assert names(tmp_path) == {cur.name, olds[-1].name, olds[-2].name}
    # A wrapper's "backup landed" reads B2; locally the fresh dump is the newest left.
    assert bp.local_dumps(tmp_path, DB)[0][1] == cur


def test_a_dump_missing_from_b2_is_uploaded_verified_then_deleted(tmp_path, caplog):
    # Its upload failed (exit 2) or was skipped: the HEAD answers a definite 404.
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h), size=200 + h) for h in (1, 2, 3, 4)]
    cur = make_dump(tmp_path, NOW)
    objs = in_b2(*olds)
    del objs[f"postgres/{olds[3].name}"]  # never landed
    s3 = FakeS3(objs)
    caplog.set_level(logging.INFO, logger="backup_postgres")

    deleted, kept = rotate(tmp_path, cur, s3)

    assert sorted(deleted) == sorted([olds[2], olds[3]]) and kept == []
    assert s3.uploads == [f"postgres/{olds[3].name}"]  # only the missing one, at its own key
    assert s3.objects[f"postgres/{olds[3].name}"] == 204
    # HEAD before (404), HEAD after the upload (size check), then the delete.
    assert s3.heads.count(f"postgres/{olds[3].name}") == 2
    assert "uploaded by this run and verified" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_failed_re_upload_keeps_the_file_and_names_it(tmp_path, caplog):
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h)) for h in (1, 2, 3, 4)]
    cur = make_dump(tmp_path, NOW)
    objs = in_b2(*olds)
    del objs[f"postgres/{olds[3].name}"]
    s3 = FakeS3(objs)
    s3.upload_error = ConnectionError("B2 refused the upload")

    deleted, kept = rotate(tmp_path, cur, s3)

    assert deleted == [olds[2]] and kept == [olds[3]] and olds[3].exists()
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and olds[3].name in errors[0] and "re-upload failed" in errors[0]


def test_a_re_upload_that_lands_with_the_wrong_size_keeps_the_file(tmp_path, caplog):
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h)) for h in (1, 2, 3)]
    cur = make_dump(tmp_path, NOW)
    objs = in_b2(olds[0], olds[1])
    s3 = FakeS3(objs)
    s3.upload_size_delta = -7
    deleted, kept = rotate(tmp_path, cur, s3)
    assert deleted == [] and kept == [olds[2]] and olds[2].exists()
    assert "size mismatch" in caplog.text


def test_a_403_is_not_a_missing_copy_nothing_uploaded(tmp_path, caplog):
    # A key without list rights gets 403 for a missing object: not definite.
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h)) for h in (1, 2, 3)]
    cur = make_dump(tmp_path, NOW)
    s3 = FakeS3(in_b2(olds[0], olds[1]))
    s3.forbidden.add(f"postgres/{olds[2].name}")
    deleted, kept = rotate(tmp_path, cur, s3)
    assert deleted == [] and kept == [olds[2]] and olds[2].exists()
    assert s3.uploads == []
    assert "403" in caplog.text


def test_size_mismatch_keeps_the_file_and_never_overwrites_the_copy(tmp_path, caplog):
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h)) for h in (1, 2, 3)]
    cur = make_dump(tmp_path, NOW)
    objs = in_b2(olds[0], olds[1])
    objs.update(in_b2(olds[2], size_delta=-1))  # a truncated upload
    s3 = FakeS3(objs)
    deleted, kept = rotate(tmp_path, cur, s3)
    assert deleted == [] and kept == [olds[2]] and olds[2].exists()
    assert "is 99 bytes, the local file 100" in caplog.text
    assert s3.uploads == []


@pytest.mark.parametrize(
    "exc, missing",
    [
        (FakeClientError(404, "404"), True),
        (FakeClientError(404, "NoSuchKey"), True),
        (FakeClientError(404, "NotFound"), True),
        (FakeClientError(404, "NoSuchBucket"), False),
        (FakeClientError(403, "403", "Forbidden"), False),
        (FakeClientError(403, "404"), False),
        (FakeClientError(500, "InternalError"), False),
        (FakeClientError(503, "SlowDown"), False),
        (ConnectionError("Could not connect to the endpoint URL"), False),
        (Exception("404 Not Found"), False),  # a 404 in the text alone is not an answer
    ],
)
def test_is_missing_object_only_on_a_definite_404(exc, missing):
    assert bp.is_missing_object(exc) is missing


def test_is_missing_object_tolerates_odd_responses():
    for response in (None, "404", {}, {"Error": "404"}, {"Error": {"Code": "404"}}, {"ResponseMetadata": 404}):
        e = Exception("x")
        e.response = response  # type: ignore[attr-defined]
        assert bp.is_missing_object(e) is False


def test_b2_copy_without_a_size_is_not_verified(tmp_path):
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h)) for h in (1, 2, 3)]
    cur = make_dump(tmp_path, NOW)

    class NoLength(FakeS3):
        def head_object(self, Bucket, Key):
            super().head_object(Bucket, Key)
            return {}

    deleted, kept = rotate(tmp_path, cur, NoLength(in_b2(*olds)))
    assert deleted == [] and kept == [olds[2]] and olds[2].exists()


def test_unreachable_b2_keeps_every_candidate(tmp_path, caplog):
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h)) for h in (1, 2, 3, 4, 5)]
    cur = make_dump(tmp_path, NOW)
    s3 = FakeS3(in_b2(*olds), head_error=ConnectionError("Could not connect to the endpoint URL"))
    deleted, kept = rotate(tmp_path, cur, s3)
    assert deleted == [] and sorted(kept) == sorted(olds[2:])
    assert all(f.exists() for f in olds)
    assert caplog.text.count("Could not connect") == 3
    assert s3.uploads == []  # unreachable is not "missing": nothing is re-uploaded


def test_s3_client_that_cannot_be_built_keeps_everything(tmp_path, caplog):
    olds = [make_dump(tmp_path, NOW - timedelta(days=d)) for d in (1, 2, 3, 9)]
    cur = make_dump(tmp_path, NOW)

    def no_boto():
        raise ImportError("No module named 'boto3'")

    deleted, kept = bp.rotate_local_backups(
        tmp_path,
        DB,
        current=cur,
        now=NOW,
        retention_days=7,
        keep=3,
        s3_bucket="auspex-backups",
        s3_client_factory=no_boto,
    )
    assert deleted == [] and sorted(kept) == sorted(olds[2:])
    assert all(f.exists() for f in olds)
    assert "could not create the S3 client: ImportError" in caplog.text


def test_age_rule_with_b2_also_waits_for_the_verified_copy(tmp_path):
    a = make_dump(tmp_path, NOW - timedelta(days=8))  # within the newest 3 but past 7 days
    cur = make_dump(tmp_path, NOW)
    deleted, kept = rotate(tmp_path, cur, FakeS3(in_b2(a, size_delta=5)))
    assert deleted == [] and kept == [a] and a.exists()
    deleted, kept = rotate(tmp_path, cur, FakeS3(in_b2(a)))
    assert deleted == [a] and kept == []


def test_no_candidates_never_builds_a_client(tmp_path):
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h)) for h in (1, 2)]
    cur = make_dump(tmp_path, NOW)

    def boom():
        raise AssertionError("no B2 call is needed when nothing is to be deleted")

    deleted, kept = rotate(tmp_path, cur, None, s3_bucket="auspex-backups", s3_client_factory=boom)
    assert deleted == [] and kept == [] and all(f.exists() for f in olds)


# ---------------------------------------------------------------- without B2


def test_without_b2_only_the_age_rule_applies(tmp_path, caplog):
    recent = [make_dump(tmp_path, NOW - timedelta(hours=6 * i)) for i in range(1, 21)]  # 5 days, 20 dumps
    stale = [make_dump(tmp_path, NOW - timedelta(days=d)) for d in (8, 30)]
    cur = make_dump(tmp_path, NOW)
    caplog.set_level(logging.INFO, logger="backup_postgres")

    deleted, kept = rotate(tmp_path, cur, None)

    assert sorted(deleted) == sorted(stale) and kept == []
    assert all(f.exists() for f in recent) and cur.exists()
    assert "only copies" in caplog.text


def test_retention_zero_disables_the_age_rule(tmp_path):
    old = make_dump(tmp_path, NOW - timedelta(days=400))
    cur = make_dump(tmp_path, NOW)
    assert rotate(tmp_path, cur, None, retention_days=0) == ([], [])
    assert old.exists()


# ---------------------------------------------------------------- the current dump


@pytest.mark.parametrize("with_b2", [True, False])
def test_the_dump_this_run_wrote_is_never_deleted(tmp_path, with_b2):
    # Clock skew: this run's stamp is the OLDEST and past retention, and
    # it is outside the newest 3 — still kept.
    cur = make_dump(tmp_path, NOW - timedelta(days=30))
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h)) for h in (1, 2, 3, 4)]
    s3 = FakeS3(in_b2(cur, *olds)) if with_b2 else None
    deleted, kept = rotate(tmp_path, cur, s3)
    assert cur.exists() and cur not in deleted and cur not in kept
    if with_b2:
        # The newest 3 by stamp are olds[0..2]; the current dump sits
        # outside them and is kept on top (N + 1 files, never N - 1).
        assert deleted == [olds[3]]
        assert f"postgres/{cur.name}" not in s3.heads
    else:
        assert deleted == []


def test_keep_below_one_is_refused(tmp_path):
    cur = make_dump(tmp_path, NOW)
    with pytest.raises(ValueError):
        rotate(tmp_path, cur, FakeS3({}), keep=0)


def test_a_failed_delete_is_kept_and_reported(tmp_path, monkeypatch, caplog):
    olds = [make_dump(tmp_path, NOW - timedelta(hours=h)) for h in (1, 2, 3, 4)]
    cur = make_dump(tmp_path, NOW)
    real_unlink = Path.unlink

    def unlink(self, *a, **kw):
        if self.name == olds[3].name:
            raise PermissionError(13, "Permission denied", str(self))
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", unlink)
    deleted, kept = rotate(tmp_path, cur, FakeS3(in_b2(*olds)))
    assert deleted == [olds[2]] and kept == [olds[3]]
    assert "the delete failed" in caplog.text and olds[3].name in caplog.text


# ---------------------------------------------------------------- main()


class MainRun:
    """main() with pg_dump and boto3 faked."""

    def __init__(self, tmp_path: Path, s3: FakeS3):
        self.dir = tmp_path
        self.s3 = s3
        self.dumped: list[Path] = []

    def past(self, n: int, *, in_b2_too: bool = True, hours: float = 1) -> list[Path]:
        now = datetime.now(timezone.utc)
        files = [make_dump(self.dir, now - timedelta(hours=hours * (i + 1))) for i in range(n)]
        if in_b2_too:
            self.s3.objects.update(in_b2(*files))
        return files

    def __call__(self, *extra: str, b2: bool = True, now: datetime | None = None) -> int:
        argv = ["--database-url", f"postgresql://u:p@db:5432/{DB}", "--local-dir", str(self.dir), *extra]
        if b2:
            argv += ["--s3-bucket", "auspex-backups"]
        return bp.main(argv, now=now)


@pytest.fixture
def run_main(tmp_path, monkeypatch):
    for var in ("BACKUP_S3_BUCKET", "BACKUP_LOCAL_KEEP", "BACKUP_LOCAL_RETENTION", "BACKUP_S3_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    run = MainRun(tmp_path, FakeS3())

    def fake_pg_dump(url, output_path):
        output_path.write_bytes(b"d" * 500)
        run.dumped.append(output_path)

    monkeypatch.setattr(bp, "run_pg_dump", fake_pg_dump)
    monkeypatch.setattr(bp, "make_s3_client", lambda endpoint_url: run.s3)
    return run


def test_main_b2_ok_uploads_then_keeps_the_newest_three(run_main):
    olds = run_main.past(6)
    assert run_main() == 0
    new = run_main.dumped[-1]
    assert run_main.s3.uploads == [f"postgres/{new.name}"]
    assert names(run_main.dir) == {new.name, olds[0].name, olds[1].name}


def test_main_failed_upload_exits_2_and_keeps_the_new_dump(run_main):
    olds = run_main.past(5)
    run_main.s3.upload_error = ConnectionError("B2 refused the upload")
    assert run_main() == 2
    new = run_main.dumped[-1]
    assert new.exists()
    # Older dumps whose own copy is verified still rotate; the new one counts toward the 3.
    assert names(run_main.dir) == {new.name, olds[0].name, olds[1].name}


def test_main_exits_3_when_rotation_keeps_an_unverified_dump(run_main, caplog):
    olds = run_main.past(4)
    run_main.s3.objects[f"postgres/{olds[3].name}"] += 1  # a copy of a different size
    assert run_main() == 3
    assert olds[3].exists() and not olds[2].exists() and run_main.dumped[-1].exists()
    assert olds[3].name in caplog.text


def _hourly(start: datetime, n: int) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(n)]


def test_main_one_failed_upload_heals_and_later_runs_stay_green(run_main):
    # The review's case: run 2's upload fails (exit 2; the DAG retry takes a
    # NEW dump), and that dump is never uploaded by its own run. Under a plain
    # "keep what is not in B2" rule every run from the 5th on exited 3 for
    # good (rc 0,2,0,0,3,3,...). Now the run that rotates it out uploads it.
    times = _hourly(datetime(2026, 11, 3, 2, 0, 1, tzinfo=timezone.utc), 8)
    rcs = []
    for i, t in enumerate(times):
        run_main.s3.upload_error = ConnectionError("B2 refused the upload") if i == 1 else None
        rcs.append(run_main(now=t))
    assert rcs == [0, 2, 0, 0, 0, 0, 0, 0]
    failed = run_main.dumped[1]
    healed_key = f"postgres/{failed.name}"
    # Uploaded exactly once, by the run that rotated it out (the 5th), then deleted.
    assert run_main.s3.uploads.count(healed_key) == 1
    assert run_main.s3.uploads.index(healed_key) == 4  # run 0 + runs 2,3,4 uploaded their own first
    assert not failed.exists()
    assert names(run_main.dir) == {f.name for f in run_main.dumped[-3:]}
    assert {f"postgres/{f.name}" for f in run_main.dumped} <= set(run_main.s3.objects)


def test_main_skip_upload_dump_is_uploaded_when_it_rotates_out(run_main, caplog):
    times = _hourly(datetime(2026, 11, 3, 2, 0, 1, tzinfo=timezone.utc), 5)
    assert run_main("--skip-upload", now=times[0]) == 0
    assert "not in B2 yet" in caplog.text
    skipped = run_main.dumped[0]
    assert run_main.s3.uploads == []
    assert [run_main(now=t) for t in times[1:]] == [0, 0, 0, 0]
    assert not skipped.exists() and f"postgres/{skipped.name}" in run_main.s3.uploads
    assert names(run_main.dir) == {f.name for f in run_main.dumped[-3:]}


def test_main_b2_outage_keeps_everything_then_recovers(run_main):
    times = _hourly(datetime(2026, 11, 3, 2, 0, 1, tzinfo=timezone.utc), 7)
    assert [run_main(now=t) for t in times[:3]] == [0, 0, 0]

    class Down(FakeS3):
        def head_object(self, Bucket, Key):
            raise ConnectionError("Could not connect to the endpoint URL")

        def upload_file(self, path, bucket, key):
            raise ConnectionError("Could not connect to the endpoint URL")

    healthy = run_main.s3
    run_main.s3 = Down()
    assert [run_main(now=t) for t in times[3:5]] == [2, 2]
    assert len(list(run_main.dir.glob("*.dump"))) == 5  # nothing deleted while B2 is down
    run_main.s3 = healthy
    assert [run_main(now=t) for t in times[5:]] == [0, 0]
    # The two outage dumps were uploaded by rotation (or are still in the newest 3).
    assert names(run_main.dir) == {f.name for f in run_main.dumped[-3:]}
    for f in run_main.dumped:
        assert f.exists() or f"postgres/{f.name}" in healthy.objects


def test_main_without_b2_is_age_only(run_main):
    recent = run_main.past(10, in_b2_too=False)
    assert run_main(b2=False) == 0
    assert run_main.s3.heads == [] and run_main.s3.uploads == []
    assert names(run_main.dir) == {run_main.dumped[-1].name, *(f.name for f in recent)}


def test_main_reads_backup_local_keep(run_main, monkeypatch):
    monkeypatch.setenv("BACKUP_LOCAL_KEEP", "5")
    olds = run_main.past(8)
    assert run_main() == 0
    assert names(run_main.dir) == {run_main.dumped[-1].name, *(f.name for f in olds[:4])}


@pytest.mark.parametrize("bad", ["0", "-1", "three", ""])
def test_main_refuses_a_bad_local_keep_with_exit_4(run_main, bad, caplog):
    assert run_main("--local-keep", bad) == 4  # not 2: that means "the upload failed"
    assert run_main.dumped == [] and "no dump taken" in caplog.text and "BACKUP_LOCAL_KEEP" in caplog.text


@pytest.mark.parametrize(
    "var, bad", [("BACKUP_LOCAL_KEEP", "0"), ("BACKUP_LOCAL_RETENTION", "-1"), ("BACKUP_LOCAL_RETENTION", "7d")]
)
def test_main_refuses_a_bad_env_setting_with_exit_4(run_main, monkeypatch, var, bad, caplog):
    monkeypatch.setenv(var, bad)
    assert run_main() == 4
    assert run_main.dumped == [] and var in caplog.text


def test_main_pg_dump_failure_rotates_nothing(run_main, monkeypatch):
    olds = run_main.past(6, in_b2_too=True, hours=24 * 3)  # 3..18 days old, half past the age rule

    def broken(url, output_path):
        output_path.write_bytes(b"partial")
        raise RuntimeError("pg_dump exited 1")

    monkeypatch.setattr(bp, "run_pg_dump", broken)
    assert run_main() == 1
    assert names(run_main.dir) == {f.name for f in olds}  # partial removed, nothing rotated
    assert run_main.s3.heads == []


def test_help_says_the_cap_needs_b2(capsys):
    with pytest.raises(SystemExit):
        bp.main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--local-keep" in out and "Ignored without BACKUP_S3_BUCKET" in out
    assert "the local copy is then the only copy" in out
    assert "exits 4" in out and "is uploaded first" in out
