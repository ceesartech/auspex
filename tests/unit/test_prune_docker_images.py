"""Unit tests for scripts/prune_docker_images.py.

A FakeDocker answers the four read commands the script issues (ps,
inspect, image ls, image inspect) from an in-memory image store shaped on
the VM's 2026-11-02 state (containerd image store: Metadata.LastTagTime is
the pull/untag time) and records every `docker image rm`. What is pinned:
images a container uses, the newest N deploys (including an unused, bad
newest deploy after a rollback to the one before it) and anything inside the
grace (boundary included) are never removed; deploys are ordered by the
EARLIEST pull time of their images; an unreadable time keeps the image and
fails the run; any docker failure, an empty container list or unreadable
container images remove nothing; the dry run removes nothing; rm never uses
-f.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prune_docker_images", REPO / "scripts" / "prune_docker_images.py")
assert spec and spec.loader
pdi = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pdi  # dataclasses resolve annotations via sys.modules
spec.loader.exec_module(pdi)

NOW = datetime(2026, 11, 3, 5, 30, tzinfo=timezone.utc)
API, AIRFLOW, FRONTEND = pdi.OWN_REPOS


def sha(n: int) -> str:
    return f"{n:040x}"


def ts(dt: datetime | None) -> str:
    return "0001-01-01T00:00:00Z" if dt is None else dt.strftime("%Y-%m-%dT%H:%M:%S.123456789Z")


class FakeDocker:
    def __init__(self):
        self.images: dict[str, dict] = {}  # id -> inspect doc
        self.containers: dict[str, str] = {}  # container id -> image id
        self.calls: list[list[str]] = []
        self.removed: list[str] = []
        self.fail: dict[str, str] = {}  # first arg(s) -> error, e.g. {"ps": "..."}
        self.rm_refuse: set[str] = set()
        self.inspect_blank = False  # `docker inspect` prints only blank lines

    def add(self, img_id: str, tags: list[str], pulled: datetime | None) -> str:
        full = "sha256:" + img_id.ljust(64, "0")
        self.images[full] = {"Id": full, "RepoTags": tags, "Metadata": {"LastTagTime": ts(pulled)}}
        return full

    def run(self, img_full_id: str) -> None:
        self.containers[f"c{len(self.containers)}"] = img_full_id

    def deploy(self, n: int, pulled: datetime, *, shared_frontend: str | None = None) -> dict[str, str]:
        ids = {
            "api": self.add(f"a{n:03d}", [f"{API}:{sha(n)}"], pulled),
            "airflow": self.add(f"b{n:03d}", [f"{AIRFLOW}:{sha(n)}"], pulled + timedelta(seconds=5)),
        }
        if shared_frontend:
            doc = self.images[shared_frontend]
            doc["RepoTags"].append(f"{FRONTEND}:{sha(n)}")
            doc["Metadata"]["LastTagTime"] = ts(pulled)
            ids["frontend"] = shared_frontend
        else:
            ids["frontend"] = self.add(f"f{n:03d}", [f"{FRONTEND}:{sha(n)}"], pulled)
        return ids

    def __call__(self, args: list[str]) -> str:
        self.calls.append(list(args))
        key = " ".join(args[:2])
        for k, err in self.fail.items():
            if key.startswith(k):
                raise pdi.DockerError(err)
        if args[:1] == ["ps"]:
            assert args == ["ps", "--all", "--quiet", "--no-trunc"]
            return "".join(f"{c}\n" for c in self.containers)
        if args[:1] == ["inspect"]:
            assert args[1:3] == ["--format", "{{.Image}}"]
            if self.inspect_blank:
                return "\n" * len(args[3:])
            return "".join(f"{self.containers[c]}\n" for c in args[3:])
        if args[:2] == ["image", "ls"]:
            return "".join(f"{i}\n" for i in self.images)
        if args[:2] == ["image", "inspect"]:
            return json.dumps([self.images[i] for i in args[2:]])
        if args[:2] == ["image", "rm"]:
            assert len(args) == 3 and not args[2].startswith("-"), args  # never -f
            ref = args[2]
            if ref in self.rm_refuse:
                raise pdi.DockerError(f"conflict: unable to remove repository reference {ref!r}")
            self.removed.append(ref)
            return f"Untagged: {ref}\n"
        raise AssertionError(f"unexpected docker call {args}")


@pytest.fixture
def vm() -> FakeDocker:
    """Shaped on 2026-11-02: 5 deploys, the newest running; third-party
    images in use; :latest leftovers untagged days ago; a stopped
    container (airflow-init) on its own image."""
    d = FakeDocker()
    t0 = datetime(2026, 10, 26, 11, 45, tzinfo=timezone.utc)
    for n, hours in enumerate([0, 30, 100, 150, 175], start=1):
        ids = d.deploy(n, t0 + timedelta(hours=hours))
    for img in ids.values():
        d.run(img)  # deploy 5 is live
    d.run(d.add("grafana_new", ["grafana/grafana:latest"], t0 + timedelta(days=4)))
    d.run(d.add("airflow_base", ["apache/airflow:3.3.0-python3.11"], t0 - timedelta(days=60)))  # stopped init
    d.add("grafana_old", [], t0 + timedelta(days=4))  # untagged when :latest moved
    d.add("postgres_old", [], t0)
    d.add("unused_tag", ["busybox:latest"], t0)
    return d


def plan_refs(plan, which):
    return sorted(r for r, _ in getattr(plan, which))


# ---------------------------------------------------------------- the plan


def test_keeps_used_images_and_newest_three_deploys_removes_the_rest(vm):
    rc = pdi.main(["--apply"], docker=vm, now=NOW)
    assert rc == 0
    removed = set(vm.removed)
    for n in (1, 2):
        assert {f"{API}:{sha(n)}", f"{AIRFLOW}:{sha(n)}", f"{FRONTEND}:{sha(n)}"} <= removed
    for n in (3, 4, 5):
        assert not any(sha(n) in r for r in removed)
    assert "sha256:" + "grafana_old".ljust(64, "0") in removed
    assert "sha256:" + "postgres_old".ljust(64, "0") in removed
    assert "busybox:latest" in removed
    assert "grafana/grafana:latest" not in removed
    assert "apache/airflow:3.3.0-python3.11" not in removed  # a stopped container still counts


def test_an_image_in_use_is_never_removed_even_when_old_and_outside_the_newest(vm):
    old_api = next(i for i, d in vm.images.items() if f"{API}:{sha(1)}" in d["RepoTags"])
    vm.run(old_api)  # a rolled-back api still on deploy 1
    pdi.main(["--apply"], docker=vm, now=NOW)
    assert f"{API}:{sha(1)}" not in vm.removed
    assert f"{AIRFLOW}:{sha(1)}" in vm.removed  # the rest of deploy 1 is still old


def test_rollback_keeps_the_bad_newer_deploy_and_two_before_it():
    # The rollback case: deploy N (6) was bad, the containers run N-1 (5),
    # and N was pulled more than 24 h ago and is unused. N is still one of
    # the newest 3 by pull time, so N, N-1 and N-2 stay; N-3 and older go.
    d = FakeDocker()
    t0 = NOW - timedelta(days=10)
    deploys = {n: d.deploy(n, t0 + timedelta(days=n)) for n in range(1, 7)}  # deploy 6 pulled 4 days ago
    for img in deploys[5].values():
        d.run(img)
    assert pdi.main(["--apply", "--keep-deploys", "3", "--grace-hours", "24"], docker=d, now=NOW) == 0
    for n in (6, 5, 4):
        assert not any(sha(n) in r for r in d.removed), n
    for n in (3, 2, 1):
        assert {f"{API}:{sha(n)}", f"{AIRFLOW}:{sha(n)}", f"{FRONTEND}:{sha(n)}"} <= set(d.removed), n


def test_keep_deploys_is_honoured(vm):
    pdi.main(["--apply", "--keep-deploys", "1"], docker=vm, now=NOW)
    assert any(sha(4) in r for r in vm.removed) and not any(sha(5) in r for r in vm.removed)


def test_grace_protects_a_fresh_pull_not_yet_running(vm):
    # A deploy that pulled 2 h ago but has not restarted the containers yet,
    # and a re-pulled :latest whose old copy was untagged 3 h ago.
    vm.deploy(6, NOW - timedelta(hours=2))
    vm.add("prom_old", [], NOW - timedelta(hours=3))
    pdi.main(["--apply", "--keep-deploys", "1"], docker=vm, now=NOW)
    assert not any(sha(6) in r for r in vm.removed)
    assert not any(sha(5) in r for r in vm.removed)  # live
    assert "sha256:" + "prom_old".ljust(64, "0") not in vm.removed
    assert any(sha(4) in r for r in vm.removed)  # past the grace, not newest, not used


def test_deploys_are_ordered_by_the_earliest_pull_of_their_images():
    # Deploy 1's frontend image is shared with (re-tagged by) deploy 3, so
    # its LastTagTime is deploy 3's; deploy 1 must still rank oldest.
    d = FakeDocker()
    t0 = NOW - timedelta(days=5)
    first = d.deploy(1, t0)
    d.deploy(2, t0 + timedelta(days=1))
    live = d.deploy(3, t0 + timedelta(days=2), shared_frontend=first["frontend"])
    for img in live.values():
        d.run(img)
    used, images = pdi.load_state(d)
    plan = pdi.plan_prune(used, images, now=NOW, keep_deploys=2, grace=timedelta(hours=24))
    assert [s for s, _ in plan.deploys] == [sha(3), sha(2), sha(1)]
    removed = plan_refs(plan, "remove")
    assert f"{API}:{sha(1)}" in removed and f"{AIRFLOW}:{sha(1)}" in removed
    # The shared frontend image is in use (deploy 3), so neither of its tags goes.
    assert not any(FRONTEND in r for r in removed)


def test_untagged_images_are_removed_by_id_and_tags_by_ref(vm):
    used, images = pdi.load_state(vm)
    plan = pdi.plan_prune(used, images, now=NOW, keep_deploys=3, grace=timedelta(hours=24))
    for ref, why in plan.remove:
        assert ref.startswith("sha256:") == why.startswith("untagged"), (ref, why)


# ---------------------------------------------------------------- fail closed


def test_unreadable_pull_time_keeps_the_image_and_fails_the_run(vm, caplog):
    vm.add("mystery", ["mystery:1"], None)
    rc = pdi.main(["--apply"], docker=vm, now=NOW)
    assert rc == 1
    assert "mystery:1" not in vm.removed
    assert "KEPT mystery:1 — its pull time is unreadable" in caplog.text
    assert "busybox:latest" in vm.removed  # the rest of the plan still ran


def test_unreadable_time_on_a_deploy_keeps_every_deploy(vm):
    doc = next(d for d in vm.images.values() if f"{API}:{sha(1)}" in d["RepoTags"])
    doc["Metadata"]["LastTagTime"] = "garbage"
    rc = pdi.main(["--apply"], docker=vm, now=NOW)
    assert rc == 1
    assert not any(r.startswith(pdi.OWN_REPOS) for r in vm.removed)


@pytest.mark.parametrize("failing", ["ps", "inspect", "image ls", "image inspect"])
def test_any_docker_read_failure_removes_nothing(vm, failing, caplog):
    vm.fail[failing] = "Cannot connect to the Docker daemon"
    rc = pdi.main(["--apply"], docker=vm, now=NOW)
    assert rc == 1 and vm.removed == []
    assert "Nothing removed" in caplog.text


def test_no_containers_listed_removes_nothing(vm, caplog):
    vm.containers.clear()
    assert pdi.main(["--apply"], docker=vm, now=NOW) == 1
    assert vm.removed == [] and "listed no containers" in caplog.text


def test_container_images_unreadable_removes_nothing(vm, caplog):
    # `docker ps` lists containers but `docker inspect` gives no image IDs:
    # every image would look unused, so refuse.
    vm.inspect_blank = True
    assert pdi.main(["--apply"], docker=vm, now=NOW) == 1
    assert vm.removed == [] and not any(c[:2] == ["image", "rm"] for c in vm.calls)
    assert "could not read the image of any container" in caplog.text


def test_grace_boundary_an_image_exactly_grace_old_is_kept():
    grace = timedelta(hours=24)
    edge = pdi.Image(id="sha256:" + "e" * 64, tags=["edge:1"], last_tag_time=NOW - grace)
    older = pdi.Image(id="sha256:" + "f" * 64, tags=["older:1"], last_tag_time=NOW - grace - timedelta(seconds=1))
    plan = pdi.plan_prune({"sha256:" + "0" * 64}, [edge, older], now=NOW, keep_deploys=3, grace=grace)
    assert plan_refs(plan, "keep") == ["edge:1"]
    assert plan_refs(plan, "remove") == ["older:1"]


def test_garbled_image_json_removes_nothing(vm):
    real = vm.__call__

    def garbled(args):
        out = real(args)
        return out[:-5] if args[:2] == ["image", "inspect"] else out

    assert pdi.main(["--apply"], docker=garbled, now=NOW) == 1
    assert not any(c[:2] == ["image", "rm"] for c in vm.calls)


def test_a_refused_rm_is_reported_and_the_rest_still_runs(vm, caplog):
    vm.rm_refuse.add("busybox:latest")
    rc = pdi.main(["--apply"], docker=vm, now=NOW)
    assert rc == 1
    assert "could not remove busybox:latest" in caplog.text
    assert f"{API}:{sha(1)}" in vm.removed


def test_dry_run_removes_nothing(vm, caplog):
    caplog.set_level(logging.INFO, logger="prune_docker_images")
    rc = pdi.main([], docker=vm, now=NOW)
    assert rc == 0
    assert not any(c[:2] == ["image", "rm"] for c in vm.calls)
    assert "REMOVE busybox:latest" in caplog.text and "DRY RUN" in caplog.text


@pytest.mark.parametrize("argv", [["--keep-deploys", "0"], ["--grace-hours", "0.5"]])
def test_refuses_settings_that_could_drop_the_rollback_target(vm, argv):
    with pytest.raises(SystemExit):
        pdi.main(["--apply", *argv], docker=vm, now=NOW)
    assert vm.calls == []


def test_parse_docker_time():
    assert pdi.parse_docker_time("2026-11-02T09:27:26.462616576Z") == datetime(
        2026, 11, 2, 9, 27, 26, 462616, tzinfo=timezone.utc
    )
    assert pdi.parse_docker_time("2026-11-02T09:27:26Z") == datetime(2026, 11, 2, 9, 27, 26, tzinfo=timezone.utc)
    assert pdi.parse_docker_time("2026-11-02T10:27:26.5+01:00") == datetime(
        2026, 11, 2, 9, 27, 26, 500000, tzinfo=timezone.utc
    )
    for bad in (None, "", "0001-01-01T00:00:00Z", "yesterday", "2026-11-02 09:27:26"):
        assert pdi.parse_docker_time(bad) is None


def test_real_runner_raises_with_stderr(monkeypatch):
    class Proc:
        returncode = 1
        stdout = ""
        stderr = "permission denied while trying to connect to the Docker daemon socket"

    monkeypatch.setattr(pdi.subprocess, "run", lambda *a, **k: Proc())
    with pytest.raises(pdi.DockerError, match="permission denied"):
        pdi.run_docker(["ps", "--all"])
