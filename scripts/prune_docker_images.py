"""Prune Docker images on the VM, keeping a local rollback target.

WHAT IT KEEPS
  * Every image any container uses, running or stopped (by image ID).
  * The images of the newest --keep-deploys deploys (default 3: the
    current one and the two before it). A deploy is one git SHA tag on
    ghcr.io/ceesartech/auspex/{api,airflow,frontend}; deploys are ordered
    by when the VM pulled them (the image's Metadata.LastTagTime, the
    earliest across the deploy's images, so an image shared with a later
    deploy does not make an old deploy look new).
  * Anything pulled or untagged less than --grace-hours ago (default 24),
    so a deploy that has pulled but not yet restarted its containers, or
    an image just pulled by hand for a rollback, is never touched.
WHAT IT REMOVES (with --apply; the default is a dry run that prints the plan)
  Every other image: older deploys' SHA tags, untagged ("dangling")
  images left when a :latest image was re-pulled, and unused third-party
  tags — one `docker image rm <ref>` each, never with -f, so Docker's own
  in-use check still stands behind this one.

WHY NOT `docker image prune -af --filter until=168h` (the old weekly task)
  With the containerd image store (Docker 29 on the VM) `until` compares
  the time an image record was created on the VM (pulled, or untagged when
  its tag moved), not the image's build date. So it (a) kept every untagged
  :latest leftover for a week after each re-pull — 4.9 GB of old grafana/
  postgres/prometheus images on 2026-11-02 — and (b) deleted the previous
  deploy's images, the rollback target, after any quiet week. Deploys
  themselves are cheap: consecutive SHA images share ~4.4 GB of layers and
  add ~10-15 MB each (api) when the dependency layers are unchanged; a
  dependency change makes one deploy carry its own ~8 GB until it drops
  out of the newest 3.

WHY 3 DEPLOYS
  The rollback target is almost always the deploy just before the current
  one (monitoring/runbooks/API_DOWN.md); one more covers a bad deploy
  followed by a quick second one. GHCR keeps every tag, so an older
  rollback re-pulls (the runbook's `dc pull api` already does).

FAIL CLOSED
  Any docker command that fails, an empty container list, or an image whose
  pull time cannot be read: nothing of that kind is removed, an ERROR names
  it, and the exit code is 1 (the DAG turns red). A refused `docker image rm`
  is reported the same way after the rest of the plan has run.

Runs daily from the disk_maintenance DAG inside the airflow scheduler
container (docker CLI + the host socket; /opt/auspex is mounted read-only):
    python /opt/auspex/scripts/prune_docker_images.py            # dry run
    python /opt/auspex/scripts/prune_docker_images.py --apply
By hand on the VM:
    docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ghcr.yml \\
        exec -T airflow-scheduler python /opt/auspex/scripts/prune_docker_images.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("prune_docker_images")

OWN_REPOS = (
    "ghcr.io/ceesartech/auspex/api",
    "ghcr.io/ceesartech/auspex/airflow",
    "ghcr.io/ceesartech/auspex/frontend",
)
SHA_TAG = re.compile(r"[0-9a-f]{40}")
_TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})")

Runner = Callable[[list[str]], str]


class DockerError(RuntimeError):
    pass


def run_docker(args: list[str]) -> str:
    """Run `docker <args>`; stdout on success, DockerError (with stderr) otherwise."""
    proc = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise DockerError(f"docker {' '.join(args[:3])} … exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
    return proc.stdout


def parse_docker_time(value: Optional[str]) -> Optional[datetime]:
    """Docker's RFC 3339 nanosecond times → aware datetime. None for a
    missing, unparseable or zero ("0001-01-01…") time: unknown."""
    if not value:
        return None
    m = _TS.fullmatch(value.strip())
    if not m:
        return None
    base, frac, tz = m.groups()
    if base.startswith("0001-"):
        return None
    iso = base + ("." + (frac or "0")[:6].ljust(6, "0")) + ("+00:00" if tz == "Z" else tz)
    return datetime.fromisoformat(iso).astimezone(timezone.utc)


@dataclass
class Image:
    id: str
    tags: list[str]
    last_tag_time: Optional[datetime]


@dataclass
class Plan:
    keep: list[tuple[str, str]]  # (ref, reason)
    remove: list[tuple[str, str]]  # (ref, reason) — a tag ref, or an image ID when untagged
    unknown: list[tuple[str, str]]  # (ref, why) kept because a pull time is unreadable (ERROR)
    deploys: list[tuple[str, datetime]]  # every SHA deploy found, newest first


def load_state(docker: Runner) -> tuple[set[str], list[Image]]:
    """(image IDs used by any container, every image)."""
    ctr_ids = docker(["ps", "--all", "--quiet", "--no-trunc"]).split()
    if not ctr_ids:
        raise DockerError("docker ps --all listed no containers — refusing: every image would look unused")
    used = {line.strip() for line in docker(["inspect", "--format", "{{.Image}}", *ctr_ids]).splitlines()}
    used.discard("")
    if not used:
        raise DockerError("could not read the image of any container — refusing")

    image_ids = sorted(set(docker(["image", "ls", "--all", "--quiet", "--no-trunc"]).split()))
    if not image_ids:
        return used, []
    raw = json.loads(docker(["image", "inspect", *image_ids]))
    images = [
        Image(
            id=d["Id"],
            tags=[t for t in (d.get("RepoTags") or []) if not t.endswith(":<none>")],
            last_tag_time=parse_docker_time((d.get("Metadata") or {}).get("LastTagTime")),
        )
        for d in raw
    ]
    return used, images


def split_ref(tag: str) -> tuple[str, str]:
    repo, _, version = tag.rpartition(":")
    return repo, version


def plan_prune(used: set[str], images: list[Image], *, now: datetime, keep_deploys: int, grace: timedelta) -> Plan:
    if keep_deploys < 1:
        raise ValueError(f"keep_deploys must be >= 1, got {keep_deploys}")
    cutoff = now - grace
    plan = Plan(keep=[], remove=[], unknown=[], deploys=[])

    # Deploys: SHA tags of our own repos, ordered by the earliest pull time
    # across that SHA's images. One unreadable time → the order is unknown,
    # so every deploy is kept (and reported).
    sha_times: dict[str, list[Optional[datetime]]] = {}
    for img in images:
        for tag in img.tags:
            repo, version = split_ref(tag)
            if repo in OWN_REPOS and SHA_TAG.fullmatch(version):
                sha_times.setdefault(version, []).append(img.last_tag_time)
    known = {sha: [t for t in ts if t is not None] for sha, ts in sha_times.items()}
    order_known = all(len(known[sha]) == len(ts) for sha, ts in sha_times.items())
    if order_known:
        deploy_time = {sha: min(ts) for sha, ts in known.items()}
        plan.deploys = sorted(deploy_time.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        kept_shas = {sha for sha, _ in plan.deploys[:keep_deploys]}
    else:
        kept_shas = set(sha_times)

    for img in images:
        refs = img.tags or [img.id]
        if img.id in used:
            plan.keep += [(r, "used by a container") for r in refs]
            continue
        for ref in refs:
            repo, version = split_ref(ref) if img.tags else ("", "")
            own_deploy = repo in OWN_REPOS and SHA_TAG.fullmatch(version) is not None
            if own_deploy and version in kept_shas:
                if order_known:
                    plan.keep.append((ref, f"one of the newest {keep_deploys} deploys"))
                else:
                    plan.keep.append((ref, "deploy order unknown"))
                    plan.unknown.append((ref, "a deploy image's pull time is unreadable, so no deploy is removed"))
                continue
            if img.last_tag_time is None:
                plan.keep.append((ref, "pull time unreadable"))
                plan.unknown.append((ref, "its pull time is unreadable"))
                continue
            if img.last_tag_time >= cutoff:
                plan.keep.append((ref, f"pulled/untagged {img.last_tag_time:%Y-%m-%d %H:%MZ}, within the grace"))
                continue
            kind = "older deploy" if own_deploy else ("untagged" if not img.tags else "unused tag")
            plan.remove.append((ref, f"{kind}, pulled/untagged {img.last_tag_time:%Y-%m-%d %H:%MZ}"))
    return plan


def main(argv: Optional[list[str]] = None, docker: Runner = run_docker, now: Optional[datetime] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep-deploys", type=int, default=3, help="Newest SHA deploys to keep locally (default 3).")
    p.add_argument(
        "--grace-hours", type=float, default=24.0, help="Never remove an image pulled/untagged this recently."
    )
    p.add_argument("--apply", action="store_true", help="Remove the planned images (default: dry run).")
    args = p.parse_args(argv)
    if args.keep_deploys < 1 or args.grace_hours < 1:
        p.error("--keep-deploys and --grace-hours must be >= 1")
    now = now or datetime.now(timezone.utc)

    try:
        used, images = load_state(docker)
    except (DockerError, json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error("Nothing removed — could not read the docker state: %s", e)
        return 1
    plan = plan_prune(used, images, now=now, keep_deploys=args.keep_deploys, grace=timedelta(hours=args.grace_hours))

    for sha, t in plan.deploys:
        logger.info("deploy %s pulled %s", sha[:12], f"{t:%Y-%m-%d %H:%MZ}")
    for ref, why in plan.keep:
        logger.info("KEEP   %s (%s)", ref, why)
    for ref, why in plan.remove:
        logger.info("REMOVE %s (%s)", ref, why)
    for ref, why in plan.unknown:
        logger.error("KEPT %s — %s (Metadata.LastTagTime); check the docker image store", ref, why)
    logger.info(
        "Plan: %d image(s), %d in use; keep %d ref(s), remove %d.",
        len(images),
        len(used),
        len(plan.keep),
        len(plan.remove),
    )

    failed: list[str] = []
    if args.apply:
        for ref, _ in plan.remove:
            try:
                docker(["image", "rm", ref])
                logger.info("removed %s", ref)
            except DockerError as e:
                logger.error("could not remove %s: %s", ref, e)
                failed.append(ref)
        if plan.remove:
            logger.info("Removed %d of %d planned.", len(plan.remove) - len(failed), len(plan.remove))
    else:
        logger.info("DRY RUN — nothing removed (pass --apply).")

    if failed or plan.unknown:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
