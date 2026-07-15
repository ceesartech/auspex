# Modal training cutover — explicit runbook

Everything needed to move retraining from the VM to Modal, and back. Follow the
phases in order. **Nothing you do before Phase 6 changes production** — training
keeps running on the VM until you flip the flag in Phase 6.

- **Where things run:** `modal secret create`, `modal run`, `modal token` → your
  **laptop** (Modal CLI, already authed). `dc …` commands → the **VM**
  (`ssh auspex`, `cd /opt/auspex`).
- **The flag:** Airflow Variable `training_backend`. `vm` (current default) =
  train on the VM. `modal` = train on Modal. Flip either way anytime, no deploy.
- **Rollback is always:** `airflow variables set training_backend vm`.

Known values (confirm against your own consoles):
- B2 bucket: **`auxpex-backups`**
- B2 endpoint: **`https://s3.us-west-004.backblazeb2.com`** (region `us-west-004`, auto-derived)
- Dump prefix: `postgres/` · Artifact prefix: `modal-train/`

On the VM, define the compose alias once per shell (all three prod overlays):
```bash
ssh auspex
cd /opt/auspex
alias dc='docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ghcr.yml'
```

---

## Phase 0 — Prerequisites (one-time, laptop)

```bash
# 1. Modal CLI present + authed (you said it is). Verify:
modal --version
modal app list          # should succeed without an auth error

# 2. A repo checkout on your laptop, on the latest main:
cd /path/to/auspex
git pull            # must include commit 1f2da16 or later
ls modal_train/train_modal.py   # exists
```

---

## Phase 1 — Rotate the B2 application key

The trial's B2 key was shared in chat; replace it. Do this in the **Backblaze
web console**:

1. **Backblaze → Account → Application Keys → "Add a New Application Key".**
   - Name: `auspex-modal`
   - Allow access to Buckets: **`auxpex-backups`** (this bucket only)
   - Type of Access: **Read and Write**
   - (Leave file-name prefix + duration empty.)
   - Click **Create New Key**.
2. Copy the two values it shows **once**:
   - `keyID`   → this is your new **`AWS_ACCESS_KEY_ID`**
   - `applicationKey` → your new **`AWS_SECRET_ACCESS_KEY`**
3. **Do not delete the old key yet.** You'll revoke it in Phase 3 after the new
   one is proven working on both the VM and Modal.

---

## Phase 2 — Create the two Modal secrets (laptop)

The Modal functions read B2 + Telegram creds from these secrets — the VM database
is never exposed. Fill in the four `<…>` values (new B2 key from Phase 1; your
Telegram bot token + chat id — the same ones already on the VM).

```bash
# B2 access for pull-dump + push-artifacts. Names MUST be exactly these
# (b2_io.py reads AWS_ACCESS_KEY_ID/SECRET + BACKUP_S3_BUCKET/ENDPOINT_URL):
modal secret create auspex-b2 \
  AWS_ACCESS_KEY_ID='<NEW_B2_KEY_ID>' \
  AWS_SECRET_ACCESS_KEY='<NEW_B2_APP_KEY>' \
  BACKUP_S3_BUCKET='auxpex-backups' \
  BACKUP_S3_ENDPOINT_URL='https://s3.us-west-004.backblazeb2.com'

# Telegram so a failing bundle pages from its own container:
modal secret create auspex-telegram \
  TELEGRAM_BOT_TOKEN='<TELEGRAM_BOT_TOKEN>' \
  TELEGRAM_CHAT_ID='<TELEGRAM_CHAT_ID>' \
  ENABLE_TELEGRAM_NOTIFICATIONS='true'
```

Verify both exist:
```bash
modal secret list      # shows auspex-b2 and auspex-telegram
```

---

## Phase 3 — Wire the VM (Modal token + prove the new B2 key)

The airflow-scheduler triggers Modal, so it needs a Modal token. Create a
**dedicated** token in the Modal dashboard (don't reuse your laptop's):

1. **Modal dashboard → Settings → API Tokens → New Token.** Name it `auspex-vm`.
   Copy the **Token ID** (`ak-…`) and **Token Secret** (`as-…`).
2. On the VM, edit `/opt/auspex/.env` and set (also update the two AWS_* lines to
   the **new** B2 key from Phase 1 — the VM's backups + the pull step use them):
   ```
   MODAL_TOKEN_ID=ak-xxxxxxxx
   MODAL_TOKEN_SECRET=as-xxxxxxxx
   AWS_ACCESS_KEY_ID=<NEW_B2_KEY_ID>
   AWS_SECRET_ACCESS_KEY=<NEW_B2_APP_KEY>
   ```
3. Recreate the two services that read those (env change needs a recreate, not a
   restart):
   ```bash
   dc up -d --force-recreate airflow-scheduler api
   ```
4. **Prove the new B2 key works on the VM** before revoking the old one — force a
   backup and watch for "Upload verified":
   ```bash
   dc exec -T api python /app/scripts/backup_postgres.py 2>&1 | grep -E "Upload verified|ERROR"
   ```
   You want a line like `Upload verified: 1.. bytes in s3://auxpex-backups/postgres/…`.
5. **Now revoke the OLD B2 key** in the Backblaze console (Application Keys →
   the old key → Delete). The new key is proven on both Modal (Phase 4 next) and
   the VM.

---

## Phase 4 — Smoke test ONE bundle (laptop + VM)

Prove the full path on soccer only. From your laptop repo checkout:
```bash
modal run modal_train/train_modal.py --run-id smoke1 --bundles soccer_match_result
```
**Expect:** a function named `soccer_match_result_training` in the Modal
dashboard; `[restore] matches rows = …`; a summary line with **served Brier
≈ 0.594–0.596**; and:
```
Artifacts in B2 under modal-train/smoke1/ — pull + gate on the VM with:
  docker compose exec -T api python /app/scripts/pull_modal_artifacts.py --run-id smoke1
```

Now verify the round-trip + gate on the VM **without touching production**
(`--shadow` gates + reports but writes nothing into staging):
```bash
dc exec -T api python /app/scripts/pull_modal_artifacts.py --run-id smoke1 --shadow
dc exec -T api cat /app/models/modal-incoming/smoke1/promote_decisions.json
```
**Expect:** `soccer_match_result` → `decision: "promote"`, reason "no incumbent
yet — promoted to seed (brier=0.59…)", and a `challenger_brier` in the 0.594–0.596
band. If the Brier is in range, Modal training is faithful.

---

## Phase 5 — Shadow a full 13-bundle run (laptop + VM)

```bash
# Laptop: all 13 named functions, in parallel (~90s wall clock):
modal run modal_train/train_modal.py --run-id shadow1
```
**Expect:** 13 functions, `ok=13 errored=0`, per-bundle served Brier (tennis/mma
may show `—` — no held-out test, which is expected and still promotes).

```bash
# VM: gate all 13, still NO swap:
dc exec -T api python /app/scripts/pull_modal_artifacts.py --run-id shadow1 --shadow
dc exec -T api cat /app/models/modal-incoming/shadow1/promote_decisions.json | python3 -m json.tool
```
**Eyeball check (the important one):** compare each bundle's `challenger_brier`
against the current live numbers from the monitor:
```bash
dc exec -T api python /app/scripts/monitor_models.py --days 30 | grep -iE "brier|soccer|nhl|nba|nfl"
```
Soccer should be ~0.594; the others in the ranges you saw in the 13-bundle trial
(NHL ~0.42–0.58, NBA ~0.35–0.47, NFL ~0.38–0.50). If nothing looks wildly off,
you're clear to cut over. (No incumbents exist yet, so every bundle shows
"promote to seed" — that's expected; the FIRST real run seeds them and every run
after is gated.)

---

## Phase 6 — Cut over (VM)

```bash
# Flip the flag:
dc exec -T airflow-scheduler airflow variables set training_backend modal

# Confirm the DAG re-parsed to the Modal graph (wait ~30s):
dc exec -T airflow-scheduler airflow tasks list retrain_models
#   → should now show: validate_data, modal_trigger, pull_and_gate,
#     swap_production, reload_api, cleanup_old_backups
#     (the train_<sport> tasks are gone)
```

Either wait for the Sunday 04:00 UTC run, or trigger one now:
```bash
dc exec -T airflow-scheduler airflow dags trigger retrain_models
```

Watch it (in the Airflow UI at `airflow.<your-domain>`, or):
```bash
dc logs -f airflow-scheduler | grep -iE "modal_trigger|pull_and_gate|swap_production|reload_api"
```
**Verify after it finishes:**
```bash
# 1. Which bundles promoted this run:
dc exec -T api cat /app/models/staging/promote_decisions.json 2>/dev/null || \
dc exec -T api sh -c 'cat /app/models/production/*/held_out_metrics.json' | head
# 2. A rollback backup was made:
dc exec -T api ls -dt /app/models/production-prev-* | head -1
# 3. The api reloaded models cleanly (no load errors):
dc logs --since 5m api | grep -iE "load_models|loaded|error" | tail
# 4. Recs still generate on the next pipeline tick (or force one):
dc exec -T api python /app/scripts/generate_recommendations.py --days 14 2>&1 | tail -3
```

---

## Phase 7 — Rollback (any time)

**Training back to the VM (instant, no deploy):**
```bash
dc exec -T airflow-scheduler airflow variables set training_backend vm
```

**Restore the previous models** (if a promoted set is bad):
```bash
dc exec -T api bash -c '
  cd /app/models &&
  PREV=$(ls -dt production-prev-* | head -1) &&
  rm -rf production && cp -a "$PREV" production'
dc restart api
```

**Force-promote a bundle the gate rejected** (e.g. you trust it despite a
within-noise regression): delete its incumbent sidecar so the next run treats it
as un-gated —
```bash
dc exec -T api rm /app/models/production/<ensemble_name>/held_out_metrics.json
# ensemble names: ensemble_soccer_match_result, ensemble_nhl_ml, ensemble_nhl_reg,
# ensemble_nhl_pl, ensemble_nhl_tot, ensemble_nba_ml, ensemble_nba_sp,
# ensemble_nba_tot, ensemble_nfl_ml, ensemble_nfl_sp, ensemble_nfl_tot,
# ensemble_tennis_ml, ensemble_mma_ml
```

---

## Phase 8 — Bake, then finalize (~1 month later)

Keep `training_backend=modal` with the VM branch retained as fallback. Each week,
glance at `promote_decisions.json` + `monitor_models.py` for drift. After several
clean runs:

```bash
# Delete the standalone trial + its Modal volumes (data-at-rest cleanup):
git rm -r modal_trial/ && git commit -m "cleanup: remove Modal trial (superseded by modal_train/)" && git push
modal volume rm auspex-trial-data --yes
modal volume rm auspex-trial-models --yes
```

Then reconsider the **box downgrade**: training is off the VM, so measure a peak
week (the `HostMemoryHigh` alert guards you). If peak RAM stays < ~6 GB and you're
not about to grow the corpus, drop CPX41 → CPX31 (≈ half cost) and re-tune the
`mem_limit`s down (api no longer needs 10 GB).

---

## Quick reference

| Action | Command (VM unless noted) |
|---|---|
| Flip to Modal | `dc exec -T airflow-scheduler airflow variables set training_backend modal` |
| Flip to VM (rollback) | `dc exec -T airflow-scheduler airflow variables set training_backend vm` |
| Current backend | `dc exec -T airflow-scheduler airflow variables get training_backend` |
| Manual Modal run (laptop) | `modal run modal_train/train_modal.py --run-id <id>` |
| Gate + stage a run | `dc exec -T api python /app/scripts/pull_modal_artifacts.py --run-id <id>` |
| Gate only, no swap | add `--shadow` to the above |
| Trigger a retrain | `dc exec -T airflow-scheduler airflow dags trigger retrain_models` |
| Per-bundle Modal logs/cost | Modal dashboard → each `<bundle>_training` function |

## Troubleshooting

- **modal_trigger fails "not authenticated":** `MODAL_TOKEN_ID/SECRET` missing or
  wrong in the VM `.env`, or the scheduler wasn't recreated after editing it.
  Re-check Phase 3, then `dc up -d --force-recreate airflow-scheduler`.
- **"No objects under s3://auxpex-backups/postgres/":** no dump exists / wrong
  bucket in the `auspex-b2` secret. Confirm `db_backup_daily` ran and the secret's
  `BACKUP_S3_BUCKET`/`ENDPOINT_URL` are correct.
- **"refusing to train on a stale dump":** the nightly backup is >30h old — fix
  `db_backup_daily` first (a Modal retrain trains on the last dump, so it must be
  fresh).
- **A bundle pages "REJECT":** its served Brier was worse than the incumbent by
  more than the 0.009 floor — production kept the old model (working as intended).
  Read `promote_decisions.json` for the numbers; force-promote via Phase 7 if you
  disagree.
- **pg_restore "unsupported version":** the Modal image ships PG17; if the VM's
  `pg_dump` ever jumps past 17, bump `postgresql-17` in `modal_train/train_modal.py`.
