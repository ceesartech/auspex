# Modal training trial

A **standalone experiment** — wired into nothing — to validate that our model
training runs correctly on [Modal](https://modal.com), **in parallel across all
13 bundles**, before we decide on a consolidated Modal-backed retrain DAG.

It does not modify `train_all_models.py`, `retrain_models.py`, or anything else
in the repo. It only *reads* the training code and *reuses* a nightly pg_dump.

## What it does

`train_trial.py` defines a Modal app that, **per bundle, in its own container,
in parallel**:

1. restores the real nightly `pg_dump` into an ephemeral local Postgres,
2. runs `python -m training.train_all_models --sport <bundle> --model-type all
   --database-url <local pg> --export-onnx` — the *identical* code path the VM
   uses (`pd.read_sql`), so there's no CSV round-trip risk,
3. captures the held-out **Brier / ECE** from the `CALIBRATION GATE` log line.

The 13 bundles are soccer + NHL(4) + NBA(3) + NFL(3) + tennis + MMA. Horse
racing is excluded (it trains via its own precompute path, not this module).

## Why this shape

- **Parallel, not sequential.** The on-VM DAG trains sequentially *only* because
  training shares the api container ("so the api container isn't running 5
  ensembles' worth of training in parallel", `retrain_models.py`). On Modal each
  bundle gets its own container, so that constraint disappears — this is the
  thing the trial proves out.
- **Restore the dump, don't ship CSVs.** Correctness first: every bundle has its
  own SQL query, so a live-DB restore is faithful for *all* of them at once and
  matches what a future DAG→Modal design would do.
- **Metric comparison, not byte comparison.** XGBoost/LightGBM are seeded but the
  torch NN is not, so artifacts won't be byte-identical to a VM run. Judge the
  held-out Brier within the **ΔBrier ≈ 0.009** noise floor the audit uses.
  Expected `soccer_match_result` ensemble Brier ≈ **0.594–0.596**.

## Run it

Prereqs: a Modal account + token (you have these), and `python -m pip install modal`.

> **Run every `modal` command from the repo root.** The app resolves
> `requirements.txt` and `services/ml-models/src` relative to the working
> directory at launch, so `cd` to the repo root first.

```bash
# 1. Auth (once). Either paste the token or use the browser flow.
modal token set --token-id <id> --token-secret <secret>
#   (or: modal setup)

# 2. Get a fresh dump onto your machine. Cheapest is straight off the VM:
scp auspex:/opt/auspex/backups/$(ssh auspex 'ls -t /opt/auspex/backups/*.dump | head -1 | xargs basename') ./dump.dump
#   (or pull the latest object from Backblaze B2 — same file.)

# 3. Stage it into the Modal volume the app reads from (/data/dump.dump).
modal volume create auspex-trial-data      # no-op if it already exists
modal volume put   auspex-trial-data ./dump.dump /dump.dump

# 4a. Smoke test on ONE bundle first (fast, ~a few min + image build the first time):
modal run modal_trial/train_trial.py --bundles soccer_match_result

# 4b. Then the full parallel fan-out (all 13 at once):
modal run modal_trial/train_trial.py
```

The first run builds the image (installs the app's requirements + CPU torch +
onnx + a Postgres server) — a few minutes, cached thereafter.

## Read the result

The local entrypoint prints a table:

```
bundle                     status    secs  held-out Brier / ECE (ensemble)
------------------------------------------------------------------------------
soccer_match_result        ok         182  Brier=0.5951  ECE=0.0210
nhl_moneyline              ok          96  Brier=...
...
parallel wall-clock: 214s   ok=13  errored=0
```

- **Correct?** Compare each bundle's Brier to a VM baseline (e.g. soccer ≈ 0.594–
  0.596). Within ~0.009 → training is faithful on Modal.
- **Parallel?** `wall-clock` should be roughly the *slowest single bundle*, not
  the sum — that's the win over the VM's sequential ~90 min.
- Trained artifacts (+ ONNX) land in the `auspex-trial-models` volume under
  `/<bundle>/`. Inspect with `modal volume ls auspex-trial-models`.

## Cost & cleanup

Pay-per-second CPU compute; a full parallel run is minutes of a few 4-vCPU
containers → cents to low single dollars, no standing cost. Clean up when done:

```bash
modal volume rm auspex-trial-data --yes
modal volume rm auspex-trial-models --yes
```

## If we like it

The trial's structure is the skeleton of the eventual design: a consolidated
`retrain_models` DAG task that triggers this per-bundle Modal function in
parallel, ships artifacts back to the VM (via B2 or a volume download), then
does the existing staging→production swap + api reload. Nothing here commits us
to that — it just gives us the numbers to decide.

## Troubleshooting

- **`pg_restore: unsupported version (1.NN) in file header`** — the container's
  `pg_restore` is older than the `pg_dump` that wrote the dump. The VM's server
  is PG15 but its api container's `pg_dump` is 17, so dumps are archive v1.16 and
  need `pg_restore >= 17`. The image installs PG17 from the PGDG apt repo for
  exactly this reason; if the VM's `pg_dump` ever jumps to 18, bump the
  `postgresql-17` pin in `train_trial.py`.

## Notes / knobs

- **Memory**: containers default to 8 GiB. Soccer is the heaviest; if a bundle
  OOMs, bump `memory=` on `@app.function`.
- **GPU**: not used — the models are CPU-bound and the NN auto-selects CPU. Add
  `gpu="T4"` to the function only if you later want to A/B a bigger NN.
- **Security**: the training data leaves the VM as a dump into a private Modal
  volume. Delete the `auspex-trial-data` volume after the trial. (And the B2 app
  key shared earlier is still worth rotating.)
