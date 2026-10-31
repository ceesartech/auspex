"""scripts/compare_team_resolution.py: the old-vs-new resolution report.
Its verdicts decide whether a resolver change ships, so a regression must
fail it and a seeded correction must not."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ctr = _load("compare_team_resolution")


def entry(slug, espn_id, name, country="Spain"):
    return {"key": f"{slug}|{espn_id}", "slug": slug, "espn_id": espn_id, "name": name, "country": country}


ENTRIES = [
    entry("esp.1", "88", "Espanyol"),
    entry("mex.1", "225", "Santos", "Mexico"),
    entry("bra.1", "2674", "Santos", "Brazil"),
    entry("esp.1", "83", "Barcelona"),
    entry("esp.1", "1", "Brand New"),
]


class TestCompare:
    def _rows(self, old, new, seed=None):
        return {r["key"]: r["verdict"] for r in ctr.compare(old, new, ENTRIES, seed or {})}

    def test_verdicts(self):
        old = {
            "esp.1|88": ("espanol", ""),  # old resolved, new lost it
            "mex.1|225": ("santos", ""),  # old resolved onto Santos FC's row
            "bra.1|2674": ("santos", ""),
            "esp.1|83": ("fcb", ""),
            "esp.1|1": (None, ""),
        }
        new = {
            "esp.1|88": (None, ""),
            "mex.1|225": ("laguna", "seed"),
            "bra.1|2674": ("santos", "seed"),
            "esp.1|83": ("fcb", "exact"),
            "esp.1|1": ("brand", "fuzzy"),
        }
        got = self._rows(old, new, seed={"225": {"evidence": "ESPN mex.1 id 225 is Santos Laguna"}})
        assert got == {
            "esp.1|88": "REGRESSION",
            "mex.1|225": "SEEDED",
            "bra.1|2674": "same",
            "esp.1|83": "same",
            "esp.1|1": "GAIN",
        }

    def test_an_unseeded_change_of_row_is_a_regression_too(self):
        old = {k: ("x", "") for k in (e["key"] for e in ENTRIES)}
        new = dict(old, **{"esp.1|83": ("y", "fuzzy")})
        assert self._rows(old, new)["esp.1|83"] == "REGRESSION"


class TestCollisions:
    def test_a_row_two_ids_resolve_to_and_a_name_two_ids_share(self):
        rows = [{**e, "new_team_id": tid} for e, tid in zip(ENTRIES, ["espanol", "santos", "santos", "fcb", None])]
        shared_rows, shared_names = ctr.collisions(rows, ENTRIES)
        assert [tid for tid, _ in shared_rows] == ["santos"]
        assert [name for name, _ in shared_names] == ["santos"]

    def test_two_ids_meeting_only_after_the_merge_collide(self):
        # ESPN 2684 (Nacional, Montevideo) lands on 'Nacional' by name; 3472 (C.D. Nacional, Madeira) on its
        # twin, which the team merge folds into 'Nacional'.
        entries = [entry("conmebol.libertadores", "2684", "Nacional"), entry("por.1", "3472", "C.D. Nacional")]
        rows = [{**entries[0], "new_team_id": "nacional"}, {**entries[1], "new_team_id": "cd-twin"}]
        assert ctr.collisions(rows, entries) == ([], [])
        shared_rows, _ = ctr.collisions(rows, entries, fold={"cd-twin": "nacional"})
        assert [tid for tid, _ in shared_rows] == ["nacional"]

    def test_the_merge_map_loader_skips_comments(self, tmp_path):
        path = tmp_path / "map.csv"
        path.write_text("# comment\nnew_team_id,old_team_id,new_name\ncd-twin,nacional,C.D. Nacional\n")
        assert ctr.load_merge_map(str(path)) == {"cd-twin": "nacional"}

    def test_one_id_in_two_competitions_is_no_collision(self):
        entries = [
            entry("eng.1", "364", "Liverpool", "England"),
            entry("uefa.champions", "364", "Liverpool", "International"),
        ]
        rows = [{**e, "new_team_id": "lfc"} for e in entries]
        assert ctr.collisions(rows, entries) == ([], [])


def test_espn_teams_reads_a_saved_response(tmp_path):
    payload = {"sports": [{"leagues": [{"teams": [{"team": {"id": "88", "displayName": "Espanyol"}}, {"team": {}}]}]}]}
    (tmp_path / "esp.1.json").write_text(json.dumps(payload), encoding="utf-8")
    assert ctr.espn_teams("esp.1", str(tmp_path), None) == [{"id": "88", "displayName": "Espanyol"}]


def test_the_default_new_version_is_this_tree(tmp_path):
    assert ctr.scripts_tree(None, None, str(tmp_path)) == str(SCRIPTS)
