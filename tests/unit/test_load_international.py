"""load_international: every row it writes is one promote_raw can promote. A
tournament with no code, or a code promote_raw.LEAGUE_MAP has no league for,
was written as league 'UNK' (and --include-friendlies as 'FRIEND'); since
2026-11-01 promote_raw refuses such raw rows and exits 3 on every run while
they sit in raw_football_data, so the loader refuses them first."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


intl = _load("load_international")
import promote_raw  # noqa: E402


def _dataset(rows: list[tuple]) -> pd.DataFrame:
    cols = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"]
    return pd.DataFrame(rows, columns=cols)


WC = ("2022-12-18", "Argentina", "France", 3, 3, "FIFA World Cup", "Lusail", "Qatar", True)
CONCACAF = ("2025-06-15", "Curaçao", "Haiti", 2, 1, "CONCACAF Gold Cup", "Houston", "USA", True)


class TestCodes:
    def test_every_tournament_code_is_an_international_league_promote_raw_files(self):
        for tournament, (code, _name) in intl.TOURNAMENT_CODES.items():
            assert promote_raw.LEAGUE_MAP.get(code, (None, None))[1] == "International", (tournament, code)

    def test_the_default_tournaments_all_have_a_code(self):
        assert intl.unknown_tournaments(intl.DEFAULT_TOURNAMENTS) == []

    def test_no_made_up_code_for_an_unknown_tournament(self):
        with pytest.raises(ValueError, match="no code in TOURNAMENT_CODES"):
            intl.normalize(_dataset([WC, (*WC[:5], "Island Games", *WC[6:])]), ["FIFA World Cup", "Island Games"])


class TestMain:
    def _run(self, monkeypatch, argv, rows=(WC, CONCACAF)):
        fetched: list = []
        written: list = []
        monkeypatch.setattr(intl, "fetch_dataset", lambda: fetched.append(1) or _dataset(list(rows)))
        monkeypatch.setattr(intl, "write_database", lambda df, url: written.append(df) or len(df))
        return intl.main([*argv, "--database-url", "pg://x"]), fetched, written

    @pytest.mark.parametrize("tournaments", ["FIFA World Cup,Island Games", "Friendly"])
    def test_a_tournament_promote_raw_could_not_file_is_refused_before_anything_is_fetched(
        self, monkeypatch, tournaments
    ):
        rc, fetched, written = self._run(monkeypatch, ["--tournaments", tournaments])
        assert rc == 2 and fetched == [] and written == []

    def test_include_friendlies_is_gone(self, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, ["--include-friendlies"])
        assert exc.value.code == 2

    def test_a_row_promote_raw_would_refuse_writes_nothing(self, monkeypatch):
        garbled = (*CONCACAF[:1], "Curaçao".encode("utf-8").decode("latin-1"), *CONCACAF[2:])
        rc, _, written = self._run(monkeypatch, ["--tournaments", "FIFA World Cup,CONCACAF Gold Cup"], (WC, garbled))
        assert rc == intl.REFUSED_EXIT_CODE == 3 and written == []

    def test_the_rows_written_are_all_promotable(self, monkeypatch):
        rc, _, written = self._run(monkeypatch, ["--tournaments", "FIFA World Cup,CONCACAF Gold Cup"])
        assert rc == 0
        (df,) = written
        assert list(df["league"]) == ["WC", "GOLD"] and list(df["HomeTeam"]) == ["Argentina", "Curaçao"]
        assert [promote_raw.raw_row_problem(r) for r in df.to_dict("records")] == [None, None]
