"""Pure parsers for live jackpot sources — fixtures are verbatim captures
from the real endpoints (prod VM, 2026-08-05)."""

import importlib.util
import sys
from pathlib import Path

_LIVE_PATH = Path(__file__).parent.parent.parent / "src" / "services" / "lottery_live.py"
_spec = importlib.util.spec_from_file_location("lottery_live", _LIVE_PATH)
lottery_live = importlib.util.module_from_spec(_spec)
sys.modules["lottery_live"] = lottery_live
_spec.loader.exec_module(lottery_live)

parse_dollar_amount = lottery_live.parse_dollar_amount
parse_megamillions_payload = lottery_live.parse_megamillions_payload
parse_powerball_api = lottery_live.parse_powerball_api
parse_powerball_homepage = lottery_live.parse_powerball_homepage


class TestDollarAmount:
    def test_forms(self):
        assert parse_dollar_amount("$786 Million") == 786e6
        assert parse_dollar_amount("$1.2 Billion") == 1.2e9
        assert parse_dollar_amount("$341,600,000") == 341_600_000.0
        assert parse_dollar_amount("no money here") is None


# Verbatim structure of the powerball.com homepage jackpot block (captured
# from the prod VM — this is what the CDN serves non-browser clients).
PB_HOMEPAGE = """
<div><span class="title-label yellow | lh-1 text-center mx-auto mb-2 py-1 px-3">
  Estimated Jackpot
</span>
<span class="game-jackpot-number text-xxxl lh-1 text-center">$786 Million</span>
</div>
<div class="row winners-group mb-3"><span class="title-label yellow | lh-1 text-center mx-auto mb-2 py-1 px-3">
  Cash Value
</span>
<span class="game-jackpot-number text-lg lh-1 text-center">$341.6 Million</span>
</div>
"""


class TestPowerballHomepage:
    def test_real_markup(self):
        r = parse_powerball_homepage(PB_HOMEPAGE)
        assert r == {"advertised": 786e6, "cash_value": 341_600_000.0}

    def test_missing_jackpot_returns_none(self):
        assert parse_powerball_homepage("<html>maintenance</html>") is None

    def test_missing_cash_still_returns_advertised(self):
        html = PB_HOMEPAGE.split("winners-group")[0]
        r = parse_powerball_homepage(html)
        assert r["advertised"] == 786e6 and r["cash_value"] == 0.0


class TestPowerballApi:
    def test_real_shape(self):
        body = '[{"field_prize_amount": "$786 Million", "field_prize_amount_cash": "$341.6 Million"}]'
        r = parse_powerball_api(body)
        assert r == {"advertised": 786e6, "cash_value": 341_600_000.0}

    def test_html_body_returns_none_for_fallback(self):
        # The CDN serves the SPA homepage to non-browser clients — the JSON
        # parse must come up None so the caller falls back to the scrape.
        assert parse_powerball_api("<!DOCTYPE html><html>...</html>") is None


class TestMegaMillions:
    def test_real_xml_wrapped_json(self):
        body = (
            '<string xmlns="http://tempuri.org/">'
            '{"Jackpot": {"NextPrizePool": 70000000.0, "NextCashValue": 29700000.0}}'
            "</string>"
        )
        r = parse_megamillions_payload(body)
        assert r == {"advertised": 70e6, "cash_value": 29.7e6}

    def test_garbage_returns_none(self):
        assert parse_megamillions_payload("<string>not json</string>") is None
        assert parse_megamillions_payload("") is None
