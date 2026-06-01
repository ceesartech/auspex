"""Unit tests for precompute_predictions_nhl — sport-aware Telegram
template + per-market thresholds (Phase 4d). Patches requests + env so
no real HTTP or shell config is touched."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ppn = _load("precompute_predictions_nhl", "precompute_predictions_nhl.py")


# Fixed match payload — every test reuses this so the assertions stay
# focused on the gate / template / threshold logic.
SAMPLE = dict(
    league_name="NHL",
    home_team="Canadiens",
    away_team="Rangers",
    match_date=datetime(2025, 3, 15, 19, 0),
    market="moneyline",
    predicted_outcome="home",
    probabilities={"home": 0.72, "away": 0.28},
)


class TestMarketThresholds:
    def test_moneyline_threshold_is_loosest(self):
        # Lock the ordering: moneyline should always be the easiest
        # gate, puck-line and total the strictest. If someone flips
        # them, low-signal markets would spam the chat.
        t = ppn.MARKET_NOTIFY_THRESHOLDS
        assert t["moneyline"] < t["puck_line"]
        assert t["moneyline"] < t["total"]
        assert t["regulation"] < t["puck_line"]

    def test_all_nhl_markets_have_a_threshold(self):
        # Every market the TASKS registry exposes for NHL must have an
        # entry — otherwise the script falls back to the 0.70 default,
        # which silently changes behavior. The fallback is for FUTURE
        # markets, not the four shipping today.
        for market in ("moneyline", "regulation", "puck_line", "total"):
            assert market in ppn.MARKET_NOTIFY_THRESHOLDS


class TestSendTelegramNhl:
    def test_below_threshold_skips(self):
        # Confidence under the gate must short-circuit before the env
        # lookup so we never accidentally fire a low-confidence alert
        # even if env is misconfigured.
        with patch.object(ppn, "requests") as mock_requests:
            sent = ppn.send_telegram_nhl(**SAMPLE, confidence=0.50, threshold=0.60)
        assert sent is False
        mock_requests.post.assert_not_called()

    def test_skip_when_disabled_env(self):
        # ENABLE_TELEGRAM_NOTIFICATIONS not set / not 'true' means no
        # network call even when over threshold.
        with (
            patch.dict(
                "os.environ",
                {
                    "ENABLE_TELEGRAM_NOTIFICATIONS": "false",
                    "TELEGRAM_BOT_TOKEN": "t",
                    "TELEGRAM_CHAT_ID": "c",
                },
                clear=False,
            ),
            patch.object(ppn, "requests") as mock_requests,
        ):
            sent = ppn.send_telegram_nhl(**SAMPLE, confidence=0.80, threshold=0.60)
        assert sent is False
        mock_requests.post.assert_not_called()

    def test_skip_when_token_missing(self):
        # Enabled but creds missing — log + skip, don't crash.
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_CHAT_ID": "c",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(ppn, "requests") as mock_requests:
            sent = ppn.send_telegram_nhl(**SAMPLE, confidence=0.80, threshold=0.60)
        assert sent is False
        mock_requests.post.assert_not_called()

    def test_sends_with_hockey_emoji_and_market_label(self):
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "12345",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(ppn, "requests") as mock_requests:
            mock_requests.post.return_value.raise_for_status.return_value = None
            sent = ppn.send_telegram_nhl(
                **{**SAMPLE, "market": "puck_line"},
                confidence=0.80,
                threshold=0.70,
            )

        assert sent is True
        # Single call, correct URL, body contains the friendly market
        # label + 🏒 — verifies we don't ship 'puck_line' or '⚽' to
        # users.
        assert mock_requests.post.call_count == 1
        kwargs = mock_requests.post.call_args.kwargs
        body_text = kwargs["json"]["text"]
        assert "🏒" in body_text
        assert "Puck Line" in body_text
        assert "puck_line" not in body_text  # raw snake_case must not leak
        assert "⚽" not in body_text

    def test_network_failure_is_swallowed(self):
        # A transient Telegram outage shouldn't fail the whole batch
        # — the function returns False and the run() loop keeps going.
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "12345",
        }

        # Local RequestException class so we don't need to import
        # requests at test-module level (the real module is patched
        # inside ppn anyway).
        class _Err(Exception):
            pass

        with patch.dict("os.environ", env, clear=False), patch.object(ppn, "requests") as mock_requests:
            mock_requests.RequestException = _Err
            mock_requests.post.side_effect = _Err("connection reset")
            sent = ppn.send_telegram_nhl(**SAMPLE, confidence=0.80, threshold=0.60)
        assert sent is False


class TestMarketDisplayLabels:
    def test_all_thresholded_markets_have_display_label(self):
        # The display map and threshold map must cover the same keys —
        # otherwise a notification fires with a raw 'puck_line' string
        # in the user-facing template.
        assert set(ppn.MARKET_DISPLAY_LABELS) == set(ppn.MARKET_NOTIFY_THRESHOLDS)
