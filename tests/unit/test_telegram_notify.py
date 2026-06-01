"""Unit tests for scripts/telegram_notify — digest rendering, chunking,
env-gating, and one-message-per-run behavior."""

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


tn = _load("telegram_notify", "telegram_notify.py")


def _alert(sport="soccer", confidence=0.72, market_label="1X2", home="Arsenal", away="Chelsea"):
    return tn.Alert(
        sport=sport,
        league_name="Premier League" if sport == "soccer" else "NHL",
        home_team=home,
        away_team=away,
        match_date=datetime(2025, 3, 15, 15, 0),
        market_label=market_label,
        predicted_outcome="home",
        confidence=confidence,
        probabilities={"home": confidence, "away": 1 - confidence},
    )


class TestRenderDigest:
    def test_empty_returns_empty_string(self):
        assert tn.render_digest([]) == ""

    def test_renders_one_pick_with_sport_emoji_and_market(self):
        out = tn.render_digest([_alert(sport="soccer")])
        assert "⚽" in out
        assert "Arsenal vs Chelsea" in out
        assert "1X2" in out
        assert "<b>" in out  # HTML formatting present

    def test_renders_mixed_sports(self):
        out = tn.render_digest(
            [
                _alert(sport="soccer", home="Arsenal", away="Chelsea"),
                _alert(sport="nhl", market_label="Puck Line", home="Canadiens", away="Rangers"),
            ]
        )
        assert "⚽" in out
        assert "🏒" in out
        assert "Arsenal vs Chelsea" in out
        assert "Canadiens vs Rangers" in out
        assert "Puck Line" in out

    def test_default_header_uses_count(self):
        out = tn.render_digest([_alert(), _alert()])
        # Header includes the count so the user sees at-a-glance how
        # many picks the digest contains.
        assert "2 high-confidence" in out

    def test_custom_header_overrides_default(self):
        out = tn.render_digest([_alert()], header="Custom title")
        assert "Custom title" in out


class TestChunking:
    def test_short_digest_is_single_chunk(self):
        out = tn._chunk_text("short message")
        assert out == ["short message"]

    def test_splits_on_blank_line_between_picks(self):
        # Build a digest that's longer than the safe limit by repeating
        # a "pick block" separated by blank lines.
        block = "pick line " * 50  # ~500 chars
        digest = "\n\n".join([block] * 10)  # ~5000 chars total
        assert len(digest) > tn._SAFE_CHUNK_SIZE
        chunks = tn._chunk_text(digest)
        assert len(chunks) > 1
        # No chunk exceeds the limit.
        assert all(len(c) <= tn._SAFE_CHUNK_SIZE for c in chunks)
        # All blocks are present across chunks (no silent drop).
        rejoined = "\n\n".join(chunks)
        assert rejoined.count(block) == 10


class TestSendTelegramDigest:
    def test_empty_alerts_returns_zero_without_calling_http(self):
        with patch.object(tn, "requests") as mock_requests:
            sent = tn.send_telegram_digest([])
        assert sent == 0
        mock_requests.post.assert_not_called()

    def test_disabled_env_returns_zero_without_calling_http(self):
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "false",
            "TELEGRAM_BOT_TOKEN": "t",
            "TELEGRAM_CHAT_ID": "c",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(tn, "requests") as mock_requests:
            sent = tn.send_telegram_digest([_alert()])
        assert sent == 0
        mock_requests.post.assert_not_called()

    def test_missing_token_returns_zero(self):
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_CHAT_ID": "c",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(tn, "requests") as mock_requests:
            sent = tn.send_telegram_digest([_alert()])
        assert sent == 0
        mock_requests.post.assert_not_called()

    def test_sends_exactly_one_message_for_small_digest(self):
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "12345",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(tn, "requests") as mock_requests:
            mock_requests.post.return_value.raise_for_status.return_value = None
            sent = tn.send_telegram_digest([_alert(), _alert(), _alert()])
        assert sent == 1
        assert mock_requests.post.call_count == 1
        # The single message body contains all three picks.
        body = mock_requests.post.call_args.kwargs["json"]["text"]
        assert body.count("Arsenal vs Chelsea") == 3

    def test_chunks_oversized_digest_into_multiple_messages(self):
        # Build 200 alerts so the rendered digest blows past 3900 chars.
        many = [_alert(home=f"Home{i}", away=f"Away{i}") for i in range(200)]
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "12345",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(tn, "requests") as mock_requests:
            mock_requests.post.return_value.raise_for_status.return_value = None
            sent = tn.send_telegram_digest(many)
        # Should chunk into multiple messages, each under the limit.
        assert sent > 1
        assert mock_requests.post.call_count == sent
        for call in mock_requests.post.call_args_list:
            body = call.kwargs["json"]["text"]
            assert len(body) <= tn._SAFE_CHUNK_SIZE

    def test_network_failure_stops_further_chunks(self):
        # If chunk 1 fails, we don't blast chunks 2/3 — better to surface
        # a partial digest than to spam retries.
        many = [_alert(home=f"H{i}", away=f"A{i}") for i in range(200)]

        class _Err(Exception):
            pass

        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "12345",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(tn, "requests") as mock_requests:
            mock_requests.RequestException = _Err
            mock_requests.post.side_effect = _Err("connection reset")
            sent = tn.send_telegram_digest(many)
        assert sent == 0
        # Exactly one attempt before giving up.
        assert mock_requests.post.call_count == 1
