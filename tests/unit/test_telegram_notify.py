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

    def test_each_sport_has_emoji_entry(self):
        # Every sport the system queues picks for must have an emoji
        # in _SPORT_EMOJI — otherwise the digest line falls back to
        # the • placeholder, which looks broken next to sibling sports.
        # If you add a new sport's recs script and don't update the
        # emoji map, this test catches it.
        for sport in ("soccer", "nhl", "nba", "nfl", "tennis", "mma", "horse_racing"):
            assert sport in tn._SPORT_EMOJI, f"_SPORT_EMOJI missing {sport!r}"
            assert tn._SPORT_EMOJI[sport], f"_SPORT_EMOJI[{sport!r}] is empty"

    def test_horse_racing_emoji_renders(self):
        # Horse racing value-bet alert: home_team=horse, away_team='Field',
        # league_name=track. The 🐎 emoji must appear in the rendered
        # output and the horse vs Field framing must survive into HTML.
        rec = tn.Alert(
            sport="horse_racing",
            league_name="Newton Abbot R3",
            home_team="Jena d'Oudairies",
            away_team="Field",
            match_date=datetime(2026, 6, 3, 15, 0),
            market_label="Win",
            predicted_outcome="Jena d'Oudairies",
            confidence=0.31,
            probabilities={"win": 0.31},
            odds_decimal=3.75,
            expected_value=0.16,
            recommended_stake=14.0,
            bookmaker="Betfair Exchange",
        )
        out = tn.render_digest([rec])
        assert "🐎" in out
        assert "Jena d'Oudairies vs Field" in out
        assert "Newton Abbot R3" in out
        assert "Win" in out
        # Value-bet mode triggers because expected_value is set.
        assert "EV +16%" in out
        assert "@ 3.75" in out
        assert "Betfair Exchange" in out

    def test_default_header_uses_count(self):
        out = tn.render_digest([_alert(), _alert()])
        # Header includes the count so the user sees at-a-glance how
        # many picks the digest contains.
        assert "2 high-confidence" in out

    def test_custom_header_overrides_default(self):
        out = tn.render_digest([_alert()], header="Custom title")
        assert "Custom title" in out

    def test_value_bet_mode_renders_ev_odds_stake(self):
        # Setting expected_value flips the formatter to value-bet mode.
        # The line should show EV %, odds, and the recommended stake —
        # NOT the raw probability breakdown the prediction mode uses.
        rec = tn.Alert(
            sport="nhl",
            league_name="NHL",
            home_team="Canadiens",
            away_team="Rangers",
            match_date=datetime(2025, 3, 15, 19, 0),
            market_label="Moneyline",
            predicted_outcome="home",
            confidence=0.62,
            probabilities={"home": 0.62, "away": 0.38},
            odds_decimal=1.85,
            expected_value=0.147,
            recommended_stake=25.0,
            bookmaker="bet365",
        )
        out = tn.render_digest([rec])
        assert "EV +15%" in out
        assert "@ 1.85" in out
        assert "stake $25" in out
        assert "bet365" in out
        assert "model 62%" in out
        assert "💰" in out  # value-bet marker emoji
        # Should NOT also dump the raw probability list.
        assert "away 38%" not in out

    def test_prediction_mode_unchanged_when_ev_absent(self):
        # Backwards compat: alerts without EV still render in the
        # original prediction format.
        out = tn.render_digest([_alert(sport="soccer")])
        assert "EV" not in out
        assert "💰" not in out
        # Raw probability breakdown is present in prediction mode.
        assert "home" in out

    def test_non_numeric_probability_value_renders_as_string(self):
        # monitor_models piggybacks on the Alert shape by stuffing a
        # markdown body into probabilities={"body": "..."}. The
        # renderer must not crash trying to format a string as %.
        alert = _alert(sport="soccer")
        alert.probabilities = {"body": "drift detected: ECE 0.15"}
        out = tn.render_digest([alert])
        # The string passes through verbatim.
        assert "body drift detected: ECE 0.15" in out


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


class _FakeRedisPipeline:
    """Minimal redis pipeline stub: collects ops and replays them on
    execute(). Good enough to verify the helper's queue semantics
    without a live redis."""

    def __init__(self, client):
        self.client = client
        self.ops = []

    def rpush(self, key, *values):
        self.ops.append(("rpush", key, values))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))
        return self

    def lrange(self, key, start, end):
        self.ops.append(("lrange", key, start, end))
        return self

    def delete(self, key):
        self.ops.append(("delete", key))
        return self

    def execute(self):
        results = []
        for op in self.ops:
            if op[0] == "rpush":
                self.client.store.setdefault(op[1], []).extend(op[2])
                results.append(len(self.client.store[op[1]]))
            elif op[0] == "expire":
                results.append(True)
            elif op[0] == "lrange":
                results.append(list(self.client.store.get(op[1], [])))
            elif op[0] == "delete":
                self.client.store.pop(op[1], None)
                results.append(1)
        return results


class _FakeRedis:
    def __init__(self):
        self.store: dict = {}

    def pipeline(self):
        return _FakeRedisPipeline(self)


class TestEnqueueAndDrain:
    def test_enqueue_empty_returns_zero_no_redis_lookup(self):
        with patch.object(tn, "_redis_client") as mock_rc:
            depth = tn.enqueue_alerts([])
        assert depth == 0
        mock_rc.assert_not_called()

    def test_enqueue_without_redis_returns_zero(self):
        with patch.object(tn, "_redis_client", return_value=None):
            depth = tn.enqueue_alerts([_alert()])
        assert depth == 0

    def test_enqueue_pushes_and_sets_ttl(self):
        fake = _FakeRedis()
        with patch.object(tn, "_redis_client", return_value=fake):
            depth = tn.enqueue_alerts([_alert(), _alert()])
        assert depth == 2
        assert len(fake.store[tn.DEFAULT_QUEUE_KEY]) == 2

    def test_enqueue_accumulates_across_calls(self):
        # Soccer pushes, then NHL pushes — drain should see both.
        fake = _FakeRedis()
        with patch.object(tn, "_redis_client", return_value=fake):
            tn.enqueue_alerts([_alert(sport="soccer")])
            depth = tn.enqueue_alerts([_alert(sport="nhl", market_label="Moneyline")])
        assert depth == 2
        assert len(fake.store[tn.DEFAULT_QUEUE_KEY]) == 2

    def test_drain_sends_one_combined_digest_with_both_sports(self):
        fake = _FakeRedis()
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "12345",
        }
        with patch.object(tn, "_redis_client", return_value=fake):
            tn.enqueue_alerts([_alert(sport="soccer", home="Arsenal", away="Chelsea")])
            tn.enqueue_alerts([_alert(sport="nhl", market_label="Moneyline", home="Canadiens", away="Rangers")])

            with patch.dict("os.environ", env, clear=False), patch.object(tn, "requests") as mock_requests:
                mock_requests.post.return_value.raise_for_status.return_value = None
                result = tn.drain_and_send_digest(header="Auspex picks")

        assert result == {"drained": 2, "sent": 1}
        body = mock_requests.post.call_args.kwargs["json"]["text"]
        # Both sports present in the same message — this is the core
        # promise of the combined digest path.
        assert "Arsenal vs Chelsea" in body
        assert "Canadiens vs Rangers" in body
        assert "⚽" in body
        assert "🏒" in body
        # Queue is empty after drain so the next pipeline run starts clean.
        assert tn.DEFAULT_QUEUE_KEY not in fake.store

    def test_drain_orders_soccer_before_nhl(self):
        # Within the digest, soccer picks should appear before NHL — the
        # frontend already shows them in this order so the digest should
        # match. Locked here so a future re-sort doesn't silently flip it.
        fake = _FakeRedis()
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "12345",
        }
        with patch.object(tn, "_redis_client", return_value=fake):
            # Push NHL first to prove the sort isn't relying on insertion
            # order.
            tn.enqueue_alerts([_alert(sport="nhl", market_label="Moneyline", home="A", away="B")])
            tn.enqueue_alerts([_alert(sport="soccer", home="C", away="D")])

            with patch.dict("os.environ", env, clear=False), patch.object(tn, "requests") as mock_requests:
                mock_requests.post.return_value.raise_for_status.return_value = None
                tn.drain_and_send_digest()

        body = mock_requests.post.call_args.kwargs["json"]["text"]
        assert body.index("C vs D") < body.index("A vs B")

    def test_drain_empty_queue_sends_nothing(self):
        fake = _FakeRedis()
        env = {
            "ENABLE_TELEGRAM_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "12345",
        }
        with (
            patch.object(tn, "_redis_client", return_value=fake),
            patch.dict("os.environ", env, clear=False),
            patch.object(tn, "requests") as mock_requests,
        ):
            result = tn.drain_and_send_digest()
        assert result == {"drained": 0, "sent": 0}
        mock_requests.post.assert_not_called()

    def test_alert_roundtrips_through_redis(self):
        # The dict serializer + parser pair must preserve every field,
        # including the datetime, otherwise the drained digest would
        # render with wrong / missing data.
        original = _alert(sport="nhl", market_label="Puck Line", home="Canadiens", away="Rangers")
        d = tn._alert_to_dict(original)
        # Sanity: datetime is JSON-safe.
        import json as _json

        encoded = _json.dumps(d)
        decoded = _json.loads(encoded)
        restored = tn._alert_from_dict(decoded)
        assert restored == original
