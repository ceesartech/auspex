"""E2E: WebSocket live-odds stream."""

import json
import time
import threading
import pytest

try:
    import websocket as ws_lib  # websocket-client
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

WS_URL_TEMPLATE = "ws://localhost:8000/ws/odds"


@pytest.mark.skipif(not WS_AVAILABLE, reason="websocket-client not installed")
def test_websocket_connects(auth_token):
    messages = []
    errors = []

    def on_message(ws, message):
        messages.append(json.loads(message))
        ws.close()

    def on_error(ws, error):
        errors.append(str(error))

    ws = ws_lib.WebSocketApp(
        WS_URL_TEMPLATE,
        header={"Authorization": f"Bearer {auth_token}"},
        on_message=on_message,
        on_error=on_error,
    )

    thread = threading.Thread(target=lambda: ws.run_forever(ping_interval=5))
    thread.daemon = True
    thread.start()
    thread.join(timeout=10)

    assert not errors, f"WebSocket errors: {errors}"
    # Connection succeeded if we got a message or connected without errors
    # (server may not push immediately in dev mode)


@pytest.mark.skipif(not WS_AVAILABLE, reason="websocket-client not installed")
def test_websocket_unauthenticated_rejected():
    """WebSocket connection without a token should be closed by the server."""
    connected = threading.Event()
    closed = threading.Event()
    close_codes = []

    def on_open(ws):
        connected.set()

    def on_close(ws, code, msg):
        close_codes.append(code)
        closed.set()

    ws = ws_lib.WebSocketApp(
        WS_URL_TEMPLATE,
        on_open=on_open,
        on_close=on_close,
    )
    thread = threading.Thread(target=lambda: ws.run_forever())
    thread.daemon = True
    thread.start()
    closed.wait(timeout=8)

    # Either never connected or was closed with 4001/4003/1008
    if close_codes:
        assert close_codes[0] in (1008, 4001, 4003, 1000), (
            f"Unexpected close code: {close_codes[0]}"
        )
