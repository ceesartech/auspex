"""WebSocket endpoints for real-time updates"""

import asyncio
import json
import logging
from typing import List, Optional

from auth.jwt_handler import verify_token
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client"""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")


manager = ConnectionManager()


@router.websocket("/predictions")
async def websocket_predictions(websocket: WebSocket, token: Optional[str] = None):
    """WebSocket for live prediction updates"""

    if token:
        try:
            verify_token(token)
        except Exception:
            await websocket.close(code=1008)
            return

    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "subscribe":
                    match_ids = message.get("match_ids", [])
                    await websocket.send_text(json.dumps({"type": "subscribed", "match_ids": match_ids}))

                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/odds")
async def websocket_odds(websocket: WebSocket, token: Optional[str] = None):
    """WebSocket for live odds updates"""

    await manager.connect(websocket)

    try:
        while True:
            await asyncio.sleep(5)

            await websocket.send_text(json.dumps({"type": "odds_update", "data": {}}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
