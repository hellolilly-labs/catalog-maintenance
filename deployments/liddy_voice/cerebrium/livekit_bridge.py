"""
LiveKit-Cerebrium WebSocket Bridge

This module bridges between Cerebrium's WebSocket interface and LiveKit's
agent protocol, allowing the voice agent to run on Cerebrium's infrastructure.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

import websockets
from livekit import agents, rtc, api

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for the LiveKit-Cerebrium bridge"""
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    cerebrium_ws_url: Optional[str] = None
    

class LiveKitCerebriumBridge:
    """
    Bridges WebSocket connections between Cerebrium and LiveKit.
    
    This allows the voice agent to receive WebSocket connections from
    Cerebrium and forward them to LiveKit's infrastructure.
    """
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.active_sessions: Dict[str, Any] = {}
        
    async def handle_cerebrium_connection(self, websocket, path):
        """
        Handle incoming WebSocket connection from Cerebrium.
        
        Args:
            websocket: The WebSocket connection from Cerebrium
            path: The connection path
        """
        session_id = None
        try:
            logger.info(f"New Cerebrium WebSocket connection from {websocket.remote_address}")
            
            # Wait for initial message with connection details
            initial_msg = await websocket.recv()
            data = json.loads(initial_msg)
            
            session_id = data.get("session_id", "unknown")
            room_name = data.get("room_name", "default")
            user_identity = data.get("identity", "user")
            
            logger.info(f"Session {session_id}: Connecting to room {room_name} as {user_identity}")
            
            # Create LiveKit token for this session
            token = self._create_livekit_token(room_name, user_identity)
            
            # Connect to LiveKit
            room = rtc.Room()
            await room.connect(self.config.livekit_url, token)
            
            # Store session info
            self.active_sessions[session_id] = {
                "websocket": websocket,
                "room": room,
                "room_name": room_name,
                "identity": user_identity
            }
            
            # Send connection success
            await websocket.send(json.dumps({
                "type": "connected",
                "session_id": session_id,
                "room_name": room_name
            }))
            
            # Start forwarding messages
            await self._forward_messages(session_id)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Session {session_id}: WebSocket connection closed")
        except Exception as e:
            logger.error(f"Session {session_id}: Error handling connection: {e}")
            if websocket.open:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
        finally:
            # Cleanup
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if "room" in session:
                    await session["room"].disconnect()
                del self.active_sessions[session_id]
    
    async def _forward_messages(self, session_id: str):
        """
        Forward messages between Cerebrium WebSocket and LiveKit.
        
        Args:
            session_id: The session identifier
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return
            
        websocket = session["websocket"]
        room = session["room"]
        
        # Create tasks for bidirectional message forwarding
        tasks = [
            asyncio.create_task(self._forward_cerebrium_to_livekit(session_id)),
            asyncio.create_task(self._forward_livekit_to_cerebrium(session_id))
        ]
        
        # Wait for either task to complete (connection closed)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
    
    async def _forward_cerebrium_to_livekit(self, session_id: str):
        """Forward messages from Cerebrium to LiveKit."""
        session = self.active_sessions.get(session_id)
        if not session:
            return
            
        websocket = session["websocket"]
        room = session["room"]
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                # Handle different message types
                if data["type"] == "audio":
                    # Forward audio data to LiveKit
                    # This would need proper audio handling
                    pass
                elif data["type"] == "control":
                    # Handle control messages
                    pass
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Session {session_id}: Cerebrium connection closed")
    
    async def _forward_livekit_to_cerebrium(self, session_id: str):
        """Forward messages from LiveKit to Cerebrium."""
        session = self.active_sessions.get(session_id)
        if not session:
            return
            
        websocket = session["websocket"]
        room = session["room"]
        
        # Set up LiveKit event handlers
        @room.on("track_published")
        def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            """Handle track published events."""
            asyncio.create_task(websocket.send(json.dumps({
                "type": "track_published",
                "participant": participant.identity,
                "track_sid": publication.sid
            })))
        
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            """Handle track subscribed events."""
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                # Set up audio forwarding
                asyncio.create_task(self._forward_audio_track(session_id, track))
        
        # Keep the coroutine running
        while session_id in self.active_sessions:
            await asyncio.sleep(1)
    
    async def _forward_audio_track(self, session_id: str, track: rtc.AudioTrack):
        """Forward audio from LiveKit track to Cerebrium WebSocket."""
        session = self.active_sessions.get(session_id)
        if not session:
            return
            
        websocket = session["websocket"]
        
        # This is a simplified example - actual implementation would need
        # proper audio frame handling and encoding
        async for frame in track:
            if websocket.open:
                await websocket.send(json.dumps({
                    "type": "audio_frame",
                    "data": frame.data.tobytes().hex(),  # Convert to hex for JSON
                    "sample_rate": frame.sample_rate,
                    "channels": frame.num_channels
                }))
    
    def _create_livekit_token(self, room_name: str, identity: str) -> str:
        """Create a LiveKit access token."""
        token = api.AccessToken(
            self.config.livekit_api_key,
            self.config.livekit_api_secret
        )
        token.with_identity(identity).with_name(identity)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
        ))
        return token.to_jwt()


async def start_bridge_server(config: BridgeConfig, host: str = "0.0.0.0", port: int = 8765):
    """
    Start the WebSocket bridge server.
    
    Args:
        config: Bridge configuration
        host: Host to bind to
        port: Port to bind to
    """
    bridge = LiveKitCerebriumBridge(config)
    
    logger.info(f"Starting LiveKit-Cerebrium bridge on {host}:{port}")
    
    async with websockets.serve(bridge.handle_cerebrium_connection, host, port):
        await asyncio.Future()  # Run forever