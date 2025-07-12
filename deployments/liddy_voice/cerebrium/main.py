#!/usr/bin/env python3
"""
Cerebrium deployment for Liddy Voice Agent

This wraps the LiveKit voice agent for deployment on Cerebrium's
serverless GPU infrastructure.
"""

import os
import sys
import asyncio
import logging
from typing import Optional

# Add packages to path for Cerebrium environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "packages")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import LiveKit agent components
try:
    from livekit import agents, api
    from livekit.agents import WorkerOptions
    from liddy_voice import voice_agent
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# Cerebrium requires functions to be defined at module level
# We'll create wrapper functions that Cerebrium can call

def initialize():
    """
    Initialize function called when container starts.
    This is where we can prewarm models and set up connections.
    """
    logger.info("Initializing Cerebrium voice agent...")
    
    # Run the prewarm function from voice_agent
    try:
        asyncio.run(voice_agent.prewarm(None))
        logger.info("Prewarm completed successfully")
    except Exception as e:
        logger.error(f"Prewarm failed: {e}")
    
    return {"status": "initialized"}


async def handle_websocket(websocket_data: dict):
    """
    Handle WebSocket connections for LiveKit voice agent.
    
    Args:
        websocket_data: WebSocket connection data from Cerebrium
    
    Returns:
        Response data for the WebSocket connection
    """
    logger.info(f"Handling WebSocket connection: {websocket_data}")
    
    # TODO: Implement WebSocket handling for LiveKit
    # This will need to bridge between Cerebrium's WebSocket and LiveKit's protocol
    
    return {"status": "websocket_handled"}


def run(request_data: dict):
    """
    Main entry point for Cerebrium REST API calls.
    
    Args:
        request_data: Request data from Cerebrium
        
    Returns:
        Response data
    """
    logger.info(f"Processing request: {request_data}")
    
    # Extract request type
    request_type = request_data.get("type", "unknown")
    
    if request_type == "health":
        return {"status": "healthy", "service": "liddy-voice-agent"}
    
    elif request_type == "create_token":
        # Create LiveKit access token
        room_name = request_data.get("room_name", "default")
        identity = request_data.get("identity", "user")
        
        try:
            # Use environment variables for LiveKit configuration
            api_key = os.getenv("LIVEKIT_API_KEY")
            api_secret = os.getenv("LIVEKIT_API_SECRET")
            
            if not api_key or not api_secret:
                return {"error": "LiveKit credentials not configured"}
            
            token = api.AccessToken(api_key, api_secret) \
                .with_identity(identity) \
                .with_name(identity) \
                .with_grants(api.VideoGrants(
                    room_join=True,
                    room=room_name,
                ))
            
            return {
                "token": token.to_jwt(),
                "room_name": room_name,
                "identity": identity
            }
            
        except Exception as e:
            logger.error(f"Failed to create token: {e}")
            return {"error": str(e)}
    
    elif request_type == "agent_request":
        # Handle agent request (similar to voice_agent.request_fnc)
        try:
            # Extract request parameters
            request = request_data.get("request", {})
            
            # Run the request function asynchronously
            result = asyncio.run(voice_agent.request_fnc(request))
            
            return {"result": result}
            
        except Exception as e:
            logger.error(f"Agent request failed: {e}")
            return {"error": str(e)}
    
    else:
        return {"error": f"Unknown request type: {request_type}"}


# Cerebrium-specific configuration
def get_cerebrium_config():
    """
    Get Cerebrium-specific configuration.
    """
    return {
        "gpu": True,  # Enable GPU for voice processing
        "cpu": 4,     # CPU cores
        "memory": 16, # GB of RAM
        "gpu_type": "NVIDIA_A10G",  # GPU type for ML inference
        "min_replicas": 0,  # Scale to zero when not in use
        "max_replicas": 10, # Maximum scaling
        "cooldown": 60,     # Seconds before scaling down
    }