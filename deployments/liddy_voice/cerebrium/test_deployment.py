#!/usr/bin/env python3
"""
Test script for Cerebrium deployment of Liddy Voice Agent.

This script tests various endpoints and functionality of the deployed agent.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

import aiohttp
import websockets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CerebriumTester:
    """Test client for Cerebrium-deployed voice agent."""
    
    def __init__(self, project_id: str, app_name: str = "liddy-voice-agent"):
        self.project_id = project_id
        self.app_name = app_name
        self.base_url = f"https://api.cortex.cerebrium.ai/v4/p-{project_id}/{app_name}"
        self.ws_url = f"wss://api.cortex.cerebrium.ai/v4/p-{project_id}/{app_name}/ws"
        
    async def test_health(self) -> Dict[str, Any]:
        """Test health check endpoint."""
        print("🏥 Testing health check...")
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/run"
            data = {"type": "health"}
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                print(f"✅ Health check: {result}")
                return result
    
    async def test_token_creation(self) -> Dict[str, Any]:
        """Test LiveKit token creation."""
        print("\n🎫 Testing token creation...")
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/run"
            data = {
                "type": "create_token",
                "room_name": "test-room",
                "identity": "test-user"
            }
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                if "token" in result:
                    print(f"✅ Token created successfully")
                    print(f"   Room: {result.get('room_name')}")
                    print(f"   Identity: {result.get('identity')}")
                else:
                    print(f"❌ Token creation failed: {result}")
                return result
    
    async def test_websocket_connection(self) -> None:
        """Test WebSocket connectivity."""
        print("\n🔌 Testing WebSocket connection...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Send initial connection message
                connection_msg = {
                    "session_id": "test-session",
                    "room_name": "test-room",
                    "identity": "test-user"
                }
                await websocket.send(json.dumps(connection_msg))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                result = json.loads(response)
                
                if result.get("type") == "connected":
                    print("✅ WebSocket connected successfully")
                    print(f"   Session ID: {result.get('session_id')}")
                else:
                    print(f"❌ WebSocket connection failed: {result}")
                
                # Close connection
                await websocket.close()
                
        except asyncio.TimeoutError:
            print("❌ WebSocket connection timed out")
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
    
    async def test_agent_request(self) -> Dict[str, Any]:
        """Test agent request handling."""
        print("\n🤖 Testing agent request...")
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/run"
            data = {
                "type": "agent_request",
                "request": {
                    "room": "test-room",
                    "participant": "test-user",
                    "metadata": {
                        "user_id": "test-123",
                        "account": "specialized.com"
                    }
                }
            }
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                if "error" not in result:
                    print("✅ Agent request processed successfully")
                else:
                    print(f"❌ Agent request failed: {result}")
                return result
    
    async def run_all_tests(self) -> None:
        """Run all tests."""
        print("🚀 Starting Cerebrium deployment tests")
        print("=" * 50)
        
        # Test REST endpoints
        await self.test_health()
        await self.test_token_creation()
        await self.test_agent_request()
        
        # Test WebSocket
        await self.test_websocket_connection()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")


async def main():
    """Main test runner."""
    # Get project ID from environment or command line
    project_id = os.getenv("CEREBRIUM_PROJECT_ID")
    
    if not project_id:
        if len(sys.argv) > 1:
            project_id = sys.argv[1]
        else:
            print("❌ Please provide CEREBRIUM_PROJECT_ID as environment variable or argument")
            print("Usage: python test_deployment.py <project-id>")
            sys.exit(1)
    
    # Create tester and run tests
    tester = CerebriumTester(project_id)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())