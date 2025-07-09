# Liddy Voice Package

Voice-enabled AI assistant with brand-aware customer service capabilities.

## Overview

The `liddy_voice` package provides voice assistant functionality with:

- **Voice Interface**: Real-time voice interaction with streaming responses
- **Brand-Aware AI**: Contextual responses based on brand research
- **RAG Search**: Retrieval-augmented generation for accurate product information
- **Session Management**: Conversation history and state tracking

## Installation

```bash
pip install -e packages/liddy_voice
```

## Key Features

### Voice Assistant
```python
from liddy_voice.sample_assistant import VoiceAssistant

assistant = VoiceAssistant(
    agent_id="specialized-assistant",
    api_key=settings.ASSEMBLYAI_API_KEY
)

# Start voice session
await assistant.start_session()
```

### Search Service
```python
from liddy_voice.search_service import VoiceOptimizedSearchService

search = VoiceOptimizedSearchService(
    brand_domain="specialized.com",
    index_name="specialized-hybrid"
)

# Search for products
results = await search.search(
    query="mountain bikes under $3000",
    top_k=5
)
```

### Session Management
```python
from liddy_voice.session_state_manager import SessionStateManager

session = SessionStateManager(agent_id="specialized-assistant")
session.add_message("user", "Show me road bikes")
history = session.get_conversation_history()
```

## Architecture

```
liddy_voice/
├── sample_assistant.py      # Main voice assistant
├── search_service.py        # RAG search service
├── session_state_manager.py # Session tracking
├── model.py                # Response generation
├── prompt_manager.py       # Dynamic prompts
└── deploy/                 # Deployment configs
```

## Deployment

### Local Development
```bash
# Run voice assistant
python -m liddy_voice.sample_assistant
```

### Google Cloud Run
```bash
# Deploy to Cloud Run
cd deployments/liddy_voice
./gcp/deploy-cloudrun.sh
```

### Docker
```bash
# Build image
docker build -f deployments/liddy_voice/Dockerfile -t liddy-voice .

# Run container
docker run -p 8080:8080 liddy-voice
```

## API Endpoints

### WebSocket Interface
```
wss://your-domain/voice
```

Messages:
- `session_update`: Session state changes
- `conversation_update`: New messages
- `transcript`: Real-time transcription
- `error`: Error messages

## Configuration

Environment variables:
```bash
# Required
OPENAI_API_KEY=your-key
PINECONE_API_KEY=your-key
ASSEMBLYAI_API_KEY=your-key

# Cloud deployment
GCP_PROJECT_ID=your-project
GCP_REGION=us-central1
```

## Dependencies

- assemblyai: Real-time voice transcription
- websockets: WebSocket communication
- fastapi: Web framework
- uvicorn: ASGI server
- pinecone-client: Vector search

## Development

When developing voice features:
1. Test with different accents and speaking speeds
2. Optimize for real-time response generation
3. Handle network interruptions gracefully
4. Keep responses concise for voice output