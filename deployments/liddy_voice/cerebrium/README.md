# Cerebrium Deployment for Liddy Voice Agent

This directory contains the configuration and scripts needed to deploy the Liddy Voice Agent on Cerebrium's serverless GPU infrastructure.

## Overview

Cerebrium provides serverless GPU infrastructure with:
- Pay-per-second billing
- Automatic scaling (including scale-to-zero)
- WebSocket and REST API support
- GPU acceleration for ML workloads
- Fast cold starts with container caching

## Prerequisites

1. Install Cerebrium CLI:
   ```bash
   pip install cerebrium
   ```

2. Create a Cerebrium account and login:
   ```bash
   cerebrium login
   ```

3. Set up environment variables in `.env` file (copy from your local setup)

## Deployment Structure

```
cerebrium/
├── main.py                    # Main Cerebrium entry point
├── livekit_bridge.py         # WebSocket bridge for LiveKit
├── cerebrium.toml            # Cerebrium configuration
├── requirements-cerebrium.txt # Python dependencies
├── deploy.sh                 # Deployment script
└── README.md                 # This file
```

## Configuration

### Hardware Configuration (cerebrium.toml)

- **CPU**: 4 cores
- **Memory**: 16GB RAM
- **GPU**: NVIDIA A10G (24GB VRAM)
- **Storage**: 10GB
- **Scaling**: 0-10 replicas with 60s cooldown

### Environment Variables

Create a `.env` file with:

```env
# AI/ML APIs
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
CEREBRAS_API_KEY=your_key

# LiveKit
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret
LIVEKIT_URL=wss://your-livekit-server.com

# Vector Search
PINECONE_API_KEY=your_key

# Redis (for product caching)
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_PASSWORD=your_password

# Speech Services
ELEVENLABS_API_KEY=your_key
DEEPGRAM_API_KEY=your_key
ASSEMBLYAI_API_KEY=your_key

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Observability
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_secret
```

## Deployment

1. **First-time setup**:
   ```bash
   cd deployments/liddy_voice/cerebrium
   ./deploy.sh
   ```

2. **Update deployment**:
   ```bash
   cerebrium deploy --name liddy-voice-agent
   ```

3. **Check deployment status**:
   ```bash
   cerebrium status liddy-voice-agent
   ```

## API Endpoints

After deployment, you'll get endpoints like:

### REST API
```
https://api.cortex.cerebrium.ai/v4/p-{project-id}/liddy-voice-agent/run
```

### WebSocket
```
wss://api.cortex.cerebrium.ai/v4/p-{project-id}/liddy-voice-agent/ws
```

## Testing

### Health Check
```bash
cerebrium run liddy-voice-agent --request '{"type": "health"}'
```

### Create LiveKit Token
```bash
cerebrium run liddy-voice-agent --request '{
  "type": "create_token",
  "room_name": "test-room",
  "identity": "test-user"
}'
```

### Agent Request
```bash
cerebrium run liddy-voice-agent --request '{
  "type": "agent_request",
  "request": {
    "room": "test-room",
    "participant": "test-user"
  }
}'
```

## Architecture

### WebSocket Bridge

The `livekit_bridge.py` module provides a bridge between Cerebrium's WebSocket interface and LiveKit's protocol:

1. Cerebrium WebSocket → Bridge → LiveKit Room
2. Handles audio forwarding, control messages, and session management
3. Maintains active sessions and cleans up on disconnect

### Request Flow

1. **REST API**: Client → Cerebrium → main.py → voice_agent
2. **WebSocket**: Client → Cerebrium WS → livekit_bridge → LiveKit

## Performance Considerations

### Cold Starts
- First request may take 10-20s due to model loading
- Subsequent requests are much faster (<1s)
- Use `min_replicas: 1` to keep one instance warm

### GPU Utilization
- A10G GPU provides excellent performance for:
  - Real-time STT/TTS processing
  - LLM inference for conversation
  - Audio processing pipelines

### Scaling
- Scales to 0 when idle (cost-effective)
- Can handle up to 10 concurrent sessions
- 60s cooldown prevents aggressive scaling

## Monitoring

### Logs
```bash
cerebrium logs liddy-voice-agent --tail
```

### Metrics
```bash
cerebrium metrics liddy-voice-agent
```

## Cost Optimization

1. **Scale to Zero**: Enabled by default (no cost when idle)
2. **Efficient Hardware**: A10G provides good price/performance
3. **Batching**: Consider batching requests where possible
4. **Caching**: Redis caching reduces repeated computations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are in requirements-cerebrium.txt
2. **Environment Variables**: Check .env file is properly configured
3. **GPU Memory**: Monitor GPU usage, may need to adjust model loading
4. **WebSocket Timeouts**: Adjust connection timeout in bridge configuration

### Debug Mode

Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

1. Set up monitoring and alerting
2. Configure custom domain for endpoints
3. Implement request authentication
4. Add performance metrics collection
5. Set up CI/CD pipeline for automatic deployments

## Support

- Cerebrium Discord: https://discord.gg/ATj6USmeE2
- Cerebrium Docs: https://docs.cerebrium.ai
- Liddy Support: support@liddy.ai