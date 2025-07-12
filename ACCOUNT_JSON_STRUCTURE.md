# Account.json Structure with Voice Settings

This document describes the structure of `account.json` files after merging the AccountManagers.

## Complete Example

```json
{
  "account": {
    "domain": "specialized.com",
    "display_name": "Specialized",
    "status": "active"
  },
  "voice": {
    "greeting": "Hey, this is Spence! How can I help?",
    "primary": {
      "provider": "elevenlabs",
      "voice_id": "8sGzMkj2HZn6rYwGx6G0",
      "voice_name": "Tomas",
      "model": "eleven_flash_v2_5",
      "stability": 0.5,
      "similarity_boost": 0.8,
      "style": 0.0,
      "use_speaker_boost": true,
      "speed": 1.0
    },
    "fallbacks": [
      {
        "provider": "google",
        "voice_id": "en-US-Chirp3-HD-Orus",
        "voice_name": "Spence"
      }
    ],
    "persona": {
      "friendliness": 0.8,
      "professionalism": 0.7,
      "enthusiasm": 0.9
    }
  },
  "search": {
    "pinecone_index": "specialized-llama-2048",
    "embedding_model": "llama-text-embed-v2"
  },
  "features": [
    "search",
    "catalog",
    "rag",
    "hybrid_search",
    "reranking",
    "variants"
  ]
}
```

## Voice Configuration Fields

### Primary Provider
The main TTS provider configuration:
- `provider`: TTS provider name (e.g., "elevenlabs", "google")
- `voice_id`: Provider-specific voice identifier
- `voice_name`: Human-readable voice name
- `model`: (ElevenLabs) Model to use (e.g., "eleven_flash_v2_5", "eleven_turbo_v2_5")
- `stability`: (ElevenLabs) Voice stability (0.0-1.0)
- `similarity_boost`: (ElevenLabs) Similarity boost (0.0-1.0)
- `style`: (ElevenLabs) Style exaggeration (0.0-1.0)
- `use_speaker_boost`: (ElevenLabs) Enable speaker boost
- `speed`: (ElevenLabs) Speaking speed multiplier

### Fallback Providers
Array of backup TTS providers used if primary fails:
- Same fields as primary provider
- Typically includes Google TTS as final fallback

### Persona Settings
- `greeting`: Initial greeting message
- Additional numeric fields become personality sliders

## Migration from Hardcoded Settings

The system will automatically fall back to hardcoded settings if no account.json exists. To migrate:

1. Create `accounts/{domain}/account.json`
2. Add the voice configuration section
3. Test with voice assistant

## Backward Compatibility

The system maintains compatibility with:
- Old schema with `agent.persona.greeting`
- Legacy method names in liddy_voice.AccountManager
- Hardcoded TTS settings for known accounts

## Example Migration Script

```python
import json
from liddy import get_account_storage_provider

async def migrate_account_to_json(account: str, tts_settings):
    """Migrate hardcoded TTS settings to account.json"""
    
    storage = get_account_storage_provider()
    
    # Build voice config from TTS settings
    voice_config = {
        "greeting": get_greeting_for_account(account),
        "primary": {
            "provider": tts_settings.primary_provider.voice_provider,
            "voice_id": tts_settings.primary_provider.voice_id,
            "voice_name": tts_settings.primary_provider.voice_name,
            # Add other fields as needed
        },
        "fallbacks": [
            {
                "provider": fb.voice_provider,
                "voice_id": fb.voice_id,
                "voice_name": fb.voice_name
            }
            for fb in tts_settings.fallback_providers
        ]
    }
    
    # Load existing or create new config
    existing = await storage.get_file(account, "account.json")
    if existing:
        config = json.loads(existing)
    else:
        config = {"account": {"domain": account}}
    
    # Add voice config
    config["voice"] = voice_config
    
    # Save back
    await storage.save_file(
        account,
        "account.json",
        json.dumps(config, indent=2)
    )
```