"""
Data models for AccountManager.

Contains both core and voice-related configuration models.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    """Search backend configuration for an account."""
    backend: str = "pinecone"  # pinecone, elasticsearch, etc.
    dense_index: Optional[str] = None
    sparse_index: Optional[str] = None
    unified_index: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class AccountVariantType:
    """Product variant configuration for an account."""
    name: str
    query_param_name: Optional[str] = None
    display_name: Optional[str] = None
    
    def __repr__(self):
        return f"AccountVariantType(name={self.name}, query_param_name={self.query_param_name})"


class TtsProviderSettings:
    """Base class for TTS provider settings."""
    
    def __init__(self, voice_provider: str, voice_id: str, voice_name: str):
        self.voice_provider = voice_provider
        self.voice_id = voice_id
        self.voice_name = voice_name
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.voice_provider,
            "voice_id": self.voice_id,
            "voice_name": self.voice_name
        }


class ElevenLabsTtsProviderSettings(TtsProviderSettings):
    """ElevenLabs-specific TTS settings."""
    
    def __init__(
        self,
        voice_id: str,
        voice_name: str,
        voice_model: Optional[str] = None,
        voice_stability: Optional[float] = None,
        voice_similarity_boost: Optional[float] = None,
        voice_style: Optional[float] = None,
        voice_use_speaker_boost: Optional[bool] = None,
        voice_speed: Optional[float] = None
    ):
        super().__init__(voice_provider="elevenlabs", voice_id=voice_id, voice_name=voice_name)
        self.voice_model = voice_model if voice_model else os.environ.get("VOICE_MODEL", "eleven_flash_v2_5")
        self.voice_stability = voice_stability if voice_stability else float(os.getenv("VOICE_STABILITY", 0.45))
        self.voice_similarity_boost = voice_similarity_boost if voice_similarity_boost else float(os.getenv("VOICE_SIMILARITY_BOOST", 0.8))
        self.voice_style = voice_style if voice_style else float(os.getenv("VOICE_STYLE", 0.0))
        self.voice_use_speaker_boost = voice_use_speaker_boost if voice_use_speaker_boost else os.getenv("VOICE_USE_SPEAKER_BOOST", "true").lower() == "true"
        self.voice_speed = voice_speed if voice_speed else float(os.getenv("VOICE_SPEED", 1.0))
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            "model": self.voice_model,
            "stability": self.voice_stability,
            "similarity_boost": self.voice_similarity_boost,
            "style": self.voice_style,
            "use_speaker_boost": self.voice_use_speaker_boost,
            "speed": self.voice_speed
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElevenLabsTtsProviderSettings":
        """Create from dictionary (e.g., from JSON)."""
        return cls(
            voice_id=data.get("voice_id"),
            voice_name=data.get("voice_name", ""),
            voice_model=data.get("model"),
            voice_stability=data.get("stability"),
            voice_similarity_boost=data.get("similarity_boost"),
            voice_style=data.get("style"),
            voice_use_speaker_boost=data.get("use_speaker_boost"),
            voice_speed=data.get("speed")
        )


class AccountTTSSettings:
    """TTS configuration for an account with primary and fallback providers."""
    
    def __init__(self, providers: List[TtsProviderSettings]):
        if not providers or len(providers) < 1:
            raise ValueError("At least one TTS provider is required")
        self.providers = providers
    
    @property
    def primary_provider(self) -> TtsProviderSettings:
        """Get the primary TTS provider."""
        return self.providers[0]
    
    @property
    def fallback_providers(self) -> List[TtsProviderSettings]:
        """Get fallback TTS providers."""
        return self.providers[1:] if len(self.providers) > 1 else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "providers": [provider.to_dict() for provider in self.providers]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccountTTSSettings":
        """Create from dictionary (e.g., from JSON)."""
        providers = []
        for provider_data in data.get("providers", []):
            if provider_data.get("provider") == "elevenlabs":
                providers.append(ElevenLabsTtsProviderSettings.from_dict(provider_data))
            else:
                providers.append(TtsProviderSettings(
                    voice_provider=provider_data.get("provider", "unknown"),
                    voice_id=provider_data.get("voice_id", ""),
                    voice_name=provider_data.get("voice_name", "")
                ))
        return cls(providers)