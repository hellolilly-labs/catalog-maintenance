from typing import List

class TtsProviderSettings:
    def __init__(self, voice_provider: str, voice_id: str, voice_name: str):
        self.voice_provider = voice_provider
        self.voice_id = voice_id
        self.voice_name = voice_name

class ElevenLabsTtsProviderSettings(TtsProviderSettings):
    def __init__(self, voice_id: str, voice_name: str, voice_model: str = None, voice_stability: float = None, voice_similarity_boost: float = None, voice_style: float = None, voice_use_speaker_boost: bool = None, voice_speed: float = None):
        super().__init__(voice_provider="elevenlabs", voice_id=voice_id, voice_name=voice_name)
        self.voice_model = voice_model if voice_model else os.environ.get("VOICE_MODEL", "eleven_flash_v2_5")
        self.voice_stability = voice_stability if voice_stability else float(os.getenv("VOICE_STABILITY", 0.45))
        self.voice_similarity_boost = voice_similarity_boost if voice_similarity_boost else float(os.getenv("VOICE_SIMILARITY_BOOST", 0.8))
        self.voice_style = voice_style if voice_style else float(os.getenv("VOICE_STYLE", 0.0))
        self.voice_use_speaker_boost = voice_use_speaker_boost if voice_use_speaker_boost else os.getenv("VOICE_USE_SPEAKER_BOOST", "true").lower() == "true"
        self.voice_speed = voice_speed if voice_speed else float(os.getenv("VOICE_SPEED", 1.0))

class AccountTTSSettings:
    def __init__(self, providers: List[TtsProviderSettings]):
        if not providers or len(providers) < 1:
            raise ValueError("At least one TTS provider is required")
        self.providers = providers
    
    @property
    def primary_provider(self) -> TtsProviderSettings:
        return self.providers[0]
    
    @property
    def fallback_providers(self) -> List[TtsProviderSettings]:
        return self.providers[1:] if len(self.providers) > 1 else []


class AccountVariantType:
    def __init__(self, name: str, query_param_name: str = None):
        self.name = name
        self.query_param_name = query_param_name

    def __repr__(self):
        return f"AccountVariantType(name={self.name}, query_param_name={self.query_param_name})"

"""
Account Manager for RAG Configuration

This module manages account-specific configurations for RAG and other services.
Now supports dynamic configuration loading from GCP storage with Redis caching.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional
import asyncio

from .account_config_loader import get_account_config_loader

logger = logging.getLogger(__name__)

# Global cache for AccountManager instances (singleton pattern)
_account_managers: Dict[str, "AccountManager"] = {}

class AccountManager:
    """
    Manager for account-specific configurations
    Supports dynamic loading from GCP storage with fallback to hardcoded values
    """
    
    def __init__(self, account: str = None):
        """
        Initialize the account manager with the specified account (sync fallback)
        
        For best performance, use get_account_manager() async factory method.
        This sync constructor only loads hardcoded configuration.
        
        Args:
            account: Account name (domain) to use for configuration
        """
        self.account = self._extract_account_name(account_str=account)
        self.config_loader = get_account_config_loader()
        self.dynamic_config = None
        
        # Sync constructor only loads hardcoded settings (KISS approach)
        self.tts_settings = self._build_hardcoded_tts_settings()
        
        # Register hardcoded config as fallback for future use
        self._register_hardcoded_fallback()
    
    @classmethod
    async def create_async(cls, account: str = None):
        """
        Create an AccountManager instance with async config loading
        
        Prefer get_account_manager() singleton factory method instead.
        
        Args:
            account: Account name (domain) to use for configuration
            
        Returns:
            AccountManager instance with dynamic config loaded
        """
        # Use singleton factory method for consistency
        return await get_account_manager(account)
    
    def _build_tts_settings_from_config(self, config: Dict[str, Any]) -> AccountTTSSettings:
        """Build TTS settings from dynamic configuration (supports old and new schema)"""
        try:
            # NEW SCHEMA: Multi-agent with agents array
            if 'agents' in config and config['agents']:
                agent_config = config['agents'][0]  # Primary agent
                voice_config = agent_config.get('voice', {})
            # OLD SCHEMA: Single agent at root level (backward compatibility)
            elif 'agent' in config:
                agent_config = config.get('agent', {})
                voice_config = agent_config.get('voice', {})
            else:
                logger.warning(f"No agent configuration found in config for {self.account}")
                return self._build_hardcoded_tts_settings()
            
            if voice_config.get('provider') == 'elevenlabs':
                primary_provider = ElevenLabsTtsProviderSettings(
                    voice_id=voice_config.get('voice_id'),
                    voice_name=voice_config.get('voice_name', 'Dynamic Voice'),
                    voice_model=voice_config.get('model', 'eleven_flash_v2_5'),
                    voice_stability=voice_config.get('stability', 0.5),
                    voice_similarity_boost=voice_config.get('similarity_boost', 0.8),
                    voice_style=voice_config.get('style', 0.0),
                    voice_use_speaker_boost=voice_config.get('use_speaker_boost', True),
                    voice_speed=voice_config.get('speed', 1.0)
                )
            else:
                # Default to ElevenLabs if not specified
                primary_provider = TtsProviderSettings(
                    voice_provider=voice_config.get('provider', 'unknown'),
                    voice_id=voice_config.get('voice_id'),
                    voice_name=voice_config.get('voice_name', '')
                )
            
            # Add Google fallback
            fallback_provider = TtsProviderSettings(
                voice_provider="google",
                voice_id="en-US-Chirp-HD-F",
                voice_name="Kate"
            )
            
            return AccountTTSSettings(providers=[primary_provider, fallback_provider])
            
        except Exception as e:
            logger.error(f"Error building TTS settings from config: {e}")
            return self._build_hardcoded_tts_settings()
    
    def _build_hardcoded_tts_settings(self) -> AccountTTSSettings:
        """Build TTS settings using hardcoded values (fallback)"""
        if "sundayriley.com" == self.account:
            return AccountTTSSettings(
                providers=[
                    ElevenLabsTtsProviderSettings(
                        voice_id="CV4xD6M8z1X1kya4Pepj",
                        voice_name="Dio",
                        voice_stability=0.75,
                        voice_similarity_boost=0.8,
                        voice_style=0.0,
                        voice_use_speaker_boost=True,
                        voice_model="eleven_turbo_v2_5",
                        # voice_id="vnd0afTMgWq4fDRVyDo3",
                        # voice_name="Deobra"
                        # voice_id="tnSpp4vdxKPjI9w0GnoV",
                        # voice_name="Hope"
                    ),
                    TtsProviderSettings(
                        voice_provider="google",
                        voice_id="en-US-Chirp-HD-F",
                        voice_name="Kate"
                    ),
                ]
            )
        elif "darakayejewelry.com" == self.account:
            return AccountTTSSettings(
                providers=[
                   ElevenLabsTtsProviderSettings(
                        # voice_id="uYXf8XasLslADfZ2MB4u",
                        voice_id="8DzKSPdgEQPaK5vKG0Rs",
                        voice_name="Vanessa",
                        voice_model="eleven_flash_v2_5",
                        voice_stability=0.35,
                        voice_similarity_boost=0.6,
                        voice_style=0.0,
                        voice_use_speaker_boost=True,
                        voice_speed=1.07,
                        # voice_id="73WKGv75PKCl1ypZdhCy",
                        # voice_name="DK"
                        # voice_id="j3QcmAr55TvFW5CDB0Q3",
                        # voice_name="Iniga"
                        # voice_id="7YaUDeaStRuoYg3FKsmU",
                        # voice_name="Callie"
                        # voice_id="CV4xD6M8z1X1kya4Pepj",
                        # voice_name="Dio"
                        # voice_id="vnd0afTMgWq4fDRVyDo3",
                        # voice_name="Deobra"
                        # voice_id="tnSpp4vdxKPjI9w0GnoV",
                        # voice_name="Hope"
                    ),
                    TtsProviderSettings(
                        voice_provider="google",
                        voice_id="en-US-Chirp-HD-F",
                        voice_name="Kate"
                    ),
                ]
            )
        elif "darakayejewelry-test.myshopify.com" == self.account:
            return AccountTTSSettings(
                providers=[
                    ElevenLabsTtsProviderSettings(
                        # voice_id="uYXf8XasLslADfZ2MB4u",
                        voice_id="8DzKSPdgEQPaK5vKG0Rs",
                        voice_name="Vanessa",
                        voice_model="eleven_flash_v2_5",
                        voice_stability=0.35,
                        voice_similarity_boost=0.75,
                        voice_style=0.0,
                        voice_use_speaker_boost=True,
                        voice_speed=1.07,
                        # voice_id="73WKGv75PKCl1ypZdhCy",
                        # voice_name="DK"
                        # voice_id="7YaUDeaStRuoYg3FKsmU",
                        # voice_name="Callie"
                        # voice_id="j3QcmAr55TvFW5CDB0Q3",
                        # voice_name="Iniga"
                        # voice_id="CV4xD6M8z1X1kya4Pepj",
                        # voice_name="Dio"
                        # voice_id="vnd0afTMgWq4fDRVyDo3",
                        # voice_name="Deobra"
                        # voice_id="tnSpp4vdxKPjI9w0GnoV",
                        # voice_name="Hope"
                    ),
                    TtsProviderSettings(
                        voice_provider="google",
                        voice_id="en-US-Chirp-HD-F",
                        voice_name="Kate"
                    ),
                ]
            )
        elif "flex-liddy.myshopify.com" == self.account or "flexfits.com" == self.account:
            return AccountTTSSettings(
                providers=[
                    ElevenLabsTtsProviderSettings(
                        voice_id="qgNVOmUjsNA8CD4Fed8D",
                        voice_name="Lindsay"
                        # voice_id="iCrDUkL56s3C8sCRl7wb",
                        # voice_name="Hope"
                        # voice_id="625jGFaa0zTLtQfxwc6Q",
                        # voice_name="Veda Sky"
                        # voice_id="8DzKSPdgEQPaK5vKG0Rs",
                        # voice_name="Vanessa"
                        # voice_id="8N2ng9i2uiUWqstgmWlH",
                        # voice_name="Beth"
                        # voice_id="YcoDoyTR36LSyeRrdtg5",
                        # voice_name="Lauren"
                        # voice_id="7YaUDeaStRuoYg3FKsmU",
                        # voice_name="Callie"
                        # voice_id="j3QcmAr55TvFW5CDB0Q3",
                        # voice_name="Iniga"
                        # voice_id="CV4xD6M8z1X1kya4Pepj",
                        # voice_name="Dio"
                        # voice_id="vnd0afTMgWq4fDRVyDo3",
                        # voice_name="Deobra"
                        # voice_id="tnSpp4vdxKPjI9w0GnoV",
                        # voice_name="Hope"
                    ),
                    TtsProviderSettings(
                        voice_provider="google",
                        voice_id="en-US-Chirp-HD-F",
                        voice_name="Kate"
                    ),
                ]
            )
        elif "balenciaga.com" == self.account:
            return AccountTTSSettings(
                providers=[
                    ElevenLabsTtsProviderSettings(
                        # voice_id="7YaUDeaStRuoYg3FKsmU",
                        # voice_name="Callie"
                        voice_id="txtf1EDouKke753vN8SL",
                        voice_name="Jeanne"
                        # voice_id="lvQdCgwZfBuOzxyV5pxu",
                        # voice_name="AudiA"
                        # voice_id="CV4xD6M8z1X1kya4Pepj",
                        # voice_name="Dio"
                        # voice_id="vnd0afTMgWq4fDRVyDo3",
                        # voice_name="Deobra"
                        # voice_id="tnSpp4vdxKPjI9w0GnoV",
                        # voice_name="Hope"
                    ),
                    TtsProviderSettings(
                        voice_provider="google",
                        voice_id="en-US-Chirp-HD-F",
                        voice_name="Kate"
                    ),
                ]
            )
        elif "gucci.com" == self.account:
            return AccountTTSSettings(
                providers=[
                    ElevenLabsTtsProviderSettings(
                        # voice_id="7YaUDeaStRuoYg3FKsmU",
                        # voice_name="Callie"
                        voice_id="8KInRSd4DtD5L5gK7itu",
                        voice_name="VIP",
                        voice_model="eleven_flash_v2_5",
                        voice_stability=0.75,
                        voice_similarity_boost=0.8,
                        voice_style=0.0,
                        voice_use_speaker_boost=True,
                        # voice_id="PS6wM7QnnPTUDE3t4bbl",
                        # voice_name="Serena"
                        # voice_id="lvQdCgwZfBuOzxyV5pxu",
                        # voice_name="AudiA"
                        # voice_id="CV4xD6M8z1X1kya4Pepj",
                        # voice_name="Dio"
                        # voice_id="vnd0afTMgWq4fDRVyDo3",
                        # voice_name="Deobra"
                        # voice_id="tnSpp4vdxKPjI9w0GnoV",
                        # voice_name="Hope"
                    ),
                    TtsProviderSettings(
                        voice_provider="google",
                        voice_id="en-US-Chirp-HD-F",
                        voice_name="Kate"
                    ),
                ]
            )
        elif "liddyai.myshopify.com" == self.account:
            return AccountTTSSettings(
                providers=[
                    ElevenLabsTtsProviderSettings(
                        voice_id="7YaUDeaStRuoYg3FKsmU",
                        voice_name="Callie"
                        # voice_id="vnd0afTMgWq4fDRVyDo3",
                        # voice_name="Deobra"
                        # voice_id="tnSpp4vdxKPjI9w0GnoV",
                        # voice_name="Hope"
                    ),
                    TtsProviderSettings(
                        voice_provider="google",
                        voice_id="en-US-Chirp-HD-F",
                        voice_name="Kate"
                    ),
                ]
            )
        elif "specialized.com" == self.account:
            return AccountTTSSettings(
                providers=[
                    ElevenLabsTtsProviderSettings(
                        voice_id="8sGzMkj2HZn6rYwGx6G0",
                        voice_name="Tomas"
                    ),
                    TtsProviderSettings(
                        voice_provider="google",
                        voice_id="en-US-Chirp3-HD-Orus",
                        voice_name="Spence"
                    )
                ]
            )
        else:
            return AccountTTSSettings(
                providers=[
                    ElevenLabsTtsProviderSettings(
                        voice_id="8sGzMkj2HZn6rYwGx6G0",
                        voice_name="Tomas"
                    ),
                    TtsProviderSettings(
                        voice_provider="google",
                        voice_id="en-US-Chirp3-HD-Orus",
                        voice_name="Spence"
                    )
                ]
            )
    
    def _register_hardcoded_fallback(self):
        """Register hardcoded configuration as fallback for this account"""
        try:
            # Build a basic fallback config structure
            hardcoded_config = {
                "account": {
                    "type": "hardcoded_fallback",
                    "status": "active",
                    "domain": self.account
                },
                "agent": {
                    "persona": {
                        "greeting": self._get_hardcoded_greeting()
                    },
                    "voice": self._extract_voice_config_from_tts_settings()
                }
            }
            
            # Register with the config loader
            self.config_loader.register_fallback_config(self.account, hardcoded_config)
            
        except Exception as e:
            logger.warning(f"Failed to register hardcoded fallback for {self.account}: {e}")
    
    def _get_hardcoded_greeting(self) -> str:
        """Get hardcoded greeting without checking dynamic config (to avoid circular dependency)"""
        if "sundayriley.com" in self.account:
            return "Hi, this is Kate. How can I help you today?"
        elif "darakayejewelry.com" in self.account:
            return "Hi, this is Dara! How can I help you today?"
        elif "darakayejewelry-test.myshopify.com" in self.account:
            return "Hi, this is Dara! How can I help you today?"
        elif "flex-liddy.myshopify.com" in self.account or "flexfits.com" in self.account:
            return "Hi, this is Lauren! How can I help you today?"
        elif "balenciaga.com" in self.account:
            return "Bonjour. I'm Bela—your Balenciaga curator. How may I assist?"
        elif "gucci.com" in self.account:
            return "Ciao, I'm Gigi, your Gucci AI Client Advisor for handbags. How may I assist you?"
        elif "liddyai.myshopify.com" in self.account:
            return "Hi, this is Callie. How can I help you today?"
        elif "specialized.com" in self.account:
            return "Hey, this is Spence! How can I help?"
        else:
            return "Hello! How can I help?"
    
    def _extract_voice_config_from_tts_settings(self) -> Dict[str, Any]:
        """Extract voice configuration from TTS settings for fallback"""
        try:
            primary_provider = self.tts_settings.primary_provider
            if isinstance(primary_provider, ElevenLabsTtsProviderSettings):
                return {
                    "provider": "elevenlabs",
                    "voice_id": primary_provider.voice_id,
                    "voice_name": primary_provider.voice_name,
                    "model": primary_provider.voice_model,
                    "stability": primary_provider.voice_stability,
                    "similarity_boost": primary_provider.voice_similarity_boost,
                    "style": primary_provider.voice_style,
                    "use_speaker_boost": primary_provider.voice_use_speaker_boost,
                    "speed": primary_provider.voice_speed
                }
            else:
                return {
                    "provider": primary_provider.voice_provider,
                    "voice_id": primary_provider.voice_id, 
                    "voice_name": primary_provider.voice_name
                }
        except Exception as e:
            logger.warning(f"Error extracting voice config: {e}")
            return {
                "provider": "elevenlabs",
                "voice_id": "8sGzMkj2HZn6rYwGx6G0",
                "voice_name": "Tomas"
            }

    def get_account(self):
        return self.account

    def set_account(self, account):
        self.account = account
        return self.account
    
    def get_variant_types(self) -> list[AccountVariantType]:
        if "specialized.com" == self.account:
            return [AccountVariantType(name="color", query_param_name="color")]
        else:
            return []
    
    def get_tts_settings(self) -> AccountTTSSettings:
        return self.tts_settings

    def get_rag_details(self) -> str:
        """Get the Pinecone index name for this account.
        
        Returns:
            str: The Pinecone index name, or empty string if no RAG index is configured
        """
        if "sundayriley.com" == self.account:
            return "sundayriley-llama-2048"
        elif "darakayejewelry.com" == self.account:
            # return "darakayejewelry-llama-2048"
            return ""
        elif "darakayejewelry-test.myshopify.com" == self.account:
            # return "darakayejewelry-llama-2048"
            return ""
        elif "balenciaga.com" == self.account:
            return "balenciaga-llama-2048"
        elif "gucci.com" == self.account:
            return "gucci-llama-2048"
        elif "specialized.com" == self.account:
            return "specialized-llama-2048"
        else:
            return ""

    def get_default_greeting(self) -> str:
        # Try dynamic configuration first
        if self.dynamic_config:
            try:
                greeting = None
                
                # NEW SCHEMA: Multi-agent with agents array
                if 'agents' in self.dynamic_config and self.dynamic_config['agents']:
                    greeting = self.dynamic_config['agents'][0].get('persona', {}).get('greeting')
                # OLD SCHEMA: Single agent at root level (backward compatibility)
                elif 'agent' in self.dynamic_config:
                    greeting = self.dynamic_config.get('agent', {}).get('persona', {}).get('greeting')
                
                if greeting:
                    return greeting
            except Exception as e:
                logger.warning(f"Error getting greeting from dynamic config: {e}")
        
        # Fall back to hardcoded greetings
        return self._get_hardcoded_greeting()

    def get_personality_sliders(self) -> Dict[str, Any]:
        """Get personality sliders from dynamic configuration (supports old and new schema)"""
        if self.dynamic_config:
            try:
                persona = None
                
                # NEW SCHEMA: Multi-agent with agents array
                if 'agents' in self.dynamic_config and self.dynamic_config['agents']:
                    persona = self.dynamic_config['agents'][0].get('persona', {})
                # OLD SCHEMA: Single agent at root level (backward compatibility)  
                elif 'agent' in self.dynamic_config:
                    persona = self.dynamic_config.get('agent', {}).get('persona', {})
                
                if persona:
                    # Extract personality slider values from persona (exclude greeting)
                    sliders = {}
                    for key, value in persona.items():
                        if key != 'greeting' and isinstance(value, (int, float)):
                            sliders[key] = value
                    return sliders
                    
            except Exception as e:
                logger.warning(f"Error getting personality sliders from dynamic config: {e}")
        
        return {}
    
    # private method to extract the account name from a string
    def _extract_account_name(self, account_str):
        # Assuming the account name is the first part of the string
        if not account_str:
            account_str = "specialized.com"
        
        # clean account to only the domain name lowercased
        # strip off protocol and www
        account_str =account_str.lower()
        account_protocol = account_str.split("://")
        if len(account_protocol) > 1:
            account_str = account_protocol[1]
        account_str = account_str.split("/")[0]
        account_dots = account_str.split(".")
        # if www, then remove it
        if account_dots[0] == "www":
            account_str = ".".join(account_dots[1:])
        # if len(account_dots) > 2:
        #     account_str = ".".join(account_dots[1:])
        # if no dot, then add .com
        if "." not in account_str:
            account_str = account_str + ".com"
        return account_str


# =============================================================================
# SINGLETON FACTORY FUNCTIONS - Clean API using singleton AccountManager pattern  
# =============================================================================

async def get_account_manager(account: str = None) -> AccountManager:
    """
    Get singleton AccountManager for account with async config loading.
    
    This is the main entry point for all account operations. Uses singleton pattern
    to ensure one AccountManager instance per account across the application.
    
    Args:
        account: Account name (domain) to use for configuration
        
    Returns:
        AccountManager instance (singleton for this account)
    """
    clean_account = AccountManager._extract_account_name(None, account)
    
    if clean_account not in _account_managers:
        # Create instance without calling __init__ to avoid sync config loading
        instance = AccountManager.__new__(AccountManager)
        instance.account = clean_account
        instance.config_loader = get_account_config_loader()
        instance.dynamic_config = None
        
        # Try to load dynamic configuration
        try:
            instance.dynamic_config = await instance.config_loader.get_account_config(instance.account)
            if instance.dynamic_config:
                logger.info(f"Loaded dynamic config for {instance.account}")
                instance.tts_settings = instance._build_tts_settings_from_config(instance.dynamic_config)
            else:
                logger.info(f"No dynamic config found for {instance.account}, using hardcoded defaults")
                instance.tts_settings = instance._build_hardcoded_tts_settings()
        except Exception as e:
            logger.warning(f"Error loading dynamic config for {instance.account}: {e}")
            logger.info(f"Falling back to hardcoded config for {instance.account}")
            instance.dynamic_config = None
            instance.tts_settings = instance._build_hardcoded_tts_settings()
        
        # Register hardcoded config as fallback for future use
        instance._register_hardcoded_fallback()
        
        # Cache the instance
        _account_managers[clean_account] = instance
    
    return _account_managers[clean_account]


def clear_account_managers():
    """Clear all AccountManager instances from memory"""
    global _account_managers
    _account_managers.clear()
    logger.info("Cleared all account managers from memory")
