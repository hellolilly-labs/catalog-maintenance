"""
Account Manager for Catalog Maintenance

Provides account-specific configuration and intelligence for search,
catalog management, and query enhancement.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from liddy.storage import AccountStorageProvider, get_account_storage_provider
from .models import (
    SearchConfig,
    AccountVariantType,
    TtsProviderSettings,
    ElevenLabsTtsProviderSettings,
    AccountTTSSettings
)

logger = logging.getLogger(__name__)

# Global cache for AccountManager instances (singleton pattern)
_account_managers: Dict[str, "AccountManager"] = {}


from liddy.account_config_loader import get_account_config_loader

class AccountManager:
    """
    Manages account-specific configuration and intelligence.
    
    This is the central configuration hub for:
    - Search backend configuration
    - Product catalog intelligence
    - Query enhancement context
    - Brand-specific settings
    """

    # _cached_account_managers = {}
    # @staticmethod
    # def get_account_manager(account: str = None):
    #     if account in AccountManager._cached_account_managers:
    #         return AccountManager._cached_account_managers[account]
    #     else:
    #         account_manager = AccountManager(account)
    #         AccountManager._cached_account_managers[account] = account_manager
    #         return account_manager

    
    def __init__(self, account: str):
        """
        Initialize AccountManager for a specific account.
        
        Args:
            account: Account/brand domain (e.g., "specialized.com")
        """
        self.account = self._normalize_account(account)
        self._search_config: Optional[SearchConfig] = None
        self._catalog_intelligence: Optional[Dict[str, Any]] = None
        self._brand_config: Dict[str, Any] = {}
        self._account_config: Optional[Dict[str, Any]] = None
        self._tts_settings: Optional[AccountTTSSettings] = None
        self._is_voice_enabled: bool = False
        
    def _normalize_account(self, account: str) -> str:
        """Normalize account name to standard format."""
        if not account:
            return "default"
            
        # Remove protocol and www
        account = account.lower()
        if "://" in account:
            account = account.split("://")[1]
        if account.startswith("www."):
            account = account[4:]
        
        # Extract domain
        account = account.split("/")[0]
        
        # Add .com if no TLD
        if "." not in account:
            account = f"{account}.com"
            
        return account
    
    def get_search_config(self) -> SearchConfig:
        """
        Get search backend configuration for this account.
        
        Returns:
            SearchConfig with backend details
        """
        if self._search_config is None:
            # Build search config based on account
            brand_name = self.account.replace('.', '-')
            
            # Default to Pinecone with separate indexes
            self._search_config = SearchConfig(
                backend="pinecone",
                dense_index=f"{brand_name}-dense",
                sparse_index=f"{brand_name}-sparse",
                config={
                    "cloud": "gcp",
                    "region": "us-central1",
                    "dense_model": "llama-text-embed-v2",
                    "sparse_model": "pinecone-sparse-english-v0",
                    "rerank_model": "bge-reranker-v2-m3"
                }
            )
            
            # Account-specific overrides
            if self.account == "specialized.com":
                self._search_config.unified_index = "specialized-llama-2048"
            elif self.account == "sundayriley.com":
                self._search_config.unified_index = "sundayriley-llama-2048"
                
        return self._search_config
    
    async def get_catalog_intelligence(self) -> str:
        """
        Get ProductCatalogResearcher intelligence for this account.
        
        Returns cached data if available, otherwise loads from storage.
        
        Returns:
            Dictionary containing catalog research data
        """
        if self._catalog_intelligence is None:
            try:
                # Try to load from ProductCatalogResearcher output
                from liddy.storage import get_account_storage_provider
                
                storage: AccountStorageProvider = get_account_storage_provider()
                content = await storage.get_research_data(self.account, "product_catalog")
                
                if content:
                    self._catalog_intelligence = content
                    logger.info(f"Loaded catalog intelligence for {self.account}")
                else:
                    logger.warning(f"No catalog intelligence found for {self.account}")
                    self._catalog_intelligence = ''
                    
            except Exception as e:
                logger.error(f"Error loading catalog intelligence: {e}")
                import traceback
                traceback.print_exc()
                self._catalog_intelligence = ''
        
        return self._catalog_intelligence
        
    async def get_query_enhancement_context(self) -> str:
        """
        Get context for query enhancement.
        
        Returns search context from catalog intelligence.
        """
        return await self.get_catalog_intelligence()
    
    async def get_search_enhancement_prompt(self, context_type: str = "product") -> str:
        """
        Get the compact search enhancement prompt from catalog intelligence.
        
        This extracts Part C (product) or Part D (knowledge) from the ProductCatalogResearcher output,
        which is a ready-to-use prompt for search query enhancement.
        
        Args:
            context_type: Either "product" or "knowledge" to get the appropriate prompt
        
        Returns:
            The extracted search enhancement prompt, or a fallback if not found
        """
        if not self._catalog_intelligence:
            await self.get_catalog_intelligence()
        
        if not self._catalog_intelligence:
            logger.warning(f"No catalog intelligence available for {self.account}")
            return self._get_fallback_search_prompt(context_type)
        
        # Extract appropriate part from the catalog intelligence
        if context_type == "knowledge":
            return self._extract_knowledge_enhancement_prompt(self._catalog_intelligence)
        else:
            return self._extract_search_enhancement_prompt(self._catalog_intelligence)
    
    def _extract_search_enhancement_prompt(self, catalog_content: str) -> str:
        """Extract the Part C search enhancement prompt from catalog research."""
        # Look for the Part C section markers
        start_marker = "**THIS SECTION IS FOR DIRECT USE IN SEARCH QUERY ENHANCEMENT - COPY AS-IS**"
        end_marker = "**END OF SEARCH ENHANCEMENT PROMPT**"
        
        start_idx = catalog_content.find(start_marker)
        if start_idx == -1:
            logger.warning(f"Search enhancement prompt not found in catalog intelligence for {self.account}")
            return self._get_fallback_search_prompt()
        
        # Move past the marker
        start_idx += len(start_marker)
        
        end_idx = catalog_content.find(end_marker, start_idx)
        if end_idx == -1:
            logger.warning(f"Search enhancement prompt end marker not found for {self.account}")
            return self._get_fallback_search_prompt()
        
        # Extract the content between markers
        prompt_content = catalog_content[start_idx:end_idx].strip()
        
        # Replace any remaining template variables
        brand_name = self.account.split('.')[0].title()
        prompt_content = prompt_content.replace("{{brand_name}}", brand_name)
        prompt_content = prompt_content.replace("{{brand_domain}}", self.account)
        
        return prompt_content
    
    def _extract_knowledge_enhancement_prompt(self, catalog_content: str) -> str:
        """Extract the Part D knowledge enhancement prompt from catalog research."""
        # Look for the Part D section markers
        start_marker = "**THIS SECTION IS FOR DIRECT USE IN KNOWLEDGE SEARCH ENHANCEMENT - COPY AS-IS**"
        end_marker = "**END OF KNOWLEDGE ENHANCEMENT PROMPT**"
        
        start_idx = catalog_content.find(start_marker)
        if start_idx == -1:
            logger.warning(f"Knowledge enhancement prompt not found in catalog intelligence for {self.account}")
            return self._get_fallback_search_prompt("knowledge")
        
        # Move past the marker
        start_idx += len(start_marker)
        
        end_idx = catalog_content.find(end_marker, start_idx)
        if end_idx == -1:
            logger.warning(f"Knowledge enhancement prompt end marker not found for {self.account}")
            return self._get_fallback_search_prompt("knowledge")
        
        # Extract the content between markers
        prompt_content = catalog_content[start_idx:end_idx].strip()
        
        # Replace any remaining template variables
        brand_name = self.account.split('.')[0].title()
        prompt_content = prompt_content.replace("{{brand_name}}", brand_name)
        prompt_content = prompt_content.replace("{{brand_domain}}", self.account)
        
        return prompt_content
    
    def _get_fallback_search_prompt(self, context_type: str = "product") -> str:
        """Get a fallback search enhancement prompt when catalog intelligence is unavailable."""
        brand_name = self.account.split('.')[0].title()
        
        if context_type == "knowledge":
            return f"""### Knowledge Search Enhancement Instructions for {brand_name}

You are enhancing knowledge/support queries for {brand_name}. Apply these general guidelines:

#### 1. Support Topics
Expand vague help requests into specific support categories like returns, warranty, sizing, troubleshooting.

#### 2. Policy Areas
Include relevant policy terms when users ask about processes or rules.

#### 3. Problem Translation
Convert problem descriptions into searchable support topics.

#### 4. Knowledge Enhancement Rules:
1. Clarify if user needs how-to, troubleshooting, or policy info
2. Add product context when relevant
3. Include {brand_name}-specific terminology
4. Focus on solving the user's problem"""
        
        else:
            return f"""### Search Enhancement Instructions for {brand_name}

You are enhancing product search queries for {brand_name}. Apply these general guidelines:

#### 1. Brand-Specific Terminology
Add "{brand_name}" and related brand terms to generic queries.

#### 2. Query Expansion
Expand queries with relevant synonyms and related terms.

#### 3. Technical Terms
Include both common and technical terms for products.

#### 4. Search Enhancement Rules:
1. Maintain the user's original intent
2. Add relevant context and specifications
3. Include brand-specific terminology
4. Keep queries natural and not overly complex"""
    
    def get_variant_types(self) -> List[Dict[str, str]]:
        """
        Get product variant types for this account.
        
        Returns:
            List of variant type configurations
        """
        # Default variants
        variants = []
        
        # Account-specific variants
        if self.account == "specialized.com":
            variants.append({
                "name": "color",
                "query_param": "color",
                "display_name": "Color"
            })
            variants.append({
                "name": "size", 
                "query_param": "size",
                "display_name": "Size"
            })
        elif self.account in ["sundayriley.com", "darakayejewelry.com"]:
            variants.append({
                "name": "size",
                "query_param": "size", 
                "display_name": "Size"
            })
            
        return variants
    
    def get_brand_config(self) -> Dict[str, Any]:
        """
        Get brand-specific configuration.
        
        Returns:
            Dictionary with brand settings
        """
        if not self._brand_config:
            self._brand_config = {
                "account": self.account,
                "display_name": self._get_display_name(),
                "currency": "USD",
                "features": self._get_brand_features()
            }
        
        return self._brand_config
    
    def _get_display_name(self) -> str:
        """Get display name for the brand."""
        display_names = {
            "specialized.com": "Specialized",
            "sundayriley.com": "Sunday Riley", 
            "darakayejewelry.com": "Dara Kaye Jewelry",
            "balenciaga.com": "Balenciaga",
            "gucci.com": "Gucci"
        }
        return display_names.get(self.account, self.account.split('.')[0].title())
    
    def _get_brand_features(self) -> List[str]:
        """Get enabled features for this brand."""
        features = ["search", "catalog"]
        
        # Add brand-specific features
        if self.account in ["specialized.com", "sundayriley.com"]:
            features.extend(["rag", "hybrid_search", "reranking"])
        
        if self.account == "specialized.com":
            features.append("variants")
            
        return features
    
    async def _load_account_config(self) -> None:
        """Load account configuration from storage."""
        if self._account_config is not None:
            return
            
        try:
            account_config_loader = get_account_config_loader()
            self._account_config = await account_config_loader.get_account_config(self.account)
            
            # storage = get_account_storage_provider()
            # config_data = await storage.read_file(self.account, "account.json")
            # if config_data:
            #     self._account_config = json.loads(config_data)
            #     logger.info(f"Loaded account config for {self.account}")
                
            #     # Load voice settings if available
            #     if "voice" in self._account_config:
            #         self._load_voice_settings_from_config()
            #         self._is_voice_enabled = True
            # else:
            #     logger.debug(f"No account.json found for {self.account}")
            #     self._account_config = {}
                
        except Exception as e:
            logger.warning(f"Error loading account config for {self.account}: {e}")
            self._account_config = {}
    
    def _load_voice_settings_from_config(self) -> None:
        """Load voice settings from account configuration."""
        if not self._account_config or "voice" not in self._account_config:
            return
            
        voice_config = self._account_config["voice"]
        
        # Build TTS settings from config
        self._tts_settings = self._build_tts_settings_from_config(voice_config)
    
    def _build_tts_settings_from_config(self, voice_config: Dict[str, Any]) -> AccountTTSSettings:
        """Build TTS settings from voice configuration."""
        providers = []
        
        # Primary provider
        primary = voice_config.get("primary", {})
        if primary.get("provider") == "elevenlabs":
            providers.append(ElevenLabsTtsProviderSettings(
                voice_id=primary.get("voice_id"),
                voice_name=primary.get("voice_name", "Dynamic Voice"),
                voice_model=primary.get("model"),
                voice_stability=primary.get("stability"),
                voice_similarity_boost=primary.get("similarity_boost"),
                voice_style=primary.get("style"),
                voice_use_speaker_boost=primary.get("use_speaker_boost"),
                voice_speed=primary.get("speed")
            ))
        else:
            providers.append(TtsProviderSettings(
                voice_provider=primary.get("provider", "unknown"),
                voice_id=primary.get("voice_id", ""),
                voice_name=primary.get("voice_name", "")
            ))
        
        # Fallback providers
        for fallback in voice_config.get("fallbacks", []):
            if fallback.get("provider") == "elevenlabs":
                providers.append(ElevenLabsTtsProviderSettings(
                    voice_id=fallback.get("voice_id"),
                    voice_name=fallback.get("voice_name", "Fallback Voice"),
                    voice_model=fallback.get("model"),
                    voice_stability=fallback.get("stability"),
                    voice_similarity_boost=fallback.get("similarity_boost"),
                    voice_style=fallback.get("style"),
                    voice_use_speaker_boost=fallback.get("use_speaker_boost"),
                    voice_speed=fallback.get("speed")
                ))
            else:
                providers.append(TtsProviderSettings(
                    voice_provider=fallback.get("provider", "google"),
                    voice_id=fallback.get("voice_id", "en-US-Chirp-HD-F"),
                    voice_name=fallback.get("voice_name", "Kate")
                ))
        
        # Default Google fallback if no fallbacks specified
        if len(providers) == 1:
            providers.append(TtsProviderSettings(
                voice_provider="google",
                voice_id="en-US-Chirp-HD-F",
                voice_name="Kate"
            ))
        
        return AccountTTSSettings(providers)
    
    async def get_tts_settings(self) -> Optional[AccountTTSSettings]:
        """
        Get TTS settings for this account.
        
        Returns:
            AccountTTSSettings if voice is configured, None otherwise
        """
        await self._load_account_config()
        
        # Use configured settings if available
        if self._tts_settings:
            return self._tts_settings
        
        # Fall back to hardcoded settings
        return self._get_hardcoded_tts_settings()
    
    def _get_hardcoded_tts_settings(self) -> Optional[AccountTTSSettings]:
        """Get hardcoded TTS settings for backward compatibility."""
        # Hardcoded settings for specific accounts
        if self.account == "sundayriley.com":
            return AccountTTSSettings([
                ElevenLabsTtsProviderSettings(
                    voice_id="CV4xD6M8z1X1kya4Pepj",
                    voice_name="Dio",
                    voice_stability=0.75,
                    voice_similarity_boost=0.8,
                    voice_style=0.0,
                    voice_use_speaker_boost=True,
                    voice_model="eleven_turbo_v2_5",
                ),
                TtsProviderSettings(
                    voice_provider="google",
                    voice_id="en-US-Chirp-HD-F",
                    voice_name="Kate"
                ),
            ])
        elif self.account == "darakayejewelry.com":
            return AccountTTSSettings([
                ElevenLabsTtsProviderSettings(
                    voice_id="8DzKSPdgEQPaK5vKG0Rs",
                    voice_name="Vanessa",
                    voice_model="eleven_flash_v2_5",
                    voice_stability=0.35,
                    voice_similarity_boost=0.6,
                    voice_style=0.0,
                    voice_use_speaker_boost=True,
                    voice_speed=1.07,
                ),
                TtsProviderSettings(
                    voice_provider="google",
                    voice_id="en-US-Chirp-HD-F",
                    voice_name="Kate"
                ),
            ])
        elif self.account == "darakayejewelry-test.myshopify.com":
            return AccountTTSSettings([
                ElevenLabsTtsProviderSettings(
                    voice_id="8DzKSPdgEQPaK5vKG0Rs",
                    voice_name="Vanessa",
                    voice_model="eleven_flash_v2_5",
                    voice_stability=0.35,
                    voice_similarity_boost=0.75,
                    voice_style=0.0,
                    voice_use_speaker_boost=True,
                    voice_speed=1.07,
                ),
                TtsProviderSettings(
                    voice_provider="google",
                    voice_id="en-US-Chirp-HD-F",
                    voice_name="Kate"
                ),
            ])
        elif self.account in ["flex-liddy.myshopify.com", "flexfits.com"]:
            return AccountTTSSettings([
                ElevenLabsTtsProviderSettings(
                    voice_id="qgNVOmUjsNA8CD4Fed8D",
                    voice_name="Lindsay"
                ),
                TtsProviderSettings(
                    voice_provider="google",
                    voice_id="en-US-Chirp-HD-F",
                    voice_name="Kate"
                ),
            ])
        elif self.account == "balenciaga.com":
            return AccountTTSSettings([
                ElevenLabsTtsProviderSettings(
                    voice_id="txtf1EDouKke753vN8SL",
                    voice_name="Jeanne"
                ),
                TtsProviderSettings(
                    voice_provider="google",
                    voice_id="en-US-Chirp-HD-F",
                    voice_name="Kate"
                ),
            ])
        elif self.account == "gucci.com":
            return AccountTTSSettings([
                ElevenLabsTtsProviderSettings(
                    voice_id="8KInRSd4DtD5L5gK7itu",
                    voice_name="VIP",
                    voice_model="eleven_flash_v2_5",
                    voice_stability=0.75,
                    voice_similarity_boost=0.8,
                    voice_style=0.0,
                    voice_use_speaker_boost=True,
                ),
                TtsProviderSettings(
                    voice_provider="google",
                    voice_id="en-US-Chirp-HD-F",
                    voice_name="Kate"
                ),
            ])
        elif self.account == "liddyai.myshopify.com":
            return AccountTTSSettings([
                ElevenLabsTtsProviderSettings(
                    voice_id="7YaUDeaStRuoYg3FKsmU",
                    voice_name="Callie"
                ),
                TtsProviderSettings(
                    voice_provider="google",
                    voice_id="en-US-Chirp-HD-F",
                    voice_name="Kate"
                ),
            ])
        elif self.account == "specialized.com":
            return AccountTTSSettings([
                ElevenLabsTtsProviderSettings(
                    voice_id="8sGzMkj2HZn6rYwGx6G0",
                    voice_name="Tomas"
                ),
                TtsProviderSettings(
                    voice_provider="google",
                    voice_id="en-US-Chirp3-HD-Orus",
                    voice_name="Spence"
                )
            ])
        
        # Default settings for unknown accounts
        return AccountTTSSettings([
            ElevenLabsTtsProviderSettings(
                voice_id="8sGzMkj2HZn6rYwGx6G0",
                voice_name="Tomas"
            ),
            TtsProviderSettings(
                voice_provider="google",
                voice_id="en-US-Chirp3-HD-Orus",
                voice_name="Spence"
            )
        ])
    
    async def get_default_greeting(self) -> str:
        """
        Get default greeting for the voice assistant.
        
        Returns:
            Greeting string
        """
        await self._load_account_config()
        
        # Check config first
        if self._account_config:
            greeting = self._account_config.get("voice", {}).get("greeting")
            if greeting:
                return greeting
            
            # Also check agent config for compatibility
            greeting = self._account_config.get("agent", {}).get("persona", {}).get("greeting")
            if greeting:
                return greeting
        
        # Hardcoded greetings
        greetings = {
            "sundayriley.com": "Hi, this is Kate. How can I help you today?",
            "specialized.com": "Hey, this is Spence! How can I help?",
            "darakayejewelry.com": "Hi, this is Dara! How can I help you today?",
            "darakayejewelry-test.myshopify.com": "Hi, this is Dara! How can I help you today?",
            "flex-liddy.myshopify.com": "Hi, this is Lauren! How can I help you today?",
            "flexfits.com": "Hi, this is Lauren! How can I help you today?",
            "balenciaga.com": "Bonjour. I'm Bela—your Balenciaga curator. How may I assist?",
            "gucci.com": "Ciao, I'm Gigi, your Gucci AI Client Advisor for handbags. How may I assist you?",
            "liddyai.myshopify.com": "Hi, this is Callie. How can I help you today?",
        }
        
        return greetings.get(self.account, "Hello! How can I help?")
    
    async def get_personality_sliders(self) -> Dict[str, float]:
        """
        Get personality sliders from configuration.
        
        Returns:
            Dictionary of personality parameters
        """
        await self._load_account_config()
        
        if self._account_config:
            persona = self._account_config.get("voice", {}).get("persona", {})
            
            # Extract numeric values (sliders)
            sliders = {}
            for key, value in persona.items():
                if key != "greeting" and isinstance(value, (int, float)):
                    sliders[key] = float(value)
            
            if sliders:
                return sliders
        
        return {}
    
    def get_rag_details(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get RAG configuration details for voice assistant.
        
        Returns:
            Tuple of (index_name, embedding_model)
        """
        # Use unified index if available
        search_config = self.get_search_config()
        
        if search_config.unified_index:
            embedding_model = search_config.config.get("dense_model", "llama-text-embed-v2")
            return (search_config.unified_index, embedding_model)
        
        # Legacy hardcoded indexes
        rag_indexes = {
            "sundayriley.com": "sundayriley-llama-2048",
            "balenciaga.com": "balenciaga-llama-2048",
            "gucci.com": "gucci-llama-2048",
            "specialized.com": "specialized-llama-2048",
        }
        
        index_name = rag_indexes.get(self.account, "")
        embedding_model = "llama-text-embed-v2"
        
        return (index_name, embedding_model)
    
    def get_voice_variant_types(self) -> List[AccountVariantType]:
        """
        Get variant types for voice assistant (legacy compatibility).
        
        Returns:
            List of AccountVariantType objects
        """
        # Convert from the main variant types
        variants = []
        for variant_dict in self.get_variant_types():
            variants.append(AccountVariantType(
                name=variant_dict["name"],
                query_param_name=variant_dict.get("query_param")
            ))
        return variants


async def get_account_manager(account: str) -> AccountManager:
    """
    Get or create AccountManager instance for an account.
    
    Uses singleton pattern for efficiency.
    
    Args:
        account: Account/brand domain
        
    Returns:
        AccountManager instance
    """
    normalized_account = AccountManager._normalize_account(None, account)
    
    if normalized_account not in _account_managers:
        instance = AccountManager(normalized_account)
        _account_managers[normalized_account] = instance
        # # Pre-load account config
        # await instance._load_account_config()
        logger.info(f"Created AccountManager for {normalized_account}")
    
    return _account_managers[normalized_account]


def clear_account_managers():
    """Clear all cached AccountManager instances."""
    _account_managers.clear()
    logger.info("Cleared all AccountManager instances")