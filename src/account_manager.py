"""
Account Manager for Catalog Maintenance

Provides account-specific configuration and intelligence for search,
catalog management, and query enhancement.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Global cache for AccountManager instances (singleton pattern)
_account_managers: Dict[str, "AccountManager"] = {}


@dataclass
class SearchConfig:
    """Search backend configuration for an account."""
    backend: str = "pinecone"  # pinecone, elasticsearch, etc.
    dense_index: Optional[str] = None
    sparse_index: Optional[str] = None
    unified_index: Optional[str] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class AccountManager:
    """
    Manages account-specific configuration and intelligence.
    
    This is the central configuration hub for:
    - Search backend configuration
    - Product catalog intelligence
    - Query enhancement context
    - Brand-specific settings
    """
    
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
            brand_name = self.account.split('.')[0]
            
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
    
    async def get_catalog_intelligence(self) -> Dict[str, Any]:
        """
        Get ProductCatalogResearcher intelligence for this account.
        
        Returns cached data if available, otherwise loads from storage.
        
        Returns:
            Dictionary containing catalog research data
        """
        if self._catalog_intelligence is None:
            try:
                # Try to load from ProductCatalogResearcher output
                from src.storage import get_account_storage_provider
                
                storage = get_account_storage_provider()
                content = await storage.load_content(
                    brand_domain=self.account,
                    content_type="research/product_catalog"
                )
                
                if content:
                    research_data = json.loads(content)
                    self._catalog_intelligence = self._extract_catalog_intelligence(research_data)
                    logger.info(f"Loaded catalog intelligence for {self.account}")
                else:
                    logger.warning(f"No catalog intelligence found for {self.account}")
                    self._catalog_intelligence = {}
                    
            except Exception as e:
                logger.error(f"Error loading catalog intelligence: {e}")
                self._catalog_intelligence = {}
        
        return self._catalog_intelligence
    
    def _extract_catalog_intelligence(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key intelligence from ProductCatalogResearcher output."""
        intelligence = {
            'descriptor_context': '',
            'search_context': '',
            'brand_insights': '',
            'category_mappings': {},
            'attribute_mappings': {},
            'quality_score': 0.0
        }
        
        if 'content' in research_data:
            content = research_data['content']
            
            # Extract descriptor generation context
            if '## Product Descriptor Generation Context' in content:
                start = content.find('## Product Descriptor Generation Context')
                end = content.find('## Product Knowledge Search Context', start)
                if end > start:
                    intelligence['descriptor_context'] = content[start:end].strip()
            
            # Extract search context
            if '## Product Knowledge Search Context' in content:
                start = content.find('## Product Knowledge Search Context')
                end = content.find('## ', start + 30) if '## ' in content[start + 30:] else len(content)
                intelligence['search_context'] = content[start:end].strip()
            
            # Extract brand insights
            if '### Brand Overview' in content:
                start = content.find('### Brand Overview')
                end = content.find('### ', start + 20) if '### ' in content[start + 20:] else start + 1000
                intelligence['brand_insights'] = content[start:end].strip()
        
        # Extract metadata
        if 'metadata' in research_data:
            metadata = research_data['metadata']
            intelligence['quality_score'] = metadata.get('quality_score', 0.0)
            
        return intelligence
    
    async def get_query_enhancement_context(self) -> str:
        """
        Get context for query enhancement.
        
        Returns search context from catalog intelligence.
        """
        intelligence = await self.get_catalog_intelligence()
        return intelligence.get('search_context', '')
    
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
        _account_managers[normalized_account] = AccountManager(normalized_account)
        logger.info(f"Created AccountManager for {normalized_account}")
    
    return _account_managers[normalized_account]


def clear_account_managers():
    """Clear all cached AccountManager instances."""
    _account_managers.clear()
    logger.info("Cleared all AccountManager instances")