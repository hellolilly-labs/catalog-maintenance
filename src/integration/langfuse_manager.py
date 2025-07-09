"""
Langfuse Integration for RAG System

Manages prompts, filter dictionaries, and search configurations in Langfuse.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not available - prompts will use local defaults")

logger = logging.getLogger(__name__)


class LangfuseRAGManager:
    """
    Manages RAG-related prompts and configurations in Langfuse.
    
    Features:
    - Filter dictionary management
    - Search prompt templates
    - Query optimization prompts
    - Brand-specific configurations
    - Version control for prompts
    """
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.brand_key = brand_domain.replace('.', '-').lower()
        
        # Initialize Langfuse if available
        self.langfuse = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse = Langfuse()
                logger.info(f"✅ Langfuse initialized for {brand_domain}")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}")
        
        # Prompt naming conventions
        self.prompt_names = {
            'filter_dictionary': f'liddy/catalog/{self.brand_key}/filter_dictionary',
            'query_optimizer': f'liddy/catalog/{self.brand_key}/query_optimizer',
            'search_enhancer': f'liddy/catalog/{self.brand_key}/search_enhancer',
            'product_presenter': f'liddy/catalog/{self.brand_key}/product_presenter',
            'filter_extractor': f'liddy/catalog/{self.brand_key}/filter_extractor'
        }
        
        # Local cache for prompts
        self.prompt_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Local fallbacks
        self.local_prompts_path = Path(f"accounts/{brand_domain}/prompts")
        self.local_prompts_path.mkdir(parents=True, exist_ok=True)
    
    def update_filter_dictionary(self, filters: Dict[str, Any]) -> bool:
        """
        Update the filter dictionary in Langfuse.
        
        Args:
            filters: Dictionary of available filters from catalog analysis
            
        Returns:
            True if update successful
        """
        # Prepare filter dictionary for prompt
        filter_dict = self._prepare_filter_dictionary(filters)
        
        # Create prompt content
        prompt_content = {
            "name": self.prompt_names['filter_dictionary'],
            "prompt": self._create_filter_dictionary_prompt(filter_dict),
            "config": {
                "brand": self.brand_domain,
                "filters": filter_dict,
                "updated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "labels": ["rag", "filters", self.brand_key],
            "tags": ["auto-generated", "catalog-sync"]
        }
        
        # Update in Langfuse
        if self.langfuse:
            try:
                # Create or update prompt
                self.langfuse.create_prompt(
                    name=self.prompt_names['filter_dictionary'],
                    prompt=prompt_content['prompt'],
                    config=prompt_content['config'],
                    labels=prompt_content['labels'],
                    tags=prompt_content['tags']
                )
                logger.info(f"✅ Updated filter dictionary in Langfuse")
                
                # Clear cache
                self.prompt_cache.pop(self.prompt_names['filter_dictionary'], None)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to update Langfuse: {e}")
        
        # Fallback to local storage
        self._save_local_prompt('filter_dictionary', prompt_content)
        return True
    
    def update_query_optimizer_prompt(self, examples: List[Dict[str, Any]]) -> bool:
        """
        Update the query optimization prompt with brand-specific examples.
        """
        prompt_template = self._create_query_optimizer_prompt(examples)
        
        prompt_content = {
            "name": self.prompt_names['query_optimizer'],
            "prompt": prompt_template,
            "config": {
                "brand": self.brand_domain,
                "examples": examples,
                "updated_at": datetime.now().isoformat()
            },
            "labels": ["rag", "query-optimization", self.brand_key]
        }
        
        if self.langfuse:
            try:
                self.langfuse.create_prompt(
                    name=self.prompt_names['query_optimizer'],
                    prompt=prompt_content['prompt'],
                    config=prompt_content['config'],
                    labels=prompt_content['labels']
                )
                logger.info("✅ Updated query optimizer prompt")
                return True
            except Exception as e:
                logger.error(f"Failed to update query optimizer: {e}")
        
        self._save_local_prompt('query_optimizer', prompt_content)
        return True
    
    def get_filter_dictionary(self) -> Dict[str, Any]:
        """
        Retrieve the current filter dictionary.
        """
        # Check cache first
        cache_key = self.prompt_names['filter_dictionary']
        if cache_key in self.prompt_cache:
            cached = self.prompt_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self.cache_duration:
                return cached['data']
        
        # Try Langfuse
        if self.langfuse:
            try:
                prompt = self.langfuse.get_prompt(
                    name=self.prompt_names['filter_dictionary']
                )
                
                if prompt and hasattr(prompt, 'config'):
                    filter_dict = prompt.config.get('filters', {})
                    
                    # Cache result
                    self.prompt_cache[cache_key] = {
                        'data': filter_dict,
                        'timestamp': datetime.now()
                    }
                    
                    return filter_dict
                    
            except Exception as e:
                logger.warning(f"Failed to get filter dictionary from Langfuse: {e}")
        
        # Fallback to local
        return self._load_local_prompt('filter_dictionary').get('config', {}).get('filters', {})
    
    def get_search_enhancement_prompt(self) -> str:
        """
        Get the prompt for enhancing search queries.
        """
        cache_key = self.prompt_names['search_enhancer']
        
        # Check cache
        if cache_key in self.prompt_cache:
            cached = self.prompt_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self.cache_duration:
                return cached['data']
        
        # Try Langfuse
        if self.langfuse:
            try:
                prompt = self.langfuse.get_prompt(
                    name=self.prompt_names['search_enhancer']
                )
                
                if prompt:
                    prompt_text = prompt.prompt
                    
                    # Cache result
                    self.prompt_cache[cache_key] = {
                        'data': prompt_text,
                        'timestamp': datetime.now()
                    }
                    
                    return prompt_text
                    
            except Exception as e:
                logger.warning(f"Failed to get search enhancer prompt: {e}")
        
        # Fallback to default
        return self._get_default_search_enhancer_prompt()
    
    def _prepare_filter_dictionary(self, raw_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare filter dictionary for use in prompts.
        """
        prepared = {
            'categories': [],
            'attributes': {},
            'price_ranges': {},
            'availability': [],
            'special_filters': {}
        }
        
        for filter_name, filter_config in raw_filters.items():
            if filter_name.startswith('_'):
                continue
            
            filter_type = filter_config.get('type')
            
            if filter_name == 'category' and filter_type == 'categorical':
                prepared['categories'] = filter_config.get('values', [])
            
            elif filter_type == 'categorical':
                prepared['attributes'][filter_name] = filter_config.get('values', [])
            
            elif filter_type == 'numeric_range':
                prepared['price_ranges'][filter_name] = {
                    'min': filter_config.get('min'),
                    'max': filter_config.get('max')
                }
            
            elif filter_type == 'multi_select':
                prepared['special_filters'][filter_name] = filter_config.get('values', [])
        
        return prepared
    
    def _create_filter_dictionary_prompt(self, filter_dict: Dict[str, Any]) -> str:
        """
        Create a prompt template with filter information.
        """
        prompt = f"""You are a product search assistant for {self.brand_domain}.

Available Filters:

CATEGORIES:
{json.dumps(filter_dict.get('categories', []), indent=2)}

ATTRIBUTES:
{json.dumps(filter_dict.get('attributes', {}), indent=2)}

PRICE RANGES:
{json.dumps(filter_dict.get('price_ranges', {}), indent=2)}

SPECIAL FILTERS:
{json.dumps(filter_dict.get('special_filters', {}), indent=2)}

When extracting filters from user queries:
1. Map user terms to the exact filter values above
2. Handle synonyms and variations
3. Extract multiple filters when present
4. Infer reasonable defaults when ambiguous

Example mappings:
- "expensive" → price range: high end of scale
- "cheap"/"affordable" → price range: low end of scale
- "latest" → sort by: newest first
- Brand nicknames → official brand names
"""
        
        return prompt
    
    def _create_query_optimizer_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """
        Create query optimization prompt with examples.
        """
        example_text = "\n\n".join([
            f"User: {ex['user_query']}\nOptimized: {ex['optimized_query']}\nFilters: {json.dumps(ex.get('filters', {}))}"
            for ex in examples[:5]
        ])
        
        return f"""Optimize search queries for {self.brand_domain} products.

Your task:
1. Enhance the query for better search results
2. Extract applicable filters
3. Determine search strategy (exact, semantic, hybrid)
4. Add relevant context and synonyms

Examples:
{example_text}

Output format:
{{
    "optimized_query": "enhanced search query",
    "filters": {{"filter_name": "value"}},
    "search_strategy": "exact|semantic|hybrid",
    "confidence": 0.0-1.0
}}
"""
    
    def _get_default_search_enhancer_prompt(self) -> str:
        """
        Default prompt for search enhancement.
        """
        return f"""Enhance this product search query for {self.brand_domain}.

Add relevant terms, synonyms, and context to improve search results.
Consider:
- Product terminology specific to this brand
- Common variations and abbreviations  
- Related features and attributes
- User intent and context

Keep the enhanced query concise and focused.
"""
    
    def _save_local_prompt(self, prompt_type: str, content: Dict[str, Any]) -> None:
        """
        Save prompt locally as backup.
        """
        file_path = self.local_prompts_path / f"{prompt_type}.json"
        
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
        
        logger.info(f"💾 Saved {prompt_type} prompt locally")
    
    def _load_local_prompt(self, prompt_type: str) -> Dict[str, Any]:
        """
        Load prompt from local storage.
        """
        file_path = self.local_prompts_path / f"{prompt_type}.json"
        
        if file_path.exists():
            with open(file_path) as f:
                return json.load(f)
        
        return {}


class RAGConfigManager:
    """
    Manages RAG configuration and settings in Langfuse.
    """
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.manager = LangfuseRAGManager(brand_domain)
        
        # Configuration keys
        self.config_keys = {
            'search_weights': 'search_weight_config',
            'sync_settings': 'sync_settings',
            'quality_thresholds': 'quality_thresholds'
        }
    
    def update_search_weights(
        self,
        default_dense: float = 0.8,
        default_sparse: float = 0.2,
        query_patterns: Optional[Dict[str, Dict[str, float]]] = None
    ) -> bool:
        """
        Update search weight configuration.
        """
        config = {
            'default': {
                'dense': default_dense,
                'sparse': default_sparse
            },
            'patterns': query_patterns or {
                'exact_match': {'dense': 0.3, 'sparse': 0.7},
                'semantic': {'dense': 0.9, 'sparse': 0.1},
                'mixed': {'dense': 0.6, 'sparse': 0.4}
            }
        }
        
        # Save configuration
        return self._save_config('search_weights', config)
    
    def update_sync_settings(
        self,
        check_interval: int = 300,
        batch_size: int = 100,
        change_threshold: int = 10
    ) -> bool:
        """
        Update synchronization settings.
        """
        config = {
            'check_interval': check_interval,
            'batch_size': batch_size,
            'change_threshold': change_threshold,
            'updated_at': datetime.now().isoformat()
        }
        
        return self._save_config('sync_settings', config)
    
    def get_search_weights(self) -> Dict[str, Any]:
        """
        Get current search weight configuration.
        """
        return self._load_config('search_weights', {
            'default': {'dense': 0.8, 'sparse': 0.2}
        })
    
    def _save_config(self, key: str, config: Dict[str, Any]) -> bool:
        """
        Save configuration to Langfuse or local storage.
        """
        # For now, save locally
        config_path = Path(f"accounts/{self.brand_domain}/rag_config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config
        if config_path.exists():
            with open(config_path) as f:
                all_config = json.load(f)
        else:
            all_config = {}
        
        # Update specific key
        all_config[key] = config
        
        # Save back
        with open(config_path, 'w') as f:
            json.dump(all_config, f, indent=2)
        
        return True
    
    def _load_config(self, key: str, default: Any) -> Any:
        """
        Load configuration from storage.
        """
        config_path = Path(f"accounts/{self.brand_domain}/rag_config.json")
        
        if config_path.exists():
            with open(config_path) as f:
                all_config = json.load(f)
                return all_config.get(key, default)
        
        return default