"""
Unified Search Service

A minimal-dependency search service that works across both catalog-maintenance
and voice-assistant projects. Uses AccountManager for configuration and
SearchRAG interface for search operations.
"""

import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from liddy.account_manager import AccountManager, get_account_manager
from liddy.llm.base import LLMModelService
from liddy.llm.simple_factory import LLMFactory
from liddy.prompt_manager import PromptManager, ChatMessageDict
from liddy.model import BasicChatMessage, UserState
from liddy.search.user_context_enhancement import format_user_context_for_prompt
from .base import BaseRAG
from .pinecone import PineconeRAG

logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
    """Metrics for search operations."""
    query: str
    enhanced_query: str
    search_time: float
    total_results: int
    filters_applied: Dict[str, Any]
    search_backend: str
    enhancements_used: List[str]


class SearchService:
    """
    Unified search service with minimal dependencies.
    
    Features:
    - Query enhancement using AccountManager intelligence
    - Filter extraction from natural language
    - Multiple search backend support (via SearchRAG interface)
    - Metrics and performance tracking
    """
    
    # Cache for search instances per account
    _search_instances: Dict[str, BaseRAG] = {}
    
    @classmethod
    async def search_products(
        cls,
        query: str,
        account: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 50,
        top_n: int = 7,
        enable_enhancement: bool = True,
        enable_filter_extraction: bool = False,
        search_mode: str = "hybrid",
        user_state: Optional[UserState] = None,
        chat_ctx: Optional[List[BasicChatMessage]] = None,
        rerank: bool = True,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Search for products with intelligent enhancements.
        
        Args:
            query: Search query
            account: Account/brand domain
            filters: Pre-extracted filters (optional)
            top_k: Number of results to return
            enable_enhancement: Whether to enhance query with catalog intelligence
            enable_filter_extraction: Whether to extract filters from query
            search_mode: 'dense', 'sparse', or 'hybrid'
            user_state: Optional user state for personalization
            chat_ctx: Optional chat context for personalization
            **kwargs: Additional search parameters
            
        Returns:
            Tuple of (search results, metrics)
        """
        start_time = time.time()
        enhancements_used = []
        
        # Get account manager
        account_manager: AccountManager = await get_account_manager(account)
        
        # Original query for metrics
        original_query = query
        
        # Extract filters if enabled and not provided
        if enable_filter_extraction and not filters:
            filters = cls._extract_filters(query, account_manager)
            if filters:
                enhancements_used.append("filter_extraction")
        
        # Enhance query if enabled
        if enable_enhancement:
            query = await cls._enhance_query(
                query, 
                account_manager,
                user_state=user_state,
                chat_context=chat_ctx
            )
            if query != original_query:
                enhancements_used.append("query_enhancement")
        
        # Get or create search instance
        search_instance = await cls._get_search_instance(account_manager)
        
        # Perform search
        results = await search_instance.search_products(
            query=query,
            filters=filters,
            top_k=top_k,
            top_n=top_n,
            search_mode=search_mode,
            rerank=rerank,
            **kwargs
        )
        
        # Create metrics
        metrics = SearchMetrics(
            query=original_query,
            enhanced_query=query,
            search_time=time.time() - start_time,
            total_results=len(results),
            filters_applied=filters or {},
            search_backend=search_instance.__class__.__name__,
            enhancements_used=enhancements_used
        )
        
        logger.info(
            f"Search completed for {account}: {len(results)} results in {metrics.search_time:.3f}s "
            f"(enhancements: {', '.join(enhancements_used) or 'none'})"
        )
        
        return results, metrics
    
    @classmethod
    async def search_knowledge(
        cls,
        query: str,
        account: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        enable_enhancement: bool = True,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], SearchMetrics]:
        """
        Search knowledge base articles.
        
        Similar to search_products but for knowledge/information namespace.
        """
        start_time = time.time()
        enhancements_used = []
        
        # Get account manager
        account_manager = await get_account_manager(account)
        
        # Original query for metrics
        original_query = query
        
        # Enhance query if enabled
        if enable_enhancement:
            query = await cls._enhance_query(
                query,
                account_manager,
                context_type="knowledge"
            )
            if query != original_query:
                enhancements_used.append("query_enhancement")
        
        # Get or create search instance
        search_instance = await cls._get_search_instance(account_manager)
        
        # Perform search
        results = await search_instance.search_knowledge(
            query=query,
            filters=filters,
            top_k=top_k,
            **kwargs
        )
        
        # Create metrics
        metrics = SearchMetrics(
            query=original_query,
            enhanced_query=query,
            search_time=time.time() - start_time,
            total_results=len(results),
            filters_applied=filters or {},
            search_backend=search_instance.__class__.__name__,
            enhancements_used=enhancements_used
        )
        
        return results, metrics
    
    @classmethod
    async def _get_search_instance(cls, account_manager: AccountManager, prewarm_level: str = "standard") -> BaseRAG:
        """Get or create search instance based on account configuration.
        
        Args:
            account_manager: Account manager instance
            prewarm_level: Level of prewarming - "minimal", "standard", or "full"
        """
        account = account_manager.account
        
        if account not in cls._search_instances:
            search_config = account_manager.get_search_config()
            
            if search_config.backend == "pinecone":
                # Create Pinecone search instance
                cls._search_instances[account] = PineconeRAG(
                    brand_domain=account,
                    dense_index_name=search_config.dense_index,
                    sparse_index_name=search_config.sparse_index,
                    # **search_config.config
                )
                await cls._search_instances[account].initialize(prewarm_level=prewarm_level)
                logger.info(f"Created SearchPinecone instance for {account} with {prewarm_level} prewarming")
            else:
                raise ValueError(f"Unsupported search backend: {search_config.backend}")
        
        return cls._search_instances[account]
    
    @classmethod
    def _get_default_product_query_enhancement_prompt(cls) -> List[ChatMessageDict]:
        """Get the default product query enhancement prompt."""
        
        system_prompt = """
        You are an expert product search assistant specializing in {{brand_name}} products. Your role is to enhance customer queries by:
        1. Understanding intent from conversation context
        2. Adding relevant brand-specific terminology
        3. Identifying implicit product features or categories
        4. Expanding ambiguous terms with appropriate product vocabulary
        
        Always maintain the original search intent while making the query more precise for {{brand_name}}'s catalog.
        """
        
        user_prompt = """
        Enhance this product search query using the provided brand-specific instructions.
        
        # Original Query:
        {{query}}
        
        # Conversation History:
        {{chat_context}}
        
        # Enhancement Instructions:
        {{enhancement_context}}
        
        # Task:
        Apply the enhancement instructions above to improve the search query. Focus on:
        1. Using the brand-specific terminology mappings provided
        2. Expanding with relevant product categories when appropriate
        3. Adding technical specifications that match the query intent
        4. Including relevant features and attributes from the instructions
        5. Applying the use case associations to understand user needs
        
        # Response Format (JSON):
        {
            "enhanced_query": "[Enhanced search query following the instructions]",
            "reasoning": "[Which specific rules/mappings you applied]",
            "confidence_score": 0.8,
            "confidence_reasoning": "[Based on how well query matched the patterns]",
            "extracted_intent": {
                "category": "[Product category if identified from instructions]",
                "features": ["feature1", "feature2"],
                "price_range": "[If mentioned]",
                "use_case": "[Intended use if identified]"
            },
            "follow_up_questions": [
                "[Question to clarify specific needs]",
                "[Question about preferences or requirements]"
            ]
        }
        """
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @classmethod
    async def _enhance_query(
        cls,
        query: str,
        account_manager: AccountManager,
        user_state: Optional[UserState] = None,
        context_type: str = "product",
        chat_context: Optional[List[BasicChatMessage]] = None
    ) -> str:
        """Enhance query using catalog intelligence."""
        try:
            # Get the appropriate enhancement prompt based on context type
            enhancement_context = await account_manager.get_search_enhancement_prompt(context_type)
            
            if not enhancement_context:
                return query

            # Get the product query enhancement prompt
            prompt_manager = PromptManager.get_prompt_manager()
            prompt_key = f"liddy/search/query_enhancement/{context_type}"
            
            try:
                # Try to get prompt from Langfuse
                prompt_client = await prompt_manager.get_prompt(prompt_name=prompt_key, prompt_type="chat", prompt=cls._get_default_product_query_enhancement_prompt())
                prompts = prompt_client.prompt
                system_prompt = next((msg["content"] for msg in prompts if msg["role"] == "system"), None)
                user_prompt = next((msg["content"] for msg in prompts if msg["role"] == "user"), None)
                
                if not system_prompt or not user_prompt:
                    raise ValueError("Invalid prompt format")
                    
            except Exception as e:
                logger.warning(f"Could not retrieve prompt '{prompt_key}' from Langfuse: {e}. Using default.")
                # Fall back to default prompt
                default_prompts = cls._get_default_product_query_enhancement_prompt()
                system_prompt = default_prompts[0]["content"]
                user_prompt = default_prompts[1]["content"]
            
            # Prepare template variables
            brand_name = account_manager.account.split('.')[0].title()
            
            # Format chat context if provided
            chat_context_str = ""
            if chat_context:
                chat_context_str = "\n".join([
                    f"{msg.role}: {msg.content}" 
                    for msg in chat_context[-11:] if hasattr(msg, 'role') and msg.role in ['assistant', 'user']  # Last 11 messages for context
                ])
            
            if user_state:
                chat_context_str = format_user_context_for_prompt(user_state)
            
            template_vars = {
                "brand_name": brand_name,
                "brand_domain": account_manager.account,
                "query": query,
                "enhancement_context": enhancement_context[:2000],  # Limit context size
                "chat_context": chat_context_str or "No previous conversation",
            }
            
            # Replace template variables
            for var, value in template_vars.items():
                system_prompt = system_prompt.replace(f"{{{{{var}}}}}", str(value))
                user_prompt = user_prompt.replace(f"{{{{{var}}}}}", str(value))
            
            # Use appropriate model based on context type
            model_name = "openai/o3-mini" if context_type == "product" else "openai/o3-mini"
            llm: LLMModelService = LLMFactory.get_service(model_name=model_name)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            start_time = time.time()
            response = await llm.chat_completion(
                model=model_name,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent enhancements
                response_format={"type": "json_object"}
            )
            logger.info(f"Query enhancement time: {time.time() - start_time:.3f}s")
            # Parse JSON response
            import json
            result = json.loads(response.get('content', '{}'))
            
            # Log enhancement details
            logger.info(
                f"Query enhanced: '{query}' -> '{result.get('enhanced_query', query)}' "
                f"(confidence: {result.get('confidence_score', 0.0)})"
            )
            
            return result.get('enhanced_query', query)
            
            # # Simple enhancement: Add brand context to ambiguous queries
            # query_lower = query.lower()
            
            # # Check if query is very short or generic
            # if len(query.split()) <= 2 or any(
            #     generic in query_lower 
            #     for generic in ['best', 'good', 'nice', 'product', 'item']
            # ):
            #     # Extract key terms from context
            #     context_terms = []
            #     if 'specializes in' in enhancement_context:
            #         match = re.search(r'specializes in ([^.]+)', enhancement_context)
            #         if match:
            #             context_terms.append(match.group(1).strip())
                
            #     if context_terms and not any(term in query_lower for term in context_terms):
            #         # Enhance with context
            #         return f"{query} {' '.join(context_terms[:1])}"
            
            # return query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    @classmethod
    def _extract_filters(
        cls,
        query: str,
        account_manager
    ) -> Dict[str, Any]:
        """Extract filters from natural language query."""
        filters = {}
        query_lower = query.lower()
        
        # Price extraction
        price_patterns = [
            (r'under \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'max'),
            (r'below \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'max'),
            (r'less than \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'max'),
            (r'over \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'min'),
            (r'above \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'min'),
            (r'more than \$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'min'),
            (r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?) ?- ?\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', 'range')
        ]
        
        for pattern, price_type in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if price_type == 'range':
                    min_price = float(match.group(1).replace(',', ''))
                    max_price = float(match.group(2).replace(',', ''))
                    filters['price'] = {'$gte': min_price, '$lte': max_price}
                elif price_type == 'max':
                    filters['price'] = {'$lte': float(match.group(1).replace(',', ''))}
                else:  # min
                    filters['price'] = {'$gte': float(match.group(1).replace(',', ''))}
                break
        
        # Category extraction based on account
        brand_config = account_manager.get_brand_config()
        if account_manager.account == "specialized.com":
            categories = {
                'mountain': 'Mountain',
                'road': 'Road',
                'gravel': 'Gravel',
                'electric': 'E-Bike',
                'kids': 'Kids'
            }
        elif account_manager.account == "sundayriley.com":
            categories = {
                'serum': 'Serum',
                'moisturizer': 'Moisturizer',
                'cleanser': 'Cleanser',
                'mask': 'Mask',
                'oil': 'Oil'
            }
        else:
            categories = {}
        
        for keyword, category in categories.items():
            if keyword in query_lower:
                filters['category'] = category
                break
        
        # Variant extraction
        variants = account_manager.get_variant_types()
        for variant in variants:
            variant_name = variant['name'].lower()
            if variant_name in query_lower:
                # Extract variant value (simplified)
                words = query_lower.split()
                for i, word in enumerate(words):
                    if word == variant_name and i + 1 < len(words):
                        filters[f"variant_{variant_name}"] = words[i + 1]
                        break
        
        return filters
    
    @classmethod
    def determine_search_weights(
        cls,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """
        Determine dense vs sparse weights for hybrid search.
        
        Args:
            query: Search query
            filters: Extracted filters
            
        Returns:
            Tuple of (dense_weight, sparse_weight)
        """
        # If filters are present, slightly favor sparse
        if filters:
            return 0.6, 0.4
        
        # Otherwise use query characteristics
        query_lower = query.lower()
        
        # Exact match indicators
        if any(indicator in query_lower for indicator in ['exact', 'model', 'sku']):
            return 0.3, 0.7
        
        # Semantic indicators
        if any(indicator in query_lower for indicator in ['comfortable', 'quality', 'best']):
            return 0.8, 0.2
        
        # Default balanced
        return 0.7, 0.3


# Convenience functions for backward compatibility
async def search_products(query: str, account: str, **kwargs) -> List[Dict[str, Any]]:
    """Simple product search without metrics."""
    results, _ = await SearchService.search_products(query, account, **kwargs)
    return results


async def search_knowledge(query: str, account: str, **kwargs) -> List[Dict[str, Any]]:
    """Simple knowledge search without metrics."""
    results, _ = await SearchService.search_knowledge(query, account, **kwargs)
    return results