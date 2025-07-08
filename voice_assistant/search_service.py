"""
Search utilities for handling query enhancement, search execution, and speech timing.
Enhanced with intelligent query optimization and brand-specific filter extraction.
"""

import asyncio
import logging
import json
import time
from typing import Any, Dict, List, Callable, Optional, Union, Tuple, Set
from livekit.agents import llm

from spence.llm_service import LlmService
from spence.product import Product
from spence.rag import PineconeRAG
from spence.model import UserState
from spence.account_manager import get_account_manager

# Type imports for return types
try:
    from .search_performance_tracker import SearchMetrics
except ImportError:
    SearchMetrics = Dict[str, Any]

# Import enhanced components from catalog-maintenance
try:
    from liddy_intelligence.agents.query_optimization_agent import QueryOptimizationAgent
    from liddy_intelligence.agents.catalog_filter_analyzer import CatalogFilterAnalyzer
    logger.info("Advanced RAG components loaded successfully")
except ImportError as e:
    logger.warning(f"Advanced RAG components not available: {e}")
    QueryOptimizationAgent = None
    CatalogFilterAnalyzer = None

# Langfuse for prompt management
try:
    from langfuse import Langfuse
    _langfuse_client = Langfuse()
    logger.info("Langfuse client initialized")
except Exception as e:
    logger.warning(f"Langfuse not available: {e}")
    _langfuse_client = None

logger = logging.getLogger(__name__)

class SearchService:
    """Service class for handling various types of searches and query enhancements"""
    
    # Cache for query optimizers per account
    _query_optimizers: Dict[str, QueryOptimizationAgent] = {}
    
    # Cache for catalog labels - stateful session-based caching
    _catalog_labels_cache: Dict[str, Dict[str, Set[str]]] = {}
    _catalog_labels_loaded: Dict[str, bool] = {}
    
    @staticmethod
    async def get_query_optimizer(account: str) -> Optional[QueryOptimizationAgent]:
        """Get or create a query optimizer for an account."""
        if not QueryOptimizationAgent:
            return None
            
        if account not in SearchService._query_optimizers:
            try:
                optimizer = QueryOptimizationAgent(account)
                SearchService._query_optimizers[account] = optimizer
                logger.info(f"Created query optimizer for {account}")
            except Exception as e:
                logger.error(f"Failed to create query optimizer for {account}: {e}")
                return None
                
        return SearchService._query_optimizers[account]

    @staticmethod
    async def enhance_query(
        query: str, 
        user_state: UserState,
        chat_ctx: llm.ChatContext, 
        system_prompt: str, 
        model_name: str = "gpt-4.1"
    ) -> str:
        """
        Enhance a search query using an LLM based on conversation context.
        
        Args:
            query: The original search query
            chat_ctx: Conversation context
            system_prompt: System prompt for the LLM
            model_name: LLM model to use
            
        Returns:
            Enhanced query string or original query if enhancement fails
        """
        if not query:
            return query
            
        try:
            # Create a new context for the query enhancement
            enhancement_ctx = llm.ChatContext([])
            
            # Add system prompt
            if system_prompt:
                enhancement_ctx.add_message(
                    role="system",
                    content=[system_prompt]
                )
                
            # Prepare the enhancement prompt
            enhancement_prompt = (
                f"Analyze this query and enhance it based on the conversation context. "
                f"Return only the enhanced query text with no explanations.\n\n"
                f"Query: {query}\n\nConversation:\n\"\"\""
            )
            
            # Extract relevant conversation history - use more context for better understanding
            messages = chat_ctx.copy().items[-50:] if len(chat_ctx.copy().items) > 50 else chat_ctx.copy().items
            for message in messages:
                if hasattr(message, "role") and message.role != "system":
                    content = message.content
                    if isinstance(content, list):
                        content = " ".join(content)
                    enhancement_prompt += f"{message.role}: {content}\n"
            
            enhancement_prompt += "\"\"\""
            
            # Add the prompt to the context
            enhancement_ctx.add_message(
                role="user",
                content=[enhancement_prompt]
            )
            
            # Get LLM service and generate enhanced query
            llm_model = LlmService.fetch_model_service_from_model(model_name=model_name, account=user_state.account, user=user_state.user_id, model_use="query_enhancement")
            
            enhanced_query = await LlmService.chat_wrapper(
                llm_service=llm_model,
                chat_ctx=enhancement_ctx,
            )
            
            logger.debug(f"Original query: {query}")
            logger.debug(f"Enhanced query: {enhanced_query}")
            
            # Return enhanced query if valid, otherwise return original
            return enhanced_query if enhanced_query else query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query  # Fall back to original query on error

    @staticmethod
    async def perform_search(
        query: str,
        search_function: Callable,
        query_enhancer: Optional[Callable] = None,
        search_params: Dict[str, Any] = None,
        user_state: UserState = None,
    ) -> Any:
        """
        Generic search function that handles query enhancement and search execution.
        
        Args:
            query: Original search query
            search_function: Async function that performs the actual search
            query_enhancer: Optional function to enhance the query
            search_params: Parameters to pass to search_function
            
        Returns:
            Search results from the search_function
        """
        if not query:
            return []
            
        search_params = search_params or {}
        
        # 1. Enhance query if enhancer is provided
        enhanced_query = query
        if query_enhancer:
            try:
                enhanced_query = await query_enhancer(query, user_state, **search_params.get("enhancer_params", {}))
            except Exception as e:
                logger.error(f"Error enhancing query: {e}")
        
        # 2. Run the search function with the enhanced query
        try:
            return await search_function(enhanced_query, **search_params)
        except Exception as e:
            logger.error(f"Error in search function: {e}")
            return []

    @staticmethod
    async def load_product_catalog_research(account: str) -> Dict[str, Any]:
        """
        Load ProductCatalogResearcher output for enhanced query understanding.
        
        Args:
            account: Account/brand identifier
            
        Returns:
            Dictionary containing product catalog research
        """
        try:
            from liddy.storage import get_account_storage_provider
            
            storage = get_account_storage_provider()
            
            # Load product catalog research
            content = await storage.load_content(
                brand_domain=account,
                content_type="research/product_catalog"
            )
            
            if content:
                research_data = json.loads(content)
                
                # Extract the key sections for query enhancement
                catalog_research = {
                    'descriptor_context': '',
                    'search_context': '',
                    'brand_insights': ''
                }
                
                # Parse the research content
                if 'content' in research_data:
                    research_content = research_data['content']
                    
                    # Extract descriptor generation context
                    if '## Product Descriptor Generation Context' in research_content:
                        desc_start = research_content.find('## Product Descriptor Generation Context')
                        desc_end = research_content.find('## Product Knowledge Search Context', desc_start)
                        if desc_end > desc_start:
                            catalog_research['descriptor_context'] = research_content[desc_start:desc_end].strip()
                    
                    # Extract search context
                    if '## Product Knowledge Search Context' in research_content:
                        search_start = research_content.find('## Product Knowledge Search Context')
                        search_end = research_content.find('## ', search_start + 30) if '## ' in research_content[search_start + 30:] else len(research_content)
                        catalog_research['search_context'] = research_content[search_start:search_end].strip()
                    
                    # Extract brand insights summary
                    if '### Brand Overview' in research_content:
                        brand_start = research_content.find('### Brand Overview')
                        brand_end = research_content.find('### ', brand_start + 20) if '### ' in research_content[brand_start + 20:] else brand_start + 1000
                        catalog_research['brand_insights'] = research_content[brand_start:brand_end].strip()
                
                logger.info(f"✅ Loaded product catalog research for {account}")
                return catalog_research
            else:
                logger.warning(f"No product catalog research found for {account}")
                return {}
                
        except Exception as e:
            logger.warning(f"Could not load product catalog research: {e}")
            return {}
    
    @staticmethod
    async def enhance_product_query(query: str, user_state: UserState, chat_ctx: llm.ChatContext, product_knowledge: str = "") -> str:
        """
        Enhance a product search query with domain-specific knowledge.
        
        Args:
            query: Original search query
            chat_ctx: Conversation context
            product_knowledge: Optional product domain knowledge
            
        Returns:
            Enhanced query string
        """
        # Try to load product catalog research if not provided
        if not product_knowledge and user_state and hasattr(user_state, 'account'):
            catalog_research = await SearchService.load_product_catalog_research(user_state.account)
            if catalog_research and catalog_research.get('search_context'):
                product_knowledge = catalog_research['search_context']
        
        system_prompt = (
            f"You are a product search assistant who is an expert at crafting product search queries. "
            f"The following is a knowledge base that will assist you in curating the user's search query. "
            f"Use this knowledge base to help you curate the perfect search query based on the initial query "
            f"as well as the context of the conversation.\n\n{product_knowledge or 'No specific product knowledge provided.'}"
        )
        return await SearchService.enhance_query(query, user_state, chat_ctx, system_prompt)
    
    @staticmethod
    def extract_user_preferences(user_state: UserState) -> Dict[str, Any]:
        """
        Extract search-relevant preferences from UserState.
        
        Args:
            user_state: Current user state
            
        Returns:
            Dictionary of user preferences for search optimization
        """
        preferences = {
            'price_range': None,
            'recent_searches': [],
            'category_interests': {},
            'search_style': 'balanced',
            'user_context': []
        }
        
        if not user_state:
            return preferences
        
        # Extract from conversation history if available
        if hasattr(user_state, 'conversation_exit_state') and user_state.conversation_exit_state:
            exit_state = user_state.conversation_exit_state
            if exit_state.transcript_summary:
                preferences['user_context'].append(f"Previous conversation: {exit_state.transcript_summary}")
        
        # Extract from sentiment analysis if available
        if hasattr(user_state, 'sentiment_analysis') and user_state.sentiment_analysis:
            sentiment = user_state.sentiment_analysis
            if hasattr(sentiment, 'details') and sentiment.details:
                preferences['user_context'].append(f"User sentiment details: {sentiment.details}")
        
        # Extract from communication directive if available
        if hasattr(user_state, 'communication_directive') and user_state.communication_directive:
            directive = user_state.communication_directive
            if directive.formality and directive.formality.score > 0.7:
                preferences['search_style'] = 'formal'
                preferences['user_context'].append("User prefers formal communication")
        
        return preferences
    
    @staticmethod
    def determine_search_weights(
        query: str,
        user_state: Optional[UserState] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Determine dense vs sparse weights based on query characteristics.
        
        Args:
            query: Search query
            user_state: Optional user state for preference-based weighting
            filters: Optional extracted filters
            
        Returns:
            Tuple of (dense_weight, sparse_weight) or (None, None) for auto
        """
        # Extract user preferences if available
        user_prefs = SearchService.extract_user_preferences(user_state) if user_state else {}
        
        # Check for exact match indicators
        exact_match_indicators = [
            'exact', 'specifically', 'model number', 'sku', '"'
        ]
        
        query_lower = query.lower()
        is_exact_search = any(indicator in query_lower for indicator in exact_match_indicators)
        
        # Check if searching for specific model/series
        is_model_search = any(indicator in query_lower for indicator in [
            'model', 'series', 'collection', 'line'
        ])
        
        # Determine weights
        if is_exact_search or is_model_search:
            # Favor sparse embeddings for exact/model searches
            return 0.3, 0.7
        elif user_prefs.get('search_style') == 'formal':
            # Slightly favor dense for formal users (more semantic)
            return 0.7, 0.3
        else:
            # Let the system auto-determine
            return None, None
    
    @staticmethod
    async def load_catalog_labels(account: str, force_reload: bool = False) -> Dict[str, Set[str]]:
        """
        Load all product labels from the catalog for semantic enhancement.
        
        Uses session-based caching for performance optimization since this is a
        stateful service with short session duration.
        
        Args:
            account: Account identifier
            force_reload: Force reload from database (bypasses cache)
            
        Returns:
            Dictionary of category -> set of labels
        """
        # Check cache first (unless force reload)
        if not force_reload and account in SearchService._catalog_labels_cache:
            logger.debug(f"Using cached catalog labels for {account}")
            return SearchService._catalog_labels_cache[account]
        
        try:
            from ..model import Product
            from ..account_manager import get_account_manager
            
            logger.info(f"Loading catalog labels for {account} from database...")
            
            # Get account manager and load products
            account_manager = await get_account_manager(account)
            products = await Product.get_products_async(account=account)
            
            # Collect all labels by category
            catalog_labels = {}
            for product in products:
                if hasattr(product, 'product_labels') and product.product_labels:
                    for category, labels in product.product_labels.items():
                        if category not in catalog_labels:
                            catalog_labels[category] = set()
                        if isinstance(labels, list):
                            catalog_labels[category].update(labels)
                        elif isinstance(labels, str):
                            catalog_labels[category].add(labels)
            
            # Cache the results for this session
            SearchService._catalog_labels_cache[account] = catalog_labels
            SearchService._catalog_labels_loaded[account] = True
            
            logger.info(f"Loaded and cached catalog labels for {account}: {len(catalog_labels)} categories")
            logger.debug(f"Categories: {list(catalog_labels.keys())}")
            
            return catalog_labels
            
        except Exception as e:
            logger.warning(f"Could not load catalog labels for {account}: {e}")
            return {}

    @staticmethod
    async def preload_catalog_labels(account: str) -> bool:
        """
        Preload catalog labels for an account to optimize subsequent searches.
        
        This should be called during Assistant initialization to ensure labels
        are ready for system prompt generation and semantic search enhancement.
        
        Args:
            account: Account identifier
            
        Returns:
            True if successfully preloaded, False otherwise
        """
        try:
            start_time = time.time()
            labels = await SearchService.load_catalog_labels(account)
            load_time = time.time() - start_time
            
            if labels:
                logger.info(f"🚀 Preloaded catalog labels for {account} in {load_time:.3f}s")
                logger.info(f"   📋 Available categories: {', '.join(labels.keys())}")
                return True
            else:
                logger.warning(f"⚠️  No catalog labels found for {account}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to preload catalog labels for {account}: {e}")
            return False

    @staticmethod
    def get_cached_catalog_labels(account: str) -> Optional[Dict[str, Set[str]]]:
        """
        Get cached catalog labels without triggering a database load.
        
        Args:
            account: Account identifier
            
        Returns:
            Cached labels or None if not loaded
        """
        return SearchService._catalog_labels_cache.get(account)

    @staticmethod
    def is_catalog_labels_loaded(account: str) -> bool:
        """Check if catalog labels are loaded for an account."""
        return SearchService._catalog_labels_loaded.get(account, False)

    @staticmethod
    def format_catalog_labels_for_system_prompt(account: str) -> str:
        """
        Format cached catalog labels for inclusion in AI assistant system prompts.
        
        This creates a structured overview of available product categories and
        labels that the AI can use for intelligent product recommendations.
        
        Args:
            account: Account identifier
            
        Returns:
            Formatted string for system prompt inclusion
        """
        labels = SearchService.get_cached_catalog_labels(account)
        
        if not labels:
            return "# PRODUCT CATEGORIES\nNo product categories loaded yet."
        
        prompt_section = "# PRODUCT CATEGORIES AND LABELS\n\n"
        prompt_section += "Use these categories to understand customer intent and make precise recommendations:\n\n"
        
        for category, category_labels in sorted(labels.items()):
            if category_labels:  # Only include categories with labels
                formatted_category = category.replace('_', ' ').title()
                sorted_labels = sorted(list(category_labels))
                
                prompt_section += f"**{formatted_category}**: {', '.join(sorted_labels[:10])}"
                if len(sorted_labels) > 10:
                    prompt_section += f" (+{len(sorted_labels) - 10} more)"
                prompt_section += "\n"
        
        prompt_section += "\n"
        prompt_section += "**Usage Guidelines:**\n"
        prompt_section += "- Match customer language to appropriate categories (e.g., 'beginner' → skill_level)\n"
        prompt_section += "- Use multiple categories for precise filtering (e.g., skin_types + benefits)\n"
        prompt_section += "- Consider category combinations for better recommendations\n"
        prompt_section += "- Explain category relevance when making suggestions\n\n"
        
        return prompt_section

    @staticmethod
    def get_category_summary_stats(account: str) -> Dict[str, Any]:
        """
        Get summary statistics about loaded catalog labels.
        
        Useful for system monitoring and prompt optimization.
        
        Args:
            account: Account identifier
            
        Returns:
            Dictionary with statistics about loaded categories
        """
        labels = SearchService.get_cached_catalog_labels(account)
        
        if not labels:
            return {"loaded": False, "categories": 0, "total_labels": 0}
        
        stats = {
            "loaded": True,
            "categories": len(labels),
            "total_labels": sum(len(category_labels) for category_labels in labels.values()),
            "category_details": {}
        }
        
        for category, category_labels in labels.items():
            stats["category_details"][category] = {
                "label_count": len(category_labels),
                "sample_labels": list(sorted(category_labels))[:5]
            }
        
        return stats

    @staticmethod
    def clear_catalog_labels_cache(account: Optional[str] = None):
        """
        Clear catalog labels cache for memory management.
        
        Args:
            account: Specific account to clear, or None to clear all
        """
        if account:
            if account in SearchService._catalog_labels_cache:
                del SearchService._catalog_labels_cache[account]
            if account in SearchService._catalog_labels_loaded:
                del SearchService._catalog_labels_loaded[account]
            logger.info(f"🧹 Cleared catalog labels cache for {account}")
        else:
            SearchService._catalog_labels_cache.clear()
            SearchService._catalog_labels_loaded.clear()
            logger.info("🧹 Cleared all catalog labels cache")

    @staticmethod
    def extract_semantic_filters(query: str, catalog_labels: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """Extract semantic filters from query using catalog label intelligence."""
        semantic_filters = {}
        query_lower = query.lower()
        
        # Common intent mappings for different domains
        intent_patterns = {
            "beginner": ["skill_level", "experience_level", "difficulty"],
            "expert": ["skill_level", "experience_level", "difficulty"], 
            "professional": ["skill_level", "target_user", "user_type"],
            "racing": ["use_cases", "intended_use", "activity_type"],
            "trails": ["riding_terrain", "terrain_type", "use_cases"],
            "mountain": ["riding_terrain", "terrain_type", "bike_type"],
            "road": ["riding_terrain", "terrain_type", "bike_type"],
            "sensitive": ["skin_types", "skin_concerns", "user_type"],
            "dry": ["skin_types", "skin_concerns", "benefits"],
            "oily": ["skin_types", "skin_concerns", "benefits"],
            "gaming": ["use_cases", "intended_use", "activity_type"],
            "streaming": ["use_cases", "intended_use", "activity_type"],
            "lightweight": ["performance_traits", "key_features", "characteristics"],
            "durable": ["performance_traits", "key_features", "characteristics"]
        }
        
        # Look for intent patterns in query
        for intent, possible_categories in intent_patterns.items():
            if intent in query_lower:
                for category in possible_categories:
                    if category in catalog_labels:
                        # Find matching labels in this category
                        matching_labels = []
                        category_labels = catalog_labels[category]
                        
                        # Direct match
                        for label in category_labels:
                            if intent in label.lower():
                                matching_labels.append(label)
                        
                        # Semantic match (e.g., "beginner" matches "entry_level")
                        if intent == "beginner" and not matching_labels:
                            for label in category_labels:
                                if any(term in label.lower() for term in ["entry", "basic", "intro", "starter"]):
                                    matching_labels.append(label)
                        elif intent == "expert" and not matching_labels:
                            for label in category_labels:
                                if any(term in label.lower() for term in ["advanced", "pro", "expert", "elite"]):
                                    matching_labels.append(label)
                        
                        if matching_labels:
                            semantic_filters[category] = matching_labels
                            break  # Found a match, move to next intent
        
        return semantic_filters

    @staticmethod
    async def enhance_product_query_with_filters(
        query: str, 
        user_state: UserState, 
        chat_ctx: llm.ChatContext, 
        account: str,
        product_knowledge: str = ""
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance product query with intelligent filter extraction and user context.
        
        Args:
            query: Original search query
            user_state: User state with context
            chat_ctx: Conversation context
            account: Account domain
            product_knowledge: Optional product domain knowledge
            
        Returns:
            Tuple of (enhanced_query, extracted_filters)
        """
        try:
            # Get query optimizer for this account
            optimizer = await SearchService.get_query_optimizer(account)
            
            if not optimizer:
                # Fallback to simple enhancement
                enhanced_query = await SearchService.enhance_product_query(
                    query, user_state, chat_ctx, product_knowledge
                )
                return enhanced_query, {}
            
            # Extract user preferences for context enrichment
            user_prefs = SearchService.extract_user_preferences(user_state)
            
            # Build comprehensive context from chat history
            context = {
                "recent_messages": [],
                "expressed_interests": [],
                "user_preferences": user_prefs,
                "conversation_stage": "unknown"
            }
            
            # Add conversation stage detection
            if user_state and hasattr(user_state, 'conversation_exit_state') and user_state.conversation_exit_state:
                if user_state.conversation_exit_state.last_interaction_time:
                    time_diff = time.time() - user_state.conversation_exit_state.last_interaction_time
                    if time_diff < 3600:  # Less than 1 hour
                        context["conversation_stage"] = "resumed_recent"
                    elif time_diff < 86400:  # Less than 1 day
                        context["conversation_stage"] = "resumed_same_day"
                    else:
                        context["conversation_stage"] = "new_conversation"
            
            # Extract conversation context - look further back for better understanding
            messages = chat_ctx.items[-30:] if len(chat_ctx.items) > 30 else chat_ctx.items
            for msg in messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    if msg.role == "user":
                        content = msg.content if isinstance(msg.content, str) else " ".join(msg.content)
                        context["recent_messages"].append(content)
                    elif msg.role == "assistant":
                        # Extract mentioned interests/preferences
                        content = msg.content if isinstance(msg.content, str) else " ".join(msg.content)
                        if any(term in content.lower() for term in ["looking for", "interested in", "need", "want", "recommend"]):
                            context["expressed_interests"].append(content)
            
            # Optimize query with filter extraction
            start_time = time.time()
            optimization_result = await optimizer.optimize_product_query(
                original_query=query,
                context=context,
                user_state=user_state.__dict__ if user_state else None
            )
            
            optimization_time = time.time() - start_time
            
            enhanced_query = optimization_result.get("optimized_query", query)
            extracted_filters = optimization_result.get("filters", {})
            
            # ENHANCEMENT: Add semantic label intelligence using cached labels
            catalog_labels = SearchService.get_cached_catalog_labels(account)
            if not catalog_labels:
                # Load if not cached (fallback, but should be preloaded)
                logger.debug(f"Catalog labels not cached for {account}, loading...")
                catalog_labels = await SearchService.load_catalog_labels(account)
            
            if catalog_labels:
                semantic_filters = SearchService.extract_semantic_filters(query, catalog_labels)
                if semantic_filters:
                    extracted_filters["semantic_labels"] = semantic_filters
                    logger.debug(f"Added semantic filters from cached product labels: {semantic_filters}")
            
            # Log performance and results
            logger.info(f"Query optimization completed in {optimization_time:.3f}s")
            logger.debug(f"Original query: {query}")
            logger.debug(f"Enhanced query: {enhanced_query}")
            logger.debug(f"Extracted filters: {extracted_filters}")
            
            # Track metrics if Langfuse is available
            if _langfuse_client:
                try:
                    _langfuse_client.generation(
                        name="query_optimization",
                        input={"query": query, "context": context},
                        output={"enhanced_query": enhanced_query, "filters": extracted_filters},
                        metadata={
                            "account": account,
                            "optimization_time": optimization_time,
                            "filter_count": len(extracted_filters)
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to track Langfuse metrics: {e}")
            
            return enhanced_query, extracted_filters
            
        except Exception as e:
            logger.error(f"Error in enhanced query optimization: {e}")
            # Fallback to simple enhancement
            enhanced_query = await SearchService.enhance_product_query(
                query, user_state, chat_ctx, product_knowledge
            )
            return enhanced_query, {}

    @staticmethod
    async def enhance_knowledge_query(query: str, user_state: UserState, chat_ctx: llm.ChatContext, knowledge_base: str = "") -> str:
        """
        Enhance a knowledge search query with domain-specific context.
        
        Args:
            query: Original search query
            chat_ctx: Conversation context
            knowledge_base: Optional knowledge base context
            
        Returns:
            Enhanced query string
        """
        system_prompt = (
            f"You are a knowledge retrieval specialist. Your job is to enhance search queries to find "
            f"relevant information in documentation, guides, and other informational content. "
            f"Use this knowledge base to improve the query precision:\n\n{knowledge_base or 'No specific knowledge base provided.'}"
        )
        return await SearchService.enhance_query(query, user_state, chat_ctx, system_prompt)

    @staticmethod
    async def search_products_rag(
        query: str, 
        account: str = None, 
        top_k: int = 35, 
        top_n: int = 10, 
        min_score: float = 0.15,
        min_n: int = 0,
        **kwargs
    ) -> List[Dict]:
        """
        Search for products using RAG.
        
        Args:
            query: Search query
            account: Account identifier
            top_k: Maximum number of documents to return
            top_n: Number of top results for re-ranking
            min_score: Minimum similarity score threshold
            min_n: Minimum number of results to consider
            
        Returns:
            List of product search results
        """
        try:
            return await PineconeRAG.rag_query_pinecone(
                query=query, 
                account=account,
                use_ranked=False, 
                namespaces=["products"],
                top_k=kwargs.get('top_k', top_k),
                top_n=kwargs.get('top_n', top_n),
                min_score=kwargs.get('min_score', min_score),
                min_n=kwargs.get('min_n', min_n)
            )
        except Exception as e:
            logger.error(f"Error in search_products_rag: {e}")
            return []
    
    @staticmethod
    async def search_products_with_context(
        query: str,
        user_state: Optional[UserState] = None,
        chat_ctx: Optional[llm.ChatContext] = None,
        account: str = None,
        product_knowledge: str = "",
        use_hybrid: bool = True,
        **kwargs
    ) -> Tuple[List[Dict], SearchMetrics]:
        """
        Unified product search with user context and performance tracking.
        
        Args:
            query: Search query
            user_state: Optional user state for context
            chat_ctx: Optional conversation context
            account: Account/brand identifier
            product_knowledge: Optional product knowledge base
            use_hybrid: Whether to use hybrid search (if available)
            **kwargs: Additional search parameters
            
        Returns:
            Tuple of (search results, performance metrics)
        """
        from .search_performance_tracker import track_search_performance
        start_time = time.time()
        
        # Enhance query with user context if available
        if chat_ctx and user_state:
            enhanced_query, filters = await SearchService.enhance_product_query_with_filters(
                query=query,
                user_state=user_state,
                chat_ctx=chat_ctx,
                account=account,
                product_knowledge=product_knowledge
            )
        else:
            enhanced_query = query
            filters = {}
        
        # Determine search weights if using hybrid
        dense_weight, sparse_weight = None, None
        if use_hybrid:
            dense_weight, sparse_weight = SearchService.determine_search_weights(
                query=enhanced_query,
                user_state=user_state,
                filters=filters
            )
        
        # Perform search
        if use_hybrid:
            # Try new hybrid search with separate indexes
            try:
                results = await SearchService.search_products_hybrid_separate_indexes(
                    query=enhanced_query,
                    account=account,
                    filters=filters,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    top_k=kwargs.get('top_k', 35),
                    top_n=kwargs.get('top_n', 10),
                    min_score=kwargs.get('min_score', 0.15),
                    rerank=kwargs.get('rerank', True)
                )
                search_type = 'hybrid_separate_indexes'
            except Exception as e:
                logger.warning(f"Hybrid search with separate indexes failed, falling back to RAG: {e}")
                results = await SearchService.search_products_rag_with_filters(
                    query=enhanced_query,
                    filters=filters,
                    account=account,
                    **kwargs
                )
                search_type = 'rag'
        else:
            # Use standard RAG search
            results = await SearchService.search_products_rag_with_filters(
                query=enhanced_query,
                filters=filters,
                account=account,
                **kwargs
            )
            search_type = 'rag'
        
        # Track performance
        metrics = track_search_performance(
            query=query,
            enhanced_query=enhanced_query,
            search_type=search_type,
            results=results,
            start_time=start_time,
            user_id=getattr(user_state, 'user_id', None) if user_state else None,
            account=account,
            filters=filters,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        
        return results, metrics
    
    @staticmethod
    async def search_products_hybrid_separate_indexes(
        query: str,
        account: str,
        filters: Dict[str, Any] = None,
        dense_weight: float = None,
        sparse_weight: float = None,
        top_k: int = 35,
        top_n: int = 10,
        min_score: float = 0.15,
        rerank: bool = True,
        **kwargs
    ) -> List[Dict]:
        """
        Hybrid search using separate dense and sparse indexes with server-side embeddings.
        
        This method uses the new Pinecone architecture with:
        - Separate indexes for dense (llama-text-embed-v2) and sparse (pinecone-sparse-english-v0)
        - Server-side embeddings for both indexes
        - Multi-stage reranking capability
        
        Args:
            query: Search query
            account: Account/brand identifier
            filters: Metadata filters to apply
            dense_weight: Weight for dense results (0-1)
            sparse_weight: Weight for sparse results (0-1)
            top_k: Maximum documents to retrieve from each index
            top_n: Number of results after reranking
            min_score: Minimum relevance score
            rerank: Whether to apply neural reranking
            
        Returns:
            List of search results with hybrid scoring
        """
        try:
            # Import the unified SearchPinecone implementation
            from liddy.search.pinecone import get_search_pinecone
            
            # Get or create SearchPinecone instance for this account
            search_pinecone = await get_search_pinecone(
                brand_domain=account,
                namespace="products"
            )
            
            # Convert filters to Pinecone format
            pinecone_filters = SearchService._convert_filters_to_pinecone(filters) if filters else None
            
            # Perform hybrid search using SearchPinecone
            results = await search_pinecone.search_products(
                query=query,
                filters=pinecone_filters,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                search_mode="hybrid",
                rerank=rerank
            )
            
            # Add matched filters information
            for result in results:
                result['matched_filters'] = SearchService._get_matched_filters(
                    result.get('metadata', {}), 
                    filters
                ) if filters else []
            
            logger.info(f"Hybrid search with SearchRAG found {len(results)} results")
            return results
            
        except ImportError as e:
            logger.warning(f"SearchRAG not available: {e}")
            # Fallback to standard RAG search
            return await SearchService.search_products_rag_with_filters(
                query=query,
                filters=filters,
                account=account,
                top_k=top_k,
                top_n=top_n,
                min_score=min_score,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to standard RAG search
            return await SearchService.search_products_rag_with_filters(
                query=query,
                filters=filters,
                account=account,
                top_k=top_k,
                top_n=top_n,
                min_score=min_score,
                **kwargs
            )
    
    @staticmethod
    async def search_products_rag_with_filters(
        query: str,
        filters: Dict[str, Any],
        account: str = None,
        top_k: int = 35,
        top_n: int = 10,
        min_score: float = 0.15,
        min_n: int = 3,
        **kwargs
    ) -> List[Dict]:
        """
        Enhanced RAG search with intelligent filter application.
        
        Args:
            query: Enhanced search query
            filters: Extracted filters from query optimization
            account: Account identifier
            top_k: Maximum documents to retrieve
            top_n: Number of results after reranking
            min_score: Minimum relevance score
            min_n: Minimum results to return
            
        Returns:
            List of product search results with metadata
        """
        try:
            # Get account manager for RAG configuration
            account_manager = await get_account_manager(account)
            rag_index = account_manager.get_rag_details()
            
            if not rag_index:
                logger.warning(f"No RAG index configured for {account}")
                return []
            
            # Initialize RAG client with standard embedding model
            rag = PineconeRAG(
                account=account,
                index_name=rag_index,
                model_name="llama-text-embed-v2",
                namespace="products"
            )
            
            # Convert filters to Pinecone format
            pinecone_filters = SearchService._convert_filters_to_pinecone(filters)
            
            # Search with filters
            start_time = time.time()
            
            if pinecone_filters:
                # Use filtered search for better precision
                results = await rag.search_with_filter(
                    query=query,
                    filter_dict=pinecone_filters,
                    namespace="products",
                    top_k=top_k,
                    top_n=top_n,
                    min_score=min_score,
                    min_n=min_n,
                    timeout=10.0
                )
            else:
                # Use standard search without filters
                results = await rag.search(
                    query=query,
                    namespace="products",
                    top_k=top_k,
                    top_n=top_n,
                    min_score=min_score,
                    min_n=min_n,
                    timeout=10.0
                )
            
            search_time = time.time() - start_time
            
            # Process and enrich results
            enriched_results = []
            for result in results:
                metadata = result.get('metadata', {})
                
                # Parse metadata if it's a string
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        pass
                
                # Add relevance information
                metadata['relevance_score'] = result.get('score', 0)
                metadata['search_rank'] = len(enriched_results) + 1
                
                enriched_results.append({
                    'id': result.get('id'),
                    'score': result.get('score', 0),
                    'metadata': metadata,
                    'matched_filters': SearchService._get_matched_filters(metadata, filters)
                })
            
            logger.info(f"Enhanced RAG search completed in {search_time:.3f}s, found {len(enriched_results)} results")
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Error in filtered product search: {e}")
            return []
    
    @staticmethod
    def _convert_filters_to_pinecone(filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert extracted filters to Pinecone query format, including dynamic product labels."""
        pinecone_filters = {}
        
        for key, value in filters.items():
            if value is None or value == "":
                continue
                
            # Handle different filter types
            if key == "price":
                # Price is typically a range [min, max]
                if isinstance(value, list) and len(value) == 2:
                    pinecone_filters["price"] = {
                        "$gte": value[0],
                        "$lte": value[1]
                    }
            elif key == "features":
                # Features are multi-select
                if isinstance(value, list) and value:
                    pinecone_filters["features"] = {"$in": value}
            elif key.startswith("label_"):
                # Handle dynamic product labels (e.g., "label_riding_terrain", "label_skill_level")
                category = key[6:]  # Remove "label_" prefix
                if isinstance(value, list) and value:
                    pinecone_filters[f"product_labels.{category}"] = {"$in": value}
                elif isinstance(value, str):
                    pinecone_filters[f"product_labels.{category}"] = {"$in": [value]}
            elif key == "semantic_labels":
                # Handle semantic label matching from intelligent query analysis
                if isinstance(value, dict):
                    for label_category, label_values in value.items():
                        if isinstance(label_values, list) and label_values:
                            pinecone_filters[f"product_labels.{label_category}"] = {"$in": label_values}
            elif key in ["category", "gender", "frame_material"]:
                # Categorical filters
                if isinstance(value, str):
                    pinecone_filters[key] = value
            elif key == "intended_use":
                # Multi-value field
                if isinstance(value, list) and value:
                    pinecone_filters["intended_use"] = {"$in": value}
                    
        return pinecone_filters
    
    @staticmethod
    def _get_matched_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> List[str]:
        """Identify which filters matched this product, including dynamic product labels."""
        matched = []
        
        for key, value in filters.items():
            if key == "price" and isinstance(value, list):
                product_price = metadata.get("price", 0)
                if value[0] <= product_price <= value[1]:
                    matched.append(f"price_{value[0]}-{value[1]}")
            elif key == "features" and isinstance(value, list):
                product_features = metadata.get("features", [])
                for feature in value:
                    if feature in product_features:
                        matched.append(f"feature_{feature}")
            elif key.startswith("label_"):
                # Handle dynamic product label matches
                category = key[6:]  # Remove "label_" prefix
                product_labels = metadata.get("product_labels", {})
                if isinstance(product_labels, dict):
                    category_labels = product_labels.get(category, [])
                    if isinstance(value, list):
                        for label_value in value:
                            if label_value in category_labels:
                                matched.append(f"label_{category}_{label_value}")
                    elif isinstance(value, str) and value in category_labels:
                        matched.append(f"label_{category}_{value}")
            elif key == "semantic_labels":
                # Handle semantic label matches
                product_labels = metadata.get("product_labels", {})
                if isinstance(product_labels, dict) and isinstance(value, dict):
                    for label_category, label_values in value.items():
                        category_labels = product_labels.get(label_category, [])
                        for label_value in label_values:
                            if label_value in category_labels:
                                matched.append(f"semantic_{label_category}_{label_value}")
            elif key in metadata:
                if metadata[key] == value:
                    matched.append(f"{key}_{value}")
                elif isinstance(value, list) and metadata[key] in value:
                    matched.append(f"{key}_{metadata[key]}")
                    
        return matched

    @staticmethod
    async def search_products_llm(query: str, products: List[Product], user_state: UserState) -> List[Product]:
        """
        Search for products using LLM for small catalogs.
        
        Args:
            query: Search query
            products: List of available products
            account: Account identifier
            
        Returns:
            List of matching Product objects
        """
        if not products:
            return []
            
        try:
            results = []
            chat_ctx = llm.ChatContext([])
            
            system_prompt = (
                f"You are an expert sales assistant curating products for a user based on their query. "
                f"Return a JSON array of product ID's from the catalog that best match the query."
                f"Do not make up product ID's or hallucinate product ID's."
            )
            
            chat_ctx.add_message(
                role="system",
                content=[system_prompt]
            )
            
            # Build product catalog text
            product_catalog = f"Return a list of products that match: \"{query}\"\n\n"
            product_catalog += "Respond with JSON format only - an array of product ID's. Include up to 7 matches.\n\n"
            product_catalog += "# PRODUCT CATALOG:\n\n"
            
            for product in products[:500]:  # Limit to 100 products to avoid context limits
                product_catalog += f"{Product.to_markdown(depth=1, product=product)}\n\n"
            
            chat_ctx.add_message(
                role="user",
                content=[product_catalog]
            )
            
            # Use capable model for search
            llm_model = LlmService.fetch_model_service_from_model(model_name="gpt-4.1", account=user_state.account, user=user_state.user_id, model_use="query_enhancement")
            search_results = await LlmService.chat_wrapper(
                llm_service=llm_model,
                chat_ctx=chat_ctx,
            )
            
            # Parse results and find products
            product_ids = LlmService.parse_json_response(search_results)
            if product_ids:
                for product_id in product_ids:
                    product = await Product.find_by_id(productId=product_id, account=user_state.account)
                    if product:
                        results.append(product)
            
            return results
        except Exception as e:
            logger.error(f"Error in search_products_llm: {e}")
            return []

    @staticmethod
    async def search_knowledge(
        query: str, 
        account: str = None, 
        top_k: int = 20, 
        top_n: int = 5, 
        min_score: float = 0.15,
        min_n: int = 0,
        **kwargs
    ) -> List[Dict]:
        """
        Search for knowledge base articles.
        
        Args:
            query: Search query
            account: Account identifier
            top_k: Maximum number of documents to return
            top_n: Number of top results for re-ranking
            min_score: Minimum similarity score threshold
            min_n: Minimum number of results to consider
            
        Returns:
            List of knowledge base search results
        """
        try:
            return await PineconeRAG.rag_query_pinecone(
                query=query,
                account=account,
                use_ranked=True, 
                namespaces=["information"],
                top_k=kwargs.get('top_k', top_k),
                top_n=kwargs.get('top_n', top_n),
                min_score=kwargs.get('min_score', min_score),
                min_n=kwargs.get('min_n', min_n)
            )
        except Exception as e:
            logger.error(f"Error in search_knowledge: {e}")
            return []

    @staticmethod
    async def unified_product_search(
        query: str,
        user_state: UserState,
        chat_ctx: llm.ChatContext,
        account: str,
        use_separate_indexes: bool = True,
        enable_research_enhancement: bool = True,
        enable_filter_extraction: bool = True,
        enable_reranking: bool = True,
        **kwargs
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Unified product search with all enhancements integrated.
        
        This is the main entry point for voice AI agents to search products with:
        - ProductCatalogResearcher knowledge enhancement
        - Separate dense/sparse indexes with server-side embeddings
        - Intelligent filter extraction from catalog labels
        - Multi-stage reranking
        - Performance tracking
        
        Args:
            query: User's search query
            user_state: Current user state
            chat_ctx: Conversation context
            account: Brand account
            use_separate_indexes: Use new separate index architecture
            enable_research_enhancement: Use ProductCatalogResearcher knowledge
            enable_filter_extraction: Extract filters from query
            enable_reranking: Apply neural reranking
            **kwargs: Additional search parameters
            
        Returns:
            Tuple of (search results, search metrics)
        """
        start_time = time.time()
        search_metrics = {
            'query': query,
            'account': account,
            'enhancements': {},
            'performance': {},
            'results': {}
        }
        
        try:
            # Step 1: Load catalog research if enabled
            product_knowledge = ""
            if enable_research_enhancement:
                research_start = time.time()
                catalog_research = await SearchService.load_product_catalog_research(account)
                if catalog_research and catalog_research.get('search_context'):
                    product_knowledge = catalog_research['search_context']
                    search_metrics['enhancements']['research_loaded'] = True
                    search_metrics['performance']['research_load_time'] = time.time() - research_start
                    logger.info(f"📚 Loaded product catalog research in {search_metrics['performance']['research_load_time']:.3f}s")
            
            # Step 2: Enhance query with filters
            enhancement_start = time.time()
            if enable_filter_extraction:
                enhanced_query, filters = await SearchService.enhance_product_query_with_filters(
                    query=query,
                    user_state=user_state,
                    chat_ctx=chat_ctx,
                    account=account,
                    product_knowledge=product_knowledge
                )
                search_metrics['enhancements']['filter_extraction'] = True
                search_metrics['enhancements']['filters_extracted'] = filters
            else:
                # Simple enhancement without filter extraction
                enhanced_query = await SearchService.enhance_product_query(
                    query, user_state, chat_ctx, product_knowledge
                )
                filters = {}
            
            search_metrics['enhancements']['enhanced_query'] = enhanced_query
            search_metrics['performance']['enhancement_time'] = time.time() - enhancement_start
            logger.info(f"🔍 Query enhanced in {search_metrics['performance']['enhancement_time']:.3f}s")
            
            # Step 3: Determine search weights
            dense_weight, sparse_weight = SearchService.determine_search_weights(
                query=enhanced_query,
                user_state=user_state,
                filters=filters
            )
            search_metrics['enhancements']['weights'] = {
                'dense': dense_weight,
                'sparse': sparse_weight
            }
            
            # Step 4: Perform search
            search_start = time.time()
            if use_separate_indexes:
                # Use new SearchRAG architecture
                results = await SearchService.search_products_hybrid_separate_indexes(
                    query=enhanced_query,
                    account=account,
                    filters=filters,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    top_k=kwargs.get('top_k', 35),
                    top_n=kwargs.get('top_n', 10),
                    min_score=kwargs.get('min_score', 0.15),
                    rerank=enable_reranking
                )
                search_metrics['results']['search_type'] = 'searchrag_hybrid'
            else:
                # Fallback to standard RAG search
                results = await SearchService.search_products_rag_with_filters(
                    query=enhanced_query,
                    filters=filters,
                    account=account,
                    **kwargs
                )
                search_metrics['results']['search_type'] = 'rag'
            
            search_metrics['performance']['search_time'] = time.time() - search_start
            search_metrics['results']['count'] = len(results)
            
            # Step 5: Post-process results
            if results:
                # Add relevance explanations
                for i, result in enumerate(results):
                    result['relevance_explanation'] = SearchService._generate_relevance_explanation(
                        result, enhanced_query, filters
                    )
                    result['display_rank'] = i + 1
                
                # Calculate result diversity
                search_metrics['results']['diversity'] = SearchService._calculate_result_diversity(results)
                
                # Top result details
                top_result = results[0]
                search_metrics['results']['top_result'] = {
                    'id': top_result.get('id'),
                    'score': top_result.get('score'),
                    'name': top_result.get('metadata', {}).get('name', 'Unknown')
                }
            
            # Total performance
            search_metrics['performance']['total_time'] = time.time() - start_time
            
            logger.info(
                f"✅ Unified search completed in {search_metrics['performance']['total_time']:.3f}s - "
                f"Found {len(results)} results using {search_metrics['results']['search_type']}"
            )
            
            return results, search_metrics
            
        except Exception as e:
            logger.error(f"Error in unified product search: {e}")
            search_metrics['error'] = str(e)
            search_metrics['performance']['total_time'] = time.time() - start_time
            
            # Fallback to basic search
            try:
                results = await SearchService.search_products_rag(
                    query=query,
                    account=account,
                    **kwargs
                )
                search_metrics['results']['search_type'] = 'rag_fallback'
                return results, search_metrics
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return [], search_metrics
    
    @staticmethod
    def _generate_relevance_explanation(result: Dict, query: str, filters: Dict) -> str:
        """Generate a brief explanation of why this result is relevant."""
        explanations = []
        
        metadata = result.get('metadata', {})
        score = result.get('score', 0)
        
        # Score-based relevance
        if score > 0.9:
            explanations.append("Very high relevance")
        elif score > 0.7:
            explanations.append("High relevance")
        elif score > 0.5:
            explanations.append("Good match")
        
        # Filter matches
        matched_filters = result.get('matched_filters', [])
        if matched_filters:
            explanations.append(f"Matches {len(matched_filters)} filters")
        
        # Source type
        if result.get('source') == 'hybrid':
            explanations.append("Strong semantic and keyword match")
        elif result.get('source') == 'dense':
            explanations.append("Strong semantic match")
        elif result.get('source') == 'sparse':
            explanations.append("Strong keyword match")
        
        return "; ".join(explanations) if explanations else "Relevant match"
    
    @staticmethod
    def _calculate_result_diversity(results: List[Dict]) -> Dict[str, Any]:
        """Calculate diversity metrics for search results."""
        if not results:
            return {'categories': 0, 'price_range': 0, 'unique_features': 0}
        
        categories = set()
        prices = []
        features = set()
        
        for result in results[:10]:  # Analyze top 10
            metadata = result.get('metadata', {})
            
            # Categories
            if 'category' in metadata:
                categories.add(metadata['category'])
            
            # Prices
            if 'price' in metadata and metadata['price'] > 0:
                prices.append(metadata['price'])
            
            # Features from labels
            for key, value in metadata.items():
                if key.startswith('label_') or key in ['key_features', 'performance_traits']:
                    if isinstance(value, list):
                        features.update(value)
                    else:
                        features.add(value)
        
        diversity = {
            'categories': len(categories),
            'price_range': max(prices) - min(prices) if prices else 0,
            'unique_features': len(features)
        }
        
        return diversity
    
    @staticmethod
    async def search_knowledge_rag_with_context(
        query: str,
        user_state: UserState,
        chat_ctx: llm.ChatContext,
        account: str = None,
        knowledge_base: str = "",
        top_k: int = 20,
        top_n: int = 5,
        min_score: float = 0.15,
        **kwargs
    ) -> List[Dict]:
        """
        Enhanced knowledge search with context-aware query optimization.
        
        Args:
            query: Search query
            user_state: User state
            chat_ctx: Conversation context
            account: Account identifier
            knowledge_base: Optional knowledge base context
            top_k: Maximum documents to retrieve
            top_n: Number of results after reranking
            min_score: Minimum relevance score
            
        Returns:
            List of knowledge base search results
        """
        try:
            # Get enhanced search service if available
            if QueryOptimizationAgent and _langfuse_client:
                try:
                    # Try to get enhanced query from Langfuse
                    full_prompt_name = f"liddy/catalog/{account}/knowledge_query_enhancement"
                    prompt = _langfuse_client.get_prompt(full_prompt_name)
                    
                    # Build context variables
                    recent_messages = []
                    for msg in chat_ctx.items[-5:]:
                        if hasattr(msg, "role") and msg.role == "user":
                            content = msg.content if isinstance(msg.content, str) else " ".join(msg.content)
                            recent_messages.append(content)
                    
                    enhanced_query = prompt.compile(
                        query=query,
                        knowledge_base=knowledge_base,
                        account=account,
                        conversation_context=" ".join(recent_messages[-2:])
                    )
                    
                    logger.debug(f"Enhanced knowledge query with Langfuse: {enhanced_query}")
                    
                except Exception as e:
                    logger.warning(f"Failed to enhance query with Langfuse: {e}")
                    # Fallback to basic enhancement
                    enhanced_query = await SearchService.enhance_knowledge_query(
                        query, user_state, chat_ctx, knowledge_base
                    )
            else:
                # Use existing enhancement method
                enhanced_query = await SearchService.enhance_knowledge_query(
                    query, user_state, chat_ctx, knowledge_base
                )
            
            # Use enhanced query for search
            results = await SearchService.search_knowledge(
                enhanced_query, account, top_k, top_n, min_score, **kwargs
            )
            
            # Process results with enhanced metadata
            processed_results = []
            for result in results:
                metadata = result.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        pass
                
                processed_results.append({
                    'id': result.get('id'),
                    'score': result.get('score', 0),
                    'text': result.get('text', ''),
                    'metadata': metadata,
                    'type': 'knowledge'
                })
            
            logger.info(f"Enhanced knowledge search completed, found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in enhanced knowledge search: {e}")
            return []

# Update test suite at the end of the file
if __name__ == "__main__":
    # Test suite for search_utils
    import argparse
    import json
    import time
    from unittest.mock import Mock, AsyncMock, patch
    
    # Configure argument parser for test selection
    parser = argparse.ArgumentParser(description="Test search utilities")
    parser.add_argument("--test", choices=["enhance", "search", "products", "knowledge", "all"], default="all",
                      help="Select which test to run")
    parser.add_argument("--real", action="store_true", help="Use real services instead of mocks")
    parser.add_argument("--query", type=str, default=None, help="Custom query to use for testing")
    parser.add_argument("--account", type=str, default="specialized.com", help="Account to use for testing")
    args = parser.parse_args()
    
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("search_utils_test")
    logger.setLevel(logging.DEBUG)
    
    print("\n===== SEARCH UTILITIES TEST SUITE =====")
    if args.real:
        print("⚠️ USING REAL SERVICES - This will make actual API calls! ⚠️")
    
    class MockSession:
        """Mock session for testing speech timing"""
        def __init__(self, speech_done=True, speech_interrupted=False):
            self.current_speech = Mock()
            self.current_speech.done = Mock(return_value=speech_done)
            self.current_speech.interrupted = speech_interrupted
            self.current_speech.wait_for_playout = AsyncMock()
            self.messages = []
            
        async def say(self, text, add_to_chat_ctx=True):
            logger.info(f"Session saying: '{text}'")
            if add_to_chat_ctx:
                self.messages.append({"role": "assistant", "content": text})
            return AsyncMock()
    
    class MockProduct:
        """Mock product for testing product search"""
        def __init__(self, id="test-1", name="Test Product", productUrl="https://example.com/test"):
            self.id = id
            self.name = name 
            self.productUrl = productUrl
            self.description = "Test product description"
            self.imageUrls = ["https://example.com/test.jpg"]
            self.sizes = ["S", "M", "L"]
            self.colors = ["Red", "Blue"]
            
    async def mock_slow_search(_query, **_kwargs):
        """Mock search that takes longer than the timeout"""
        await asyncio.sleep(0.8)  # Simulate slow search
        return [{"id": "result1", "score": 0.95, "text": "Test result"}]
        
    async def mock_fast_search(_query, **_kwargs):
        """Mock search that completes before the timeout"""
        await asyncio.sleep(0.1)  # Simulate fast search
        return [{"id": "result2", "score": 0.85, "text": "Fast result"}]
    
    async def test_enhance_query():
        """Test query enhancement with LLM responses"""
        print("\n----- Testing Query Enhancement -----")
        
        # Create test chat context with conversation history
        test_ctx = llm.ChatContext([])
        test_ctx.add_message(role="user", content="I need a mountain bike")
        test_ctx.add_message(role="assistant", content="What type of trails do you plan to ride?")
        test_ctx.add_message(role="user", content="Mostly downhill and some technical trails")
        
        # Use custom query if provided
        query = args.query or "mountain bike"
        print(f"Testing with query: '{query}'")
        
        if args.real:
            try:
                # Test with real LLM service
                print("\nTesting with real LLM service...")
                
                # Test basic query enhancement
                enhanced = await SearchService.enhance_query(
                    query, 
                    UserState(account=args.account, user_id="test_user"),
                    test_ctx,
                    "You are a query enhancement specialist"
                )
                print(f"Original query: '{query}'")
                print(f"Enhanced query: '{enhanced}'")
                
                # Test domain-specific enhancements
                product_enhanced = await SearchService.enhance_product_query(
                    query, 
                    UserState(account=args.account, user_id="test_user"),
                    test_ctx,
                    "Product knowledge base: Specialized offers Stumpjumper, Demo, and Enduro models for mountain biking."
                )
                print(f"Product-specific enhanced query: '{product_enhanced}'")
                
                knowledge_enhanced = await SearchService.enhance_knowledge_query(
                    query.replace("bike", "bike maintenance") if "bike" in query else query + " maintenance", 
                    test_ctx,
                    "Knowledge base covers: chain lubrication, brake adjustments, tire pressure, and suspension setup."
                )
                print(f"Knowledge-specific enhanced query: '{knowledge_enhanced}'")
                
            except Exception as e:
                print(f"❌ Error with real LLM service: {e}")
        else:
            # Use mocks for testing
            with patch("spence.llm_service.LlmService.fetch_model_service_from_model") as mock_fetch:
                with patch("spence.llm_service.LlmService.chat_wrapper") as mock_chat:
                    # Set up mocks
                    mock_fetch.return_value = AsyncMock()
                    mock_chat.return_value = f"enhanced query for {query} with additional context"
                    
                    # Test basic query enhancement
                    enhanced = await SearchService.enhance_query(
                        query, 
                        test_ctx,
                        "You are a query enhancement specialist"
                    )
                    print(f"Original query: '{query}'")
                    print(f"Enhanced query: '{enhanced}'")
                    
                    # Test domain-specific enhancements
                    product_enhanced = await SearchService.enhance_product_query(
                        query, 
                        test_ctx,
                        "Product knowledge base: Specialized offers Stumpjumper, Demo, and Enduro models for mountain biking."
                    )
                    print(f"Product-specific enhanced query: '{product_enhanced}'")
                    
                    knowledge_enhanced = await SearchService.enhance_knowledge_query(
                        query.replace("bike", "bike maintenance") if "bike" in query else query + " maintenance", 
                        test_ctx,
                        "Knowledge base covers: chain lubrication, brake adjustments, tire pressure, and suspension setup."
                    )
                    print(f"Knowledge-specific enhanced query: '{knowledge_enhanced}'")
        
        print("✅ Query enhancement tests completed")
    
    async def test_search_timing():
        """Test search timing behavior for slow and fast searches"""
        print("\n----- Testing Search Timing -----")
        
        # Test 1: Fast search should not display wait message
        print("\nTest: Fast search (should NOT display wait message)")
        fast_session = MockSession()
        _result = await SearchService.perform_search(
            query="quick search",
            search_function=mock_fast_search,
            timeout_seconds=0.5,
            wait_message="Please wait..."
        )
        if len(fast_session.messages) == 0:
            print("✅ Passed: No wait message displayed for fast search")
        else:
            print("❌ Failed: Wait message was incorrectly displayed")
        
        # Test 2: Slow search should display wait message
        print("\nTest: Slow search (SHOULD display wait message)")
        slow_session = MockSession()
        _result = await SearchService.perform_search(
            query="slow search",
            search_function=mock_slow_search,
            timeout_seconds=0.2,
            wait_message="Please wait..."
        )
        if len(slow_session.messages) == 1:
            print("✅ Passed: Wait message displayed for slow search")
        else:
            print("❌ Failed: Wait message was not displayed")
        
        # Test 3: Ongoing speech that finishes during search
        print("\nTest: Ongoing speech interaction")
        speech_session = MockSession(speech_done=False)
        
        # Create a side effect that makes the speech finish after a delay
        speech_finish_event = asyncio.Event()
        
        async def mock_wait_for_playout():
            await asyncio.sleep(0.1)
            speech_session.current_speech.done = Mock(return_value=True)
            speech_finish_event.set()
        
        speech_session.current_speech.wait_for_playout.side_effect = mock_wait_for_playout
        
        _result = await SearchService.perform_search(
            query="speech test",
            search_function=mock_slow_search,
            timeout_seconds=0.2
        )
        
        if speech_session.current_speech.wait_for_playout.called:
            print("✅ Passed: Search waited for speech to complete")
        else:
            print("❌ Failed: Search did not wait for speech")
        
        print("✅ Search timing tests completed")
    
    async def test_product_search():
        """Test product search functionality"""
        print("\n----- Testing Product Search -----")
        
        # Use custom query if provided
        query = args.query or "mountain bike"
        print(f"Testing with query: '{query}'")
        
        if args.real:
            try:
                # Test with real RAG services
                print("\nTesting with real RAG service...")
                print("This may take a moment...")
                
                # Test RAG-based product search
                start_time = time.time()
                results = await SearchService.search_products_rag(query, account=args.account)
                elapsed = time.time() - start_time
                print(f"RAG search completed in {elapsed:.2f}s")
                print(f"Found {len(results)} products via RAG")
                
                # Display top results
                for i, result in enumerate(results[:3]):
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            pass
                    
                    print(f"\nResult {i+1}:")
                    print(f"  ID: {result.get('id', 'N/A')}")
                    print(f"  Score: {result.get('score', 0):.4f}")
                    print(f"  Name: {metadata.get('name', 'N/A')}")
                    if 'description' in metadata:
                        desc = metadata['description']
                        if len(desc) > 100:
                            desc = desc[:97] + "..."
                        print(f"  Description: {desc}")
                
                # For small catalogs, also test LLM-based search
                try:
                    print("\nFetching products for LLM-based search...")
                    products = await Product.get_products_async(account=args.account)
                    if len(products) < 50:  # Only test LLM search with small catalogs
                        print(f"Testing LLM search with {len(products)} products...")
                        start_time = time.time()
                        llm_results = await SearchService.search_products_llm(query, products, user_state=args.user_state)
                        elapsed = time.time() - start_time
                        print(f"LLM search completed in {elapsed:.2f}s")
                        print(f"Found {len(llm_results)} products via LLM search")
                        
                        for i, product in enumerate(llm_results[:3]):
                            print(f"\nLLM Result {i+1}:")
                            print(f"  ID: {product.id}")
                            print(f"  Name: {product.name}")
                            print(f"  URL: {product.productUrl}")
                    else:
                        print(f"Skipping LLM search test - catalog too large ({len(products)} products)")
                except Exception as e:
                    print(f"❌ Error with LLM-based product search: {e}")
                
            except Exception as e:
                print(f"❌ Error testing real product search: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Create mock product data
            mock_products = [
                MockProduct(id="bike1", name="Trail Bike", productUrl="https://example.com/bike1"),
                MockProduct(id="bike2", name="Road Bike", productUrl="https://example.com/bike2"),
            ]
            
            # Test RAG-based product search
            print("\nTest: RAG-based product search (mocked)")
            mock_rag_results = [
                {
                    "id": "bike1", 
                    "score": 0.95, 
                    "metadata": {
                        "id": "bike1",
                        "name": "Trail Bike",
                        "description": "A great trail bike",
                        "categories": ["Mountain", "Trail"],
                        "original_price": "$2,999"
                    }
                }
            ]
            
            with patch("spence.rag.PineconeRAG.rag_query_pinecone", new=AsyncMock(return_value=mock_rag_results)):
                results = await SearchService.search_products_rag(query, account="test")
                print(f"Found {len(results)} products via RAG")
                print(f"First result: {results[0]['id'] if results else 'None'}")
            
            # Test LLM-based product search
            print("\nTest: LLM-based product search (mocked)")
            with patch("spence.product.Product.to_markdown", return_value="Mocked product markdown"):
                with patch("spence.llm_service.LlmService.fetch_model_service_from_model", return_value=AsyncMock()):
                    with patch("spence.llm_service.LlmService.chat_wrapper", new=AsyncMock(return_value='["https://example.com/bike1"]')):
                        with patch("spence.llm_service.LlmService.parse_json_response", return_value=["https://example.com/bike1"]):
                            with patch("spence.product.Product.find_by_url", return_value=mock_products[0]):
                                results = await SearchService.search_products_llm(query, mock_products, user_state=UserState(account=args.account, user_id="test_user"))
                                print(f"Found {len(results)} products via LLM search")
                                if results:
                                    print(f"First result: {results[0].name}")
        
        print("✅ Product search tests completed")
        
    async def test_knowledge_search():
        """Test knowledge search functionality"""
        print("\n----- Testing Knowledge Search -----")
        
        # Use custom query if provided
        query = args.query or "bike maintenance"
        print(f"Testing with query: '{query}'")
        
        if args.real:
            try:
                # Test with real knowledge search
                print("\nTesting with real knowledge search...")
                print("This may take a moment...")
                
                start_time = time.time()
                results = await SearchService.search_knowledge(query, account=args.account)
                elapsed = time.time() - start_time
                
                print(f"Knowledge search completed in {elapsed:.2f}s")
                print(f"Found {len(results)} knowledge articles")
                
                # Display top results
                for i, result in enumerate(results[:3]):
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            pass
                    
                    print(f"\nArticle {i+1}:")
                    print(f"  ID: {result.get('id', 'N/A')}")
                    print(f"  Score: {result.get('score', 0):.4f}")
                    print(f"  Title: {metadata.get('title', 'N/A')}")
                    
                    # Get text content if available
                    text = result.get('text', '')
                    if text and isinstance(text, str):
                        if len(text) > 100:
                            text = text[:97] + "..."
                        print(f"  Content: {text}")
                
            except Exception as e:
                print(f"❌ Error testing real knowledge search: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Test with mocked knowledge search
            mock_knowledge_results = [
                {
                    "id": "article1", 
                    "score": 0.88, 
                    "text": "How to maintain your bike chain",
                    "metadata": {
                        "title": "Bike Maintenance Guide",
                        "category": "maintenance",
                        "content_type": "information"
                    }
                }
            ]
            
            with patch("spence.rag.PineconeRAG.rag_query_pinecone", new=AsyncMock(return_value=mock_knowledge_results)):
                results = await SearchService.search_knowledge(query, account="test")
                print(f"Found {len(results)} knowledge articles")
                if results:
                    if isinstance(results[0].get('metadata'), str):
                        # Handle case where metadata might be a JSON string
                        metadata = json.loads(results[0].get('metadata'))
                    else:
                        metadata = results[0].get('metadata', {})
                        
                    print(f"First result: {metadata.get('title', 'Untitled')}")
        
        print("✅ Knowledge search tests completed")
        
    async def run_all_tests():
        """Run all test cases based on command line arguments"""
        if args.test in ["enhance", "all"]:
            await test_enhance_query()
            
        if args.test in ["search", "all"]:
            await test_search_timing()
            
        if args.test in ["products", "all"]:
            await test_product_search()
            
        if args.test in ["knowledge", "all"]:
            await test_knowledge_search()
            
        print("\n===== TEST SUITE COMPLETE =====")
    
    # Execute the tests
    asyncio.run(run_all_tests())
