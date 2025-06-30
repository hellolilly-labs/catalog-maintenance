"""
Enhanced Search Service with UserState RAG Integration

This is an enhanced version of search_service.py that integrates
user preferences and search history for better RAG performance.
"""

from typing import Dict, List, Any, Tuple, Optional
import logging
import json

from livekit.agents import llm
from voice_assistant.model import UserState
from voice_assistant.search_service import SearchService

logger = logging.getLogger(__name__)


class EnhancedSearchService(SearchService):
    """
    Enhanced search service that leverages UserState for personalized RAG search.
    """
    
    @staticmethod
    def extract_user_preferences(user_state: UserState) -> Dict[str, Any]:
        """
        Extract RAG-relevant preferences from UserState.
        
        This method safely extracts user preferences that can be used
        to enhance search quality. It handles the existing UserState
        structure and provides defaults when fields don't exist.
        """
        preferences = {
            'preferred_brands': [],
            'price_range': None,
            'recent_searches': [],
            'category_interests': {},
            'search_style': 'balanced',
            'user_context': []
        }
        
        # Extract from conversation history if available
        if hasattr(user_state, 'conversation_exit_state') and user_state.conversation_exit_state:
            exit_state = user_state.conversation_exit_state
            if exit_state.transcript_summary:
                preferences['user_context'].append(f"Previous conversation: {exit_state.transcript_summary}")
        
        # Extract from sentiment analysis if available
        if hasattr(user_state, 'sentiment_analysis') and user_state.sentiment_analysis:
            sentiment = user_state.sentiment_analysis
            if hasattr(sentiment, 'details') and sentiment.details:
                # Extract any preference-related details
                preferences['user_context'].append(f"User sentiment details: {sentiment.details}")
        
        # Extract from communication directive if available
        if hasattr(user_state, 'communication_directive') and user_state.communication_directive:
            directive = user_state.communication_directive
            if directive.formality and directive.formality.score > 0.7:
                preferences['search_style'] = 'formal'
                preferences['user_context'].append("User prefers formal communication")
        
        # Future: When we add search_preferences to UserState, extract here
        # if hasattr(user_state, 'search_preferences') and user_state.search_preferences:
        #     prefs = user_state.search_preferences
        #     preferences['preferred_brands'] = prefs.preferred_brands
        #     preferences['price_range'] = prefs.price_range
        
        return preferences
    
    @staticmethod
    async def enhance_product_query_with_user_context(
        query: str,
        user_state: UserState,
        chat_ctx: llm.ChatContext,
        account: str,
        product_knowledge: str = ""
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced version that incorporates user preferences into query optimization.
        """
        # Extract user preferences
        user_prefs = EnhancedSearchService.extract_user_preferences(user_state)
        
        # Extract conversation context - look further back for better understanding
        messages = chat_ctx.items[-30:] if len(chat_ctx.items) > 30 else chat_ctx.items
        
        conversation_context = []
        for msg in messages:
            role = msg.role if hasattr(msg, 'role') else 'unknown'
            content = msg.content if hasattr(msg, 'content') else str(msg)
            conversation_context.append(f"{role}: {content}")
        
        # Build enhanced context including user preferences
        context_for_optimization = f"""
Current Query: {query}

User Preferences:
- Preferred Brands: {', '.join(user_prefs['preferred_brands']) if user_prefs['preferred_brands'] else 'None specified'}
- Price Range: {f"${user_prefs['price_range'][0]}-${user_prefs['price_range'][1]}" if user_prefs['price_range'] else 'No preference'}
- Search Style: {user_prefs['search_style']}
- Recent Context: {'; '.join(user_prefs['user_context'][:2]) if user_prefs['user_context'] else 'None'}

Recent Conversation:
{chr(10).join(conversation_context[-10:])}

Product Knowledge:
{product_knowledge}
"""
        
        # Use the existing enhancement logic with enriched context
        enhanced_query, filters = await SearchService.enhance_product_query_with_filters(
            query=query,
            user_state=user_state,
            chat_ctx=chat_ctx,
            account=account,
            product_knowledge=context_for_optimization
        )
        
        # Add user preference filters if not already present
        if user_prefs['preferred_brands'] and 'brand' not in filters:
            filters['brand'] = {'$in': user_prefs['preferred_brands']}
        
        if user_prefs['price_range'] and 'price' not in filters:
            filters['price'] = {
                '$gte': user_prefs['price_range'][0],
                '$lte': user_prefs['price_range'][1]
            }
        
        return enhanced_query, filters
    
    @staticmethod
    def determine_search_weights(
        query: str,
        user_state: UserState,
        filters: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Determine dense vs sparse weights based on query and user preferences.
        
        Returns:
            Tuple of (dense_weight, sparse_weight) or (None, None) for auto
        """
        # Extract user preferences
        user_prefs = EnhancedSearchService.extract_user_preferences(user_state)
        
        # Check for exact match indicators
        exact_match_indicators = [
            'exact',
            'specifically',
            'model number',
            'sku',
            '"'  # Quoted search
        ]
        
        query_lower = query.lower()
        is_exact_search = any(indicator in query_lower for indicator in exact_match_indicators)
        
        # Check if searching for specific brand
        is_brand_search = 'brand' in filters or any(
            brand.lower() in query_lower 
            for brand in user_prefs.get('preferred_brands', [])
        )
        
        # Determine weights
        if is_exact_search or is_brand_search:
            # Favor sparse embeddings for exact/brand searches
            return 0.3, 0.7
        elif user_prefs['search_style'] == 'formal':
            # Slightly favor dense for formal users (more semantic)
            return 0.7, 0.3
        else:
            # Let the system auto-determine
            return None, None
    
    @staticmethod
    async def search_products_with_user_context(
        query: str,
        user_state: UserState,
        chat_ctx: llm.ChatContext,
        account: str,
        **kwargs
    ) -> List[Dict]:
        """
        Complete search flow with user context integration.
        """
        # Enhance query with user context
        enhanced_query, filters = await EnhancedSearchService.enhance_product_query_with_user_context(
            query=query,
            user_state=user_state,
            chat_ctx=chat_ctx,
            account=account,
            product_knowledge=kwargs.get('product_knowledge', '')
        )
        
        # Determine search weights based on user preferences
        dense_weight, sparse_weight = EnhancedSearchService.determine_search_weights(
            query=enhanced_query,
            user_state=user_state,
            filters=filters
        )
        
        # Log for debugging
        logger.info(f"Enhanced search - Query: '{enhanced_query}', Filters: {filters}, "
                   f"Weights: dense={dense_weight}, sparse={sparse_weight}")
        
        # Perform search with enhanced parameters
        # Note: This would integrate with the new hybrid search in the catalog-maintenance project
        results = await SearchService.search_products_rag_with_filters(
            query=enhanced_query,
            filters=filters,
            account=account,
            **kwargs
        )
        
        # Track search performance (placeholder for future implementation)
        # EnhancedSearchService._track_search_performance(
        #     user_state=user_state,
        #     query=query,
        #     enhanced_query=enhanced_query,
        #     results=results
        # )
        
        return results
    
    @staticmethod
    def _track_search_performance(
        user_state: UserState,
        query: str,
        enhanced_query: str,
        results: List[Dict],
        clicked_results: Optional[List[str]] = None
    ):
        """
        Track search performance for future optimization.
        
        This is a placeholder for when we add search_history to UserState.
        """
        # Future implementation when UserState is extended:
        # if not hasattr(user_state, 'search_history'):
        #     return
        #
        # search_history = user_state.search_history
        # search_history.add_query(
        #     query=query,
        #     enhanced_query=enhanced_query,
        #     result_count=len(results),
        #     clicked_results=clicked_results or []
        # )
        pass


# Example usage in sample_assistant.py:
"""
# Instead of:
enhanced_query, extracted_filters = await SearchService.enhance_product_query_with_filters(...)

# Use:
enhanced_query, extracted_filters = await EnhancedSearchService.enhance_product_query_with_user_context(
    query=query,
    user_state=self.session.userdata,
    chat_ctx=self.chat_ctx,
    account=self._account,
    product_knowledge=self._prompt_manager.product_search_knowledge or ""
)

# For complete integration:
results = await EnhancedSearchService.search_products_with_user_context(
    query=query,
    user_state=self.session.userdata,
    chat_ctx=self.chat_ctx,
    account=self._account
)
"""