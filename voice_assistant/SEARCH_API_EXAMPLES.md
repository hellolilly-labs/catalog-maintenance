# Search API Usage Examples

This document provides practical code examples for integrating the enhanced search functionality into your voice assistant.

## Complete Integration Example

```python
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from voice_assistant.search_service import SearchService
from voice_assistant.catalog_maintenance_api import CatalogMaintenanceAPI
from src.search.separate_index_hybrid_search import SeparateIndexHybridSearch

logger = logging.getLogger(__name__)

class EnhancedVoiceAssistant:
    """
    Complete voice assistant with integrated enhanced search capabilities.
    """
    
    def __init__(self, account: str):
        self.account = account
        self.search_cache = {}
        self.catalog_api = CatalogMaintenanceAPI()
        self.hybrid_search = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all search components."""
        try:
            # Initialize search indexes
            self.hybrid_search = SeparateIndexHybridSearch(
                brand_domain=self.account,
                dense_index_name=f"{self.account.split('.')[0]}-dense",
                sparse_index_name=f"{self.account.split('.')[0]}-sparse"
            )
            await self.hybrid_search.initialize()
            
            # Preload catalog data
            await SearchService.preload_catalog_labels(self.account)
            catalog_research = await SearchService.load_product_catalog_research(self.account)
            
            # Store for system prompt
            self.catalog_research = catalog_research
            self.initialized = True
            
            logger.info(f"✅ Enhanced search initialized for {self.account}")
            
        except Exception as e:
            logger.error(f"Failed to initialize search: {e}")
            raise
    
    async def search_products(
        self, 
        query: str, 
        user_state: Any,
        chat_ctx: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main search method with full enhancement pipeline.
        """
        if not self.initialized:
            await self.initialize()
        
        # Use unified search with all enhancements
        results, metrics = await SearchService.unified_product_search(
            query=query,
            user_state=user_state,
            chat_ctx=chat_ctx,
            account=self.account,
            **kwargs
        )
        
        # Format response for voice
        response = self._format_voice_response(results, metrics)
        
        return {
            'results': results,
            'metrics': metrics,
            'voice_response': response
        }
    
    def _format_voice_response(
        self, 
        results: List[Dict], 
        metrics: Dict[str, Any]
    ) -> str:
        """Format search results for voice output."""
        if not results:
            return "I couldn't find any products matching your criteria. Could you provide more details?"
        
        # Single result
        if len(results) == 1:
            product = results[0]['metadata']
            return (f"I found the {product['name']} "
                   f"for ${product.get('price', 'price not available')}. "
                   f"{results[0].get('relevance_explanation', '')}")
        
        # Multiple results
        response = f"I found {len(results)} products. "
        
        # Mention top 3
        for i, result in enumerate(results[:3]):
            product = result['metadata']
            if i == 0:
                response += f"The best match is {product['name']} "
            elif i == 1:
                response += f"followed by {product['name']} "
            elif i == 2:
                response += f"and {product['name']}. "
        
        response += "Would you like to hear more about any of these?"
        
        return response
```

## Search Enhancement Examples

### 1. Query Enhancement with Brand Context

```python
async def enhance_query_with_brand_context(
    query: str,
    account: str,
    chat_context: Any
) -> str:
    """
    Enhance query using ProductCatalogResearcher knowledge.
    """
    # Load brand research
    catalog_research = await SearchService.load_product_catalog_research(account)
    
    # Extract key brand terminology
    brand_terms = catalog_research.get('brand_terminology', {})
    product_categories = catalog_research.get('product_categories', [])
    
    # Build enhanced query
    enhanced_query = query
    
    # Add brand-specific synonyms
    for term, synonyms in brand_terms.items():
        if term.lower() in query.lower():
            enhanced_query += f" ({' OR '.join(synonyms)})"
    
    # Add category context if detected
    for category in product_categories:
        if category.lower() in query.lower():
            category_context = catalog_research.get(f'category_{category}_context', '')
            if category_context:
                enhanced_query += f" {category_context}"
    
    return enhanced_query
```

### 2. Filter Extraction and Application

```python
async def extract_and_apply_filters(
    query: str,
    account: str
) -> Dict[str, Any]:
    """
    Extract filters from natural language query.
    """
    # Load catalog labels
    catalog_labels = await SearchService.get_catalog_labels(account)
    
    filters = {}
    
    # Price extraction
    price_patterns = [
        (r'under \$?(\d+)', lambda m: {'price': {'$lte': int(m.group(1))}}),
        (r'over \$?(\d+)', lambda m: {'price': {'$gte': int(m.group(1))}}),
        (r'between \$?(\d+) and \$?(\d+)', lambda m: {
            'price': {'$gte': int(m.group(1)), '$lte': int(m.group(2))}
        })
    ]
    
    import re
    for pattern, extractor in price_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            filters.update(extractor(match))
            break
    
    # Category extraction
    if 'categories' in catalog_labels:
        for category in catalog_labels['categories']:
            if category.lower() in query.lower():
                filters['category'] = category
                break
    
    # Feature extraction
    if 'features' in catalog_labels:
        extracted_features = []
        for feature in catalog_labels['features']:
            if feature.lower() in query.lower():
                extracted_features.append(feature)
        
        if extracted_features:
            filters['features'] = {'$in': extracted_features}
    
    # Size extraction
    size_words = ['small', 'medium', 'large', 'xl', 'xxl', 'xs']
    for size in size_words:
        if size in query.lower():
            filters['size'] = size.upper()
            break
    
    return filters
```

### 3. Multi-Stage Search Pipeline

```python
class MultiStageSearchPipeline:
    """
    Implements a multi-stage search pipeline with progressive refinement.
    """
    
    async def search_with_refinement(
        self,
        initial_query: str,
        session: Any,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Perform multi-stage search with automatic refinement.
        """
        # Stage 1: Broad semantic search
        stage1_results, stage1_metrics = await SearchService.unified_product_search(
            query=initial_query,
            user_state=session.user_state,
            chat_ctx=session.chat_ctx,
            account=session.account,
            search_mode="dense",
            top_k=100,  # Get more candidates
            enable_reranking=False  # No reranking yet
        )
        
        # Stage 2: Apply filters if too many results
        if len(stage1_results) > 50:
            # Extract filters from query
            filters = await extract_and_apply_filters(initial_query, session.account)
            
            if filters:
                stage2_results, _ = await SearchService.unified_product_search(
                    query=initial_query,
                    user_state=session.user_state,
                    chat_ctx=session.chat_ctx,
                    account=session.account,
                    filters=filters,
                    search_mode="hybrid",
                    top_k=50
                )
            else:
                stage2_results = stage1_results[:50]
        else:
            stage2_results = stage1_results
        
        # Stage 3: Neural reranking with context
        if stage2_results:
            # Prepare reranking context
            reranking_context = {
                'user_preferences': session.user_state.preferences,
                'previous_purchases': session.user_state.purchase_history,
                'session_context': session.chat_ctx
            }
            
            # Rerank with enhanced context
            reranked_results = await self._neural_rerank(
                query=initial_query,
                candidates=stage2_results,
                context=reranking_context,
                top_n=max_results
            )
            
            return reranked_results
        
        return []
    
    async def _neural_rerank(
        self,
        query: str,
        candidates: List[Dict],
        context: Dict[str, Any],
        top_n: int
    ) -> List[Dict]:
        """Apply neural reranking with context."""
        # Implementation would use Pinecone's reranking or custom model
        # This is a simplified example
        from pinecone import Pinecone
        pc = Pinecone()
        
        # Prepare documents for reranking
        documents = []
        for candidate in candidates:
            doc_text = self._build_reranking_text(candidate, context)
            documents.append({
                'id': candidate['id'],
                'text': doc_text
            })
        
        # Rerank
        reranked = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=f"{query} {context.get('user_preferences', '')}",
            documents=documents,
            top_n=top_n
        )
        
        # Map back to original results
        reranked_results = []
        for item in reranked.data:
            for candidate in candidates:
                if candidate['id'] == item.id:
                    candidate['rerank_score'] = item.score
                    reranked_results.append(candidate)
                    break
        
        return reranked_results
```

## Conversation Context Examples

### 1. Multi-Turn Search Refinement

```python
class ConversationalSearchManager:
    """
    Manages multi-turn conversational search with context preservation.
    """
    
    def __init__(self):
        self.search_sessions = {}
    
    async def handle_search_turn(
        self,
        user_id: str,
        utterance: str,
        session: Any
    ) -> Dict[str, Any]:
        """
        Handle a search turn in conversation.
        """
        # Get or create search session
        if user_id not in self.search_sessions:
            self.search_sessions[user_id] = {
                'history': [],
                'filters': {},
                'preferences': {}
            }
        
        search_session = self.search_sessions[user_id]
        
        # Determine if this is a refinement or new search
        is_refinement = self._is_refinement(utterance, search_session)
        
        if is_refinement:
            # Build cumulative query
            cumulative_query = self._build_cumulative_query(
                utterance, 
                search_session
            )
            
            # Inherit and update filters
            filters = search_session['filters'].copy()
            new_filters = await extract_and_apply_filters(utterance, session.account)
            filters.update(new_filters)
            
            # Search with context
            results, metrics = await SearchService.unified_product_search(
                query=cumulative_query,
                user_state=session.user_state,
                chat_ctx=session.chat_ctx,
                account=session.account,
                filters=filters,
                user_context={'search_history': search_session['history']}
            )
            
            # Update session
            search_session['history'].append({
                'query': utterance,
                'type': 'refinement',
                'results_count': len(results),
                'timestamp': datetime.now()
            })
            search_session['filters'] = filters
            
        else:
            # New search
            results, metrics = await SearchService.unified_product_search(
                query=utterance,
                user_state=session.user_state,
                chat_ctx=session.chat_ctx,
                account=session.account
            )
            
            # Reset session
            search_session['history'] = [{
                'query': utterance,
                'type': 'initial',
                'results_count': len(results),
                'timestamp': datetime.now()
            }]
            search_session['filters'] = metrics.get('enhancements', {}).get('filters_extracted', {})
        
        return {
            'results': results,
            'metrics': metrics,
            'session_state': search_session,
            'is_refinement': is_refinement
        }
    
    def _is_refinement(self, utterance: str, session: Dict) -> bool:
        """Determine if utterance is a search refinement."""
        refinement_indicators = [
            'also', 'but', 'and', 'or', 'except', 
            'without', 'under', 'over', 'cheaper', 
            'more expensive', 'different', 'similar'
        ]
        
        # Check for refinement indicators
        utterance_lower = utterance.lower()
        has_indicator = any(ind in utterance_lower for ind in refinement_indicators)
        
        # Check if recent search exists
        has_recent_search = (
            session.get('history') and 
            (datetime.now() - session['history'][-1]['timestamp']).seconds < 300
        )
        
        return has_indicator and has_recent_search
    
    def _build_cumulative_query(
        self, 
        new_utterance: str, 
        session: Dict
    ) -> str:
        """Build query incorporating conversation history."""
        # Get recent queries
        recent_queries = [
            h['query'] for h in session['history'][-3:]
            if h['type'] in ['initial', 'refinement']
        ]
        
        # Combine with new utterance
        combined = ' '.join(recent_queries + [new_utterance])
        
        # Remove duplicates while preserving order
        words = combined.split()
        seen = set()
        unique_words = []
        for word in words:
            if word.lower() not in seen:
                seen.add(word.lower())
                unique_words.append(word)
        
        return ' '.join(unique_words)
```

### 2. Preference Learning

```python
class SearchPreferenceLearner:
    """
    Learns user preferences from search interactions.
    """
    
    def __init__(self):
        self.user_preferences = {}
    
    async def learn_from_interaction(
        self,
        user_id: str,
        query: str,
        results: List[Dict],
        selected_index: Optional[int] = None,
        feedback: Optional[str] = None
    ):
        """
        Learn preferences from user interactions.
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'preferred_categories': {},
                'price_range': {'min': float('inf'), 'max': 0},
                'preferred_features': {},
                'brand_affinity': {}
            }
        
        prefs = self.user_preferences[user_id]
        
        # Learn from selection
        if selected_index is not None and selected_index < len(results):
            selected = results[selected_index]['metadata']
            
            # Update category preference
            category = selected.get('category')
            if category:
                prefs['preferred_categories'][category] = \
                    prefs['preferred_categories'].get(category, 0) + 1
            
            # Update price range
            price = selected.get('price')
            if price:
                prefs['price_range']['min'] = min(prefs['price_range']['min'], price)
                prefs['price_range']['max'] = max(prefs['price_range']['max'], price)
            
            # Update feature preferences
            features = selected.get('features', [])
            for feature in features:
                prefs['preferred_features'][feature] = \
                    prefs['preferred_features'].get(feature, 0) + 1
        
        # Learn from feedback
        if feedback:
            # Simple sentiment analysis
            positive_words = ['good', 'great', 'perfect', 'love', 'exactly']
            negative_words = ['bad', 'wrong', 'not', 'different', 'expensive']
            
            is_positive = any(word in feedback.lower() for word in positive_words)
            is_negative = any(word in feedback.lower() for word in negative_words)
            
            # Adjust preferences based on feedback
            if is_negative and results:
                # Reduce preference for shown results
                for result in results[:3]:
                    category = result['metadata'].get('category')
                    if category and category in prefs['preferred_categories']:
                        prefs['preferred_categories'][category] *= 0.9
    
    def get_preference_context(self, user_id: str) -> Dict[str, Any]:
        """Get user preference context for search enhancement."""
        if user_id not in self.user_preferences:
            return {}
        
        prefs = self.user_preferences[user_id]
        
        # Get top preferences
        top_categories = sorted(
            prefs['preferred_categories'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        top_features = sorted(
            prefs['preferred_features'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'preferred_categories': [cat for cat, _ in top_categories],
            'price_preference': prefs['price_range'],
            'preferred_features': [feat for feat, _ in top_features]
        }
```

## Error Handling and Fallbacks

```python
class RobustSearchHandler:
    """
    Handles search with comprehensive error handling and fallbacks.
    """
    
    async def search_with_fallbacks(
        self,
        query: str,
        session: Any,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Perform search with multiple fallback strategies.
        """
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                # Try enhanced search first
                if attempt == 0:
                    results, metrics = await SearchService.unified_product_search(
                        query=query,
                        user_state=session.user_state,
                        chat_ctx=session.chat_ctx,
                        account=session.account,
                        use_separate_indexes=True,
                        enable_research_enhancement=True,
                        enable_filter_extraction=True,
                        enable_reranking=True
                    )
                    
                    if results:
                        return {
                            'results': results,
                            'metrics': metrics,
                            'fallback_level': 0
                        }
                
                # Fallback 1: Disable enhancements
                elif attempt == 1:
                    results, metrics = await SearchService.unified_product_search(
                        query=query,
                        user_state=session.user_state,
                        chat_ctx=session.chat_ctx,
                        account=session.account,
                        use_separate_indexes=True,
                        enable_research_enhancement=False,
                        enable_filter_extraction=False,
                        enable_reranking=False
                    )
                    
                    if results:
                        return {
                            'results': results,
                            'metrics': metrics,
                            'fallback_level': 1
                        }
                
                # Fallback 2: Use standard RAG
                elif attempt == 2:
                    from spence.rag import PineconeRAG
                    rag = PineconeRAG(session.account)
                    
                    results = await rag.search(
                        query=query,
                        top_k=10
                    )
                    
                    if results:
                        return {
                            'results': results,
                            'metrics': {'fallback': 'standard_rag'},
                            'fallback_level': 2
                        }
                
            except Exception as e:
                last_error = e
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                attempt += 1
                
                # Wait before retry
                await asyncio.sleep(0.5 * attempt)
        
        # All attempts failed
        logger.error(f"All search attempts failed. Last error: {last_error}")
        
        return {
            'results': [],
            'metrics': {
                'error': str(last_error),
                'attempts': max_retries
            },
            'fallback_level': -1
        }
```

## Testing and Validation

```python
async def test_search_integration():
    """
    Test the complete search integration.
    """
    # Initialize assistant
    assistant = EnhancedVoiceAssistant("specialized.com")
    await assistant.initialize()
    
    # Test queries
    test_queries = [
        "I need a mountain bike under $2000",
        "Show me road bikes for beginners",
        "What's your most popular bike?",
        "I want something similar to the Tarmac SL7",
        "bikes with electronic shifting"
    ]
    
    # Mock session
    class MockSession:
        def __init__(self):
            self.account = "specialized.com"
            self.user_state = type('obj', (object,), {
                'preferences': {},
                'purchase_history': []
            })
            self.chat_ctx = []
    
    session = MockSession()
    
    # Run tests
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        
        try:
            result = await assistant.search_products(
                query=query,
                user_state=session.user_state,
                chat_ctx=session.chat_ctx
            )
            
            print(f"✅ Found {len(result['results'])} results")
            print(f"📊 Search time: {result['metrics']['performance']['total_time']:.3f}s")
            print(f"🎯 Voice response: {result['voice_response']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_search_integration())
```

## Performance Benchmarking

```python
async def benchmark_search_performance():
    """
    Benchmark search performance across different configurations.
    """
    configurations = [
        {
            'name': 'Full Enhancement',
            'params': {
                'use_separate_indexes': True,
                'enable_research_enhancement': True,
                'enable_filter_extraction': True,
                'enable_reranking': True
            }
        },
        {
            'name': 'No Reranking',
            'params': {
                'use_separate_indexes': True,
                'enable_research_enhancement': True,
                'enable_filter_extraction': True,
                'enable_reranking': False
            }
        },
        {
            'name': 'Basic Hybrid',
            'params': {
                'use_separate_indexes': True,
                'enable_research_enhancement': False,
                'enable_filter_extraction': False,
                'enable_reranking': False
            }
        }
    ]
    
    # Test queries
    queries = [
        "mountain bike",
        "road bike under $3000 for beginners",
        "bikes similar to specialized tarmac with electronic shifting"
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n📊 Testing: {config['name']}")
        
        for query in queries:
            start_time = time.time()
            
            try:
                search_results, metrics = await SearchService.unified_product_search(
                    query=query,
                    user_state=None,
                    chat_ctx=[],
                    account="specialized.com",
                    **config['params']
                )
                
                elapsed = time.time() - start_time
                
                results.append({
                    'config': config['name'],
                    'query': query,
                    'result_count': len(search_results),
                    'total_time': elapsed,
                    'search_time': metrics['performance']['search_time'],
                    'top_score': search_results[0]['score'] if search_results else 0
                })
                
                print(f"  ✓ {query[:30]}... - {elapsed:.3f}s, {len(search_results)} results")
                
            except Exception as e:
                print(f"  ✗ {query[:30]}... - Error: {e}")
    
    # Analyze results
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("\n📈 Performance Summary:")
    print(df.groupby('config')[['total_time', 'result_count', 'top_score']].mean())
    
    return df
```

## Next Steps

1. **Implement A/B testing** to compare search configurations
2. **Add user feedback collection** to improve search quality
3. **Build recommendation engine** on top of search
4. **Create analytics dashboard** for search monitoring
5. **Extend to multi-language support**
6. **Implement personalization** based on user history
7. **Add voice-specific optimizations** for natural language queries