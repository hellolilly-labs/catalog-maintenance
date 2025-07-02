# Voice Assistant Search Integration Guide

This comprehensive guide explains how to integrate the enhanced search functionality into your voice AI sales agent, with detailed examples and best practices.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Integration Steps](#integration-steps)
- [Advanced Usage](#advanced-usage)
- [Search Metrics](#search-metrics)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [Performance Optimization](#performance-optimization)
- [Real-World Examples](#real-world-examples)

## Overview

The enhanced search system provides state-of-the-art product discovery through:

### 1. **Separate Dense/Sparse Indexes** with Server-Side Embeddings
- **Dense Index**: `llama-text-embed-v2` (1024 dimensions) - Captures semantic meaning
- **Sparse Index**: `pinecone-sparse-english-v0` - Neural sparse model 23% better than BM25

### 2. **ProductCatalogResearcher Integration** 
- Leverages 8-phase brand research for query understanding
- Provides brand-specific context and terminology
- Enhances queries with domain knowledge

### 3. **Intelligent Filter Extraction**
- Automatically identifies filters from natural language
- Maps to brand-specific catalog labels
- Supports category, price, features, and custom attributes

### 4. **Multi-Stage Reranking**
- Initial retrieval from both indexes
- Score combination with adaptive weights
- Neural reranking with `bge-reranker-v2-m3`
- Relevance explanations for transparency

## Quick Start

### Basic Usage

```python
from voice_assistant.search_service import SearchService

# Perform unified search with all enhancements
results, metrics = await SearchService.unified_product_search(
    query="I need a mountain bike for beginners",
    user_state=user_state,
    chat_ctx=chat_context,
    account="specialized.com"
)

# Process results
for result in results[:5]:
    print(f"Product: {result['metadata']['name']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Relevance: {result['relevance_explanation']}")
```

### Configuration Options

```python
# Full control over search behavior
results, metrics = await SearchService.unified_product_search(
    query=query,
    user_state=user_state,
    chat_ctx=chat_context,
    account=account,
    use_separate_indexes=True,          # Use new architecture
    enable_research_enhancement=True,   # Use ProductCatalogResearcher
    enable_filter_extraction=True,      # Extract filters from query
    enable_reranking=True,              # Apply neural reranking
    top_k=50,                          # Retrieve top 50 from indexes
    top_n=10,                          # Return top 10 after reranking
    min_score=0.15                     # Minimum relevance threshold
)
```

## Integration Steps

### 1. Initialize Search Service

```python
# In your Assistant initialization
async def initialize_assistant(account: str):
    # Preload catalog labels for better performance
    await SearchService.preload_catalog_labels(account)
    
    # Load product catalog research
    catalog_research = await SearchService.load_product_catalog_research(account)
    
    # Use in system prompt
    system_prompt = f"""
    You are a knowledgeable sales assistant for {account}.
    
    {SearchService.format_catalog_labels_for_system_prompt(account)}
    
    Brand Insights:
    {catalog_research.get('brand_insights', 'Loading...')}
    """
```

### 2. Handle Search Queries

```python
async def handle_product_search(query: str, session):
    try:
        # Use unified search
        results, metrics = await SearchService.unified_product_search(
            query=query,
            user_state=session.user_state,
            chat_ctx=session.chat_ctx,
            account=session.account
        )
        
        # Log performance
        logger.info(f"Search completed in {metrics['performance']['total_time']:.3f}s")
        logger.info(f"Enhanced query: {metrics['enhancements']['enhanced_query']}")
        logger.info(f"Filters extracted: {metrics['enhancements']['filters_extracted']}")
        
        # Present results
        if results:
            await present_search_results(results, session)
        else:
            await session.say("I couldn't find any products matching your criteria.")
            
    except Exception as e:
        logger.error(f"Search error: {e}")
        await session.say("I'm having trouble searching right now. Let me try again.")
```

### 3. Use Specific Search Modes

```python
# Dense-only search (semantic understanding)
results = await hybrid_search.search(
    query=query,
    search_mode="dense",
    top_k=20
)

# Sparse-only search (exact matching)
results = await hybrid_search.search(
    query=query,
    search_mode="sparse",
    top_k=20
)

# Hybrid search with custom weights
results = await hybrid_search.search(
    query=query,
    search_mode="hybrid",
    dense_weight=0.7,
    sparse_weight=0.3
)
```

## Search Metrics

The unified search returns comprehensive metrics:

```python
metrics = {
    'query': 'original query',
    'account': 'brand.com',
    'enhancements': {
        'research_loaded': True,
        'enhanced_query': 'enhanced version',
        'filters_extracted': {...},
        'weights': {'dense': 0.7, 'sparse': 0.3}
    },
    'performance': {
        'research_load_time': 0.125,
        'enhancement_time': 0.234,
        'search_time': 0.456,
        'total_time': 0.815
    },
    'results': {
        'search_type': 'hybrid_separate_indexes',
        'count': 10,
        'diversity': {
            'categories': 3,
            'price_range': 1000,
            'unique_features': 15
        },
        'top_result': {
            'id': '12345',
            'score': 0.95,
            'name': 'Product Name'
        }
    }
}
```

## Best Practices

1. **Preload Catalog Labels**: Call `preload_catalog_labels()` during initialization
2. **Use Conversation Context**: Always pass the full chat context for better query understanding
3. **Monitor Performance**: Track search metrics to optimize configuration
4. **Handle Fallbacks**: The system automatically falls back to standard RAG if needed
5. **Leverage Filters**: Use extracted filters to provide more relevant results

## Troubleshooting

### Search Returns No Results
- Check if indexes exist: `{brand}-dense` and `{brand}-sparse`
- Verify ProductCatalogResearcher has run for the brand
- Check minimum score threshold (default: 0.15)

### Slow Performance
- Enable index preloading during initialization
- Reduce `top_k` parameter (default: 35)
- Disable research enhancement for simple queries

### Filter Extraction Issues
- Ensure catalog labels are loaded
- Check if QueryOptimizationAgent is available
- Verify filter dictionary exists for the brand

## Architecture

```
User Query
    ↓
ProductCatalogResearcher Knowledge Enhancement
    ↓
Query Optimization & Filter Extraction
    ↓
Separate Index Search
    ├── Dense Index (llama-text-embed-v2)
    └── Sparse Index (pinecone-sparse-english-v0)
    ↓
Result Combination & Scoring
    ↓
Neural Reranking
    ↓
Final Results with Explanations
```

## Advanced Usage

### Dynamic Weight Adjustment

The system automatically adjusts dense vs sparse weights based on query characteristics:

```python
# Query analysis for weight determination
async def determine_search_weights(query: str):
    query_lower = query.lower()
    
    # Exact match indicators favor sparse search
    if any(indicator in query_lower for indicator in ['exact', 'model', 'sku', '"']):
        return {'dense': 0.3, 'sparse': 0.7}
    
    # Semantic indicators favor dense search
    if any(indicator in query_lower for indicator in ['like', 'similar', 'comfortable', 'best']):
        return {'dense': 0.8, 'sparse': 0.2}
    
    # Default balanced approach
    return {'dense': 0.7, 'sparse': 0.3}
```

### Custom Filter Handling

```python
# Extract and apply custom filters
async def search_with_custom_filters(query: str, session):
    # Extract filters from natural language
    filters = await SearchService.extract_filters_from_query(
        query=query,
        account=session.account,
        catalog_labels=session.catalog_labels
    )
    
    # Apply price range filter
    if "under $500" in query:
        filters['price'] = {'$lte': 500}
    
    # Apply category filter
    if filters.get('extracted_category'):
        filters['category'] = filters['extracted_category']
    
    # Perform search with filters
    results, metrics = await SearchService.unified_product_search(
        query=query,
        user_state=session.user_state,
        chat_ctx=session.chat_ctx,
        account=session.account,
        filters=filters
    )
```

### Multi-Turn Conversation Support

```python
async def handle_followup_search(followup_query: str, session):
    # Get previous search context
    previous_results = session.search_history[-1] if session.search_history else None
    
    # Build enhanced query with context
    if previous_results:
        context_enhanced_query = f"{followup_query} (previous: {previous_results['query']})"
        
        # Inherit filters from previous search
        inherited_filters = previous_results.get('filters', {})
        
        results, metrics = await SearchService.unified_product_search(
            query=context_enhanced_query,
            user_state=session.user_state,
            chat_ctx=session.chat_ctx,
            account=session.account,
            filters=inherited_filters
        )
```

## Performance Optimization

### 1. Preloading Strategy

```python
# Startup preloading for optimal performance
async def initialize_search_system(account: str):
    # Preload in parallel
    tasks = [
        SearchService.preload_catalog_labels(account),
        SearchService.load_product_catalog_research(account),
        SearchService.initialize_indexes(account)
    ]
    
    catalog_labels, catalog_research, indexes_ready = await asyncio.gather(*tasks)
    
    return {
        'catalog_labels': catalog_labels,
        'catalog_research': catalog_research,
        'indexes_ready': indexes_ready
    }
```

### 2. Caching Strategy

```python
# Implement search result caching
class SearchCache:
    def __init__(self, ttl_seconds=300):  # 5-minute cache
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get_cache_key(self, query: str, filters: dict, account: str):
        return f"{account}:{query}:{json.dumps(filters, sort_keys=True)}"
    
    async def get_or_search(self, query: str, **kwargs):
        cache_key = self.get_cache_key(query, kwargs.get('filters', {}), kwargs['account'])
        
        # Check cache
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                return cached_result, {'cache_hit': True}
        
        # Perform search
        results, metrics = await SearchService.unified_product_search(query, **kwargs)
        
        # Cache results
        self.cache[cache_key] = ((results, metrics), time.time())
        
        return results, metrics
```

### 3. Batch Processing

```python
# Process multiple searches efficiently
async def batch_search(queries: List[str], session):
    tasks = []
    
    for query in queries:
        task = SearchService.unified_product_search(
            query=query,
            user_state=session.user_state,
            chat_ctx=session.chat_ctx,
            account=session.account,
            enable_reranking=False  # Disable for batch processing
        )
        tasks.append(task)
    
    # Execute in parallel
    results = await asyncio.gather(*tasks)
    
    # Optional: Apply cross-query reranking
    return optimize_batch_results(results)
```

## Real-World Examples

### Example 1: Voice Shopping Assistant

```python
class VoiceShoppingAssistant:
    async def handle_product_inquiry(self, transcript: str, session):
        # Enhance query with shopping context
        shopping_context = {
            'user_preferences': session.user_state.preferences,
            'previous_purchases': session.user_state.purchase_history,
            'budget_range': session.user_state.budget
        }
        
        # Search with all enhancements
        results, metrics = await SearchService.unified_product_search(
            query=transcript,
            user_state=session.user_state,
            chat_ctx=session.chat_ctx,
            account=session.account,
            user_context=shopping_context
        )
        
        # Generate voice response
        if results:
            top_product = results[0]
            response = f"I found the {top_product['metadata']['name']} for ${top_product['metadata']['price']}. "
            response += f"It's {top_product['relevance_explanation']}. Would you like to hear more details?"
            
            await session.say(response)
            
            # Store for follow-up
            session.current_product = top_product
        else:
            await session.say("I couldn't find products matching your description. Could you tell me more about what you're looking for?")
```

### Example 2: Conversational Product Discovery

```python
class ConversationalProductDiscovery:
    async def refine_search(self, refinement: str, session):
        # Build cumulative query
        if session.search_refinements:
            cumulative_query = f"{session.original_query} {' '.join(session.search_refinements)} {refinement}"
        else:
            cumulative_query = f"{session.original_query} {refinement}"
        
        # Add refinement to history
        session.search_refinements.append(refinement)
        
        # Search with accumulated context
        results, metrics = await SearchService.unified_product_search(
            query=cumulative_query,
            user_state=session.user_state,
            chat_ctx=session.chat_ctx,
            account=session.account,
            enable_filter_extraction=True,
            top_k=20  # Fewer results for refinements
        )
        
        # Analyze refinement effectiveness
        if metrics['results']['count'] < 5:
            await session.say(f"I've narrowed it down to {metrics['results']['count']} options based on your preferences.")
        elif metrics['results']['count'] > 50:
            await session.say("I still have many options. What's most important to you?")
```

### Example 3: Comparison Shopping

```python
async def handle_comparison_request(products_to_compare: List[str], session):
    # Search for each product
    comparison_results = []
    
    for product_name in products_to_compare:
        results, _ = await SearchService.unified_product_search(
            query=product_name,
            user_state=session.user_state,
            chat_ctx=session.chat_ctx,
            account=session.account,
            search_mode="sparse",  # Use sparse for specific products
            top_k=1  # Get best match only
        )
        
        if results:
            comparison_results.append(results[0])
    
    # Generate comparison
    return generate_product_comparison(comparison_results)
```

## API Reference

### Core Methods

```python
# Main unified search method
async def unified_product_search(
    query: str,
    user_state: UserState,
    chat_ctx: llm.ChatContext,
    account: str,
    filters: Optional[Dict[str, Any]] = None,
    use_separate_indexes: bool = True,
    enable_research_enhancement: bool = True,
    enable_filter_extraction: bool = True,
    enable_reranking: bool = True,
    search_mode: str = "hybrid",  # "hybrid", "dense", "sparse"
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
    user_context: Optional[Dict[str, Any]] = None,
    top_k: int = 35,
    top_n: int = 10,
    min_score: float = 0.15
) -> Tuple[List[Dict], Dict[str, Any]]

# Preload catalog labels
async def preload_catalog_labels(account: str) -> Dict[str, Set[str]]

# Load product catalog research
async def load_product_catalog_research(account: str) -> Dict[str, Any]

# Format labels for system prompt
def format_catalog_labels_for_system_prompt(account: str) -> str

# Extract filters from query
async def extract_filters_from_query(
    query: str,
    account: str,
    catalog_labels: Dict[str, Set[str]]
) -> Dict[str, Any]
```

## Configuration

### Environment Variables

```bash
# Pinecone configuration
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=us-east-1

# Model configuration
DENSE_EMBEDDING_MODEL=llama-text-embed-v2
SPARSE_EMBEDDING_MODEL=pinecone-sparse-english-v0
RERANKING_MODEL=bge-reranker-v2-m3

# Performance settings
SEARCH_CACHE_TTL=300
MAX_CONCURRENT_SEARCHES=10
DEFAULT_TOP_K=35
DEFAULT_MIN_SCORE=0.15
```

### Index Naming Convention

```
Dense Index: {brand}-dense
Sparse Index: {brand}-sparse
Namespace: products
```

## Monitoring and Analytics

### Search Quality Metrics

```python
# Track search quality
class SearchQualityMonitor:
    async def track_search_quality(self, query: str, results: List[Dict], user_feedback: Optional[str]):
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result_count': len(results),
            'top_score': results[0]['score'] if results else 0,
            'score_distribution': self.calculate_score_distribution(results),
            'category_diversity': self.calculate_category_diversity(results),
            'user_feedback': user_feedback
        }
        
        # Log to analytics service
        await self.analytics.log_event('search_quality', metrics)
```

### Performance Monitoring

```python
# Monitor search performance
async def monitor_search_performance(metrics: Dict[str, Any]):
    if metrics['performance']['total_time'] > 1.0:
        logger.warning(f"Slow search detected: {metrics['performance']['total_time']:.2f}s")
        
        # Analyze bottlenecks
        bottlenecks = []
        if metrics['performance']['research_load_time'] > 0.3:
            bottlenecks.append('research_loading')
        if metrics['performance']['search_time'] > 0.5:
            bottlenecks.append('index_search')
        
        # Log for optimization
        logger.info(f"Bottlenecks: {bottlenecks}")
```

## Migration Guide

### From Single Index to Separate Indexes

```python
# Check if using old single index
if await is_using_single_index(account):
    logger.info(f"Migrating {account} to separate indexes...")
    
    # Create new indexes
    await create_separate_indexes(account)
    
    # Re-ingest products
    await reingest_products_to_separate_indexes(account)
    
    # Update search configuration
    config.use_separate_indexes = True
```

## Next Steps

1. **Monitor Search Quality**: Track metrics and user satisfaction
2. **A/B Testing**: Compare different configurations
3. **Personalization**: Add user preference learning
4. **Voice Optimization**: Tune for voice-specific queries
5. **Multi-language Support**: Extend to support multiple languages
6. **Recommendation Engine**: Build on search for recommendations
7. **Analytics Dashboard**: Create real-time search analytics