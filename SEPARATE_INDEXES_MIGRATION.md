# Separate Indexes Migration Guide

## Overview

This guide documents the migration from a single hybrid index to Pinecone's recommended separate indexes architecture for dense and sparse embeddings.

## Architecture Changes

### Previous: Single Hybrid Index
```
specialized-hybrid-v2
├── Dense vectors
├── Sparse vectors
└── Shared metadata
```

### New: Separate Indexes
```
specialized-dense-v2
├── Dense embeddings (2048d)
├── Metadata
└── Text for reranking

specialized-sparse-v2
├── Sparse embeddings (50000d)
├── Metadata (identical to dense)
└── BM25-style keyword matching
```

## Key Benefits

1. **Flexibility**: Can perform sparse-only or dense-only searches
2. **Scalability**: Add sparse search to existing dense indexes
3. **Control**: Different configurations per index type
4. **Performance**: Optimize each index independently

## Implementation

### 1. Updated Dependencies
```bash
# Old
pinecone-client>=3.0.0

# New
pinecone[grpc]>=7.3.0
```

### 2. Index Creation
```python
# Dense index
pc.create_index(
    name="brand-dense-v2",
    dimension=2048,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Sparse index
pc.create_index(
    name="brand-sparse-v2", 
    dimension=50000,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    sparse_values=True
)
```

### 3. Consistent ID Linkage
Both indexes use the same product ID, enabling result merging:
```python
# Same ID in both indexes
dense_vector = {'id': 'product-123', 'text': '...', 'metadata': {...}}
sparse_vector = {'id': 'product-123', 'sparse_values': {...}, 'metadata': {...}}
```

## Descriptor Quality

Our descriptor generation follows these best practices:

### 1. **Length Optimization**
- Target: 50-200 words for voice AI
- Current: Enhanced descriptors combine multiple sources

### 2. **Content Structure**
```
[Brand Name] [Product Name] [Natural Description] [Category Context] 
[Price Point] [Key Features] [Use Cases]
```

### 3. **Voice Optimization**
- Conversational tone ("perfect for", "ideal for")
- Natural language flow
- Minimal technical jargon

### 4. **Search Optimization**
- Product name and variations
- Category and subcategories
- Price range keywords
- Feature keywords

## Dynamic Metadata Schema

Metadata is generated dynamically per brand based on:

### 1. **Universal Fields** (All Brands)
- `id`, `name`, `price`, `category`, `brand`
- `description`, `image_url`

### 2. **Filter Metadata** (Brand-Specific)
Extracted from catalog analysis:
- Specialized: `frame_material`, `wheel_size`, `gender`
- Balenciaga: `collection`, `season`, `size`
- Sunday Riley: `skin_type`, `concerns`, `ingredients`

### 3. **Content Fields**
- `enhanced_descriptor`: RAG-optimized text
- `voice_summary`: 30-50 word summary
- `key_selling_points`: Top 3 features
- `search_keywords`: Extracted terms

## Query Expansion Implementation

### 1. **Synonym Mapping** (Future)
```python
query_synonyms = {
    "bike": ["bicycle", "cycle"],
    "comfortable": ["comfort", "comfy", "cushioned"],
    "lightweight": ["light", "ultralight", "featherweight"]
}
```

### 2. **Category Expansion**
```python
category_hierarchy = {
    "road": ["road bike", "racing bike", "performance bike"],
    "mountain": ["mtb", "mountain bike", "trail bike"]
}
```

### 3. **Brand-Specific Terms**
Loaded from brand research and catalog analysis.

## Migration Steps

### For New Deployments
1. Use `separate_index_ingestion.py` directly
2. Create both indexes with proper names
3. Ingest catalog into both indexes

### Index Naming Convention
```
{brand}-dense-v2    # Dense embeddings
{brand}-sparse-v2   # Sparse embeddings
```

## Usage Examples

### 1. Ingestion
```python
from src.ingestion.separate_index_ingestion import SeparateIndexIngestion

ingestion = SeparateIndexIngestion(
    brand_domain="specialized.com",
    dense_index_name="specialized-dense-v2",
    sparse_index_name="specialized-sparse-v2"
)

# Create indexes
await ingestion.create_indexes()

# Ingest catalog
stats = await ingestion.ingest_catalog("catalog.json")
```

### 2. Search
```python
from src.search.separate_index_hybrid_search import SeparateIndexHybridSearch

search = SeparateIndexHybridSearch(
    brand_domain="specialized.com",
    dense_index_name="specialized-dense-v2",
    sparse_index_name="specialized-sparse-v2"
)

# Hybrid search
results = await search.search(
    query="lightweight carbon road bike",
    search_mode="hybrid",
    rerank=True
)

# Sparse-only search (exact matches)
results = await search.search(
    query="Tarmac SL7",
    search_mode="sparse"
)
```

## Performance Expectations

### Search Latency
- Dense-only: ~100-150ms
- Sparse-only: ~50-100ms
- Hybrid + reranking: ~200-300ms

### Accuracy Improvements
- Exact product searches: 70%+ improvement with sparse
- Natural language: 30%+ improvement with hybrid
- Technical queries: 50%+ improvement with hybrid

## Next Steps

1. **Immediate**: Update Pinecone client
2. **Next**: Implement query expansion
3. **Future**: Add preference learning
4. **Long-term**: Multi-language support