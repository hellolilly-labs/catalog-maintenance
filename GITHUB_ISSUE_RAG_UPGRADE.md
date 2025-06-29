# GitHub Issue: Upgrade RAG Ingestion System for Universal Product Discovery

## 🎯 Objective
Upgrade the current Pinecone RAG ingestion system to support universal product discovery across all brand types (fashion, beauty, cycling, jewelry, etc.) with enhanced metadata, hybrid search capabilities, and automatic synchronization.

## 📋 Background
The current `pinecone_setup.py` ingestion system works but lacks:
- Filter metadata integration for intelligent query understanding
- Hybrid search (dense + sparse embeddings) for optimal accuracy
- Automatic synchronization when catalogs change
- Integration with our catalog filter analysis system
- Voice-optimized descriptors for natural conversation

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Universal RAG Ingestion Pipeline             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Catalog Processing                                      │
│     ├─► Brand-agnostic filter extraction                  │
│     ├─► Enhanced descriptor generation                     │
│     └─► Change detection & versioning                     │
│                                                             │
│  2. Hybrid Embeddings                                      │
│     ├─► Dense: Semantic understanding (llama-text-embed)   │
│     ├─► Sparse: Keyword precision (BM25/SPLADE)          │
│     └─► Metadata: Structured filtering                    │
│                                                             │
│  3. Pinecone Integration                                   │
│     ├─► Incremental updates (add/update/delete)           │
│     ├─► Multi-namespace support                           │
│     └─► Version control & rollback                        │
│                                                             │
│  4. System Synchronization                                 │
│     ├─► Langfuse prompt updates                           │
│     ├─► Query optimizer cache invalidation                │
│     └─► Performance monitoring                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## ✅ Implementation Tasks

### Phase 1: Enhanced Metadata & Universal Processing ⏱️ Week 1 ✅ COMPLETE
- [x] Create `UniversalProductProcessor` class
  - [x] Brand-agnostic field detection
  - [x] Dynamic descriptor generation
  - [x] Voice optimization logic
- [x] Integrate `CatalogFilterAnalyzer`
  - [x] Extract filters during ingestion
  - [x] Generate filter dictionaries
  - [x] Store filter metadata with products
- [x] Implement `EnhancedDescriptorGenerator` integration
  - [x] Voice-optimized descriptions
  - [x] Key selling points extraction
  - [x] Natural language use cases
- [x] Update metadata structure
  - [x] Add filter fields
  - [x] Include search keywords
  - [x] Store both original and enhanced content

**Phase 1 Deliverables:**
- `src/ingestion/universal_product_processor.py` - Universal product processing
- `src/ingestion/pinecone_ingestion.py` - Enhanced ingestion system
- `ingest_products_enhanced.py` - CLI tool
- `test_universal_processor.py` - Test suite
- Full documentation in `src/ingestion/README.md`

### Phase 2: Hybrid Search Implementation ⏱️ Week 2 🚧 IN PROGRESS
- [x] Implement sparse embedding generation
  - [x] BM25-based token scoring
  - [x] Important term extraction
  - [x] Brand/model name prioritization
- [x] Update Pinecone record structure
  - [x] Add sparse_values field
  - [x] Maintain dense embeddings
  - [x] Enrich metadata fields
- [x] Create hybrid search logic
  - [x] Weighted combination (dynamic, not fixed 80/20)
  - [x] Dynamic weight adjustment
  - [ ] Performance testing

**Phase 2 Progress:**
- Created `SparseEmbeddingGenerator` with BM25-inspired scoring
- Integrated sparse embeddings into Pinecone ingestion pipeline
- Built `HybridSearchEngine` with dynamic weight determination
- Added `HybridQueryOptimizer` for intent-based query optimization
- Created comprehensive test suite for hybrid search scenarios
- Full documentation in `docs/PHASE2_HYBRID_SEARCH.md`

### Phase 3: Automatic Synchronization ⏱️ Week 3
- [ ] Implement change detection
  - [ ] Content hashing for products
  - [ ] Timestamp tracking
  - [ ] Diff generation
- [ ] Build incremental update system
  - [ ] Add new products
  - [ ] Update changed products
  - [ ] Remove deleted products
  - [ ] Batch optimization
- [ ] Create sync triggers
  - [ ] File system watcher
  - [ ] Scheduled jobs
  - [ ] Manual trigger API
  - [ ] Webhook support

### Phase 4: System Integration ⏱️ Week 4
- [ ] Langfuse integration
  - [ ] Update filter dictionaries in prompts
  - [ ] Store brand terminology
  - [ ] Version control
- [ ] Cache management
  - [ ] Invalidate query optimizer caches
  - [ ] Update filter summaries
  - [ ] Refresh enhanced descriptors
- [ ] Monitoring & observability
  - [ ] Ingestion metrics
  - [ ] Search performance tracking
  - [ ] Error alerting
  - [ ] Quality metrics

## 📊 Success Metrics
- **Search Accuracy**: 30-50% improvement in product discovery relevance
- **Brand Precision**: 95%+ accuracy for brand/model name queries
- **Processing Speed**: < 5 minutes for 10,000 product catalog
- **Sync Latency**: < 1 minute from catalog change to searchable
- **Voice Quality**: Natural, conversational product descriptions

## 🔧 Technical Specifications

### Enhanced Metadata Structure
```python
{
    "id": "product_123",
    "values": dense_embedding,  # From enhanced descriptor
    "sparse_values": {          # Keyword importance
        "indices": [token_ids],
        "values": [weights]
    },
    "metadata": {
        # Universal fields
        "brand": "Balenciaga",
        "name": "Le Cagole Shoulder Bag",
        "category": "handbags",
        "price": 2950,
        
        # Dynamic fields (brand-specific)
        "filters": {
            "style": ["edgy", "contemporary"],
            "size": "medium",
            "material": "leather"
        },
        
        # Voice optimization
        "enhanced_descriptor": "...",
        "key_selling_points": [...],
        "search_keywords": [...]
    }
}
```

### Universal Product Processing
```python
# Works for ANY product type
product = {
    "name": "Product Name",
    "brand": "Brand",
    "price": 100,
    # ... any other fields
}

processor = UniversalProductProcessor(brand_domain)
enhanced = processor.process(product)
# Automatically extracts relevant fields and creates optimized content
```

## 🚀 Rollout Plan
1. **Development**: Build and test with 3 diverse brands (fashion, beauty, sports)
2. **Staging**: Deploy to test environment, validate with real data
3. **Production**: Gradual rollout with monitoring
4. **Optimization**: Tune weights and parameters based on usage

## 📝 Notes
- Must maintain backward compatibility during migration
- All processing must be brand-agnostic (no hardcoded logic)
- Voice optimization is critical for the AI sales agent use case
- Performance at scale is important (some brands have 50k+ products)

## 🔗 Related
- #15 Voice Assistant RAG Integration
- #23 Query Optimization Agent
- #31 Catalog Filter Analysis

---

**Priority**: High
**Type**: Enhancement
**Component**: RAG/Search
**Milestone**: Q1 2025