# Ingestion Pipeline Guide

## Current Implementation Status

### ✅ Completed Components
1. **Universal Product Processor** - Handles any brand/product format
2. **Unified Descriptor Generator** - Creates RAG-optimized descriptions with 4 components
3. **Sparse Embeddings Generator** - BM25-style keyword matching
4. **Separate Index Architecture** - Dense and sparse indexes
5. **Dynamic Metadata Schema** - Brand-agnostic filter extraction

### 🔧 **Consolidated Scripts**
- **Main Ingestor**: `ingest_products_separate_indexes.py` (modern separate indexes approach)
- **Deprecated**: `ingest_products_enhanced.py` (removed - legacy single index)
- **Deprecated**: `ingest_product_catalog.py` (consolidated into main script)

### ⚠️ Important Notes Before Running

1. **Vocabulary Building**: The sparse embeddings generator needs to build vocabulary from your entire catalog first. This is a one-time process per brand.

2. **Environment Variables**: Ensure you have:
   ```bash
   export PINECONE_API_KEY="your-api-key"
   export OPENAI_API_KEY="your-api-key"  # If using OpenAI embeddings
   ```

3. **Index Names**: We use the pattern:
   - Dense: `{brand}-dense-v2`
   - Sparse: `{brand}-sparse-v2`

## Step-by-Step Ingestion Process

### 0. Populate descriptors

```bash
python pre_generate_descriptors.py specialized.com
```

### 1. Analyze Filters Only (Optional - Fast Analysis)

```bash
python ingest_products_separate_indexes.py specialized.com catalog.json --filters-only
```

This will:
- Analyze catalog structure and extract available filters
- Generate filter dictionary and human-readable summary
- NO vector processing or Pinecone interaction
- Fast way to understand your catalog structure

### 2. Preview Your Catalog (Recommended Before Full Ingestion)

```bash
python ingest_products_separate_indexes.py specialized.com catalog.json --preview
```

This will:
- Show how products will be processed with UnifiedDescriptorGenerator
- Display all 4 components: descriptor, search terms, key points, voice summary
- Show extracted metadata fields and filter labels
- Generate and save filter summary
- NO data sent to Pinecone

### 3. Verify Descriptor Quality

```bash
python verify_descriptor_quality.py flexfits.com
```

This will analyze:
- Descriptor length (optimal: 50-200 words)
- Content completeness
- Voice optimization
- Search keyword coverage

### 4. Create Indexes and Ingest

```bash
# Create indexes if they don't exist and ingest
python ingest_product_catalog.py flexfits.com --create-indexes

# Or if indexes already exist
python ingest_product_catalog.py flexfits.com
```

### 5. Force Full Re-ingestion (if needed)

```bash
python ingest_product_catalog.py flexfits.com --force
```

## Expected Catalog Format

The script handles various JSON formats:

### Format 1: Array of products
```json
[
  {"name": "Product 1", "price": 100, ...},
  {"name": "Product 2", "price": 200, ...}
]
```

### Format 2: Object with products key
```json
{
  "products": [
    {"name": "Product 1", "price": 100, ...},
    {"name": "Product 2", "price": 200, ...}
  ]
}
```

### Format 3: Single product
```json
{"name": "Product 1", "price": 100, ...}
```

## What Happens During Ingestion

1. **Product Processing**
   - Extracts universal fields (name, price, category, etc.)
   - Generates unique IDs
   - Creates filter metadata

2. **Descriptor Enhancement**
   - Generates voice-optimized descriptions
   - Creates search keywords
   - Builds key selling points

3. **Vocabulary Building** (first time only)
   - Analyzes all products for important terms
   - Creates sparse embedding vocabulary
   - Saves to `accounts/{brand}/sparse_vocabulary.json`

4. **Index Population**
   - Dense index: Text sent for server-side embedding
   - Sparse index: BM25-style keyword vectors
   - Both indexes share same product IDs

5. **Change Detection**
   - Tracks product hashes
   - Only updates changed products
   - Handles additions, updates, deletions

## Output Files

After ingestion, you'll have:

```
accounts/{brand_domain}/
├── filter_dictionary.json      # Extracted filter labels (JSON)
├── filter_summary.md           # Human-readable filter summary (NEW)
├── sparse_vocabulary.json      # Sparse embedding vocabulary
└── catalog_insights.json       # Catalog analysis results

data/sync_state/
└── {brand_domain}_separate_indexes.json  # Change tracking
```

## Troubleshooting

### Issue: "No vocabulary found"
**Solution**: The sparse embeddings generator needs to build vocabulary first. This happens automatically on first run.

### Issue: "Index not found"
**Solution**: Use `--create-indexes` flag to create indexes automatically.

### Issue: "Products not updating"
**Solution**: Use `--force` flag to force re-ingestion of all products.

### Issue: "Memory error with large catalogs"
**Solution**: Reduce `--batch-size` (default is 100).

## Next Steps After Ingestion

1. **Test Search**:
   ```python
   from src.search.separate_index_hybrid_search import SeparateIndexHybridSearch
   
   search = SeparateIndexHybridSearch(
       brand_domain="specialized.com",
       dense_index_name="specialized-dense-v2",
       sparse_index_name="specialized-sparse-v2"
   )
   
   results = await search.search("carbon road bike", top_k=10)
   ```

2. **Monitor Performance**:
   - Check index statistics in Pinecone console
   - Review search latency
   - Analyze result quality

3. **Optimize**:
   - Adjust descriptor generation if needed
   - Fine-tune search weights
   - Add more metadata fields

## Ready to Test?

The implementation is complete enough for testing! Start with:

```bash
# Preview first
python ingest_products_separate_indexes.py specialized.com catalog.json --preview

# Then ingest
python ingest_products_separate_indexes.py specialized.com catalog.json --create-indexes
```

## Testing the search functionality

```bash
python test_search_comparison.py --account flexfits.com --ingest-baseline