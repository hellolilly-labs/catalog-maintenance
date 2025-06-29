# Enhanced RAG Ingestion System

## Overview

The enhanced ingestion system provides universal product processing and intelligent Pinecone ingestion for any brand or product type. It features:

- 🌐 **Universal Processing**: Works with fashion, beauty, sports, jewelry, electronics - any product type
- 🎯 **Voice Optimization**: Generates descriptors optimized for voice AI conversations
- 🔍 **Hybrid Search**: Combines dense embeddings (semantic) with sparse embeddings (keywords)
- 📊 **Smart Filters**: Automatically extracts and indexes product attributes
- 🔄 **Incremental Updates**: Only processes changed products for efficiency

## Quick Start

```bash
# Basic usage
python ingest_products_enhanced.py specialized.com data/products.json --index specialized-llama-2048

# Preview processing without ingesting
python ingest_products_enhanced.py balenciaga.com catalog.json --index balenciaga-llama-2048 --preview

# Force update all products
python ingest_products_enhanced.py sundayriley.com products.json --index sundayriley-llama-2048 --force

# Test search after ingestion
python ingest_products_enhanced.py gucci.com catalog.json --index gucci-llama-2048 --test
```

## How It Works

### 1. Universal Product Processing

The `UniversalProductProcessor` automatically detects and extracts relevant fields from any product structure:

```python
# Input: Raw product (any format)
{
    "productId": "BAG-789",
    "title": "Le Cagole Shoulder Bag",
    "brand_name": "Balenciaga",
    "price": 2950,
    ...
}

# Output: Standardized format with enhancements
{
    "id": "BAG-789",
    "universal_fields": {
        "identifier": "BAG-789",
        "name": "Le Cagole Shoulder Bag",
        "brand": "Balenciaga",
        "price": 2950.0,
        ...
    },
    "enhanced_descriptor": "Balenciaga Le Cagole Shoulder Bag. Iconic shoulder bag...",
    "voice_summary": "Balenciaga Le Cagole Shoulder Bag. Iconic design...",
    "search_keywords": ["balenciaga", "le cagole", "shoulder bag", ...],
    "key_selling_points": ["Iconic design", "Premium leather", ...],
    "filter_metadata": {
        "brand": "Balenciaga",
        "category": "handbags",
        "price_range": "luxury",
        ...
    }
}
```

### 2. Hybrid Embeddings

Each product gets both dense and sparse embeddings:

- **Dense Embeddings**: Semantic understanding ("luxury handbag" ≈ "designer purse")
- **Sparse Embeddings**: Exact matching (brand names, model numbers, specific terms)

### 3. Smart Filter Extraction

The system automatically discovers and extracts filters from your catalog:

```
category (categorical):
  Categories: 5
  Sample: handbags, shoes, accessories, clothing, jewelry

price (numeric_range):
  Range: 50 to 5000

style (multi_select):
  Options: 12
  Top 5: contemporary, classic, edgy, minimalist, bold
```

### 4. Incremental Updates

The system tracks product changes using content hashing:

1. New products → Added to index
2. Changed products → Updated in index
3. Deleted products → Removed from index
4. Unchanged products → Skipped (saves time)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Product Catalog (JSON)                      │
└──────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Universal Product Processor                     │
│  • Field detection      • Voice optimization                 │
│  • Keyword extraction   • Filter metadata                    │
└──────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               Pinecone Ingestion System                      │
│  • Change detection     • Hybrid embeddings                  │
│  • Batch processing     • Incremental updates                │
└──────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pinecone Index                            │
│  • Dense vectors        • Sparse vectors                     │
│  • Rich metadata        • Filter indices                     │
└─────────────────────────────────────────────────────────────┘
```

## Product Format Examples

The system handles various product formats automatically:

### Fashion/Retail
```json
{
    "productId": "SHOE-123",
    "name": "Classic Sneakers",
    "brand": "Nike",
    "price": 120,
    "categories": ["Footwear", "Sneakers"],
    "colors": ["White", "Black"],
    "sizes": ["7", "8", "9", "10", "11"]
}
```

### Beauty/Skincare
```json
{
    "sku": "SERUM-001",
    "product_name": "Vitamin C Serum",
    "vendor": "Brand Name",
    "current_price": 65.00,
    "category": "Skincare/Serums",
    "key_ingredients": ["Vitamin C", "Hyaluronic Acid"],
    "skin_types": ["All Types"]
}
```

### Jewelry
```json
{
    "item_id": "RING-456",
    "title": "Diamond Solitaire Ring",
    "price": {"amount": 3500, "currency": "USD"},
    "materials": {"metal": "18k Gold", "stone": "Diamond"},
    "specifications": {"carat": 1.0, "clarity": "VS1"}
}
```

## Configuration

### Environment Variables

```bash
# Required
export PINECONE_API_KEY="your-api-key"

# Optional
export LANGFUSE_PUBLIC_KEY="your-key"
export LANGFUSE_SECRET_KEY="your-secret"
```

### Index Requirements

Your Pinecone index should be configured with:
- Dimension: 2048 (for llama-text-embed-v2)
- Metric: Cosine
- Pod type: s1 or higher for hybrid search

## Advanced Usage

### Custom Batch Size

For large catalogs, adjust the batch size:

```bash
python ingest_products_enhanced.py brand.com catalog.json --index brand-index --batch-size 100
```

### Multiple Namespaces

Separate different content types:

```bash
# Products
python ingest_products_enhanced.py brand.com products.json --index brand-index --namespace products

# Information/Docs
python ingest_products_enhanced.py brand.com docs.json --index brand-index --namespace information
```

### Monitoring

The system provides detailed statistics:

```
📊 Ingestion Results:
  Added: 245
  Updated: 18
  Deleted: 3
  Errors: 0
  Duration: 45.23s
```

## Troubleshooting

### Common Issues

1. **"No products found in catalog"**
   - Check your JSON structure
   - Products should be in a list or under a key like "products", "items", etc.

2. **"Embedding failures"**
   - Ensure text isn't too long (max ~8000 tokens)
   - Check for special characters that might cause issues

3. **"Slow ingestion"**
   - Reduce batch size for better progress visibility
   - Use incremental updates (avoid --force unless necessary)

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
python ingest_products_enhanced.py ...
```

## Best Practices

1. **Run filter analysis first**: Use `ingest_product_catalog.py` to analyze filters before ingestion
2. **Test with preview**: Always use `--preview` first to check processing
3. **Monitor changes**: Review the change detection results before large updates
4. **Version control**: Keep your catalog files in version control
5. **Regular syncs**: Set up scheduled ingestion for automatic updates

## Next Steps

After successful ingestion:

1. **Test search quality** with the `--test` flag
2. **Update voice assistant** prompts with filter dictionaries
3. **Monitor search performance** in your application
4. **Iterate on descriptors** based on user feedback

For more information, see the [main documentation](../../README.md).