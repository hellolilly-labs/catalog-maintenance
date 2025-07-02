# Descriptor Generation Guide

## Overview

The catalog maintenance system now supports **pre-generation and caching of product descriptors** directly in `products.json`. This approach:

1. **Uses Brand Research**: Leverages voice, style, and positioning research for enhanced quality
2. **Caches Descriptors**: Stores descriptors in products.json for reuse
3. **Quality-Based Regeneration**: Automatically regenerates low-quality descriptors
4. **Efficient Ingestion**: Uses cached descriptors during vector ingestion

## Architecture

### Components

1. **ResearchAwareDescriptorGenerator**
   - Loads brand research data (voice, style, positioning)
   - Generates descriptors using LLM with brand context
   - Assesses descriptor quality (0.0-1.0 scale)
   - Caches descriptors in products.json

2. **Quality Assessment**
   - Length check (optimal: 50-150 words)
   - Product name and category mentions
   - Feature coverage
   - Natural language quality
   - Complete sentence structure

3. **Brand Research Integration**
   - Voice attributes (tone, style, formality)
   - Messaging pillars
   - Design philosophy
   - Target audience insights

## Workflow

### Step 1: Run Brand Research (if not already done)

```bash
python brand_researcher.py specialized.com --all
```

This generates:
- `foundation_research_output.json`
- `voice_messaging_research_output.json`
- `product_style_research_output.json`
- And other research outputs

### Step 2: Pre-generate Descriptors

```bash
# Basic usage
python pre_generate_descriptors.py specialized.com accounts/specialized.com/products.json

# Force regeneration of all descriptors
python pre_generate_descriptors.py --force specialized.com products.json

# Set custom quality threshold (default: 0.8)
python pre_generate_descriptors.py --quality-threshold 0.9 specialized.com products.json
```

### Step 3: Review Generated Descriptors

After generation, your `products.json` will include:

```json
{
  "products": [
    {
      "id": "bike-001",
      "name": "Specialized Tarmac SL7",
      "price": 4500,
      "descriptor": "The Specialized Tarmac SL7 represents the pinnacle of road cycling performance...",
      "descriptor_metadata": {
        "generated_at": "2024-01-20T10:30:00",
        "quality_score": 0.92,
        "generator_version": "2.0",
        "uses_brand_research": true
      }
    }
  ]
}
```

### Step 4: Ingest with Cached Descriptors

```bash
python ingest_products_separate_indexes.py specialized.com accounts/specialized.com/products.json --create-indexes
```

The ingestion system automatically:
1. Detects if brand research exists
2. Uses ResearchAwareDescriptorGenerator if research is found
3. Leverages cached descriptors from products.json
4. Only regenerates if descriptor is missing or below quality threshold

## Quality Thresholds

| Score | Quality Level | Action |
|-------|--------------|--------|
| 0.9-1.0 | Excellent | Keep cached |
| 0.8-0.89 | Good | Keep cached (default threshold) |
| 0.7-0.79 | Fair | Regenerate if threshold > 0.7 |
| < 0.7 | Poor | Always regenerate |

## Benefits

1. **Consistency**: All descriptors follow brand voice and style
2. **Performance**: No need to generate descriptors during ingestion
3. **Quality Control**: Automatic regeneration of poor descriptors
4. **Transparency**: Quality scores and metadata tracked
5. **Flexibility**: Can force regeneration or adjust thresholds

## Example Descriptor Enhancement

### Before (No Brand Research)
```
"A high-performance road bike with carbon frame and electronic shifting."
```

### After (With Brand Research)
```
"The Tarmac SL7 embodies Specialized's relentless pursuit of speed and efficiency. 
This pro-tour proven machine features our lightest ever frame construction paired 
with precision electronic shifting. Built for riders who demand uncompromising 
performance, it delivers explosive acceleration and confident handling whether 
you're attacking climbs or descending at speed. A true race bike that transforms 
every ride into a performance breakthrough."
```

## API Usage

```python
from src.catalog.research_aware_descriptor_generator import ResearchAwareDescriptorGenerator

# Initialize generator
generator = ResearchAwareDescriptorGenerator("specialized.com")

# Process and cache descriptors
enhanced_products, filter_labels = generator.process_and_cache_descriptors(
    "products.json",
    force_regenerate=False,
    quality_threshold=0.8
)
```

## Troubleshooting

### No Brand Research Found
If you see this warning, run brand research first:
```bash
python brand_researcher.py [brand_domain] --all
```

### Low Quality Scores
Check that:
1. Product data includes name, category, features
2. Brand research completed successfully
3. LLM API keys are configured

### Descriptors Not Caching
Ensure:
1. Write permissions for products.json
2. Valid JSON structure
3. Sufficient product metadata

## Next Steps

After pre-generating descriptors:
1. Review generated content for accuracy
2. Run ingestion pipeline
3. Test search quality
4. Monitor RAG performance

The pre-generated descriptors significantly improve:
- Search relevance
- Voice AI response quality
- Brand consistency
- Customer engagement