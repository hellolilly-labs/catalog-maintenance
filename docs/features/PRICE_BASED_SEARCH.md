# Price-Based Search Enhancement

## Overview

The price-based search enhancement system enables natural language price queries in product search, with deep integration into voice assistants and catalog management. The system understands various price expressions, industry-specific terminology, and provides contextual price positioning.

## Architecture

### Core Components

1. **IndustryTerminologyResearcher** (`packages/liddy_intelligence/research/industry_terminology_researcher.py`)
   - Researches industry-specific price terminology
   - Identifies brand-specific tier indicators
   - Classifies slang and community terms

2. **PriceStatisticsAnalyzer** (`packages/liddy_intelligence/catalog/price_statistics_analyzer.py`)
   - Analyzes catalog price distributions
   - Detects multi-modal distributions
   - Generates category-specific statistics

3. **PriceDescriptorUpdater** (`packages/liddy_intelligence/catalog/price_descriptor_updater.py`)
   - Updates product descriptors with price information
   - Adds semantic price context
   - Integrates terminology research

4. **Enhanced Search Service** (`packages/liddy/search/service.py`)
   - Extracts price values from queries
   - Normalizes price expressions
   - Matches products by price criteria

## Usage

### Running Terminology Research

```bash
# Research industry terminology for a brand
python run/research_industry_terminology.py specialized.com

# Force refresh existing research
python run/research_industry_terminology.py specialized.com --force-refresh
```

### Analyzing Price Distribution

```bash
# Analyze catalog pricing
python run/analyze_price_distribution.py specialized.com

# View detailed statistics
python run/analyze_price_distribution.py specialized.com --detailed
```

### Updating Descriptors with Prices

```bash
# Update all product descriptors
python run/update_descriptor_prices.py specialized.com

# Check status without updating
python run/update_descriptor_prices.py specialized.com --check-only
```

### Complete Workflow

```bash
# Ensure research runs before descriptor generation
./scripts/ensure_research_before_descriptors.sh specialized.com

# Update descriptors with research integration
./scripts/update_descriptors_with_research.sh specialized.com
```

## Query Examples

### Basic Price Queries
- "bikes under $500"
- "road bikes between $1000 and $2000"
- "premium mountain bikes"
- "budget friendly options"

### Natural Language Variations
- "show me something affordable"
- "what's your most expensive bike?"
- "find me a mid-range road bike"
- "any bikes on sale?"

### Industry-Specific Terms
- "show me S-Works bikes" (premium tier)
- "find Comp level mountain bikes" (mid-tier)
- "base model road bikes" (entry tier)

## Configuration

### Terminology Research Output

The system generates research in markdown format with:
- Price tier terminology with definitions
- Industry slang with usage classifications
- Technical terms and specifications
- Brand-specific naming patterns

### Price Statistics

Analyzes and stores:
- Overall price distribution
- Category-specific statistics
- Multi-modal distribution detection
- Semantic phrase mappings

### Descriptor Enhancement

Adds to product descriptors:
- Current price information
- Sale/discount details
- Price tier context
- Value positioning

## Integration with Voice Assistant

The voice assistant uses price enhancements to:

1. **Understand Price Queries**
   ```python
   # Voice wrapper extracts price context
   price_filter = self._extract_price_filter(processed_query)
   ```

2. **Filter Results**
   ```python
   # Apply price filtering to search results
   if price_filter:
       filtered_results = [r for r in results if price_matches(r, price_filter)]
   ```

3. **Provide Context**
   ```python
   # Include price in summaries
   summary = f"{product.name} - {price_range} - {tier_context}"
   ```

## Content Classification System

### Term Classifications

1. **"safe"** - Terms the AI can use freely
   - Professional terminology
   - Standard industry terms
   - Brand-specific model names

2. **"understand_only"** - Terms the AI understands but avoids
   - Crude slang
   - Informal expressions
   - Context-dependent terms

3. **"offensive"** - Terms the AI never uses
   - Derogatory language
   - Offensive slang
   - Inappropriate content

### Example Classifications

```json
{
  "general_slang": {
    "granny gear": {
      "definition": "Easiest gear ratio for climbing",
      "usage": "safe"
    },
    "DTF": {
      "definition": "Down to f***",
      "usage": "understand_only"
    },
    "caucacity": {
      "definition": "Audacious behavior attributed to white people",
      "usage": "offensive"
    }
  }
}
```

## Testing

### Test Price Enhancement

```bash
# Run comprehensive tests
python run/test_price_enhancement.py specialized.com

# Test specific queries
python run/test_price_enhancement.py specialized.com --query "bikes under 500"
```

### Expected Results

The test script validates:
- Price extraction from queries
- Descriptor price updates
- Search result filtering
- Voice summary generation

## Troubleshooting

### Common Issues

1. **Missing Terminology Research**
   - Run: `python run/research_industry_terminology.py [brand]`
   - The system will auto-run if missing (configurable)

2. **Incorrect Price Tiers**
   - Check price distribution: `python run/analyze_price_distribution.py [brand]`
   - Review multi-modal detection results

3. **Outdated Descriptors**
   - Update prices: `python run/update_descriptor_prices.py [brand]`
   - Force refresh: add `--force-refresh` flag

## Future Enhancements

1. **Dynamic Pricing**
   - Real-time price updates
   - Competitor price tracking
   - Price history integration

2. **Advanced Queries**
   - "bikes with best discount"
   - "price dropped in last week"
   - "compare prices across categories"

3. **Personalization**
   - User budget preferences
   - Price sensitivity scoring
   - Recommendation tuning