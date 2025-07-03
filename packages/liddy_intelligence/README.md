# Liddy Intelligence Package

AI-powered brand research and product catalog intelligence system.

## Overview

The `liddy_intelligence` package provides comprehensive brand intelligence gathering and product catalog processing:

- **Brand Research Pipeline**: 8-phase AI research system
- **Product Catalog Ingestion**: Universal product processing and vector database integration
- **Intelligent Agents**: Specialized AI agents for various tasks
- **LLM Integration**: Multi-provider LLM support (OpenAI, Anthropic, Gemini)

## Installation

```bash
pip install -e packages/liddy_intelligence
```

## Key Features

### 8-Phase Brand Research Pipeline
```python
from liddy_intelligence.research import BrandResearcher

researcher = BrandResearcher()
await researcher.run_research("specialized.com", phase="foundation_research")
```

Phases:
1. Foundation Research - Company history, mission, values
2. Market Positioning - Competitive landscape analysis
3. Product Style - Design language and aesthetics
4. Customer Cultural - Psychology and behavior patterns
5. Voice Messaging - Brand voice and communication style
6. Interview Synthesis - Human insights integration
7. Linearity Analysis - Brand consistency analysis
8. Research Integration - Final synthesis and validation

### Product Catalog Ingestion
```python
from liddy_intelligence.ingestion import SeparateIndexIngestion

ingestor = SeparateIndexIngestion(
    brand_domain="specialized.com",
    dense_index_name="specialized-dense",
    sparse_index_name="specialized-sparse"
)
await ingestor.ingest_catalog()
```

### Intelligent Agents
```python
from liddy_intelligence.agents import CatalogFilterAnalyzer

analyzer = CatalogFilterAnalyzer("specialized.com")
filters = await analyzer.analyze_product_catalog(products)
```

## Architecture

```
liddy_intelligence/
├── research/          # Brand research modules
│   ├── foundation_research.py
│   ├── market_positioning_research.py
│   └── ...
├── ingestion/         # Catalog ingestion
│   ├── core/         # Core modules
│   └── scripts/      # CLI scripts
├── agents/           # AI agents
├── llm/             # LLM providers
├── catalog/         # Catalog processing
└── prompts/         # Prompt management
```

## Command Line Tools

### Brand Research
```bash
# Run specific research phase
python -m liddy_intelligence.research.brand_researcher specialized.com --phase foundation_research

# Run all phases
python -m liddy_intelligence.research.brand_researcher specialized.com --phase all
```

### Catalog Ingestion
```bash
# Ingest product catalog
python -m liddy_intelligence.ingestion.scripts.ingest_catalog specialized.com

# Preview processing
python -m liddy_intelligence.ingestion.scripts.ingest_catalog specialized.com --preview

# Generate descriptors
python -m liddy_intelligence.ingestion.scripts.pre_generate_descriptors specialized.com
```

## Configuration

Set environment variables:
```bash
# Required
OPENAI_API_KEY=your-key
PINECONE_API_KEY=your-key

# Optional
ANTHROPIC_API_KEY=your-key
GEMINI_API_KEY=your-key
LANGFUSE_PUBLIC_KEY=your-key
LANGFUSE_SECRET_KEY=your-key
```

## Dependencies

- openai: GPT models
- anthropic: Claude models (optional)
- google-generativeai: Gemini models (optional)
- pinecone-client: Vector database
- langfuse: Prompt management
- tavily-python: Web search
- rank-bm25: Sparse embeddings

## Development

This package contains the core AI intelligence functionality.
When adding new research phases or agents, follow the established patterns.