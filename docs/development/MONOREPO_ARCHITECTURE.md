# Monorepo Architecture Guide

This document describes the architecture and organization of the Liddy AI Platform monorepo.

## Overview

The Liddy platform is organized as a Python monorepo with three main packages:

```
catalog-maintenance/
├── packages/
│   ├── liddy/              # Core shared functionality
│   ├── liddy_intelligence/ # AI-powered brand research & catalog processing
│   └── liddy_voice/        # Real-time voice assistant
├── run/                    # CLI runner scripts
├── deployments/            # Deployment configurations
├── scripts/               # Automation & workflow scripts
├── tests/                 # Test suites
└── docs/                  # Documentation
```

## Package Architecture

### 1. `liddy` - Core Package

The foundation package providing shared functionality:

```
liddy/
├── __init__.py
├── config.py              # Centralized configuration
├── storage.py             # Storage abstraction (local/GCS)
├── account_manager.py     # Account configuration
├── prompt_manager.py      # Prompt management
├── models/
│   ├── product.py         # Product data model
│   └── product_manager.py # Product catalog management
└── search/
    ├── base.py           # Search interfaces
    ├── pinecone.py       # Pinecone integration
    └── service.py        # Unified search service
```

**Key Features:**
- Storage abstraction supporting local filesystem and Google Cloud Storage
- Centralized configuration management with environment variables
- Product data models and catalog management
- Search infrastructure with Pinecone vector database

### 2. `liddy_intelligence` - Intelligence Engine

AI-powered brand research and catalog processing:

```
liddy_intelligence/
├── research/              # 8-phase brand research pipeline
│   ├── base_researcher.py # Base class with quality evaluation
│   ├── foundation_research.py
│   ├── market_positioning_research.py
│   ├── product_style_research.py
│   ├── customer_cultural_research.py
│   ├── voice_messaging_research.py
│   ├── interview_synthesis_research.py
│   ├── linearity_analysis_research.py
│   └── research_integration.py
├── llm/                   # Multi-provider LLM integration
│   ├── simple_factory.py  # Factory pattern
│   ├── openai_service.py
│   ├── anthropic_service.py
│   └── gemini_service.py
├── ingestion/            # Product catalog ingestion
│   ├── core/             # Core modules
│   │   ├── universal_product_processor.py
│   │   ├── separate_index_ingestion.py
│   │   └── sparse_embeddings.py
│   └── scripts/          # CLI tools
│       ├── ingest_catalog.py
│       └── pre_generate_descriptors.py
├── agents/               # Specialized AI agents
├── catalog/              # Catalog processing
└── workflow/             # Workflow management
```

**Key Features:**
- 8-phase sequential brand research pipeline with quality evaluation
- Multi-provider LLM support (OpenAI, Anthropic, Gemini)
- Universal product processing for any catalog format
- Hybrid search with dense and sparse embeddings
- Intelligent agents for specialized tasks

### 3. `liddy_voice` - Voice Assistant

Real-time voice-enabled AI assistant:

```
liddy_voice/
├── sample_assistant.py    # Main voice assistant
├── search_service.py      # Voice-optimized search
├── session_state_manager.py # Conversation tracking
├── model.py              # Response generation
├── prompt_manager.py     # Dynamic prompts
└── rag.py               # RAG integration
```

**Key Features:**
- Real-time voice interaction with AssemblyAI
- WebSocket-based communication
- Session state management
- RAG-powered responses
- Brand-aware conversational AI

## Import Patterns

### Cross-Package Imports
Use absolute imports for dependencies between packages:

```python
# In liddy_intelligence
from liddy.storage import get_account_storage_provider
from liddy.models.product import Product

# In liddy_voice
from liddy.search import SearchService
from liddy_intelligence.agents import CatalogFilterAnalyzer
```

### Within-Package Imports
Use relative imports within the same package:

```python
# In liddy_intelligence/research/foundation_research.py
from .base_researcher import BaseResearcher
from ..llm.simple_factory import LLMFactory
```

### Runner Scripts
Runner scripts add packages to path:

```python
# In run/brand_research.py
sys.path.insert(0, os.path.join(project_root, 'packages'))
from liddy_intelligence.research.brand_researcher import main
```

## Key Design Patterns

### 1. Factory Pattern for LLMs
```python
factory = LLMFactory()
service = factory.get_service("openai")  # or "anthropic", "gemini"
```

### 2. Storage Abstraction
```python
storage = get_account_storage_provider()  # Auto-detects local vs GCS
content = await storage.read_file(account, file_path)
```

### 3. Quality Evaluation Loop
- Each research phase includes quality evaluation
- Automatic retry with feedback (max 3 attempts)
- 8.0+ quality threshold required

### 4. Separate Index Architecture
- Dense index for semantic search
- Sparse index for keyword precision
- Consistent ID linkage between indexes

## Configuration

### Environment Variables
Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=your-key
PINECONE_API_KEY=your-key

# Optional
ANTHROPIC_API_KEY=your-key
GEMINI_API_KEY=your-key
LANGFUSE_PUBLIC_KEY=your-key
LANGFUSE_SECRET_KEY=your-key
ASSEMBLYAI_API_KEY=your-key

# Storage
STORAGE_TYPE=local  # or 'gcs'
GCS_BUCKET=your-bucket
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

### Settings Access
```python
from liddy.config import get_settings

settings = get_settings()
api_key = settings.openai_api_key
```

## Deployment

### Local Development
```bash
# Install packages in development mode
pip install -e packages/liddy
pip install -e packages/liddy_intelligence
pip install -e packages/liddy_voice

# Run services
python run/brand_research.py specialized.com
python run/ingest_catalog.py specialized.com
python run/voice_main.py
```

### Docker
```bash
# Build from monorepo root
docker build -f deployments/liddy_voice/Dockerfile -t liddy-voice .
```

### Google Cloud Run
```bash
cd deployments/liddy_voice
./gcp/deploy-cloudrun.sh
```

## Testing

### Unit Tests
```bash
pytest tests/core/
pytest tests/intelligence/
pytest tests/voice/
```

### Integration Tests
```bash
python tests/integration/test_monorepo_structure.py
```

## Development Guidelines

1. **Package Selection**: Choose the appropriate package for new features
2. **Import Discipline**: Follow established import patterns
3. **Test Coverage**: Add tests for new functionality
4. **Documentation**: Update package READMEs
5. **Quality Standards**: Maintain 8.0+ quality threshold for AI outputs
6. **Error Handling**: Use comprehensive retry logic with backoff
7. **Logging**: Use structured logging with appropriate levels

## Migration from Legacy Structure

Old code in `src/` contains import redirects. Update imports:

```python
# Old
from src.storage import get_account_storage_provider

# New
from liddy.storage import get_account_storage_provider
```

See `MONOREPO_MIGRATION_COMPLETE.md` for full migration details.