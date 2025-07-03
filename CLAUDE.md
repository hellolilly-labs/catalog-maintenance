# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Brand Intelligence & Catalog Maintenance System** organized as a Python monorepo. The system orchestrates an 8-phase AI-powered brand research pipeline, performs comprehensive brand intelligence gathering, product catalog ingestion, and knowledge base creation to enable AI-powered brand-aware customer service agents.

**Core Technology**: Python 3.12+ with async/await patterns, LLM integration (OpenAI, Anthropic, Gemini), cloud storage (Google Cloud, Pinecone vector DB), and real-time voice capabilities (AssemblyAI).

## Monorepo Structure

```
catalog-maintenance/
├── packages/
│   ├── liddy/              # Core shared functionality
│   ├── liddy_intelligence/ # Brand research & catalog processing
│   └── liddy_voice/        # Voice assistant capabilities
├── run/                    # Runner scripts for CLI usage
├── deployments/            # Deployment configurations
├── scripts/               # Automation and workflow scripts
├── tests/                 # Test suites
└── src/                   # Legacy code (being migrated)
```

## Common Commands

### Development Environment
```bash
# Activate virtual environment
source activate_venv.sh

# Install all packages in development mode
./scripts/setup_dev.sh

# Install dependencies
pip install -r requirements.txt
```

### Brand Research Pipeline Commands
```bash
# Master control script for all operations
./scripts/brand_manager.sh onboard specialized.com
./scripts/brand_manager.sh status specialized.com
./scripts/brand_manager.sh next-step specialized.com
./scripts/brand_manager.sh resume specialized.com

# Individual research phases
python run/brand_research.py specialized.com --phase foundation_research
python run/brand_research.py specialized.com --phase all

# Batch operations
./scripts/batch_operations.sh --operation status-check --discover
./scripts/brand_manager.sh dashboard
```

### Catalog Ingestion Commands
```bash
# Ingest product catalog
python run/ingest_catalog.py specialized.com

# Preview ingestion
python run/ingest_catalog.py specialized.com --preview

# Generate descriptors
python run/generate_descriptors.py specialized.com
```

### Voice Assistant Commands
```bash
# Run voice assistant
python run/voice_main.py

# Test voice search
python run/test_voice_search.py specialized.com
```

### Testing & Validation
```bash
# Run test suites
pytest tests/core/
pytest tests/intelligence/
pytest tests/voice/

# Run specific test files
python tests/intelligence/test_unified_ingestion_core.py
```

## Architecture Overview

### Package Architecture

**liddy** (Core Package):
- Storage abstraction (local/GCS)
- Data models (Product, Brand)
- Configuration management
- Shared utilities

**liddy_intelligence** (Intelligence Package):
- 8-phase brand research pipeline
- Product catalog ingestion
- LLM integration (multi-provider)
- Intelligent agents

**liddy_voice** (Voice Package):
- Real-time voice assistant
- RAG-powered search
- Session management
- WebSocket API

### Core Research Pipeline (8 Phases)
The system follows a sequential 8-phase brand intelligence pipeline:

1. **Foundation Research** (`liddy_intelligence.research.foundation_research`) - Company history, mission, values
2. **Market Positioning** (`liddy_intelligence.research.market_positioning_research`) - Competitive landscape analysis  
3. **Product Style** (`liddy_intelligence.research.product_style_research`) - Design language, aesthetics
4. **Customer Cultural** (`liddy_intelligence.research.customer_cultural_research`) - Psychology, behavior patterns
5. **Voice Messaging** (`liddy_intelligence.research.voice_messaging_research`) - Brand voice and communication style
6. **Interview Synthesis** (`liddy_intelligence.research.interview_synthesis_research`) - Human insights integration
7. **Linearity Analysis** (`liddy_intelligence.research.linearity_analysis_research`) - Brand consistency analysis
8. **Research Integration** (`liddy_intelligence.research.research_integration`) - Final synthesis and validation

### Key Architectural Components

**LLM Factory Pattern** (`liddy_intelligence.llm`):
- `simple_factory.py` - Model-agnostic LLM service factory
- `anthropic_service.py`, `openai_service.py`, `gemini_service.py` - Provider implementations
- `prompt_manager.py` - Langfuse integration for prompt management

**Research Framework** (`liddy_intelligence.research`):
- `base_researcher.py` - Base class with integrated quality evaluation (8.0+ threshold)
- Quality evaluation with feedback loops and automatic retry (max 3 attempts)
- Phase-specific researchers inherit from BaseResearcher

**Ingestion System** (`liddy_intelligence.ingestion`):
- `core/` - Core ingestion modules (UniversalProductProcessor, SeparateIndexIngestion)
- `scripts/` - CLI tools for catalog processing
- Supports separate dense/sparse indexes for hybrid search

**Workflow State Management** (`liddy_intelligence.workflow`):
- `workflow_state_manager.py` - Tracks pipeline progress and enables resume functionality
- `research_phase_tracker.py` - Individual phase progress tracking

**Storage & Configuration**:
- `liddy.storage` - Account-based storage provider (Google Cloud Storage)
- `liddy.config` - Centralized configuration with environment awareness
- `local/account_storage/` - File-based research data storage

### Quality Evaluation System
The system includes an integrated LLM-based quality evaluation framework:
- Quality scores on 0.0-10.0 scale with configurable thresholds
- 5 criteria assessment: accuracy, completeness, consistency, authenticity, actionability
- Automatic feedback loops with improvement suggestions
- Quality metadata stored in `research_metadata.json` files alongside research content

### Important Integration Patterns

**Package Imports**:
- Use absolute imports for cross-package dependencies: `from liddy.storage import get_account_storage_provider`
- Use relative imports within packages: `from ..research.base_researcher import BaseResearcher`
- Runner scripts add packages to path for CLI usage

**Langfuse Integration**: 
- Uses chat template pattern for prompt management
- Prompts stored with naming convention: `liddy/catalog/quality/{phase}_evaluator`
- Automatic fallback to default prompts if Langfuse unavailable

**KISS Principle**: 
- Stateful applications prefer in-memory solutions over external caches
- Quality results stored in existing metadata files rather than separate systems
- Simple, predictable data flows over complex optimizations

**Error Handling**:
- Comprehensive retry logic with exponential backoff
- Graceful fallbacks for optional services (Anthropic, Gemini)
- Progress tracking enables resume from interruption points

## Script-Based Workflow Management

The `scripts/` directory contains comprehensive workflow automation:

- **`brand_manager.sh`** - Master control script for single-brand operations
- **`batch_operations.sh`** - Multi-brand parallel processing
- **`brand_onboarding.sh`** - Complete zero-to-AI-agent pipeline setup
- **`brand_status.sh`** - Health monitoring and alerts
- **`brand_maintenance.sh`** - Smart refresh and maintenance routines

These scripts provide intelligent workflow state management with "next step" recommendations and resume capabilities.

## Key Project Context

**Monorepo Migration**: The project has been restructured into a monorepo with namespace packages. Legacy code in `src/` is being migrated.

**Storage Strategy**: Account-based storage under `accounts/{brand}/` with:
- `research/{phase}/research.md` - Research content
- `research/{phase}/research_metadata.json` - Quality metadata
- `products.json` - Product catalog
- `filter_dictionary.json` - Extracted filters

**Cost Optimization**: Smart refresh strategies to minimize LLM API costs while maintaining data freshness.

**Documentation Integration**: Follow existing patterns for GitHub issue tracking, commit message formats, and project context preservation.

## Environment Setup

**Python Version**: 3.12+
**Key Dependencies**: openai, anthropic (optional), google-cloud-storage, pinecone-client, langfuse, tavily-python, pydantic, aiohttp, assemblyai, websockets

**Configuration**: Environment variables in `.env` file (see `environment.example.txt`)
**Authentication**: Google Cloud service account key in `auth/` directory

## Development Guidelines

1. **Package Development**: When adding features, determine the appropriate package (core, intelligence, or voice)
2. **Import Patterns**: Follow the established import patterns (absolute for cross-package, relative within)
3. **Runner Scripts**: Add runner scripts in `run/` for CLI access to package functionality
4. **Testing**: Add tests in the appropriate `tests/` subdirectory
5. **Documentation**: Update package README files when adding significant features