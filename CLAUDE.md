# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Brand Intelligence & Catalog Maintenance System** that orchestrates an 8-phase AI-powered brand research pipeline. The system performs comprehensive brand intelligence gathering, product catalog ingestion, and knowledge base creation to enable AI-powered brand-aware customer service agents.

**Core Technology**: Python 3.12+ with async/await patterns, LLM integration (OpenAI, Anthropic, Gemini), and cloud storage (Google Cloud, Pinecone vector DB).

## Common Commands

### Development Environment
```bash
# Activate virtual environment
source activate_venv.sh

# Install dependencies
pip install -r requirements.txt

# Run Python scripts
python brand_researcher.py --brand specialized.com --auto-continue
```

### Brand Research Pipeline Commands
```bash
# Master control script for all operations
./scripts/brand_manager.sh onboard specialized.com
./scripts/brand_manager.sh status specialized.com
./scripts/brand_manager.sh next-step specialized.com
./scripts/brand_manager.sh resume specialized.com

# Individual research phases
python brand_researcher.py --brand specialized.com --phase foundation_research
python brand_researcher.py --brand specialized.com --phase all

# Batch operations
./scripts/batch_operations.sh --operation status-check --discover
./scripts/brand_manager.sh dashboard
```

### Testing & Validation
```bash
# Run individual test files
python test_specialized.py
python test_web_search_only.py

# Test specific components
pytest tests/test_configuration.py
pytest tests/test_openai_service.py
```

## Architecture Overview

### Core Research Pipeline (8 Phases)
The system follows a sequential 8-phase brand intelligence pipeline orchestrated by `brand_researcher.py`:

1. **Foundation Research** (`src/research/foundation_research.py`) - Company history, mission, values
2. **Market Positioning** (`src/research/market_positioning_research.py`) - Competitive landscape analysis  
3. **Product Style** (`src/research/product_style_research.py`) - Design language, aesthetics
4. **Customer Cultural** (`src/research/customer_cultural_research.py`) - Psychology, behavior patterns
5. **Voice Messaging** (`src/research/voice_messaging_research.py`) - Brand voice and communication style
6. **Interview Synthesis** (`src/research/interview_synthesis_research.py`) - Human insights integration
7. **Linearity Analysis** (`src/research/linearity_analysis_research.py`) - Brand consistency analysis
8. **Research Integration** (`src/research/research_integration.py`) - Final synthesis and validation

### Key Architectural Components

**LLM Factory Pattern** (`src/llm/`):
- `simple_factory.py` - Model-agnostic LLM service factory
- `anthropic_service.py`, `openai_service.py`, `gemini_service.py` - Provider implementations
- `prompt_manager.py` - Langfuse integration for prompt management

**Research Framework** (`src/research/`):
- `base_researcher.py` - Base class with integrated quality evaluation (8.0+ threshold)
- Quality evaluation with feedback loops and automatic retry (max 3 attempts)
- Phase-specific researchers inherit from BaseResearcher

**Workflow State Management** (`src/workflow/`):
- `workflow_state_manager.py` - Tracks pipeline progress and enables resume functionality
- `research_phase_tracker.py` - Individual phase progress tracking

**Storage & Configuration**:
- `src/storage.py` - Account-based storage provider (Google Cloud Storage)
- `configs/settings.py` - Centralized configuration with environment awareness
- `local/account_storage/` - File-based research data storage

### Quality Evaluation System
The system includes an integrated LLM-based quality evaluation framework:
- Quality scores on 0.0-10.0 scale with configurable thresholds
- 5 criteria assessment: accuracy, completeness, consistency, authenticity, actionability
- Automatic feedback loops with improvement suggestions
- Quality metadata stored in `research_metadata.json` files alongside research content

### Important Integration Patterns

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

**Active Development**: Currently implementing Phase 1 Quality Evaluation Framework with LLM-based quality assessment and feedback loops.

**Storage Strategy**: Account-based storage under `local/account_storage/{brand}/` with research phases, metadata, and progress tracking.

**Cost Optimization**: Smart refresh strategies to minimize LLM API costs while maintaining data freshness.

**Documentation Integration**: Follow existing patterns in `.cursorrules` for GitHub issue tracking, commit message formats, and project context preservation in `COPILOT_NOTES.md` and `PROJECT_FILEMAP.md`.

## Environment Setup

**Python Version**: 3.12+
**Key Dependencies**: openai, anthropic (optional), google-cloud-storage, pinecone-client, langfuse, tavily-python, pydantic, aiohttp

**Configuration**: Environment variables in `.env` file (see `environment.example.txt`)
**Authentication**: Google Cloud service account key in `auth/` directory