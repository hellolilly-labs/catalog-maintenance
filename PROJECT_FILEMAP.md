# PROJECT_FILEMAP.md - Catalog Maintenance Architecture Map

## Project Overview
**Catalog Maintenance** - Zero-to-RAG automation system that transforms any brand URL into a fully operational AI sales agent with comprehensive product catalog, brand intelligence, and optimized search capabilities.

**Current Status**: Phase 1 Foundation Implementation  
**Architecture Pattern**: KISS (Keep It Simple Stupid) with proven components

---

## Core Files & Architecture

### 🏗️ Foundation Components

#### **Configuration & Environment**
```
├── env.txt                           # Environment variables (API keys, settings)
├── environment.example.txt           # Template for environment setup
├── requirements.txt                  # Python dependencies
└── configs/                          # [PLANNED] Centralized configuration
    └── settings.py                   # Pydantic-based settings management
```

#### **Storage Layer (Existing - Proven)**
```
├── storage.py                        # ✅ CORE: Account-based storage abstraction
│                                     # Patterns: GCP + local, compression, metadata
│                                     # Usage: AccountStorageProvider for all storage
└── accounts/                         # Account-isolated storage structure
    └── <brand_url>/                  # Per-brand storage organization
        ├── products/                 # Product catalog data
        ├── knowledge/                # Brand knowledge and documentation
        └── [FUTURE] brand_details.md # Brand intelligence (Phase 0)
```

#### **Product Data Model (Existing - Proven)**
```
└── src/models/
    ├── __init__.py
    ├── product.py                    # ✅ CORE: Product model with metadata handling
    └── base.py                       # Base model patterns and validation
```

### 🤖 LLM Services (Multi-Provider Strategy)

#### **Existing LLM Services (Proven)**
```
└── src/llm/
    ├── __init__.py
    ├── anthropic_service.py          # ✅ EXISTING: Claude models
    ├── gemini_service.py             # ✅ EXISTING: Google Gemini
    ├── base.py                       # ✅ EXISTING: LlmModelService interface
    └── openai_service.py             # 🔄 PLANNED: Complete provider suite
```

**Usage Pattern**: All LLM services extend `LlmModelService` with standardized error handling, token management, and retry logic.

### 🔧 Phase 1 Implementation (Current Sprint)

#### **Descriptor Generation (Issue #5)**
```
└── src/
    └── descriptor.py                 # 🔄 IMPLEMENTING: LLM-powered descriptor generation
                                      # Uses proven sizing instruction exactly as provided
                                      # Multi-provider routing for optimal quality
```

#### **Vector Storage (Issue #6)**
```
└── src/
    └── pinecone_client.py            # 🔄 IMPLEMENTING: Dynamic index management
                                      # Pattern: <env>--<brand_url>--dense/sparse
                                      # Scalable to 100s-1000s of brands
```

#### **Orchestration (Issue #7)**
```
└── src/
    └── product_ingestor.py           # 🔄 IMPLEMENTING: End-to-end pipeline
                                      # Coordinates: Load → Process → Store → Vector
                                      # Batch processing with error handling
```

### 🔄 Workflow Management (Future Integration)

#### **Brand Workflow State Management**
```
└── src/workflow/
    ├── __init__.py                   # ✅ CREATED: Workflow package
    └── workflow_state_manager.py     # ✅ CREATED: Brand onboarding state tracking
```

#### **CLI & Automation Scripts**
```
└── scripts/
    ├── README.md                     # ✅ CREATED: Script documentation
    ├── brand_manager.sh              # ✅ CREATED: Main CLI interface
    ├── brand_onboarding.sh           # ✅ CREATED: New brand setup
    ├── brand_status.sh               # ✅ CREATED: Status checking
    ├── brand_maintenance.sh          # ✅ CREATED: Ongoing maintenance
    └── batch_operations.sh           # ✅ CREATED: Bulk operations
```

---

## Integration Patterns & Key Decisions

### 🎯 Decision #1: Multi-Provider LLM Strategy
**Pattern**: OpenAI + Anthropic + Gemini with intelligent routing
- **Descriptor Generation**: OpenAI GPT-4-turbo (creative excellence)
- **Sizing Analysis**: Anthropic Claude (superior reasoning) 
- **Brand Research**: Advanced models (o1, Claude 3.5 Sonnet)

### 🎯 Decision #2: KISS Storage Approach
**Pattern**: Leverage existing `storage.py` AccountStorageProvider
- ✅ **Keep**: Proven storage abstraction with GCP + local modes
- ❌ **Avoid**: Building new storage layers

### 🎯 Decision #3: Proven Prompt Reuse
**Pattern**: Use working LLM instructions exactly as provided
```python
# Proven sizing instruction (exact implementation)
sizing_prompt = """Given these product details and the sizing chart, 
find the correct sizing and create a 'sizing' field with the appropriate 
size information in JSON format."""
```

### 🎯 Decision #4: Dynamic Pinecone Index Naming
**Pattern**: `<env>--<brand_url>--<type>` for scalable organization
```python
# Examples:
"dev--specialized.com--dense"
"prod--nike.com--sparse"  
"staging--gucci.com--dense"
```

### 🎯 Decision #5: Vertical-Agnostic Design
**Pattern**: Auto-detect product categories, no hardcoded assumptions
- ✅ **Use**: Generic prompts that adapt to any brand/vertical
- ❌ **Avoid**: Hardcoded cycling/fashion/tech specific logic

---

## Phase 0: Brand Intelligence (Future Foundation)

### 🧠 Brand Research Architecture
```
└── src/research/                     # [PHASE 0] Advanced brand intelligence
    ├── brand_researcher.py           # Multi-phase research with web search
    ├── web_search.py                 # Tavily + GCP + direct scraping
    ├── brand_analyzer.py             # LLM-based brand analysis
    └── quality_evaluator.py          # Research quality scoring
```

### 📊 Brand Intelligence Storage
```
└── accounts/<brand_url>/
    ├── brand_details.md              # [PHASE 0] Comprehensive brand profile
    ├── research_phases/              # [PHASE 0] Phase-based research cache
    ├── brand_interviews/             # [PHASE 0] AI Brand Ethos transcripts
    └── research_quality/             # [PHASE 0] Quality evaluation results
```

---

## Development Workflow & Patterns

### 🔄 GitHub Integration Pattern
**Branch**: `feature/phase1-ingestion-foundation`
**Issues**: Epic #1 with child issues #2-#7
**Commits**: Reference issue numbers with descriptive messages

### 🧪 Testing Strategy
```
└── tests/
    ├── unit/                         # Component-level testing
    ├── integration/                  # End-to-end pipeline testing  
    └── fixtures/                     # Test data and mocks
```

### 📝 Documentation Pattern
- **COPILOT_NOTES.md**: Master context with architectural decisions
- **PROJECT_FILEMAP.md**: This file - architecture and integration points
- **ROADMAP/**: Implementation strategy and success metrics

---

## Anti-Patterns to Avoid

### ❌ Critical Anti-Patterns
- **Storage**: Don't rebuild storage.py - it works well
- **Prompts**: Don't modify proven LLM instructions that work
- **Hardcoding**: Don't assume specific verticals (cycling, fashion, etc.)
- **Brand Lists**: Don't hardcode brand monitoring (breaks at scale)
- **Index Names**: Don't use static Pinecone index names
- **Single Provider**: Don't depend on one LLM service without fallbacks

### ❌ Scalability Anti-Patterns  
- Manual brand discovery instead of filesystem-based
- Hardcoded vertical classification vs auto-detection
- One-size-fits-all processing vs adaptive patterns
- Monolithic brand intelligence vs phase-based modularity

---

## Current Implementation Status

### ✅ Completed Foundation
- Project structure and architecture documentation
- Existing LLM services (Anthropic, Gemini) 
- Robust storage layer with account patterns
- Product model with comprehensive metadata handling
- Workflow state management and CLI scripts

### 🔄 Phase 1 Active Implementation (Epic #1)
- **Issue #2**: ✅ PROJECT_FILEMAP.md (This document)
- **Issue #3**: 🔄 OpenAI service implementation
- **Issue #4**: 🔄 Configuration management system  
- **Issue #5**: 🔄 Descriptor & sizing generator
- **Issue #6**: 🔄 Pinecone client abstraction
- **Issue #7**: 🔄 Product ingestor orchestrator

### 🎯 Next Phase Preparation
- Brand intelligence research infrastructure (Phase 0)
- Advanced RAG optimization and query transformation
- AI sales agent persona generation with avatars
- Scalable monitoring for 100s-1000s of brands

---

## Success Criteria & Integration Points

### 🎯 Phase 1 Success Metrics
- End-to-end product catalog ingestion functional
- LLM-generated descriptors and sizing for any brand
- Dynamic Pinecone index management with proper naming
- Batch processing with comprehensive error handling
- Configuration-driven operation across environments

### 🔗 Key Integration Points
1. **storage.py** ↔ **product_ingestor.py**: Product loading and storage
2. **LLM services** ↔ **descriptor.py**: Multi-provider text generation
3. **pinecone_client.py** ↔ **product_ingestor.py**: Vector storage coordination
4. **configs/settings.py** ↔ **All components**: Centralized configuration
5. **Product model** ↔ **All processors**: Data structure consistency

### 🚀 Ready for Scale
- Architecture supports 100s-1000s of brands
- No hardcoded assumptions about verticals or brands
- Dynamic resource allocation and batch processing
- Phase-based brand intelligence with intelligent caching
- Zero-to-RAG automation pipeline ready for any e-commerce brand

---

**Last Updated**: December 2024  
**Architecture Status**: Foundation → Implementation Ready  
**Integration Pattern**: KISS + Proven Components + Dynamic Scaling 