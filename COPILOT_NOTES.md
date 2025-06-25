# COPILOT_NOTES.md - AI Agent Context Document
## Catalog Maintenance Development Guide

### Project Overview & Mission
- **Project Name**: Catalog Maintenance  
- **Status**: Phase 1 - Foundation Implementation
- **Primary Goal**: Build scalable Python ingestion pipeline for product catalogs and knowledge bases
- **Business Context**: Enable AI-powered retail experiences by maintaining synchronized vector databases with product and knowledge data

### Architecture Evolution & Key Decisions

#### Decision #1: LLM Provider Strategy (December 2024)
**Context**: Need reliable LLM services for descriptor and sizing generation across different use cases
**Decision**: Multi-provider approach with OpenAI, Anthropic, and Gemini services (OpenAI implemented, others planned)
**Rationale**: Different models excel at different tasks; router allows optimal model selection per use case

**Current Status**: OpenAI service complete, router ready for additional providers

#### Decision #2: KISS Storage Approach (December 2024)
**Context**: Existing storage.py provides comprehensive GCP + local storage with account patterns
**Decision**: Leverage existing AccountStorageProvider patterns rather than rebuilding
**Rationale**: Proven storage layer with backup, compression, and metadata handling already exists

#### Decision #3: Proven Sizing Instruction (December 2024)
**Context**: User provided working LLM instruction for sizing generation that produces good results
**Decision**: Use proven sizing prompt exactly as provided, with JSON response format
**Rationale**: Don't fix what's not broken; leverage known-working prompt patterns

#### Decision #4: Dynamic Pinecone Index Naming (December 2024)
**Context**: Need to support multiple brands and environments with organized index structure
**Decision**: Pattern `<env>--<brand_url>--dense/sparse` for index names
**Rationale**: Clear separation between environments and brands; scales to many accounts

#### Decision #5: Vertical-Agnostic Design (December 2024)
**Context**: Initial implementation was hardcoded to cycling/bike vertical and specialized.com brand
**Decision**: Auto-detect product verticals from categories and brand analysis; use generic prompts
**Rationale**: System must work with any brand/vertical (cycling, fashion, beauty, tech, etc.) without hardcoded assumptions

#### Decision #6: Brand Intelligence Generation (December 2024)
**Context**: System needed deep brand context to generate truly brand-aware descriptors and support AI sales agents
**Decision**: Add Phase 0 - automated brand research using advanced LLMs (o1, Claude 3.5 Sonnet) with web search
**Rationale**: Zero-to-RAG automation requires comprehensive brand understanding; enhances descriptor quality and provides AI sales agent context

#### Decision #7: Deep Research vs Speed (December 2024)
**Context**: Brand intelligence quality directly impacts descriptor authenticity and AI sales agent effectiveness
**Decision**: Implement 5-15 minute deep research process with 15+ sources, 5 LLM analysis rounds, quality controls
**Rationale**: Superficial research produces superficial brand intelligence; deep research (8-12 min target) produces authentic brand voice and strategic insights worth the cost

#### Decision #8: Phase-Based Research Architecture (December 2024)
**Context**: Different aspects of brand intelligence have different update frequencies; fashion brands need frequent style updates while core values remain stable
**Decision**: Split brand research into 6 modular phases with independent caching and selective refresh capabilities
**Rationale**: Massive cost savings (80%+ reduction for incremental updates), enables real-time brand evolution tracking, and allows targeted updates for specific business needs

#### Decision #9: Quality Evaluation & Feedback Loops (December 2024)
**Context**: Brand intelligence quality directly impacts descriptor authenticity and AI sales agent effectiveness; need systematic quality assurance
**Decision**: Implement LLM-based quality evaluation for each research phase with feedback loops and automatic re-runs
**Rationale**: Ensures consistent high-quality brand intelligence; feedback loops improve results; quality scores enable data-driven optimization

#### Decision #10: AI Brand Ethos Voice Interview Integration (December 2024)
**Context**: Direct brand representative interviews provide the most authentic brand voice and strategic insights
**Decision**: Integrate AI Brand Ethos Voice Interview transcripts as premium brand intelligence source with dedicated processing pipeline
**Rationale**: Human-in-the-loop brand intelligence provides unmatched authenticity; strategic interviews capture insights impossible to gather through web research

#### Decision #11: Scalable Brand Monitoring Architecture (December 2024)
**Context**: System must support 100s-1000s of brands without hardcoded brand lists or vertical assumptions  
**Decision**: Implement data-driven dynamic architecture with automatic brand discovery, vertical detection, and configuration-driven refresh rules
**Rationale**: Hardcoded brand lists and manual vertical classification cannot scale beyond dozens of brands; enterprise deployment requires self-discovering, self-classifying, and self-optimizing monitoring systems

#### Decision #12: Linear vs Non-Linear Shopping Behavior Classification (December 2024)
**Context**: Shopping behavior exists on a spectrum from objective/specification-driven (linear) to subjective/emotion-driven (non-linear), affecting optimal descriptor generation and AI sales agent conversation strategies
**Decision**: Implement LLM-based dynamic linearity classification for products and brands, with adaptive descriptor generation and brand voice modulation based on detected shopping behavior patterns
**Rationale**: Product linearity fundamentally affects how customers evaluate and purchase items; dynamic LLM classification enables personalized, psychology-matched messaging without hardcoded category assumptions

#### Decision #13: Linearity-Specific Intelligence Requirements (December 2024)
**Context**: Different shopping psychology requires different TYPES of brand intelligence - technical brands need engineering philosophy while fashion brands need undocumented style guides
**Decision**: Implement linearity-aware brand interview evaluation, intelligence gap analysis, and linearity-specific RAG structuring to capture and organize the right types of information for each brand's shopping psychology
**Rationale**: Generic brand intelligence collection misses critical information types; fashion brands need style guide capture while technical brands need competitive analysis - RAG must be structured differently for optimal retrieval by shopping psychology

#### Decision #14: Brand-Specific RAG Query Transformation (December 2024)
**Context**: User queries need brand-specific transformation to optimize search effectiveness; different brands require different search approaches based on their shopping psychology patterns
**Decision**: Generate brand-specific LLM instructions that transform user queries into structured JSON output for both product and knowledge search, with linearity-aware transformation strategies
**Rationale**: Technical brands need specification-focused query transformations while lifestyle brands need emotion/style transformations; enables optimal search relevance without hardcoding brand categories; leverages brand intelligence and linearity analysis for dynamic generation

#### Decision #15: AI Sales Agent Persona Generation (December 2024)
**Context**: Complete the zero-to-RAG pipeline with AI sales agent readiness; need brand-specific personas that translate shopping psychology into natural conversation guidance
**Decision**: Generate brand-specific AI sales agent personas with names, avatars, and system prompts using brand intelligence and linearity analysis, with live persona protection and manual promotion workflows
**Rationale**: AI sales agents need authentic brand voice and shopping psychology-matched conversation strategies; Replicate avatar generation provides professional visual identity; manual promotion protects live customer-facing personas while enabling continuous improvement through A/B testing

### Current System State
- **Completed**: Basic project structure, comprehensive configuration system, robust storage layer, comprehensive Product model, OpenAI LLM service
- **In Progress**: Phase 1 foundation implementation (Epic #8)
- **Next Phase**: Descriptor generation, Pinecone client, product ingestor orchestrator

### Recent Context & Temporal Notes
*Note: This section provides immediate context for recent work alongside GitHub commit history*

#### Recent Sessions (December 2024)
- **Phase 1 Planning**: Analyzed ROADMAP, existing code, and created implementation plan
- **GitHub Setup**: Created feature branch and comprehensive issue tracking (#1-#7)
- **LLM Integration**: Identified need for OpenAI service to complete provider suite
- **Brand Intelligence Vision**: Added comprehensive Phase 0 for automated brand research
- **Zero-to-RAG Pipeline**: Designed complete system from brand URL to RAG-ready catalog
- **Configuration System**: ✅ Completed comprehensive centralized configuration management with pydantic BaseSettings

#### Active Issues & PRs
- **Epic #8**: Phase 1 - Ingestion & Catalog Maintenance Foundation
- **Issue #9**: ✅ Create project context files (COMPLETE)
- **Issue #10**: ✅ Complete LLM provider suite with OpenAI service (COMPLETE)
- **Issue #11**: ✅ Configuration management system (COMPLETE)
- **Issue #12**: 🔄 Build descriptor & sizing generator with proven prompts (NEXT)
- **Issue #13**: 🔄 Create Pinecone client abstraction  
- **Issue #14**: 🔄 Build product ingestor orchestrator

#### Immediate Next Steps
- [x] Create COPILOT_NOTES.md with comprehensive project context
- [x] Create PROJECT_FILEMAP.md documenting current architecture
- [x] Complete OpenAI service implementation following existing patterns
- [x] Set up configs/settings.py with environment management
- [ ] Implement DescriptorGenerator using proven sizing instruction
- [ ] Build Pinecone client abstraction with dynamic index naming
- [ ] Create product ingestor orchestrator

#### Recent Architectural Decisions
- **Dynamic Index Naming**: Enables multi-brand, multi-environment support
- **LLM Router Pattern**: Allows optimal model selection for different tasks
- **KISS Storage Integration**: Leverage existing proven storage patterns
- **Brand Intelligence Generation**: Phase 0 automated brand research for zero-to-RAG pipeline
- **Advanced LLM Integration**: o1 and Claude 3.5 Sonnet for deep brand analysis
- **Centralized Configuration**: ✅ Complete pydantic-based settings with environment awareness and validation

### Integration Patterns & Conventions

#### LLM Service Pattern
```python
class LlmService(LlmModelService):
    async def chat_completion(self, system, messages, model, **kwargs):
        # Standard interface for all providers
        # Error handling with exponential backoff
        # Token counting and conversation truncation
```

#### Storage Pattern
```python
# Use existing AccountStorageProvider
storage = get_account_storage_provider()
products = await storage.get_product_catalog(account)
```

#### Product Processing Pattern
```python
# Leverage existing Product model
product = Product.from_metadata(raw_data)
if not product.descriptor or is_stale(product.descriptor):
    product.descriptor = await generate_descriptor(product)
```

### Security & Compliance Framework
- API keys managed through environment variables
- GCP credentials via service account keys
- Data processing follows account isolation patterns
- No sensitive data in logs or version control

### Success Metrics & KPIs
- **Technical**: Ingestion throughput, error rates, vector search accuracy
- **Business**: Product catalog freshness, search relevance, system uptime
- **Quality**: LLM generation consistency, sizing accuracy, descriptor quality

### Common Patterns & Anti-patterns

#### ✅ Recommended Patterns
- Use existing LLM services with proper error handling
- Leverage AccountStorageProvider for all storage operations
- Follow dynamic index naming for Pinecone organization
- Apply proven LLM prompts rather than reinventing
- Use batch processing with retry logic for reliability
- Auto-detect verticals from product data; avoid hardcoded industry assumptions
- Use generic prompts that adapt to any product category/brand
- Generate brand intelligence before product processing for brand-aware descriptors
- Cache brand_details.md for 30 days to minimize research API costs
- Use advanced LLMs (o1, Claude 3.5 Sonnet) for comprehensive brand analysis
- Invest 5-15 minutes per brand for deep research (not superficial web search)
- Gather 15+ diverse sources and perform 5 LLM analysis rounds for quality
- Enforce minimum research time and source count to prevent shallow analysis
- Use phase-based research architecture for modular brand intelligence updates
- Refresh only relevant phases (product_style for collections, voice_messaging for campaigns)
- Leverage different cache durations per phase (6mo foundation, 1mo voice)
- Implement quality evaluation for each research phase with feedback loops
- Use Langfuse for all brand research prompt management and versioning
- Process AI Brand Ethos Voice Interview transcripts as premium brand intelligence
- Automatically integrate interview insights into relevant research phases
- **SCALE**: Use dynamic brand discovery and automatic vertical detection for 100s-1000s of brands
- **SCALE**: Implement configuration-driven refresh rules, not hardcoded brand lists or verticals  
- **SCALE**: Use batch processing with intelligent criteria-based brand selection
- **SCALE**: Build self-discovering, self-classifying, and self-optimizing monitoring systems
- **LINEARITY**: Use LLM-based dynamic linearity classification for products and shopping behavior
- **LINEARITY**: Adapt descriptor generation and brand voice based on detected linearity patterns
- **LINEARITY**: Generate psychology-matched messaging without hardcoded category assumptions
- **LINEARITY**: Integrate linearity analysis into both brand intelligence and product RAG systems
- **INTELLIGENCE**: Capture linearity-specific intelligence types - engineering philosophy for technical brands, style guides for fashion brands
- **INTERVIEWS**: Use linearity-aware interview evaluation to identify intelligence gaps and recommend specific interviews
- **RAG**: Structure knowledge chunks differently by linearity for optimal retrieval and conversation adaptation
- **QUERY TRANSFORMATION**: Generate brand-specific LLM instructions for query transformation using linearity analysis
- **SEARCH OPTIMIZATION**: Apply brand-specific query transformations with JSON output for filters, boost terms, and search strategy
- **NO HARDCODING**: Use LLM evaluation to determine if brands need custom transformations; generate all prompts dynamically
- **AI PERSONA GENERATION**: Create brand-specific AI sales agent personas with linearity-aware conversation strategies
- **LIVE PERSONA PROTECTION**: Never automatically override live personas; use manual promotion workflows only
- **AVATAR GENERATION**: Replicate google/imagen-4 integration for professional, brand-aligned avatar images

#### ❌ Anti-patterns to Avoid
- Building new storage abstraction when storage.py works well
- Modifying proven LLM prompts that already work
- Single-provider LLM dependency without fallbacks
- Hard-coded index names or bucket names
- Processing products individually instead of batching
- Hardcoding vertical/industry assumptions (cycling, fashion, etc.)
- Brand-specific logic that doesn't scale to other verticals
- Regenerating brand intelligence on every run (expensive and unnecessary)
- Manual brand research when automated generation is available
- Ignoring brand intelligence context in descriptor generation
- Superficial brand research with only 1-2 sources or single LLM pass
- Rushing brand intelligence generation for speed over quality
- Skipping research quality controls and minimum time investment
- Re-researching entire brand when only specific phases need updates
- Using monolithic brand intelligence without phase-based modularity
- Ignoring phase-specific cache durations and update triggers
- Skipping quality evaluation and accepting low-quality research results
- Hardcoding prompts instead of using Langfuse prompt management
- Ignoring available AI Brand Ethos Voice Interview transcripts
- Manual interview processing when automated systems are available
- **CRITICAL**: Hardcoded brand lists in monitoring systems (breaks at 100s-1000s of brands)
- **CRITICAL**: Manual vertical classification instead of automatic detection
- **CRITICAL**: Code-based refresh rules instead of configuration-driven architecture
- **CRITICAL**: Hardcoded product category linearity assumptions instead of LLM-based dynamic classification
- **CRITICAL**: One-size-fits-all descriptor generation ignoring shopping behavior psychology
- **CRITICAL**: Static brand voice without adaptation to product linearity spectrum
- **CRITICAL**: Generic brand intelligence collection ignoring linearity-specific requirements
- **CRITICAL**: Missing style guide capture for fashion brands or engineering philosophy for technical brands
- **CRITICAL**: Uniform RAG structuring instead of linearity-optimized knowledge organization
- **CRITICAL**: One-size-fits-all query transformation instead of brand-specific optimization
- **CRITICAL**: Hardcoded query transformation rules instead of LLM-generated brand-specific prompts
- **CRITICAL**: Ignoring linearity patterns when generating query transformation strategies
- **CRITICAL**: Automatically overriding live AI sales agent personas without human approval
- **CRITICAL**: Generic persona generation ignoring brand intelligence and shopping psychology
- **CRITICAL**: Poor quality avatar generation or missing visual identity for AI agents

### Development Guidelines
- Follow GitHub workflow with feature branches and issues
- Reference issue numbers in all commits
- Use existing error handling and retry patterns
- Maintain compatibility with existing Product and storage models
- Write comprehensive logging for debugging and monitoring

### Getting Started Guide for New Contributors
1. Review ROADMAP/ingestion_and_catalog_maintenance.md for context
2. Examine existing LLM services in src/llm/ for patterns
3. Understand storage.py and Product model capabilities  
4. Check current GitHub issues for active work
5. Follow proven patterns rather than rebuilding components

### Phase 1 Implementation Context
- **Epic #1**: Foundation setup with LLM integration
- **Key Files**: configs/settings.py, src/descriptor.py, src/pinecone_client.py, src/product_ingestor.py
- **Integration Points**: Existing LLM services, storage.py, Product model, Pinecone setup patterns
- **Success Criteria**: End-to-end product catalog ingestion with LLM-generated descriptors and sizing 