# Catalog Maintenance: Zero-to-RAG Pipeline

A comprehensive system for automated brand research, product catalog ingestion, and knowledge base management. Designed to take any brand from **zero to full RAG-ready** with minimal manual intervention.

**Target**: Universal brand support with automated brand intelligence and vertical detection.
**End User**: AI sales agents accessing rich brand context and product knowledge.

---

## Pipeline Overview

### Phase 0: Brand Intelligence Generation (NEW)
**Goal**: Automated brand research and profile generation
- Web scraping and research using advanced LLMs (o1, Claude 3.5 Sonnet)
- Generate comprehensive `brand_details.md` with brand DNA
- Store in GCP bucket: `accounts/<brand_url>/brand_details.md`
- Ingest brand profile into RAG for AI sales agent context

### Phase 1: Product Catalog Ingestion
**Goal**: Intelligent product processing with brand-aware descriptors
- Auto-detect vertical from products and brand intelligence
- Generate descriptors informed by brand voice and positioning
- Create vectors with rich brand context

### Phase 2: Knowledge Base Ingestion
**Goal**: Comprehensive knowledge management
- Process PDFs, HTML, and brand documentation
- Include brand intelligence in knowledge corpus
- Support AI sales agent with complete brand understanding

---

## 1. Project Structure

```
catalog-maintenance/
├── configs/
│   └── settings.py            # bucket names, index prefixes, models, timeouts
├── src/
│   ├── models/
│   │   ├── product.py         # Product data class definition
│   │   ├── brand.py           # Brand intelligence data model (NEW)
│   │   ├── linearity.py       # Linear vs non-linear shopping behavior models (NEW)
│   │   ├── product_manager.py # Memory-based product catalog management
│   │   └── product_catalog_cache.py  # Product caching utilities
│   ├── llm/
│   │   ├── __init__.py        # LLM package exports
│   │   ├── llm_service.py     # Base LLM service (clean, no fine-tuning)
│   │   ├── openai_service.py  # OpenAI GPT integration
│   │   ├── llm_router.py      # Multi-provider routing
│   │   ├── chat_model.py      # Chat message models
│   │   ├── llm_errors.py      # LLM error handling
│   │   └── prompt_manager.py  # Langfuse prompt management
│   ├── research/              # Phase-Based Brand Intelligence with Quality Control (Phase 0)
│   │   ├── brand_researcher.py # main orchestrator with phase management
│   │   ├── research_phases/   # modular phase implementations
│   │   │   ├── __init__.py    # phase registry and factory
│   │   │   ├── foundation.py  # core brand identity research
│   │   │   ├── market_positioning.py  # competitive landscape
│   │   │   ├── product_style.py      # products & collections
│   │   │   ├── customer_cultural.py  # audience & culture
│   │   │   ├── voice_messaging.py    # brand voice & campaigns
│   │   │   ├── interview_insights.py # AI Brand Ethos Voice Interview synthesis
│   │   │   ├── linearity_analysis.py # brand/product linearity patterns (NEW)
│   │   │   └── integration.py        # phase synthesis and conflict resolution
│   │   ├── quality/           # quality evaluation and feedback systems
│   │   │   ├── __init__.py    # quality evaluation framework
│   │   │   ├── phase_evaluator.py    # LLM-based quality evaluation
│   │   │   ├── feedback_loops.py     # improvement feedback and re-run logic
│   │   │   └── quality_metrics.py    # quality criteria and scoring
│   │   ├── interviews/        # AI Brand Ethos Voice Interview processing
│   │   │   ├── __init__.py    # interview processing framework
│   │   │   ├── transcript_processor.py # interview transcript analysis
│   │   │   ├── interview_classifier.py # classify interview types
│   │   │   └── insight_extractor.py    # extract insights from interviews
│   │   ├── web_search.py      # multi-source search capabilities (Tavily, GCP, direct)
│   │   ├── brand_analyzer.py  # LLM-powered analysis with Langfuse prompts
│   │   └── phase_cache.py     # phase-based caching system with quality tracking
│   ├── linearity/             # Linear vs Non-Linear Shopping Behavior Analysis (NEW)
│   │   ├── __init__.py        # linearity analysis framework
│   │   ├── classifier.py      # LLM-based dynamic linearity classification
│   │   ├── adaptive_voice.py  # brand voice adaptation based on linearity
│   │   └── psychology_engine.py # shopping psychology analysis and insights
│   ├── product_ingestor.py    # full + incremental product sync orchestrator
│   ├── knowledge_ingestor.py  # PDF & HTML document ingestion (Phase 2)
│   ├── descriptor.py          # linearity-aware descriptor & sizing generators (UPDATED)
│   ├── pinecone_client.py     # clean Pinecone wrapper with dynamic naming
│   ├── pinecone_setup.py      # legacy reference (to be deprecated)
│   └── storage.py             # GCP + local storage abstraction
├── tests/                     # unit + integration tests
├── notebooks/
│   ├── manual_testing.ipynb   # Interactive testing and index queries
│   └── brand_research.ipynb   # Brand intelligence testing (NEW)
├── requirements.txt           # includes langfuse>=2.0.0, search APIs
└── environment.example.txt    # environment variables documentation
```

---

## 2. Configuration

* **GCP buckets**

  * Prod: `liddy-account-documents`
  * Dev:  `liddy-account-documents-dev`
  * **File structure**:
    ```
    accounts/<brand_url>/
    ├── brand_details.md          # Generated brand intelligence (Phase 0)
    ├── products.json             # Product catalog data
    └── knowledge/                # Brand documents and manuals
        ├── manuals/
        ├── marketing/
        └── brand_profile/        # Additional brand assets
    ```

* **Index naming**

  * Dense:  `<env>--<brand_url>--dense`
  * Sparse: `<env>--<brand_url>--sparse`

* **Vertical-Agnostic Design**

  * **Auto-detection**: System detects product vertical from categories and brand analysis
  * **Generic prompts**: LLM prompts adapt to any industry (cycling, fashion, beauty, etc.)
  * **Brand intelligence**: No hardcoded vertical assumptions

* **Linear vs Non-Linear Shopping Behavior Classification (NEW)**

  * **Dynamic Linearity Detection**: LLM-based classification of product shopping psychology without hardcoded categories
  * **Adaptive Descriptor Generation**: Descriptors adapt from technical/specification-focused to emotional/lifestyle-focused based on detected shopping behavior
  * **Brand Voice Modulation**: Brand voice adapts to product linearity spectrum for psychology-matched messaging
  * **RAG Integration**: Both brand intelligence and product knowledge structured to support linearity-aware AI sales agent conversations

* **Brand Intelligence Generation**

  * **Advanced LLMs**: OpenAI o1, Claude 3.5 Sonnet for research and analysis
  * **Web search integration**: Google Search API, Bing Search for brand research
  * **Research depth**: Company history, mission, target audience, positioning
  * **Brand voice analysis**: Tone, messaging style, key differentiators
  * **Competitive intelligence**: Market position, unique selling propositions

* **LLM & Prompt Management**

  * **Langfuse integration** with prefix `liddy/catalog/`
  * **Multi-provider support**: OpenAI (primary), Anthropic, Gemini
  * **Clean architecture**: No fine-tuning code, focused on inference
  * **Prompt versioning**: Langfuse manages prompt templates and versions

---

## 3. Dependency Setup

```bash
pip install -r requirements.txt
# requirements.txt includes:
#   langfuse>=2.0.0, pinecone-client, google-cloud-storage, openai, 
#   pdfminer.six, beautifulsoup4, requests, tenacity
#   google-search-results, serpapi, selenium  # Brand research APIs
```

* Set env vars:

  ```bash
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-key.json
  export PINECONE_API_KEY=…
  export PINECONE_ENVIRONMENT=…
  export ENV=dev       # or prod
  export OPENAI_API_KEY=…
  export LANGFUSE_PUBLIC_KEY=…
  export LANGFUSE_SECRET_KEY=…
  export LANGFUSE_HOST=https://cloud.langfuse.com  # or your instance
  
  # Brand Intelligence & Web Search
  export GOOGLE_SEARCH_API_KEY=…
  export GOOGLE_SEARCH_ENGINE_ID=…
  export SERPAPI_API_KEY=…
  ```

---

## 4. Phase 0: Brand Intelligence Generation

### 4.1 Brand Research Pipeline

```bash
python src/research/brand_researcher.py --brand specialized.com --force-regenerate
```

**Goal**: Automated generation of comprehensive brand profiles using advanced LLMs and web research.

**Output**: `brand_details.md` file containing rich brand intelligence for:
- Enhanced descriptor generation with brand voice
- AI sales agent context and personality
- Accurate vertical detection and positioning
- Complete zero-to-RAG brand onboarding

### 4.2 Phase-Based Research Architecture

**Research Philosophy**: Modular brand intelligence with selective refresh capabilities
**Full Research**: 10-20 minutes for comprehensive new brand analysis  
**Incremental Updates**: 2-5 minutes for selective phase refresh

#### **Research Phase Structure**

Brand research is divided into **6 distinct phases**, each with different refresh frequencies and cache durations:

```python
research_phases = {
    "foundation": {
        "cache_duration": "6-12 months",  # Very stable
        "triggers": ["major company changes", "leadership shifts"],
        "research_time": "3-5 minutes",
        "langfuse_prompt": "liddy/catalog/research/foundation",
        "quality_threshold": 8.0
    },
    "market_positioning": {
        "cache_duration": "3-6 months",   # Moderate change
        "triggers": ["new competitors", "industry shifts"], 
        "research_time": "2-4 minutes",
        "langfuse_prompt": "liddy/catalog/research/market_positioning",
        "quality_threshold": 7.5
    },
    "product_style": {
        "cache_duration": "1-3 months",   # Frequent for fashion
        "triggers": ["new collections", "product launches"],
        "research_time": "2-3 minutes",
        "langfuse_prompt": "liddy/catalog/research/product_style",
        "quality_threshold": 8.0
    },
    "customer_cultural": {
        "cache_duration": "2-4 months",   # Evolves with society
        "triggers": ["demographic shifts", "cultural trends"],
        "research_time": "2-3 minutes",
        "langfuse_prompt": "liddy/catalog/research/customer_cultural",
        "quality_threshold": 7.5
    },
    "voice_messaging": {
        "cache_duration": "1-2 months",   # Changes with campaigns
        "triggers": ["rebranding", "new campaigns"],
        "research_time": "1-2 minutes",
        "langfuse_prompt": "liddy/catalog/research/voice_messaging",
        "quality_threshold": 8.5
    },
    "interview_insights": {
        "cache_duration": "auto",         # Updates when new interviews added
        "triggers": ["new brand interview transcript"],
        "research_time": "3-5 minutes",
        "langfuse_prompt": "liddy/catalog/research/interview_synthesis",
        "quality_threshold": 9.0         # Highest quality for direct brand voice
    },
    "linearity_analysis": {
        "cache_duration": "3-6 months",   # Semi-stable brand shopping behavior patterns
        "triggers": ["product catalog changes", "brand positioning shifts"],
        "research_time": "2-4 minutes",
        "langfuse_prompt": "liddy/catalog/research/linearity_analysis",
        "quality_threshold": 8.0
    },
    "integration": {
        "cache_duration": "auto",         # Runs when any phase updates
        "triggers": ["any phase refresh"],
        "research_time": "1-2 minutes",
        "langfuse_prompt": "liddy/catalog/research/integration",
        "quality_threshold": 8.0
    }
}
```

#### **Phase 1: Foundation Research (3-5 minutes)**

**Focus**: Core brand identity that rarely changes
**Cache Duration**: 6-12 months (most stable)
**Research Sources**:
   * Company founding story, history, timeline
   * Mission, vision, core values statements
   * Leadership team and organizational culture
   * Legal structure, ownership, headquarters
   * Patents, foundational innovations

#### **Phase 2: Market Positioning Research (2-4 minutes)**

**Focus**: Competitive landscape and market position
**Cache Duration**: 3-6 months (moderate stability)
**Research Sources**:
   * Direct and indirect competitor analysis
   * Market share and industry position
   * Pricing strategy and value proposition
   * Strategic partnerships and alliances
   * Industry awards and recognition

#### **Phase 3: Product & Style Intelligence (2-3 minutes)**

**Focus**: Current products, collections, and style evolution
**Cache Duration**: 1-3 months (frequent updates, especially fashion)
**Research Sources**:
   * Current product catalogs and collections
   * New product launches and announcements  
   * Design philosophy and aesthetic trends
   * Seasonal updates and style guides
   * Product reviews and performance data

**Fashion Brand Example**:
```python
# Fashion brands need frequent Phase 3 updates
seasonal_triggers = [
    "spring_collection_launch",
    "fall_winter_preview", 
    "holiday_special_edition",
    "collaboration_announcements"
]
```

#### **Phase 4: Customer & Cultural Intelligence (2-3 minutes)**

**Focus**: Target audience and cultural relevance
**Cache Duration**: 2-4 months (evolves with society)
**Research Sources**:
   * Customer demographics and psychographics
   * Social media engagement and community
   * Customer reviews and sentiment analysis
   * Cultural trends and brand relevance
   * Influencer partnerships and endorsements

#### **Phase 5: Voice & Messaging Analysis (1-2 minutes)**

**Focus**: Current brand voice and communication style
**Cache Duration**: 1-2 months (changes with campaigns)
**Research Sources**:
   * Recent marketing campaigns and messaging
   * Social media voice and tone analysis
   * Customer service communication style
   * PR statements and public communications
   * Content strategy and editorial voice

#### **Phase 6: AI Brand Ethos Voice Interview Synthesis (3-5 minutes)**

**Focus**: Process and synthesize direct brand voice from interview transcripts
**Cache Duration**: Auto-updates when new interviews are added
**Research Sources**:
   * CEO/Founder interviews on brand vision and mission
   * CMO interviews on marketing strategy and voice
   * Product team interviews on design philosophy
   * Customer experience team interviews on audience insights
   * Executive interviews on competitive positioning

**Interview Integration Process**:
```python
# Check for new interview transcripts
interview_files = scan_brand_interviews(brand_url)
if new_interviews_detected(interview_files):
    # Process each interview transcript
    for interview in new_interviews:
        insights = extract_interview_insights(interview)
        validate_insights_quality(insights, threshold=9.0)
    
    # Synthesize with existing research phases
    synthesized_insights = synthesize_interview_data(insights)
    update_phase("interview_insights", synthesized_insights)
```

#### **Phase 7: Linearity Analysis (2-4 minutes)**

**Focus**: Analyze brand and product positioning on the linear vs non-linear shopping behavior spectrum
**Cache Duration**: 3-6 months (semi-stable patterns)
**Research Sources**:
   * Product catalog analysis for specification vs aesthetic focus
   * Brand messaging analysis for technical vs emotional language patterns
   * Customer review sentiment analysis for purchase decision factors
   * Competitive positioning analysis for market approach
   * Target audience shopping behavior patterns

**LLM Analysis Process**:
```python
# Dynamic linearity classification using LLMs
async def analyze_brand_linearity_patterns(brand_url: str, research_data: dict):
    """Use LLMs to classify brand/product linearity without hardcoded categories"""
    
    # Analyze product catalog patterns
    product_analysis = await llm_router.chat_completion(
        model="claude-4.0-sonnet",
        system=await prompt_manager.get_prompt("liddy/catalog/research/product_linearity_classifier"),
        messages=[{
            "role": "user",
            "content": f"Analyze this product catalog for shopping behavior patterns:\n{product_catalog_sample}"
        }]
    )
    
    # Analyze brand messaging patterns
    brand_voice_analysis = await llm_router.chat_completion(
        model="claude-4.0-sonnet", 
        system=await prompt_manager.get_prompt("liddy/catalog/research/brand_voice_linearity"),
        messages=[{
            "role": "user",
            "content": f"Analyze this brand messaging for technical vs emotional emphasis:\n{brand_messaging_data}"
        }]
    )
    
    # Synthesize linearity insights
    return synthesize_linearity_analysis(product_analysis, brand_voice_analysis)
```

### 4.8 Linearity-Specific Intelligence Requirements

**Critical Insight**: Different shopping psychology requires different **TYPES** of brand intelligence, not just different presentation of the same information.

#### **Linear Brand Intelligence (Technical/Objective)**
```python
linear_intelligence_requirements = {
    "engineering_principles": "Core design methodology, manufacturing philosophy",
    "performance_standards": "Quality metrics, testing protocols, benchmarks", 
    "technical_advantages": "Unique engineering solutions, patents, innovations",
    "competitive_analysis": "Head-to-head specification comparisons",
    "value_propositions": "Price/performance ratios, ROI justification",
    "customer_education": "How to evaluate specs, use case matching"
}
```

**Brand Interview Focus (Linear)**:
- **CTO/Engineering Lead**: Technical philosophy, innovation approach
- **Quality Director**: Testing protocols, standards, certifications  
- **Product Manager**: Customer technical needs, specification education
- **Customer Support**: Most common technical questions, support patterns

#### **Non-Linear Brand Intelligence (Aesthetic/Subjective)**
```python
nonlinear_intelligence_requirements = {
    "design_language": "Visual codes, signature elements, aesthetic principles",
    "style_guides": "Often undocumented! Color palettes, proportions, materials",
    "creative_influences": "Cultural, artistic, historical inspirations",
    "lifestyle_positioning": "Target aspiration, identity expression, cultural fit",
    "brand_personality": "Emotional associations, storytelling themes",
    "seasonal_evolution": "How collections develop, creative direction patterns"
}
```

**Brand Interview Focus (Non-Linear)**:
- **Creative Director**: Design philosophy, aesthetic vision, style guide details
- **Brand Director**: Lifestyle positioning, cultural relevance, aspiration creation
- **Head of Design**: Creative process, inspiration sources, aesthetic consistency
- **CMO**: Brand personality, emotional triggers, storytelling approach

#### **The Style Guide Problem (Fashion Brands)**
High fashion brands like Balenciaga often have **undocumented style guides** that can only be captured through brand interviews:

```python
fashion_style_capture_questions = [
    "Do you have internal style guides or brand books that define your aesthetic?",
    "What visual elements must be present for something to 'look like' your brand?", 
    "What proportion, silhouette, or fit principles define your design language?",
    "How do you maintain aesthetic consistency across different product categories?",
    "What cultural references or artistic influences shape your creative direction?",
    "How do you evolve your aesthetic while maintaining brand recognition?",
    "What aesthetic mistakes or 'off-brand' elements do you avoid?"
]
```

#### **Brand Interview Evaluation & Gap Analysis**
```python
class LinearityBasedInterviewEvaluator:
    async def evaluate_intelligence_gaps(self, brand_url: str, linearity_pattern: str):
        """Determine what additional interviews are needed based on linearity type"""
        
        if linearity_pattern == 'highly_nonlinear':
            # Fashion/lifestyle brands need style guide capture
            if not self.has_style_guide_documentation(brand_url):
                recommend_interview('creative_director', priority='critical')
            if not self.has_lifestyle_positioning(brand_url):
                recommend_interview('brand_director', priority='high')
                
        elif linearity_pattern == 'highly_linear':
            # Technical brands need engineering philosophy capture
            if not self.has_engineering_philosophy(brand_url):
                recommend_interview('cto_engineering_lead', priority='critical')
            if not self.has_competitive_analysis(brand_url):
                recommend_interview('product_manager', priority='high')
        
        return gaps, recommendations
```

#### **Phase 8: Integration & Synthesis (1-2 minutes)**

**Focus**: Combine all phases into unified brand intelligence
**Triggers**: Runs automatically when any phase is refreshed
**Process**:
   * Cross-phase validation and consistency checking
   * Conflict resolution between different phases
   * Interview insights integration with research data
   * Linearity analysis integration across all phases
   * Unified brand_details.md generation with phase metadata
   * Quality scoring and confidence assessment

### 4.3 Langfuse Prompt Management & Quality Evaluation

#### **Prompt Management Strategy**

All research phases use Langfuse-managed prompts with versioning and A/B testing:

```python
# Langfuse prompt structure for brand research
langfuse_prompts = {
    "liddy/catalog/research/foundation": {
        "description": "Extract core brand identity and founding story",
        "variables": ["brand_name", "research_data", "source_urls"],
        "version": "1.2.3",
        "quality_metrics": ["accuracy", "completeness", "consistency"]
    },
    "liddy/catalog/research/interview_synthesis": {
        "description": "Synthesize brand insights from interview transcripts", 
        "variables": ["interview_transcript", "brand_context", "interview_type"],
        "version": "1.0.1",
        "quality_metrics": ["authenticity", "insight_depth", "voice_capture"]
    }
}
```

#### **Quality Evaluation & Feedback Loop**

Each research phase is evaluated by an LLM judge using specific quality criteria:

```python
# Quality evaluation process
async def evaluate_phase_quality(phase_name: str, phase_output: dict, brand_url: str):
    evaluator_prompt = await prompt_manager.get_prompt(
        f"liddy/catalog/quality/{phase_name}_evaluator"
    )
    
    # LLM evaluation with specific criteria
    evaluation = await llm_router.chat_completion(
        model="claude-4.0-sonnet",  # High-quality evaluator
        system=evaluator_prompt.system_message,
        messages=[{
            "role": "user", 
            "content": f"Evaluate this {phase_name} research:\n{phase_output}"
        }]
    )
    
    quality_score = extract_quality_score(evaluation)
    improvement_feedback = extract_improvement_suggestions(evaluation)
    
    # Store evaluation results
    await store_quality_evaluation(brand_url, phase_name, {
        "score": quality_score,
        "feedback": improvement_feedback,
        "evaluator_model": "claude-3-5-sonnet",
        "timestamp": datetime.now(),
        "criteria_met": analyze_criteria_compliance(evaluation)
    })
    
    return quality_score, improvement_feedback

# Feedback loop with re-runs
async def research_phase_with_quality_control(phase_name: str, brand_url: str):
    max_attempts = 3
    quality_threshold = research_phases[phase_name]["quality_threshold"]
    
    for attempt in range(max_attempts):
        # Execute research phase
        phase_output = await execute_research_phase(phase_name, brand_url)
        
        # Evaluate quality
        quality_score, feedback = await evaluate_phase_quality(
            phase_name, phase_output, brand_url
        )
        
        if quality_score >= quality_threshold:
            logger.info(f"Phase {phase_name} passed quality check: {quality_score}/10")
            return phase_output
        
        elif attempt < max_attempts - 1:
            logger.warning(f"Phase {phase_name} quality below threshold ({quality_score}/{quality_threshold}). Re-running with feedback.")
            # Add feedback to next attempt context
            add_improvement_context(phase_name, feedback)
        
        else:
            logger.error(f"Phase {phase_name} failed quality check after {max_attempts} attempts")
            # Mark as low confidence but proceed
            phase_output["quality_warning"] = True
            phase_output["final_score"] = quality_score
            return phase_output
```

#### **Quality Metrics by Phase**

```python
quality_criteria = {
    "foundation": {
        "accuracy": "Historical facts verified across multiple sources",
        "completeness": "All core brand elements identified",
        "consistency": "No contradictions in brand narrative"
    },
    "voice_messaging": {
        "authenticity": "Voice patterns match brand's actual communication",
        "currency": "Reflects most recent brand messaging",
        "distinctiveness": "Captures unique brand voice characteristics"
    },
    "interview_insights": {
        "authenticity": "Accurately represents interviewee's stated views",
        "insight_depth": "Extracts meaningful strategic insights",
        "voice_capture": "Preserves authentic brand voice from source"
    }
}
```

### 4.4 AI Brand Ethos Voice Interview Integration

#### **Interview Types & Integration**

```python
interview_types = {
    "ceo_brand_vision": {
        "focus": "Overall brand strategy, mission, long-term vision",
        "integration_phases": ["foundation", "market_positioning"],
        "priority": "highest"
    },
    "cmo_marketing_strategy": {
        "focus": "Brand voice, messaging, customer targeting",
        "integration_phases": ["voice_messaging", "customer_cultural"],
        "priority": "high"
    },
    "designer_product_philosophy": {
        "focus": "Product design principles, aesthetic vision",
        "integration_phases": ["product_style", "foundation"],
        "priority": "high"
    },
    "cx_customer_insights": {
        "focus": "Customer behavior, preferences, pain points",
        "integration_phases": ["customer_cultural"],
        "priority": "medium"
    }
}
```

#### **Interview Processing Pipeline**

```python
# Automatic interview detection and processing
async def process_brand_interviews(brand_url: str):
    interview_dir = f"accounts/{brand_url}/brand_interviews/"
    
    # Scan for new interview transcripts
    new_interviews = detect_new_transcripts(interview_dir)
    
    for interview_file in new_interviews:
        # Extract interview metadata from filename/content
        interview_type = classify_interview_type(interview_file)
        interview_date = extract_date_from_filename(interview_file)
        
        # Process interview with specialized prompt
        insights = await extract_interview_insights(
            transcript=load_transcript(interview_file),
            interview_type=interview_type,
            brand_context=load_existing_brand_context(brand_url)
        )
        
        # Quality evaluation for interview insights
        quality_score = await evaluate_interview_quality(insights, interview_type)
        
        if quality_score >= 9.0:  # High threshold for direct brand voice
            # Update relevant research phases
            affected_phases = interview_types[interview_type]["integration_phases"]
            for phase in affected_phases:
                await update_phase_with_interview_insights(phase, insights)
            
            # Trigger integration phase refresh
            await refresh_integration_phase(brand_url)
        
        # Store interview processing metadata
        await update_interview_metadata(interview_file, {
            "processed_date": datetime.now(),
            "quality_score": quality_score,
            "insights_extracted": len(insights),
            "phases_updated": affected_phases
        })
```

### 4.5 Enhanced Command Line Interface

```bash
# Full brand research (all 7 phases including interview synthesis) - New brands
python src/research/brand_researcher.py --brand specialized.com --full-research

# Selective phase refresh - Existing brands
python src/research/brand_researcher.py --brand specialized.com --phases foundation,market_positioning
python src/research/brand_researcher.py --brand specialized.com --phases product_style  # Fashion collection update
python src/research/brand_researcher.py --brand specialized.com --phases voice_messaging  # Campaign update

# AI Brand Ethos Voice Interview processing
python src/research/brand_researcher.py --brand specialized.com --process-interviews
python src/research/brand_researcher.py --brand specialized.com --interview-file "2024-12-15_ceo_brand_vision.txt"

# Quality evaluation and feedback loops
python src/research/brand_researcher.py --brand specialized.com --evaluate-quality --phases all
python src/research/brand_researcher.py --brand specialized.com --phases voice_messaging --quality-threshold 8.5

# Automated triggers based on cache expiration
python src/research/brand_researcher.py --brand specialized.com --check-stale-phases
python src/research/brand_researcher.py --brand specialized.com --auto-refresh --include-interviews

# Force refresh with quality control
python src/research/brand_researcher.py --brand specialized.com --phases product_style --force-refresh --max-quality-attempts 3

# Batch operations with quality monitoring
python src/research/brand_researcher.py --batch --phases product_style --brands "fashion-brand1.com,fashion-brand2.com" --quality-report

# Interview-specific commands
python src/research/brand_researcher.py --brand specialized.com --scan-new-interviews
python src/research/brand_researcher.py --brand specialized.com --interview-impact-analysis
python src/research/brand_researcher.py --all-brands --process-pending-interviews
```

### 4.4 Phase Storage Structure

```
accounts/<brand_url>/
├── brand_details.md                    # Final integrated brand intelligence
├── research_phases/
│   ├── foundation.json                 # Core brand identity (6-12mo cache)
│   ├── market_positioning.json         # Competitive landscape (3-6mo cache)
│   ├── product_style.json              # Products & collections (1-3mo cache)
│   ├── customer_cultural.json          # Audience & culture (2-4mo cache)
│   ├── voice_messaging.json            # Brand voice & campaigns (1-2mo cache)
│   ├── interview_insights.json         # AI Brand Ethos Voice Interview synthesis
│   └── integration_metadata.json       # Synthesis tracking and conflicts
├── brand_interviews/                   # AI Brand Ethos Voice Interview transcripts
│   ├── 2024-12-15_ceo_brand_vision.txt        # CEO interview on brand vision
│   ├── 2024-12-20_cmo_marketing_strategy.txt  # CMO interview on marketing
│   ├── 2025-01-10_designer_product_philosophy.txt # Designer on product philosophy
│   └── interview_metadata.json        # Interview tracking and analysis
├── research_quality/                   # Quality evaluation and feedback
│   ├── phase_evaluations.json         # LLM quality scores for each phase
│   ├── improvement_feedback.json      # Feedback for phase re-runs
│   └── quality_history.json           # Quality tracking over time
└── research_history/
    ├── 2024-12-15_full_research.json   # Historical research snapshots
    └── 2024-12-20_product_style.json   # Incremental updates
```

### 4.5 Brand Details Schema (Phase-Based with Metadata)

Generated `brand_details.md` contains comprehensive brand intelligence with research provenance:

```markdown
# Brand Intelligence: {Brand Name}
*Generated: {timestamp} | Total Research Time: {duration} minutes | Sources: {source_count}*

## Research Metadata & Phase History
- **Foundation Phase**: Last updated {date} | Quality: {8.2/10} | Confidence: {High/Medium/Low} | Next refresh: {date}
- **Market Positioning**: Last updated {date} | Quality: {7.8/10} | Confidence: {High/Medium/Low} | Next refresh: {date}  
- **Product & Style**: Last updated {date} | Quality: {8.5/10} | Confidence: {High/Medium/Low} | Next refresh: {date}
- **Customer & Cultural**: Last updated {date} | Quality: {7.9/10} | Confidence: {High/Medium/Low} | Next refresh: {date}
- **Voice & Messaging**: Last updated {date} | Quality: {8.7/10} | Confidence: {High/Medium/Low} | Next refresh: {date}
- **Interview Insights**: {count} interviews processed | Quality: {9.1/10} | Latest: {date} | Type: {CEO/CMO/Designer}
- **Linearity Analysis**: Last updated {date} | Quality: {8.4/10} | Brand Pattern: {Technical/Emotional/Mixed} | Product Range: {0.2-0.9}
- **Integration Quality Score**: {8.3/10} based on phase consistency and cross-validation
- **Phase Conflicts Detected**: {any inconsistencies between phases and interviews}
- **Quality Re-runs**: {count} phases improved through feedback loops
- **Historical Snapshots**: {count} research iterations tracked

## Company Overview
- **Founded**: Year and founding story *(Confidence: High/Medium/Low)*
- **Mission**: Core mission and values *(Source: Official website/Annual report)*
- **Vision**: Long-term company vision
- **Revenue/Scale**: Company size and market presence

## Target Audience (Psychographic Profile)
- **Primary Demographics**: Age, income, lifestyle, geography
- **Psychographics**: Values, interests, motivations, pain points
- **Customer Personas**: 2-3 detailed customer archetypes
- **Use Cases**: Primary and secondary use cases with context
- **Customer Language**: How customers describe the brand

## Brand Positioning & Strategy
- **Market Category**: Primary vertical and sub-verticals
- **Competitive Position**: How they differentiate in market
- **Key Differentiators**: Unique selling propositions with evidence
- **Price Positioning**: Value, premium, luxury with justification
- **Strategic Focus**: Current business priorities and initiatives

## Brand Voice & Messaging Framework
- **Tone Analysis**: Professional, casual, technical, friendly *(with examples)*
- **Key Themes**: Innovation, performance, sustainability *(frequency analysis)*
- **Language Patterns**: Technical depth, accessibility, emotional appeals
- **Messaging Pillars**: 3-5 core brand messages with supporting evidence
- **Content Style**: Blog tone, social media voice, marketing approach

## Product Philosophy & Approach
- **Design Principles**: What drives product development *(with examples)*
- **Quality Standards**: Manufacturing and materials approach
- **Innovation Focus**: Technology, sustainability, performance priorities
- **Product Range**: Breadth, categorization, and portfolio strategy
- **Development Process**: How they approach new product creation

## Brand Heritage & Evolution
- **Founding Story**: Origin story with cultural context
- **Key Milestones**: Major developments with dates and impact
- **Brand Evolution**: How positioning/messaging has changed over time
- **Leadership Influence**: Founder/CEO impact on brand direction
- **Cultural Impact**: Influence on industry, community, or culture

## Competitive Intelligence
- **Direct Competitors**: 3-5 main competitive brands with analysis
- **Competitive Advantages**: What specifically sets them apart
- **Market Challenges**: Industry pressures and brand responses
- **Innovation Leadership**: Areas of technological or market leadership
- **Pricing Strategy**: How they position vs competitors

## Market Context & Industry Position
- **Industry Trends**: Relevant market dynamics affecting the brand
- **Customer Sentiment**: Overall brand perception from reviews/social
- **Growth Trajectory**: Recent performance and market expansion
- **Partnerships**: Key strategic relationships and collaborations
- **Future Outlook**: Strategic direction and market opportunities

## Shopping Behavior & Linearity Analysis *(NEW)*
- **Overall Brand Pattern**: {Technical/Emotional/Mixed} - {0.6 linearity score}
- **Product Linearity Range**: {0.2 - 0.9} across product catalog
- **Category Patterns**: 
  - High-Performance Products: {0.8-0.9} - Technical/specification focus
  - Lifestyle Products: {0.2-0.4} - Emotional/experience focus
  - Mainstream Products: {0.5-0.7} - Balanced approach
- **Customer Shopping Psychology**: 
  - Research-Driven Segment: {40%} - Values specs, comparisons, expert opinions
  - Inspiration-Driven Segment: {25%} - Values style, emotion, lifestyle alignment
  - Mixed Decision-Makers: {35%} - Balances function and form
- **Optimal Descriptor Strategy**: {Brand-specific recommendations}
- **AI Sales Agent Conversation Approach**: {Adaptive strategies by product type}
- **Brand Voice Adaptation Guidelines**: {Technical vs emotional emphasis by product}

---
*Research Sources: {detailed source list with URLs and confidence ratings}*
```

### 4.4 Research Quality Controls & Implementation

#### **Ensuring Research Depth (Not Speed)**

```python
# Brand Research Quality Controls
class BrandResearchPipeline:
    MIN_RESEARCH_TIME = 300  # 5 minutes minimum
    MAX_RESEARCH_TIME = 900  # 15 minutes maximum
    MIN_SOURCES = 15         # Minimum data sources
    MIN_LLM_ROUNDS = 5       # Minimum analysis rounds
    
    async def research_brand(self, brand_url: str) -> BrandIntelligence:
        start_time = time.time()
        
        # Phase 1: Comprehensive data gathering
        sources = await self._gather_comprehensive_sources(brand_url)
        if len(sources) < self.MIN_SOURCES:
            await self._expand_search_scope(brand_url)
            
        # Phase 2: Multi-round LLM analysis with different prompts
        analysis_rounds = []
        for round_num in range(self.MIN_LLM_ROUNDS):
            analysis = await self._analyze_with_prompt_variation(sources, round_num)
            analysis_rounds.append(analysis)
            
        # Phase 3: Cross-validation and synthesis
        brand_intelligence = await self._synthesize_and_validate(analysis_rounds)
        
        # Quality check: Ensure minimum research time for thoroughness
        elapsed = time.time() - start_time
        if elapsed < self.MIN_RESEARCH_TIME:
            await self._additional_deep_dive(brand_intelligence, remaining_time)
            
        return brand_intelligence
```

#### **Multi-Source Research Strategy**

1. **Primary Sources (Official)**:
   ```python
   official_sources = [
       f"https://{brand_url}/about",
       f"https://{brand_url}/mission", 
       f"https://{brand_url}/values",
       f"https://{brand_url}/story",
       f"https://{brand_url}/leadership",
       # + investor relations, press releases, etc.
   ]
   ```

2. **Secondary Sources (Market Intelligence)**:
   ```python
   research_queries = [
       f"{brand_name} company history founding story",
       f"{brand_name} target audience customer demographics", 
       f"{brand_name} competitors comparison analysis",
       f"{brand_name} brand positioning strategy",
       f"{brand_name} customer reviews sentiment analysis",
       # + 10-15 additional targeted queries
   ]
   ```

3. **Tertiary Sources (Cultural Context)**:
   ```python
   contextual_research = [
       f"{brand_name} industry influence thought leadership",
       f"{brand_name} awards recognition certifications",
       f"{brand_name} partnerships collaborations",
       f"{brand_name} innovation patents technology",
       # + cultural impact, community presence
   ]
   ```

#### **Advanced LLM Analysis Rounds**

Each analysis round uses different advanced models and prompt strategies:

```python
analysis_rounds = [
    {
        "model": "o1-preview",
        "focus": "Pattern identification and data synthesis",
        "prompt": "Analyze this brand data for patterns..."
    },
    {
        "model": "claude-3-5-sonnet", 
        "focus": "Brand voice and messaging analysis",
        "prompt": "Extract authentic brand voice patterns..."
    },
    {
        "model": "o1-mini",
        "focus": "Competitive positioning analysis", 
        "prompt": "Compare this brand against competitors..."
    },
    {
        "model": "claude-3-5-sonnet",
        "focus": "Target audience profiling",
        "prompt": "Build detailed customer personas..."
    },
    {
        "model": "o1-preview",
        "focus": "Cross-validation and consistency",
        "prompt": "Verify consistency across sources..."
    }
]
```

#### **Research Time Investment Justification**

**5-15 minutes is appropriate because**:
- **Data Gathering**: 3-7 minutes for 15-30 diverse sources
- **LLM Analysis**: 2-5 minutes for 5 rounds of advanced model analysis
- **Synthesis**: 1-3 minutes for cross-validation and final generation
- **Quality Control**: Built-in verification and depth checking

**This produces**:
✅ **Authentic brand voice** for descriptor generation
✅ **Deep target audience insights** for AI sales agent conversations  
✅ **Competitive intelligence** for positioning and differentiation
✅ **Cultural context** for authentic brand representation
✅ **Strategic insight** for business-aware AI interactions

### 4.6 Phase-Based Use Cases & Refresh Strategies

#### **Fashion Brand Example: Seasonal Collection Updates**
```python
# Gucci launches Fall 2024 collection
python src/research/brand_researcher.py --brand gucci.com --phases product_style --force-refresh

# Result: Updated style intelligence without re-researching company history
# Cost: $1-2 vs $6-8 for full research
# Time: 2-3 minutes vs 15 minutes
# Impact: Fresh collection data for descriptor generation
```

#### **Tech Startup Example: Rapid Product Evolution** 
```python
# Startup pivots product strategy or launches new features
python src/research/brand_researcher.py --brand startup.com --phases market_positioning,product_style

# Result: Updated competitive position and product focus
# Preserves: Stable company foundation and cultural data
```

#### **Luxury Brand Example: Campaign Refresh**
```python
# Luxury brand launches new advertising campaign with different voice
python src/research/brand_researcher.py --brand luxury-brand.com --phases voice_messaging

# Result: Updated brand voice for authentic descriptor generation
# Preserves: Core brand heritage and positioning (expensive to research)
```

#### **B2B Brand Example: Competitive Landscape Shift**
```python
# New competitor enters market or major acquisition happens
python src/research/brand_researcher.py --brand b2b-company.com --phases market_positioning

# Result: Updated competitive intelligence and positioning
# Preserves: Customer profiles and brand voice (still relevant)
```

### 4.7 Integration with Product Processing Pipeline

Phase-based brand intelligence directly enhances product catalog processing:

```python
# Product descriptor generation with phase-aware context
async def generate_brand_aware_descriptor(product: Product, brand_url: str):
    # Load specific phases needed for descriptor generation
    product_style = await load_research_phase(brand_url, "product_style")
    voice_messaging = await load_research_phase(brand_url, "voice_messaging") 
    customer_cultural = await load_research_phase(brand_url, "customer_cultural")
    
    # Generate descriptor with fresh style and voice intelligence
    prompt = build_descriptor_prompt(
        product=product,
        style_context=product_style,
        brand_voice=voice_messaging,
        target_audience=customer_cultural
    )
    
    return await llm_router.generate(prompt)
```

**Benefits of Phase-Based Integration**:
- **Always Fresh Style Data**: Fashion descriptors use latest collection intelligence
- **Authentic Voice**: Descriptors match current brand messaging and campaigns  
- **Cost Efficiency**: Only refresh phases that impact descriptor quality
- **Performance**: Faster descriptor generation with targeted brand context

---

## 5. Phase 1: Product Catalog Ingestion

### 5.1 Linearity-Aware Brand Processing

1. **Load brand intelligence**: Load `brand_details.md` with linearity analysis phase
2. **List** all `products.json` files under `accounts/` in your GCP bucket
3. **For each** brand URL: Download and parse JSON → list of `Product` objects
4. **Classify product linearity**: Use LLM-based dynamic classification for each product
5. **Adapt processing strategy**: Apply linearity-aware descriptor generation and voice modulation

### 5.2 Dynamic Linearity Classification

```python
class LinearityAwareProductProcessor:
    async def process_product_catalog(self, brand_url: str, products: List[Product]):
        """Process products with dynamic linearity classification"""
        
        # Load brand linearity patterns from research
        brand_intelligence = await self.load_brand_intelligence(brand_url)
        brand_linearity_patterns = brand_intelligence.get('linearity_analysis', {})
        
        processed_products = []
        for product in products:
            # Classify individual product linearity using LLMs
            linearity_score = await self.classify_product_linearity(product, brand_linearity_patterns)
            
            # Generate linearity-aware descriptor
            descriptor = await self.generate_adaptive_descriptor(product, brand_intelligence, linearity_score)
            
            # Update product with linearity metadata
            product.linearity_score = linearity_score
            product.descriptor = descriptor
            product.shopping_psychology = self.determine_shopping_psychology(linearity_score)
            
            processed_products.append(product)
        
        return processed_products
    
    async def classify_product_linearity(self, product: Product, brand_patterns: dict) -> float:
        """Use LLM to classify product linearity dynamically"""
        
        classification_prompt = await self.prompt_manager.get_prompt(
            "liddy/catalog/product/linearity_classifier"
        )
        
        result = await self.llm_router.chat_completion(
            model="claude-4.0-sonnet",
            system=classification_prompt.system_message,
            messages=[{
                "role": "user",
                "content": f"""
                Classify this product's shopping behavior pattern:
                
                Product: {product.name}
                Category: {product.category}
                Description: {product.description}
                Features: {product.features}
                Brand Context: {brand_patterns.get('overall_pattern', 'mixed')}
                
                Return a linearity score from 0.0 (fully subjective/emotional) to 1.0 (fully objective/technical)
                with reasoning for the classification.
                """
            }]
        )
        
        return self.extract_linearity_score(result)
```

### 5.3 Adaptive Descriptor Generation

* **Input**: product fields + brand intelligence + dynamic linearity classification
* **Linearity-Aware Prompts**: Adapt descriptor style based on shopping psychology
* **Brand Voice Modulation**: Adjust brand voice emphasis based on product linearity
* **Logic**: Proven sizing instruction preserved, enhanced with linearity context

```python
class LinearityAdaptiveDescriptorGenerator:
    async def generate_adaptive_descriptor(self, product: Product, brand_intelligence: dict, linearity_score: float):
        """Generate descriptors that match shopping psychology"""
        
        if linearity_score > 0.7:  # Highly linear/objective
            return await self.generate_technical_descriptor(product, brand_intelligence)
        elif linearity_score < 0.3:  # Highly non-linear/subjective  
            return await self.generate_emotional_descriptor(product, brand_intelligence)
        else:  # Mixed - balance both approaches
            return await self.generate_balanced_descriptor(product, brand_intelligence, linearity_score)
    
    async def generate_technical_descriptor(self, product: Product, brand_intelligence: dict):
        """Focus on specifications, performance, value, reliability"""
        
        technical_prompt = await self.prompt_manager.get_prompt(
            "liddy/catalog/descriptors/technical_linear"
        )
        
        # Extract technical brand voice elements
        brand_voice = brand_intelligence.get('voice_messaging', {})
        technical_themes = self.extract_technical_themes(brand_voice)
        
        return await self.llm_router.chat_completion(
            model="gpt-4",
            system=technical_prompt.format(
                brand_voice=technical_themes,
                performance_focus=brand_intelligence.get('key_differentiators', []),
                target_audience="research-oriented, specification-focused buyers"
            ),
            messages=[{
                "role": "user",
                "content": f"Generate technical descriptor for: {product.to_dict()}"
            }]
        )
    
    async def generate_emotional_descriptor(self, product: Product, brand_intelligence: dict):
        """Focus on lifestyle, emotion, identity, inspiration"""
        
        emotional_prompt = await self.prompt_manager.get_prompt(
            "liddy/catalog/descriptors/emotional_nonlinear"
        )
        
        # Extract emotional brand voice elements
        brand_voice = brand_intelligence.get('voice_messaging', {})
        emotional_themes = self.extract_emotional_themes(brand_voice)
        
        return await self.llm_router.chat_completion(
            model="gpt-4",
            system=emotional_prompt.format(
                brand_voice=emotional_themes,
                lifestyle_focus=brand_intelligence.get('target_audience', {}),
                target_audience="style-conscious, emotion-driven, experience-focused buyers"
            ),
            messages=[{
                "role": "user", 
                "content": f"Generate lifestyle descriptor for: {product.to_dict()}"
            }]
        )
```

---

## 6. Phase 2: Knowledge Base Ingestion

### 6.1 Linearity-Aware Knowledge Structuring

* **Include brand_details.md**: Auto-ingest generated brand profile with linearity analysis into knowledge base
* **Linearity-Enhanced Chunks**: All knowledge chunks enriched with linearity metadata for appropriate retrieval
* **Psychology-Matched Information**: Structure knowledge to support both technical and emotional conversation styles
* **AI sales agent ready**: Complete brand context with shopping psychology insights for adaptive conversations

### 6.2 RAG Integration with Shopping Psychology

```python
class LinearityAwareKnowledgeIngestor:
    async def ingest_brand_knowledge(self, brand_url: str):
        """Ingest knowledge with linearity awareness for AI sales agents"""
        
        # Load brand intelligence with linearity analysis
        brand_intelligence = await self.load_brand_intelligence(brand_url)
        brand_linearity_patterns = brand_intelligence.get('linearity_analysis', {})
        
        # Process brand profile for RAG
        brand_chunks = await self.chunk_brand_intelligence(brand_intelligence)
        
        # Process product catalog with linearity metadata
        products = await self.load_products(brand_url)
        product_chunks = []
        
        for product in products:
            # Add linearity context to product chunks
            linearity_context = {
                'linearity_score': product.linearity_score,
                'shopping_psychology': product.shopping_psychology,
                'conversation_style': self.determine_conversation_style(product.linearity_score),
                'emphasis_points': self.extract_emphasis_points(product, brand_linearity_patterns)
            }
            
            chunks = await self.create_linearity_aware_chunks(product, linearity_context)
            product_chunks.extend(chunks)
        
        # Ingest with linearity metadata for retrieval
        await self.ingest_chunks_with_linearity(brand_chunks + product_chunks, brand_url)
    
    def determine_conversation_style(self, linearity_score: float) -> dict:
        """Determine optimal AI sales agent conversation approach"""
        
        if linearity_score > 0.7:
            return {
                'approach': 'consultative_technical',
                'language': 'precise_data_driven',
                'focus': 'specifications_performance',
                'questions': 'needs_requirements_based',
                'information_depth': 'detailed_technical'
            }
        elif linearity_score < 0.3:
            return {
                'approach': 'inspirational_discovery',
                'language': 'evocative_lifestyle',
                'focus': 'emotion_experience',
                'questions': 'preference_aspiration_based',
                'information_depth': 'experiential_storytelling'
                         }
         else:
             return {
                 'approach': 'balanced_adaptive',
                 'language': 'accessible_comprehensive',
                 'focus': 'function_and_form',
                 'questions': 'mixed_discovery',
                 'information_depth': 'balanced_technical_emotional'
             }

### 6.3 Linearity-Specific Knowledge Structuring Examples

#### **Non-Linear RAG Optimization (Fashion/Lifestyle)**
```python
# Style guide chunk optimized for aesthetic queries
style_guide_chunk = {
    'content': """
    Brand Aesthetic & Design Language:
    - Visual Identity: Oversized silhouettes, architectural shapes, monochromatic palettes
    - Signature Elements: Logo treatments, statement colors, technical fabrics
    - Creative Philosophy: Disruption of luxury conventions, cultural commentary
    - Cultural Influences: Streetwear meets couture, youth culture appropriation
    """,
    'metadata': {
        'chunk_type': 'aesthetic_guide',
        'query_types': ['style_questions', 'aesthetic_guidance', 'design_inspiration'],
        'conversation_style': 'inspirational_discovery',
        'linearity_optimized': 'high_nonlinear'
    }
}
```

#### **Linear RAG Optimization (Technical/Performance)**
```python
# Technical specifications chunk optimized for performance queries
technical_specs_chunk = {
    'content': """
    Engineering & Performance Standards:
    - Technical Philosophy: Precision engineering, rigorous testing protocols
    - Performance Metrics: Frame stiffness, aerodynamic efficiency, weight optimization
    - Quality Standards: Carbon fiber lay-up processes, stress testing, durability certification
    - Innovation Focus: Advanced materials research, biomechanical optimization
    """,
    'metadata': {
        'chunk_type': 'technical_specs',
        'query_types': ['specification_questions', 'performance_comparisons', 'technical_details'],
        'conversation_style': 'consultative_technical',
        'linearity_optimized': 'high_linear'
    }
}
```

---

## 7. Zero-to-RAG Pipeline Execution

### 7.1 Command-Line Interface

```bash
# Phase 0: Brand Intelligence Generation

## Full Brand Research (New Brands)
python src/research/brand_researcher.py --brand specialized.com --full-research

## Selective Phase Updates (Existing Brands) 
python src/research/brand_researcher.py --brand gucci.com --phases product_style               # New collection
python src/research/brand_researcher.py --brand startup.com --phases market_positioning       # Competitive shift  
python src/research/brand_researcher.py --brand luxury.com --phases voice_messaging           # Campaign update
python src/research/brand_researcher.py --brand specialized.com --phases linearity_analysis   # Refresh shopping behavior patterns

## Automated Maintenance
python src/research/brand_researcher.py --brand specialized.com --auto-refresh                # Check stale phases
python src/research/brand_researcher.py --all-brands --check-stale-phases                     # Batch maintenance

## Batch Fashion Brand Updates
python src/research/brand_researcher.py --batch --phases product_style --brands "gucci.com,prada.com,versace.com"

# Phase 1: Product Catalog Ingestion (Linearity-Aware)
python src/product_ingestor.py --full-sync --brand specialized.com                            # Uses latest brand intelligence + linearity analysis
python src/product_ingestor.py --incremental --ids 123,456 --refresh-descriptors              # Force linearity-aware descriptor refresh
python src/product_ingestor.py --analyze-linearity --brand specialized.com                    # Analyze and classify product linearity patterns

# Phase 2: Knowledge Base Ingestion (Psychology-Aware)
python src/knowledge_ingestor.py --brand specialized.com                                       # Includes brand intelligence + linearity metadata
python src/knowledge_ingestor.py --brand specialized.com --linearity-focus technical          # Optimize for technical/linear conversations
python src/knowledge_ingestor.py --brand specialized.com --linearity-focus emotional          # Optimize for emotional/non-linear conversations

# Complete Zero-to-RAG Pipeline (Phase-Aware)
python src/research/brand_researcher.py --brand new-brand.com --full-research                 # 15-20 minutes deep research
python src/product_ingestor.py --full-sync --brand new-brand.com                             # Brand-aware descriptors
python src/knowledge_ingestor.py --brand new-brand.com                                        # Complete knowledge base
```

### 7.2 Interactive Testing

```bash
# Interactive testing notebooks
jupyter notebook notebooks/manual_testing.ipynb
jupyter notebook notebooks/brand_research.ipynb

# Testing includes:
# - Brand intelligence generation and analysis
# - Web search and research capabilities
# - Brand-aware descriptor generation
# - Complete zero-to-RAG pipeline testing
```

---

## 8. Implementation Strategy & Execution Plan

### 8.1 Implementation Phases & Dependencies

#### **Phase 0A: Core Infrastructure (Week 1-2)**
**Dependencies**: None
**Deliverables**:
- [ ] Basic project structure setup and validation
- [ ] Environment configuration with all required API keys
- [ ] Core storage abstraction (storage.py) with GCP + local modes
- [ ] Basic LLM service integration (OpenAI primary, others optional initially)
- [ ] Langfuse connection and basic prompt management
- [ ] Configuration management system (src/config/settings.py)

**Validation Checkpoints**:
- [ ] Can store/retrieve files from GCP bucket (test with small file)
- [ ] Can make successful LLM calls with Langfuse tracking
- [ ] Environment variables properly loaded and validated
- [ ] All dependencies install correctly

**Estimated Time**: 1-2 weeks
**Success Criteria**: Basic infra tests pass, environment fully configured

#### **Phase 0B: Workflow State Foundation (Week 2-3)**
**Dependencies**: Core Infrastructure complete
**Deliverables**:
- [ ] Workflow state manager implementation (src/workflow/workflow_state_manager.py)
- [ ] Brand discovery and state tracking from filesystem
- [ ] CLI integration for workflow commands (next-step, resume, workflow, history)
- [ ] Integration with brand manager scripts
- [ ] Basic error handling and state persistence

**Validation Checkpoints**:
- [ ] Can track brand workflow states correctly
- [ ] CLI commands work for status/next-step/resume operations
- [ ] State persistence working across script executions
- [ ] Workflow transitions follow correct logic

**Estimated Time**: 1 week
**Success Criteria**: Workflow management fully functional for basic operations

#### **Phase 0C: Brand Research Infrastructure (Week 3-5)**
**Dependencies**: Workflow State Foundation complete
**Deliverables**:
- [ ] Web search integration (Tavily primary, with fallbacks)
- [ ] Basic brand researcher with foundation phase implementation
- [ ] Quality evaluation framework with LLM judges
- [ ] Phase-based storage structure and caching logic
- [ ] Research phase base classes and abstractions

**Validation Checkpoints**:
- [ ] Can research single brand (foundation phase only) successfully
- [ ] Quality evaluation produces consistent 7.0+ scores
- [ ] Results stored in correct JSON format with metadata
- [ ] Caching prevents unnecessary re-research

**Estimated Time**: 2 weeks
**Success Criteria**: Foundation phase research working reliably with quality control

#### **Phase 0D: Complete Brand Research Pipeline (Week 5-7)**
**Dependencies**: Brand Research Infrastructure complete
**Deliverables**:
- [ ] All 7 research phases implemented (foundation through ai_persona_generation)
- [ ] Interview processing pipeline for brand voice capture
- [ ] Linearity analysis integration for shopping psychology
- [ ] Batch processing capabilities with parallel execution
- [ ] Complete CLI interface with all research commands
- [ ] Integration phase for cross-validation and synthesis

**Validation Checkpoints**:
- [ ] Full brand research produces comprehensive brand_details.md
- [ ] Phase caching working correctly with different expiration times
- [ ] Interview processing extracting quality insights
- [ ] Batch operations handle multiple brands efficiently
- [ ] Linearity analysis provides actionable shopping psychology insights

**Estimated Time**: 2 weeks  
**Success Criteria**: Complete brand research pipeline operational with 8.0+ quality scores

#### **Phase 1: Product & Knowledge Integration (Week 7-8)**
**Dependencies**: Complete Brand Research Pipeline
**Deliverables**:
- [ ] Product ingestor with brand-aware descriptor generation
- [ ] Knowledge ingestor with brand intelligence integration
- [ ] Linearity-aware processing for products and knowledge
- [ ] Complete zero-to-RAG pipeline integration
- [ ] Validation and testing of end-to-end pipeline

**Validation Checkpoints**:
- [ ] Products processed with brand-appropriate descriptors
- [ ] Knowledge base includes brand intelligence correctly
- [ ] Linearity analysis improves conversation appropriateness
- [ ] Complete pipeline: brand URL → RAG-ready system

**Estimated Time**: 1 week
**Success Criteria**: Full zero-to-RAG automation working end-to-end

### 8.2 Testing Strategy & Validation Framework

#### **Unit Testing Requirements**
```python
# Core unit tests to implement
unit_tests = [
    "test_workflow_state_transitions",      # All valid state changes
    "test_research_phase_caching",          # Cache expiration logic
    "test_quality_evaluation_scoring",      # Consistent quality metrics
    "test_brand_intelligence_synthesis",    # Cross-phase integration
    "test_linearity_classification",        # Shopping psychology accuracy
    "test_storage_abstraction",             # GCP and local modes
    "test_llm_service_routing",            # Multi-provider handling
    "test_error_handling_recovery"         # Graceful failure modes
]
```

#### **Integration Testing Scenarios**

**Scenario 1: New Brand Complete Onboarding**
```bash
# Test complete pipeline from scratch
1. Start: ./scripts/brand_manager.sh next-step newbrand.com
   Expected: "Start brand onboarding" (not_started state)

2. Execute: ./scripts/brand_manager.sh onboard newbrand.com  
   Expected: Complete pipeline with state transitions:
   not_started → research_in_progress → research_complete → 
   catalog_in_progress → catalog_complete → knowledge_in_progress → 
   knowledge_complete → rag_in_progress → rag_complete → 
   persona_in_progress → persona_complete → pipeline_complete

3. Validate outputs:
   - accounts/newbrand.com/brand_details.md (comprehensive)
   - accounts/newbrand.com/workflow_state.json (pipeline_complete)
   - All research phases cached with quality scores 7.0+

4. Test interruption and resume:
   - Kill process during catalog_in_progress
   - Resume: ./scripts/brand_manager.sh resume newbrand.com
   - Expected: Continues from catalog ingestion step
```

**Scenario 2: Existing Brand Selective Refresh**
```bash
# Test smart maintenance and cost optimization
1. Brand with pipeline_complete state (6 months old)
2. Execute: ./scripts/brand_manager.sh refresh existingbrand.com
3. Expected: Only stale phases refreshed (should be 1-3 phases max)
4. Validate cost: <$5 total (vs $8-15 for full research)
5. Confirm brand_details.md updated with fresh data only
```

**Scenario 3: Batch Operations & Error Handling**
```bash
# Test batch processing resilience
1. Mixed brand states: 3 brands (not_started, research_complete, pipeline_complete)
2. Execute: ./scripts/brand_manager.sh batch-next-steps
3. Expected: Correct next action for each brand
4. Test batch refresh with intentional failures:
   - One brand has network issues
   - One brand hits API rate limits
   - Expected: Other brands continue processing, errors logged properly
```

#### **Quality Assurance Benchmarks**
```python
qa_benchmarks = {
    "research_quality": {
        "average_score": "8.0+ across all phases",
        "consistency": "95% of brands score 7.5+",
        "authenticity": "Brand voice matches actual communication patterns"
    },
    "performance": {
        "full_research_time": "8-12 minutes average (5-15 min acceptable)",
        "phase_refresh_time": "1-4 minutes per phase",
        "workflow_operations": "<2 seconds response time"
    },
    "cost_efficiency": {
        "new_brand_cost": "$8-15 for complete research",
        "maintenance_cost": "$1-4 for typical refresh",
        "quality_rerun_cost": "<$2 additional for improvements"
    },
    "reliability": {
        "pipeline_success_rate": "95%+ end-to-end completion",
        "resume_success_rate": "98%+ successful workflow resumption",
        "batch_operation_resilience": "Continue processing despite individual failures"
    }
}
```

### 8.3 Error Handling & Recovery Strategy

#### **Research Phase Error Handling**
```python
class ComprehensiveErrorHandler:
    async def handle_research_failure(self, phase_name: str, error: Exception, attempt: int):
        """Robust error handling for all research scenarios"""
        
        error_handlers = {
            "RateLimitError": self.handle_rate_limit,
            "WebSearchError": self.handle_search_failure, 
            "LLMError": self.handle_llm_failure,
            "QualityThresholdError": self.handle_quality_failure,
            "NetworkError": self.handle_network_failure,
            "StorageError": self.handle_storage_failure
        }
        
        handler = error_handlers.get(type(error).__name__, self.handle_unknown_error)
        return await handler(phase_name, error, attempt)
    
    async def handle_rate_limit(self, phase_name: str, error: Exception, attempt: int):
        """Handle API rate limiting with exponential backoff"""
        if attempt < 3:
            wait_time = min(300, 2 ** attempt * 30)  # Max 5 minute wait
            logger.warning(f"Rate limited on {phase_name}, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
            return "retry"
        return "fail_with_partial"
    
    async def handle_search_failure(self, phase_name: str, error: Exception, attempt: int):
        """Fallback through search providers: Tavily → GCP → Direct scraping"""
        if "tavily" in error.source and attempt == 1:
            return await self.try_gcp_search(phase_name)
        elif "gcp" in error.source and attempt == 2:
            return await self.try_direct_scraping(phase_name)
        return "fail_with_partial"
    
    async def handle_quality_failure(self, phase_name: str, error: Exception, attempt: int):
        """Quality below threshold - retry with feedback"""
        if attempt < 3:
            await self.add_improvement_context(phase_name, error.feedback)
            return "retry_with_feedback"
        return "accept_lower_quality"  # Mark as low confidence but proceed
```

#### **Workflow Recovery Patterns**
```python
workflow_recovery = {
    "interrupted_onboarding": {
        "detection": "Workflow state exists but pipeline incomplete",
        "recovery": "Resume from last successful step",
        "validation": "Verify previous steps before continuing"
    },
    "corrupted_state": {
        "detection": "Workflow state inconsistent with filesystem",
        "recovery": "Rebuild state from filesystem analysis", 
        "validation": "Cross-check with existing outputs"
    },
    "partial_failure": {
        "detection": "Some phases complete, others failed",
        "recovery": "Retry failed phases only",
        "validation": "Ensure phase dependencies met"
    },
    "network_interruption": {
        "detection": "Network errors during processing",
        "recovery": "Resume with exponential backoff",
        "validation": "Verify data integrity before proceeding"
    }
}
```

### 8.4 Configuration Management System

```python
# src/config/settings.py
from pydantic import BaseSettings, Field
from typing import Dict, Optional

class Settings(BaseSettings):
    """Centralized configuration management"""
    
    # Environment
    ENV: str = Field(default="dev", description="Environment: dev, staging, prod")
    
    # Storage Configuration
    GCP_BUCKET_PROD: str = "liddy-account-documents"
    GCP_BUCKET_DEV: str = "liddy-account-documents-dev"
    
    # LLM Service Configuration
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    # Langfuse Configuration
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str  
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # Web Search Configuration
    TAVILY_API_KEY: Optional[str] = None
    GOOGLE_SEARCH_API_KEY: Optional[str] = None
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = None
    
    # Research Configuration
    MIN_RESEARCH_TIME: int = Field(default=300, description="Minimum research time in seconds")
    MAX_RESEARCH_TIME: int = Field(default=900, description="Maximum research time in seconds") 
    MIN_SOURCES: int = Field(default=15, description="Minimum sources per research phase")
    MIN_LLM_ROUNDS: int = Field(default=5, description="Minimum LLM analysis rounds")
    DEFAULT_QUALITY_THRESHOLD: float = Field(default=8.0, description="Default quality threshold")
    
    # Phase Cache Durations (in days)
    CACHE_DURATIONS: Dict[str, int] = {
        "foundation": 180,          # 6 months - very stable
        "market_positioning": 120,  # 4 months - moderate change
        "product_style": 60,        # 2 months - frequent for fashion
        "customer_cultural": 90,    # 3 months - evolves with society
        "voice_messaging": 30,      # 1 month - changes with campaigns
        "linearity_analysis": 120,  # 4 months - semi-stable patterns
        "rag_optimization": 180,    # 6 months - rarely needs updates
        "ai_persona_generation": 0  # Manual only - never auto-refresh
    }
    
    # Performance Configuration  
    MAX_PARALLEL_BRANDS: int = Field(default=5, description="Max brands to process in parallel")
    DEFAULT_TIMEOUT: int = Field(default=120, description="Default operation timeout in seconds")
    
    @property
    def current_bucket(self) -> str:
        """Get bucket name for current environment"""
        return self.GCP_BUCKET_PROD if self.ENV == "prod" else self.GCP_BUCKET_DEV
    
    @property  
    def cache_duration_for_phase(self, phase_name: str) -> int:
        """Get cache duration for specific research phase"""
        return self.CACHE_DURATIONS.get(phase_name, 90)  # Default 3 months
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
```

### 8.5 Monitoring & Observability

#### **Key Metrics Dashboard**
```python
# src/monitoring/metrics.py
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class PipelineMetrics:
    """Track key pipeline performance metrics"""
    
    # Research Pipeline Metrics
    research_success_rate: float        # Target: 95%+
    average_research_time: float        # Target: 8-12 minutes
    quality_score_average: float        # Target: 8.0+
    cost_per_brand: float              # Target: $8-15
    phase_failure_rates: Dict[str, float]  # Track by phase
    
    # Workflow Management Metrics
    state_transition_accuracy: float   # Target: 99%+
    resume_success_rate: float         # Target: 98%+
    checkpoint_reliability: float      # Target: 99.9%+
    
    # System Performance Metrics
    api_response_times: Dict[str, float]  # LLM, Storage, Search
    error_rates: Dict[str, float]         # By component
    resource_utilization: Dict[str, float]  # Memory, CPU

class MetricsCollector:
    def __init__(self):
        self.metrics = []
        self.start_time = time.time()
    
    def record_research_completion(self, brand_url: str, duration: float, 
                                 quality_score: float, cost: float):
        """Record successful research completion"""
        self.metrics.append({
            "type": "research_completion",
            "brand_url": brand_url,
            "duration": duration,
            "quality_score": quality_score,
            "cost": cost,
            "timestamp": time.time()
        })
    
    def record_workflow_transition(self, brand_url: str, from_state: str, 
                                 to_state: str, success: bool):
        """Record workflow state transition"""
        self.metrics.append({
            "type": "workflow_transition", 
            "brand_url": brand_url,
            "from_state": from_state,
            "to_state": to_state,
            "success": success,
            "timestamp": time.time()
        })
    
    def generate_summary_report(self) -> PipelineMetrics:
        """Generate summary metrics report"""
        research_metrics = [m for m in self.metrics if m["type"] == "research_completion"]
        workflow_metrics = [m for m in self.metrics if m["type"] == "workflow_transition"]
        
        return PipelineMetrics(
            research_success_rate=len(research_metrics) / max(1, len([m for m in self.metrics if "research" in m["type"]])),
            average_research_time=sum(m["duration"] for m in research_metrics) / max(1, len(research_metrics)),
            quality_score_average=sum(m["quality_score"] for m in research_metrics) / max(1, len(research_metrics)),
            cost_per_brand=sum(m["cost"] for m in research_metrics) / max(1, len(research_metrics)),
            phase_failure_rates=self._calculate_phase_failure_rates(),
            state_transition_accuracy=len([m for m in workflow_metrics if m["success"]]) / max(1, len(workflow_metrics)),
            resume_success_rate=self._calculate_resume_success_rate(),
            checkpoint_reliability=self._calculate_checkpoint_reliability(),
            api_response_times=self._get_api_response_times(),
            error_rates=self._calculate_error_rates(),
            resource_utilization=self._get_resource_utilization()
        )
```

#### **Simple Alerting for Critical Issues**
```python
# src/monitoring/alerts.py
class SimpleAlerting:
    def check_and_alert(self, metrics: PipelineMetrics):
        """Check metrics and log alerts for critical issues"""
        
        alerts = []
        
        if metrics.research_success_rate < 0.90:
            alerts.append(f"CRITICAL: Research success rate low: {metrics.research_success_rate:.2%}")
        
        if metrics.quality_score_average < 7.5:
            alerts.append(f"WARNING: Quality scores dropping: {metrics.quality_score_average:.1f}")
        
        if metrics.cost_per_brand > 20:
            alerts.append(f"WARNING: Costs too high: ${metrics.cost_per_brand:.2f} per brand")
        
        if metrics.state_transition_accuracy < 0.95:
            alerts.append(f"CRITICAL: Workflow state issues: {metrics.state_transition_accuracy:.2%}")
        
        # Log alerts (could extend to email/Slack in future)
        for alert in alerts:
            logger.error(f"ALERT: {alert}")
        
        return alerts
```

### 8.6 Simple Deployment Strategy

#### **Environment Setup**
```bash
# Development Environment
export ENV=dev
export GCP_BUCKET_NAME=liddy-account-documents-dev

# Production Environment  
export ENV=prod
export GCP_BUCKET_NAME=liddy-account-documents

# Shared Configuration
export LANGFUSE_HOST=https://cloud.langfuse.com
```

#### **Deployment Checklist**
- [ ] Environment variables configured correctly
- [ ] GCP storage bucket access verified
- [ ] All required API keys present and valid
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Basic smoke tests pass
- [ ] Log rotation configured (if long-running)

#### **Version Tracking**
```python
# Simple version tracking in code
__version__ = "1.0.0"
__build_date__ = "2024-01-XX"
__commit_hash__ = "auto-generated"  # Could be injected during build
```

---

## 9. Success Metrics & KPIs
