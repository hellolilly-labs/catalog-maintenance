# Research-to-Template Variable Mapping Guide

This document explains how to extract template variables from the 8-phase brand research pipeline.

## Variable Extraction Mapping

### Section 1: Brand DNA & Identity

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{BRAND_NAME}}` | All phases | Brand name used throughout |
| `{{ASSISTANT_NAME}}` | Voice Messaging Research | Look for suggested persona names or create based on brand personality |
| `{{BRAND_PHILOSOPHY}}` | Foundation Research | Core philosophy or founding principle |
| `{{BRAND_MISSION}}` | Foundation Research | Mission statement |
| `{{BRAND_VALUES_LIST}}` | Foundation Research | Core values (format: Value1 · Value2 · Value3) |
| `{{BRAND_VOICE_DESCRIPTION}}` | Voice Messaging Research | One-line description of brand personality |
| `{{SIGNATURE_ELEMENTS}}` | Product Style + Foundation | Key technologies, materials, or distinctive features |
| `{{KEY_HERITAGE_POINTS}}` | Foundation Research | Major milestones and achievements |

### Section 2: Customer Segments

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{CUSTOMER_SEGMENTS_SECTION}}` | Customer Cultural Research | Format each segment with name, percentage, description, and response strategy |

### Section 3: Tone Controls

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{MINIMAL_WORDS}}` through `{{DEEPDIVE_WORDS}}` | Voice Messaging Research | Adapt based on brand's typical verbosity preferences |
| `{{MINIMAL_DESCRIPTION}}` through `{{DEEPDIVE_DESCRIPTION}}` | Voice Messaging Research | Create persona intensity descriptions matching brand voice |

### Section 4: Communication Framework

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{COMMUNICATION_PATTERN}}` | Voice Messaging Research | Extract the brand's typical message structure |
| `{{SPEAKING_RULES_LIST}}` | Voice Messaging + Linearity Analysis | Compile communication guidelines and consistency rules |

### Section 5: Consultation Flow

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{DISCOVERY_PATTERN}}` | Customer Cultural + Voice Messaging | Create discovery questions based on customer priorities |
| `{{RESPONSE_FRAMEWORK}}` | Voice Messaging Research | Define how to structure responses |

### Section 6: Sales & Fulfillment

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{SALES_AND_FULFILLMENT_SECTION}}` | Market Positioning + Business Context | Define sales approach and fulfillment options |

### Section 7: Tool Orchestration

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{TOOL_ORCHESTRATION_TABLE}}` | Customer Cultural + Product Research | Map customer journey to tool usage |

### Section 8: Technical Authority

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{TERMINOLOGY_SECTION}}` | Industry Terminology + Product Style | Extract key terms and their explanations |
| `{{BRAND_VOCABULARY_SECTION}}` | Voice Messaging + Foundation | Preferred words, power words, and phrases |

### Section 9: Policies & Guardrails

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{ALWAYS_POLICIES}}` | All phases, especially Linearity Analysis | What the brand always does |
| `{{NEVER_POLICIES}}` | Linearity Analysis + Voice Messaging | What to avoid |
| `{{RESPONSE_MONITORING_RULES}}` | Voice Messaging Research | Quality control metrics |

### Internal Section Variables

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{LLM_PARAMETERS}}` | Technical requirements | Standard parameters + brand-specific adjustments |
| `{{SEGMENT_DETECTION_TABLE}}` | Customer Cultural Research | Keywords that indicate each segment |
| `{{TRUST_PHRASES}}` | Voice Messaging + Foundation | Authority-building statements |
| `{{OBJECTION_HANDLING_MATRIX}}` | Customer Cultural + Market Positioning | Common concerns and responses by segment |
| `{{PROOF_POINTS}}` | Product Style + Foundation | Concrete evidence and achievements |
| `{{CLOSING_PHRASES}}` | Voice Messaging Research | Action-oriented closings matching brand voice |

### Opening Greeting

| Variable | Research Source | Extraction Method |
|----------|----------------|-------------------|
| `{{GREETING_MESSAGE}}` | Voice Messaging + Brand Voice | Create welcoming opener that establishes persona |

## Runtime vs Template Variables

### Template Variables (Static - Filled from Research)
- All brand identity elements
- Customer segments and characteristics
- Communication patterns and rules
- Brand vocabulary and terminology
- Policies and guidelines

### Runtime Variables (Dynamic - Filled at Execution)
- `{{SLIDER_SETTINGS}}` - User's selected verbosity level
- Product inventory status
- Current promotions or pricing
- User's conversation history
- Detected customer segment

## Research Quality Indicators

When extracting variables, look for:
1. **Consistency** across research phases
2. **Specificity** rather than generic statements
3. **Evidence-based** insights with examples
4. **Customer-validated** preferences
5. **Measurable** elements where possible

## Missing Research Handling

If certain research elements are missing:
1. Note as `{{VARIABLE_NAME: PENDING}}` in template
2. Use Research Integration phase to fill gaps
3. Consider running targeted research for critical missing elements
4. Default to conservative, brand-safe options

## Continuous Improvement

The template should evolve based on:
1. Conversation analytics showing what works
2. Customer feedback on AI interactions
3. Brand evolution and new research insights
4. Competitive landscape changes
5. Technology capabilities expansion