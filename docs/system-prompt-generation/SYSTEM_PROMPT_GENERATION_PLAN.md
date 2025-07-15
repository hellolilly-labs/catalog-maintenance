# System Prompt Generation Plan (v2.0)
## From Brand Research to Optimized AI Persona with Runtime Security

### Overview
This plan outlines a production-ready system that generates brand-specific AI persona system prompts and hardens the LiveKit runtime with security and quality agents. The system:
- Validates complete brand research before generation (≥7.0 quality)
- Integrates with Langfuse for staging/production workflow
- Preserves manual modifications through AST-based diffing
- Maintains ≤3,000 token limits with GPT-4 prefix caching optimization
- Provides runtime security against prompt injection and echo behavior
- Enables continuous improvement through performance metrics

---

## Architecture Overview

### Build-Time Components
| Component | Purpose |
|-----------|---------|
| **SystemPromptGenerator** | Orchestrates full pipeline |
| **ResearchValidator** | Confirms all research phases complete and ≥7 quality |
| **VariableExtractor** | Maps research into template variables |
| **ModificationPreserver** | Detects & reapplies manual edits via AST diff |
| **TokenOptimizer** | Targets ≤2,500 tokens; hard-caps at 3,000 |
| **PromptVersionManager** | Staging ↔ Production in Langfuse |

### Runtime Components (LiveKit)
| Component | Purpose |
|-----------|---------|
| **PromptSanitizer** | Blocks malicious or hidden instructions |
| **EchoMonitor** | Logs echo events; injects note if repetition persists |
| **LLM Node** | ChatCompletion (`presence_penalty: 0.7`, `temperature: 0.8`) |
| **product_search · knowledge_search** | Existing RAG tool calls |

### Data Flow
```
Research Phases → Validation → Variable Extraction → Template Population
                                         ↓
Token Optimization (≤3,000) → Langfuse Staging → Manual QA → Production
                                         ↓
                    Runtime: PromptSanitizer → LLM → EchoMonitor
```

---

## Phase 1: Research Validation & Completion

### 1.1 Research Validation System
Following the pattern from `unified_descriptor_generator.py`:

```python
class ResearchValidator:
    def __init__(self, brand_domain: str):
        self.workflow_manager = get_workflow_manager()
        self.required_phases = [
            'foundation', 'market_positioning', 'product_style',
            'customer_cultural', 'voice_messaging', 'interview_synthesis',
            'linearity_analysis', 'research_integration'
        ]
    
    async def validate_and_complete(self, auto_run: bool = True) -> Dict[str, Any]:
        """Validate all research phases, optionally running missing ones"""
        progress = await self.workflow_manager.get_research_progress_summary(self.brand_domain)
        
        if progress['missing_phases']:
            if auto_run:
                logger.info(f"🔄 Running {len(progress['missing_phases'])} missing research phases")
                for phase in progress['missing_phases']:
                    await self._run_research_phase(phase)
            else:
                raise ValueError(f"Missing required research: {progress['missing_phases']}")
        
        return progress
```

### 1.2 Quality Validation
Ensure research meets quality thresholds:

```python
async def validate_research_quality(self) -> Dict[str, float]:
    """Check quality scores for all completed research"""
    quality_scores = {}
    
    for phase in self.required_phases:
        metadata_path = f"research/{phase}/research_metadata.json"
        metadata = await self.storage.load_json(metadata_path)
        
        quality_score = metadata.get('quality_score', 0.0)
        if quality_score < 7.0:  # Minimum threshold
            logger.warning(f"⚠️ {phase} quality below threshold: {quality_score}")
        
        quality_scores[phase] = quality_score
    
    return quality_scores
```

### 1.3 Core Variable Extraction
Extract these essential elements from research phases:

**From Foundation Research:**
- Brand mission (1 line)
- Core values (3-5 words)
- Key philosophy/tagline
- 2-3 heritage milestones

**From Voice Messaging Research:**
- Brand voice description (1 line)
- Communication pattern
- Power words and phrases
- Tone preferences

**From Customer Cultural Research:**
- 3 primary segments with %
- Segment characteristics (5-7 words each)
- Detection cues (keywords)

**From Product/Technical Research:**
- 5-7 signature technologies
- Tier structure
- Key metrics/proof points

### 1.4 Variable Extraction Implementation

```python
class VariableExtractor:
    """Extract template variables from research phases"""
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.storage = get_account_storage_provider()
        self.extractors = {
            'foundation': FoundationExtractor(),
            'voice_messaging': VoiceMessagingExtractor(),
            'customer_cultural': CustomerCulturalExtractor(),
            # ... other phase extractors
        }
    
    async def extract_all_variables(self) -> Dict[str, str]:
        """Extract variables from all research phases"""
        all_variables = {}
        
        for phase_name, extractor in self.extractors.items():
            research_content = await self._load_research(phase_name)
            phase_variables = extractor.extract(research_content)
            all_variables.update(phase_variables)
        
        # Add computed variables
        all_variables['SLIDER_SETTINGS'] = self._generate_slider_settings()
        all_variables['VERSION'] = self._get_next_version()
        
        return all_variables
```

---

## Phase 2: Langfuse Integration & Version Management

### 2.1 Prompt Version Manager

```python
class PromptVersionManager:
    """Manage prompt versions in Langfuse with staging/production workflow"""
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.prompt_manager = PromptManager()
        self.production_key = f"{brand_domain}/full_instructions"
        self.staging_key = f"{brand_domain}/full_instructions_staging"
    
    async def load_production_prompt(self) -> Optional[LangfusePrompt]:
        """Load current production prompt from Langfuse"""
        try:
            return self.prompt_manager.get_prompt(self.production_key)
        except Exception as e:
            logger.info(f"No production prompt found: {e}")
            return None
    
    async def load_staging_prompt(self) -> Optional[LangfusePrompt]:
        """Load current staging prompt from Langfuse"""
        try:
            return self.prompt_manager.get_prompt(self.staging_key)
        except Exception as e:
            logger.info(f"No staging prompt found: {e}")
            return None
    
    async def save_to_staging(self, prompt_content: str, metadata: Dict[str, Any]):
        """Save new prompt version to staging"""
        labels = [
            "staging",
            self.brand_domain,
            f"version:{metadata['version']}",
            f"generated:{datetime.now().isoformat()}",
            f"token_count:{metadata['token_count']}"
        ]
        
        # Create or update staging prompt
        self.prompt_manager.create_prompt(
            name=self.staging_key,
            prompt=prompt_content,
            labels=labels,
            config={
                "metadata": metadata,
                "generated_from_research": True,
                "parent_version": metadata.get('parent_version')
            }
        )
    
    async def promote_to_production(self, approval_metadata: Dict[str, Any]):
        """Promote staging prompt to production"""
        staging = await self.load_staging_prompt()
        if not staging:
            raise ValueError("No staging prompt to promote")
        
        # Update labels for production
        labels = [
            "production",
            self.brand_domain,
            f"promoted:{datetime.now().isoformat()}",
            f"promoted_by:{approval_metadata.get('user', 'system')}"
        ]
        
        # Create production version
        self.prompt_manager.create_prompt(
            name=self.production_key,
            prompt=staging.prompt,
            labels=labels,
            config=staging.config
        )
```

### 2.2 Manual Modification Preservation (v2.0 - AST-Based)

```python
import ast
from typing import Dict, List, Any, Union
from dataclasses import dataclass

@dataclass
class PromptModification:
    """Represents a detected modification in the prompt"""
    type: str  # 'addition', 'deletion', 'modification'
    section: str  # Which section was modified
    content: str  # The actual content change
    line_range: tuple  # (start_line, end_line)
    context: str  # Surrounding context for accurate reapplication

class ModificationPreserver:
    """AST-based detection and preservation of manual modifications"""
    
    def __init__(self):
        # Use markdown-it-py for robust AST parsing (handles edge cases)
        try:
            from markdown_it import MarkdownIt
            self.md_parser = MarkdownIt()
            self.use_library_parser = True
        except ImportError:
            # Fallback to regex patterns if library unavailable
            self.use_library_parser = False
            self.section_hierarchy = {
                'h1': r'^# (.+)$',
                'h2': r'^## (.+)$', 
                'h3': r'^### (.+)$',
                'table': r'^\|.*\|$',
                'code_block': r'^```',
                'list_item': r'^[\*\-\+] ',
                'numbered': r'^\d+\. ',
                'internal_block': r'^⟦.*⟧$'
            }
    
    def parse_prompt_ast(self, prompt: str) -> Dict[str, Any]:
        """Parse prompt into AST-like structure"""
        lines = prompt.split('\n')
        sections = {}
        current_section = None
        current_subsection = None
        current_content = []
        
        for i, line in enumerate(lines):
            line_type = self._classify_line(line)
            
            if line_type == 'h1':
                if current_section and current_content:
                    sections[current_section] = {
                        'content': '\n'.join(current_content),
                        'subsections': sections.get(current_section, {}).get('subsections', {}),
                        'line_range': (sections.get(current_section, {}).get('start_line', i), i-1)
                    }
                current_section = line.strip('# ')
                sections[current_section] = {'start_line': i, 'subsections': {}}
                current_content = [line]
                current_subsection = None
                
            elif line_type == 'h2':
                if current_subsection and current_content:
                    sections[current_section]['subsections'][current_subsection] = {
                        'content': '\n'.join(current_content),
                        'line_range': (sections[current_section]['subsections'].get(current_subsection, {}).get('start_line', i), i-1)
                    }
                current_subsection = line.strip('# ')
                sections[current_section]['subsections'][current_subsection] = {'start_line': i}
                current_content = [line]
                
            else:
                current_content.append(line)
        
        # Close final section
        if current_section and current_content:
            if current_subsection:
                sections[current_section]['subsections'][current_subsection]['content'] = '\n'.join(current_content)
            else:
                sections[current_section]['content'] = '\n'.join(current_content)
        
        return sections
    
    def extract_modifications(self, production: str, base_template: str) -> List[PromptModification]:
        """Use AST-based comparison to detect precise modifications"""
        prod_ast = self.parse_prompt_ast(production)
        base_ast = self.parse_prompt_ast(base_template)
        
        modifications = []
        
        # Compare sections
        for section_name in set(prod_ast.keys()) | set(base_ast.keys()):
            if section_name not in base_ast:
                # New section added
                modifications.append(PromptModification(
                    type='addition',
                    section=section_name,
                    content=prod_ast[section_name]['content'],
                    line_range=prod_ast[section_name].get('line_range', (0, 0)),
                    context=f"Added section: {section_name}"
                ))
            elif section_name not in prod_ast:
                # Section removed
                modifications.append(PromptModification(
                    type='deletion',
                    section=section_name,
                    content=base_ast[section_name]['content'],
                    line_range=base_ast[section_name].get('line_range', (0, 0)),
                    context=f"Removed section: {section_name}"
                ))
            else:
                # Compare content
                modifications.extend(self._compare_section_content(
                    section_name,
                    prod_ast[section_name],
                    base_ast[section_name]
                ))
        
        return modifications
    
    def apply_modifications(self, base_prompt: str, modifications: List[PromptModification]) -> str:
        """Apply preserved modifications using AST-based reconstruction"""
        base_ast = self.parse_prompt_ast(base_prompt)
        
        # Apply modifications in reverse line order to maintain positions
        sorted_mods = sorted(modifications, key=lambda m: m.line_range[0], reverse=True)
        
        lines = base_prompt.split('\n')
        
        for mod in sorted_mods:
            if mod.type == 'addition':
                # Insert new content
                lines = self._insert_content_at_position(lines, mod)
            elif mod.type == 'modification':
                # Replace existing content
                lines = self._replace_content_at_position(lines, mod)
            elif mod.type == 'deletion':
                # Remove content (usually we skip deletions in regeneration)
                continue
        
        return '\n'.join(lines)
    
    def _classify_line(self, line: str) -> str:
        """Classify line type for AST parsing"""
        import re
        for pattern_name, pattern in self.section_hierarchy.items():
            if re.match(pattern, line.strip()):
                return pattern_name
        return 'content'
    
    def _compare_section_content(self, section_name: str, prod_section: Dict, base_section: Dict) -> List[PromptModification]:
        """Compare individual section content"""
        modifications = []
        
        prod_content = prod_section.get('content', '')
        base_content = base_section.get('content', '')
        
        if prod_content != base_content:
            # Use more sophisticated diff for content
            import difflib
            differ = difflib.SequenceMatcher(None, base_content, prod_content)
            
            for tag, i1, i2, j1, j2 in differ.get_opcodes():
                if tag == 'insert':
                    modifications.append(PromptModification(
                        type='addition',
                        section=section_name,
                        content=prod_content[j1:j2],
                        line_range=(j1, j2),
                        context=f"Addition in {section_name}"
                    ))
                elif tag == 'replace':
                    modifications.append(PromptModification(
                        type='modification',
                        section=section_name,
                        content=prod_content[j1:j2],
                        line_range=(i1, i2),
                        context=f"Modification in {section_name}"
                    ))
        
        return modifications
```

### 2.3 Content Prioritization
Apply the "Visible vs Internal" rule:
- **Visible**: Rules, patterns, essential guidance
- **Internal**: Lists, matrices, examples, detailed segments

---

## Phase 3: System Prompt Generator Implementation

### 3.1 Main Generator Class

```python
class SystemPromptGenerator:
    """Generate system prompts from brand research with version management"""
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.research_validator = ResearchValidator(brand_domain)
        self.variable_extractor = VariableExtractor(brand_domain)
        self.version_manager = PromptVersionManager(brand_domain)
        self.modification_preserver = ModificationPreserver()
        self.token_optimizer = TokenOptimizer()
        self.template_path = "docs/system-prompt-generation/GENERIC_AI_PERSONA_TEMPLATE.md"
        
    async def generate(
        self,
        force_regenerate: bool = False,
        auto_run_research: bool = True,
        preserve_modifications: bool = True,
        dry_run: bool = False
    ) -> GenerationResult:
        """Generate system prompt with full workflow"""
        
        # Step 1: Validate research completion
        logger.info(f"🔍 Validating research for {self.brand_domain}")
        await self.research_validator.validate_and_complete(auto_run=auto_run_research)
        quality_scores = await self.research_validator.validate_research_quality()
        
        # Step 2: Load existing prompts
        logger.info("📥 Loading existing prompts from Langfuse")
        production_prompt = await self.version_manager.load_production_prompt()
        staging_prompt = await self.version_manager.load_staging_prompt()
        
        # Step 3: Extract variables from research
        logger.info("🔤 Extracting variables from research")
        variables = await self.variable_extractor.extract_all_variables()
        
        # Step 4: Detect manual modifications
        modifications = {}
        if preserve_modifications and production_prompt:
            logger.info("🔎 Detecting manual modifications")
            base_template = await self._generate_base_from_template(variables)
            modifications = self.modification_preserver.extract_modifications(
                production_prompt.prompt,
                base_template
            )
        
        # Step 5: Generate new prompt
        logger.info("🤖 Generating new system prompt")
        new_prompt = await self._generate_prompt(variables, modifications)
        
        # Step 6: Optimize tokens
        logger.info("📏 Optimizing token count")
        optimized_prompt, token_count = self.token_optimizer.optimize(new_prompt)
        
        # Step 7: Validate result
        validation_result = self._validate_prompt(optimized_prompt, token_count)
        if not validation_result.is_valid:
            raise ValueError(f"Generated prompt validation failed: {validation_result.errors}")
        
        # Step 8: Save to staging (unless dry run)
        if not dry_run:
            logger.info("💾 Saving to Langfuse staging")
            metadata = self._create_metadata(
                variables, modifications, quality_scores, token_count
            )
            await self.version_manager.save_to_staging(optimized_prompt, metadata)
        
        return GenerationResult(
            prompt=optimized_prompt,
            token_count=token_count,
            modifications_preserved=len(modifications) > 0,
            quality_scores=quality_scores,
            metadata=metadata
        )
```

### 3.2 Token Policy (v2.0)

The new architecture defines three token tiers for different sections:

| Section | Target | Hard Limit | Purpose |
|---------|--------|------------|---------|
| Brand Table | ≤800 | 900 | Core brand identity & rules |
| Tool Usage | ≤1700 | 1900 | Product/knowledge search flows |
| INTERNAL | ≤2500 | 2700 | Lists, examples, reference data |
| **Total** | **≤3000** | **3500** | Complete prompt |

### 3.3 Token Optimizer (v2.0)

```python
class TokenOptimizer:
    """Optimize prompts to fit token limits while preserving content"""
    
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        # v2.0 token targets
        self.section_targets = {
            "brand_table": (1000, 1100),   # Increased for heritage stories
            "tool_usage": (1500, 1700),    # Reduced as rarely hits 1700
            "internal": (2500, 2700),
            "total": (3000, 3500)
        }
    
    def optimize(self, prompt: str) -> Tuple[str, int]:
        """Optimize prompt to fit within token limits"""
        sections = self._parse_sections(prompt)
        token_counts = self._count_section_tokens(sections)
        
        # Check if already within limits
        if self._within_limits(token_counts):
            return prompt, sum(token_counts.values())
        
        # Progressive optimization by section
        optimized_sections = {}
        for section_name, content in sections.items():
            target, hard_limit = self.section_targets.get(section_name, (1000, 1200))
            optimized_content = self._optimize_section(
                content, 
                section_name,
                target,
                hard_limit
            )
            optimized_sections[section_name] = optimized_content
        
        # Reconstruct prompt
        optimized_prompt = self._reconstruct_prompt(optimized_sections)
        final_count = self._count_tokens(optimized_prompt)
        
        # Log prefix caching metrics for monitoring
        cached_tokens = min(final_count, 2048)  # GPT-4 prefix cache limit
        cache_efficiency = cached_tokens / final_count if final_count > 0 else 0
        logger.info(f"💾 Prefix cached = {cached_tokens}/{final_count} tokens ({cache_efficiency:.1%} efficiency)")
        
        # Final validation
        if final_count > self.section_targets["total"][1]:
            raise ValueError(f"Cannot optimize prompt below {self.section_targets['total'][1]} tokens")
        
        return optimized_prompt, final_count
    
    def _optimize_section(self, content: str, section: str, target: int, limit: int) -> str:
        """Optimize individual section based on type"""
        current = self._count_tokens(content)
        if current <= target:
            return content
        
        # Section-specific strategies
        if section == "brand_table":
            strategies = [
                self._compress_brand_essence,
                self._shorten_philosophy,
                self._abbreviate_values
            ]
        elif section == "tool_usage":
            strategies = [
                self._condense_tool_steps,
                self._remove_tool_examples,
                self._abbreviate_tool_descriptions
            ]
        elif section == "internal":
            strategies = [
                self._move_to_external_reference,
                self._compress_lists,
                self._remove_redundant_examples
            ]
        else:
            strategies = [
                self._compress_whitespace,
                self._abbreviate_common_terms,
                self._remove_optional_content
            ]
        
        optimized = content
        for strategy in strategies:
            optimized = strategy(optimized)
            new_count = self._count_tokens(optimized)
            if new_count <= target:
                return optimized
        
        # If still over, apply aggressive trimming
        if new_count > limit:
            optimized = self._aggressive_trim(optimized, limit)
        
        return optimized
```

### 3.3 Structured Sections (Visible)

**Section 1: Brand Essence** (≤70 tokens)
- Philosophy, mission, values, signature tech, voice
- Single table format
- Include heritage tagline

**Section 2: Tone Controls** (≤40 tokens)
- Single unified slider table
- 5 levels with word counts
- "+1 for more detail" rule

**Section 3: Speaking Rules** (≤70 tokens)
- 7-8 numbered rules max
- Include echo guard
- Benefit → proof (≤10 words) → question pattern
- Reference segments by name only

**Section 4: Consultation Flow** (≤40 tokens)
- 3-step discovery pattern
- Response framework (1 line)

**Section 5: Sales & Fulfillment** (≤60 tokens)
- Core policies (bullets)
- Easy returns mention
- Fulfillment options (brief)

**Section 6: Tool Usage** (≤50 tokens)
- 5-step table format
- 15-word closing phrase
- "(stay within word cap)" reminder

**Section 7: Terminology** (≤30 tokens)
- 1 line on tier vocabulary
- 2-3 translation examples

**Section 8: Policies** (≤20 tokens)
- Single line of key policies

### 2.2 Internal Section (≤600 tokens)
Structure for quick reference:
- LLM parameters
- Customer segments (detailed)
- Detection cues (table)
- Trust builders (bullets)
- Objection matrix (table)
- Metrics library
- Closing phrases

---

## Phase 3: Optimization Rules

### 3.1 Token Reduction Strategies

**Length Control:**
- Single source of truth for word limits (Tone Controls only)
- Remove all duplicate length references
- Use table formats over prose

**Content Compression:**
- Combine related concepts
- Use " · " separators for lists
- Abbreviate where clear (e.g., "Body Geo")

**Strategic Placement:**
- Long lists → Internal section
- Examples → Internal section  
- Detailed explanations → Cut or simplify

### 3.2 Voice Optimization

**Anti-Echo Measures:**
- Explicit "never repeat user's words" rule
- Vary response starters
- Monitor for echo patterns

**Brevity Enforcement:**
- Word count limits in slider
- Monitoring rule for averages
- Short, action-oriented closings

### 3.4 Runtime Security Components (LiveKit Integration)

The voice assistant system requires runtime security to prevent prompt injection and monitor echo behavior:

```python
class PromptSanitizer:
    """Runtime protection against prompt injection and malicious instructions"""
    
    def __init__(self):
        self.injection_patterns = [
            r'ignore previous instructions',
            r'forget everything',
            r'(?i)(?<![\w])system (prompt|message)',  # Look-behind for jailbreaks
            r'you are now',
            r'roleplay as',
            r'pretend to be',
            r'act as if',
            r'⟦.*⟧',  # Block attempts to inject INTERNAL markers
        ]
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.6, 
            'high': 0.9
        }
    
    async def sanitize_input(self, user_input: str, conversation_context: List[Dict]) -> Dict[str, Any]:
        """Sanitize user input and detect potential injection attempts"""
        
        # Check for injection patterns
        injection_score = self._calculate_injection_risk(user_input)
        
        # Context analysis for sophisticated attacks
        context_risk = self._analyze_conversation_context(conversation_context)
        
        # Combined risk assessment
        total_risk = max(injection_score, context_risk)
        
        if total_risk > self.severity_thresholds['high']:
            return {
                'allowed': False,
                'reason': 'High risk prompt injection detected',
                'sanitized_input': None,
                'risk_level': 'high',
                'log_alert': True
            }
        elif total_risk > self.severity_thresholds['medium']:
            # Allow but sanitize
            sanitized = self._sanitize_suspicious_content(user_input)
            return {
                'allowed': True,
                'reason': 'Medium risk - content sanitized',
                'sanitized_input': sanitized,
                'risk_level': 'medium',
                'log_alert': True
            }
        else:
            return {
                'allowed': True,
                'reason': 'Input appears safe',
                'sanitized_input': user_input,
                'risk_level': 'low',
                'log_alert': False
            }
    
    def _calculate_injection_risk(self, text: str) -> float:
        """Calculate injection risk score 0.0-1.0"""
        import re
        risk_score = 0.0
        
        for pattern in self.injection_patterns:
            if re.search(pattern, text.lower()):
                risk_score += 0.2
        
        # Check for excessive special characters
        special_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        if special_ratio > 0.3:
            risk_score += 0.3
        
        # Check for suspiciously long inputs
        if len(text) > 2000:
            risk_score += 0.2
        
        return min(risk_score, 1.0)

class EchoMonitor:
    """Monitor and prevent echo behavior in real-time"""
    
    def __init__(self):
        self.recent_responses = []
        self.max_history = 10
        self.echo_threshold = 0.8  # Similarity threshold
        self.intervention_threshold = 3  # Consecutive echoes before intervention
    
    async def check_response(self, user_input: str, ai_response: str, conversation_id: str) -> Dict[str, Any]:
        """Check if AI response exhibits echo behavior"""
        
        # Calculate similarity between input and response
        echo_score = self._calculate_echo_similarity(user_input, ai_response)
        
        # Track in conversation history
        self.recent_responses.append({
            'input': user_input,
            'response': ai_response,
            'echo_score': echo_score,
            'conversation_id': conversation_id,
            'timestamp': datetime.now()
        })
        
        # Maintain history size
        if len(self.recent_responses) > self.max_history:
            self.recent_responses.pop(0)
        
        # Check for echo pattern
        is_echo = echo_score > self.echo_threshold
        consecutive_echoes = self._count_consecutive_echoes()
        
        result = {
            'is_echo': is_echo,
            'echo_score': echo_score,
            'consecutive_count': consecutive_echoes,
            'requires_intervention': consecutive_echoes >= self.intervention_threshold,
            'suggested_action': None
        }
        
        if result['requires_intervention']:
            result['suggested_action'] = self._generate_intervention_note()
            # Temporary presence_penalty reduction to break echo pattern
            result['temp_penalty_adjustment'] = {'presence_penalty': 0.5}
        
        return result
    
    def _calculate_echo_similarity(self, input_text: str, response_text: str) -> float:
        """Calculate similarity score between input and response"""
        from difflib import SequenceMatcher
        
        # Normalize text
        input_normalized = ' '.join(input_text.lower().split())
        response_normalized = ' '.join(response_text.lower().split())
        
        # Calculate similarity
        matcher = SequenceMatcher(None, input_normalized, response_normalized)
        return matcher.ratio()
    
    def _count_consecutive_echoes(self) -> int:
        """Count consecutive echo occurrences"""
        count = 0
        for response_data in reversed(self.recent_responses):
            if response_data['echo_score'] > self.echo_threshold:
                count += 1
            else:
                break
        return count
    
    def _generate_intervention_note(self) -> str:
        """Generate intervention message for persistent echo behavior"""
        return "🔄 I notice I might be repeating your words. Let me focus on providing helpful information about our products instead. What specific product category interests you?"

class RuntimeSecurityManager:
    """Coordinates runtime security components"""
    
    def __init__(self):
        self.prompt_sanitizer = PromptSanitizer()
        self.echo_monitor = EchoMonitor()
        self.security_metrics = {
            'blocked_inputs': 0,
            'sanitized_inputs': 0,
            'echo_detections': 0,
            'interventions': 0
        }
    
    async def process_conversation_turn(
        self, 
        user_input: str, 
        conversation_context: List[Dict], 
        conversation_id: str
    ) -> Dict[str, Any]:
        """Process a complete conversation turn with security checks"""
        
        # Step 1: Input sanitization
        sanitization_result = await self.prompt_sanitizer.sanitize_input(
            user_input, conversation_context
        )
        
        if not sanitization_result['allowed']:
            self.security_metrics['blocked_inputs'] += 1
            return {
                'proceed': False,
                'reason': sanitization_result['reason'],
                'sanitized_input': None,
                'security_action': 'blocked'
            }
        
        if sanitization_result['risk_level'] == 'medium':
            self.security_metrics['sanitized_inputs'] += 1
        
        return {
            'proceed': True,
            'reason': 'Input cleared security checks',
            'sanitized_input': sanitization_result['sanitized_input'],
            'security_action': 'allowed'
        }
    
    async def process_response(
        self, 
        user_input: str, 
        ai_response: str, 
        conversation_id: str
    ) -> Dict[str, Any]:
        """Process AI response for echo detection"""
        
        echo_result = await self.echo_monitor.check_response(
            user_input, ai_response, conversation_id
        )
        
        if echo_result['is_echo']:
            self.security_metrics['echo_detections'] += 1
        
        if echo_result['requires_intervention']:
            self.security_metrics['interventions'] += 1
            return {
                'allow_response': False,
                'intervention_message': echo_result['suggested_action'],
                'echo_detected': True,
                'llm_adjustments': echo_result.get('temp_penalty_adjustment', {})
            }
        
        return {
            'allow_response': True,
            'intervention_message': None,
            'echo_detected': echo_result['is_echo']
        }
```

---

## Phase 4: Runtime Performance Metrics

### 4.1 Concrete Performance Metrics

The v2.0 architecture defines specific performance targets:

| Metric | Target | Critical Threshold | Monitoring |
|--------|--------|-------------------|------------|
| **Echo Rate** | <2% | >5% | Real-time detection |
| **Avg Response Words** | 15-45 | >80 | Per-conversation tracking |
| **Security Blocks** | <1% | >3% | Security event logging |
| **Prompt Generation Time** | <30s | >60s | Build-time monitoring |
| **Token Efficiency** | >85% | <70% | Token usage analysis |

### 4.2 Performance Monitoring Implementation

```python
class PerformanceMetrics:
    """Track and analyze system performance metrics"""
    
    def __init__(self):
        self.metrics_store = {}
        self.alert_thresholds = {
            'echo_rate': 0.05,
            'avg_response_words': 80,
            'security_block_rate': 0.001,  # Tightened to <0.1% for false positives
            'generation_time': 60,
            'token_efficiency': 0.70,
            '95th_percentile_response_latency': 200  # ms - customers notice tail latency
        }
    
    async def track_conversation_metrics(self, conversation_data: Dict) -> None:
        """Track real-time conversation metrics"""
        conversation_id = conversation_data['conversation_id']
        
        # Calculate response word count
        response_words = len(conversation_data['ai_response'].split())
        
        # Update running averages
        self._update_metric('avg_response_words', response_words)
        
        # Track echo detection
        if conversation_data.get('echo_detected'):
            self._update_metric('echo_rate', 1)
        else:
            self._update_metric('echo_rate', 0)
        
        # Check for threshold violations
        await self._check_alert_thresholds()
    
    async def track_generation_metrics(self, generation_data: Dict) -> None:
        """Track prompt generation performance"""
        generation_time = generation_data['duration_seconds']
        token_count = generation_data['token_count']
        target_tokens = generation_data['target_tokens']
        
        # Calculate token efficiency
        token_efficiency = min(target_tokens / token_count, 1.0) if token_count > 0 else 0
        
        self._update_metric('generation_time', generation_time)
        self._update_metric('token_efficiency', token_efficiency)
        
        await self._check_alert_thresholds()
    
    def _update_metric(self, metric_name: str, value: float) -> None:
        """Update running metric with exponential moving average"""
        if metric_name not in self.metrics_store:
            self.metrics_store[metric_name] = {
                'current_value': value,
                'samples': 1,
                'alpha': 0.1  # EMA smoothing factor
            }
        else:
            current = self.metrics_store[metric_name]['current_value']
            alpha = self.metrics_store[metric_name]['alpha']
            self.metrics_store[metric_name]['current_value'] = (1 - alpha) * current + alpha * value
            self.metrics_store[metric_name]['samples'] += 1
    
    async def _check_alert_thresholds(self) -> None:
        """Check if any metrics exceed alert thresholds"""
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in self.metrics_store:
                current_value = self.metrics_store[metric_name]['current_value']
                
                if metric_name in ['echo_rate', 'avg_response_words', 'security_block_rate', 'generation_time']:
                    # Higher is worse
                    if current_value > threshold:
                        await self._send_alert(metric_name, current_value, threshold)
                else:
                    # Lower is worse (token_efficiency)
                    if current_value < threshold:
                        await self._send_alert(metric_name, current_value, threshold)
    
    async def _send_alert(self, metric_name: str, current_value: float, threshold: float) -> None:
        """Send performance alert"""
        logger.warning(f"🚨 Performance Alert: {metric_name} = {current_value:.3f} (threshold: {threshold:.3f})")
        
        # Could integrate with external alerting systems here
        # e.g., Slack, PagerDuty, etc.
```

---

## Phase 5: Runner Scripts & CLI Tools

### 4.1 System Prompt Generator Runner

**File**: `run/generate_system_prompt.py`

```python
import argparse
import asyncio
import json
from pathlib import Path
from liddy_intelligence.prompts.system_prompt_generator import SystemPromptGenerator

async def main():
    parser = argparse.ArgumentParser(
        description="Generate AI persona system prompts from brand research"
    )
    parser.add_argument('brand', help='Brand domain (e.g., specialized.com)')
    parser.add_argument('--force', action='store_true', 
                       help='Force regeneration even if recent prompt exists')
    parser.add_argument('--auto-run-research', action='store_true', default=True,
                       help='Automatically run missing research phases')
    parser.add_argument('--no-preserve-mods', action='store_true',
                       help='Do not preserve manual modifications')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without saving to Langfuse')
    parser.add_argument('--output', type=str, help='Save prompt to file')
    parser.add_argument('--json', action='store_true', help='Output metadata as JSON')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SystemPromptGenerator(args.brand)
    
    try:
        # Generate prompt
        result = await generator.generate(
            force_regenerate=args.force,
            auto_run_research=args.auto_run_research,
            preserve_modifications=not args.no_preserve_mods,
            dry_run=args.dry_run
        )
        
        # Output results
        if args.json:
            print(json.dumps(result.metadata, indent=2))
        else:
            print(f"\n✅ System Prompt Generated Successfully!")
            print(f"   Token Count: {result.token_count}")
            print(f"   Modifications Preserved: {result.modifications_preserved}")
            print(f"   Average Quality Score: {sum(result.quality_scores.values())/len(result.quality_scores):.1f}")
            
            if args.dry_run:
                print("\n⚠️  DRY RUN - Prompt not saved to Langfuse")
        
        # Save to file if requested
        if args.output:
            Path(args.output).write_text(result.prompt)
            print(f"\n💾 Prompt saved to: {args.output}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
```

### 4.2 Prompt Promotion Tool

**File**: `run/promote_prompt.py`

```python
import argparse
import asyncio
from datetime import datetime
from liddy_intelligence.prompts.prompt_version_manager import PromptVersionManager

async def main():
    parser = argparse.ArgumentParser(
        description="Promote staging prompts to production"
    )
    parser.add_argument('brand', help='Brand domain')
    parser.add_argument('--user', required=True, help='User approving promotion')
    parser.add_argument('--reason', help='Reason for promotion')
    parser.add_argument('--compare', action='store_true', 
                       help='Show diff before promoting')
    
    args = parser.parse_args()
    
    manager = PromptVersionManager(args.brand)
    
    # Load prompts
    staging = await manager.load_staging_prompt()
    production = await manager.load_production_prompt()
    
    if not staging:
        print("❌ No staging prompt found")
        return 1
    
    # Show comparison if requested
    if args.compare and production:
        print("\n📊 DIFFERENCES: Production → Staging")
        # Show diff
        import difflib
        diff = difflib.unified_diff(
            production.prompt.splitlines(),
            staging.prompt.splitlines(),
            lineterm='',
            fromfile='production',
            tofile='staging'
        )
        for line in diff:
            print(line)
    
    # Confirm promotion
    print(f"\n🚀 Promoting staging prompt to production for {args.brand}")
    response = input("Continue? (y/N): ")
    
    if response.lower() != 'y':
        print("❌ Promotion cancelled")
        return 1
    
    # Promote
    await manager.promote_to_production({
        'user': args.user,
        'reason': args.reason,
        'timestamp': datetime.now().isoformat()
    })
    
    print("✅ Prompt promoted to production!")
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
```

---

## Phase 5: Workflow Integration

### 5.1 Brand Manager Script Updates

Add to `scripts/brand_manager.sh`:

```bash
# Generate system prompt
generate-prompt)
    echo "🤖 Generating system prompt for $BRAND..."
    
    # Check research status first
    RESEARCH_STATUS=$(python run/workflow_status.py "$BRAND" --json | jq -r '.completion_percentage')
    if (( $(echo "$RESEARCH_STATUS < 100" | bc -l) )); then
        echo "⚠️  Research incomplete ($RESEARCH_STATUS%). Running missing phases..."
    fi
    
    # Generate prompt
    python run/generate_system_prompt.py "$BRAND" \
        --auto-run-research \
        --output "local/prompts/${BRAND}_system_prompt_$(date +%Y%m%d).md"
    ;;

# View prompt status
prompt-status)
    echo "📊 Prompt status for $BRAND..."
    python run/compare_prompts.py "$BRAND" --tokens
    ;;

# Promote to production
promote-prompt)
    echo "🚀 Promoting staging prompt to production..."
    python run/promote_prompt.py "$BRAND" \
        --user "$USER" \
        --compare
    ;;
```

### 5.2 Automated Workflow

**File**: `scripts/prompt_automation.sh`

```bash
#!/bin/bash
# Automated prompt generation and testing

BRAND=$1
SLACK_WEBHOOK=${SLACK_WEBHOOK:-""}

# Generate new prompt
echo "🤖 Generating prompt for $BRAND..."
OUTPUT=$(python run/generate_system_prompt.py "$BRAND" --json)

if [ $? -eq 0 ]; then
    TOKEN_COUNT=$(echo "$OUTPUT" | jq -r '.token_count')
    QUALITY=$(echo "$OUTPUT" | jq -r '.average_quality')
    
    # Notify success
    MESSAGE="✅ Prompt generated for $BRAND - Tokens: $TOKEN_COUNT, Quality: $QUALITY"
    echo "$MESSAGE"
    
    # Send to Slack if configured
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$MESSAGE\"}" \
            "$SLACK_WEBHOOK"
    fi
else
    echo "❌ Prompt generation failed"
    exit 1
fi
```

---

## Phase 6: Quality Assurance & Testing

### 6.1 Prompt Validator

```python
class PromptValidator:
    """Validate generated prompts meet requirements"""
    
    def validate(self, prompt: str, token_count: int) -> ValidationResult:
        errors = []
        warnings = []
        
        # Check token count
        if token_count > 1500:
            errors.append(f"Token count {token_count} exceeds limit")
        elif token_count > 1400:
            warnings.append(f"Token count {token_count} approaching limit")
        
        # Check required sections
        required_sections = [
            "## 1 Brand Essence",
            "## 2 Tone Controls",
            "## 3 Speaking Rules",
            "⟦ INTERNAL"
        ]
        
        for section in required_sections:
            if section not in prompt:
                errors.append(f"Missing required section: {section}")
        
        # Check SLIDER_BLOCK variable
        if "{{SLIDER_SETTINGS}}" in prompt:
            errors.append("Unresolved SLIDER_SETTINGS variable")
        
        # Validate rules
        if "never repeat" not in prompt.lower():
            warnings.append("Missing echo guard rule")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

### 6.2 Integration Tests

```python
class SystemPromptIntegrationTests:
    """Test the full prompt generation pipeline"""
    
    async def test_research_validation(self):
        """Test that missing research is detected"""
        generator = SystemPromptGenerator("test.com")
        # Remove a research phase
        # Verify validation catches it
    
    async def test_modification_preservation(self):
        """Test that manual mods are preserved"""
        # Create base prompt
        # Add manual modifications
        # Generate new version
        # Verify mods preserved
    
    async def test_token_optimization(self):
        """Test token count stays within limits"""
        # Generate with lots of content
        # Verify optimization works
    
    async def test_langfuse_integration(self):
        """Test staging/production workflow"""
        # Generate to staging
        # Promote to production
        # Verify both versions exist
```

---

## Phase 7: Monitoring & Analytics

### 7.1 Prompt Performance Tracking

```python
class PromptPerformanceTracker:
    """Track how prompts perform in production"""
    
    async def track_metrics(self, brand: str, conversation_id: str):
        metrics = {
            'avg_response_words': 0,
            'echo_frequency': 0,
            'conversion_rate': 0,
            'user_satisfaction': 0,
            'prompt_version': '',
            'timestamp': datetime.now()
        }
        
        # Store in analytics database
        await self.store_metrics(brand, metrics)
    
    async def generate_report(self, brand: str, date_range: tuple):
        """Generate performance report for prompt versions"""
        # Compare metrics across versions
        # Identify improvements/regressions
```

### 7.2 A/B Testing Framework

```python
class PromptABTester:
    """A/B test prompt variations"""
    
    def assign_variant(self, user_id: str) -> str:
        """Assign user to test variant"""
        # Hash user_id for consistent assignment
        # Return 'control' or 'variant'
    
    async def get_prompt_for_variant(self, brand: str, variant: str):
        """Get appropriate prompt for variant"""
        if variant == 'control':
            return await self.load_production_prompt(brand)
        else:
            return await self.load_staging_prompt(brand)
```

---

## Phase 8: Documentation & Training

### 8.1 User Documentation

**File**: `docs/system-prompt-generation/USER_GUIDE.md`

Contents:
- How to generate prompts
- Understanding the staging/production workflow
- Interpreting quality scores
- Making manual modifications
- Troubleshooting common issues

### 8.2 Developer Documentation

**File**: `docs/system-prompt-generation/DEVELOPER_GUIDE.md`

Contents:
- Architecture overview
- Adding new extractors
- Modifying the template
- Extending the token optimizer
- Integration with other systems

---

## Implementation Timeline (v2.0)

### Phase 1 (Week 1-2): Core Build-Time Components
- [ ] ResearchValidator with ≥7.0 quality thresholds
- [ ] VariableExtractor base classes with phase-specific extractors
- [ ] PromptVersionManager with Langfuse staging/production workflow
- [ ] Basic SystemPromptGenerator with research validation

### Phase 2 (Week 3-4): Advanced Build-Time Features
- [ ] AST-based ModificationPreserver
- [ ] v2.0 TokenOptimizer with section-specific targets (800/1700/2500 tokens)
- [ ] Enhanced research validation pipeline
- [ ] Template system with variable mapping

### Phase 3 (Week 5-6): Runtime Security Components
- [ ] PromptSanitizer for injection protection
- [ ] EchoMonitor with similarity detection
- [ ] RuntimeSecurityManager coordination
- [ ] Performance metrics tracking system

### Phase 4 (Week 7-8): Integration & Production Readiness
- [ ] Runner scripts and CLI tools
- [ ] Workflow integration with brand_manager.sh
- [ ] Comprehensive testing suite
- [ ] Performance monitoring and alerting
- [ ] Documentation and team training

---

## Success Criteria (v2.0)

### 1. **Build-Time Functionality**
   - ✅ Validates all 8 research phases complete with ≥7.0 quality
   - ✅ Preserves manual modifications using AST-based diffing
   - ✅ Manages Langfuse staging/production workflow
   - ✅ Stays within new token limits (≤3,000 total, sectioned targets)
   - ✅ Integrates with existing brand research pipeline

### 2. **Runtime Security**
   - ✅ Blocks prompt injection attempts (>90% detection rate)
   - ✅ Detects and prevents echo behavior (<2% echo rate)
   - ✅ Maintains conversation quality under security monitoring
   - ✅ Provides real-time performance metrics

### 3. **Performance Targets**
   - ✅ Echo rate: <2% (critical: >5%)
   - ✅ Avg response words: 15-45 (critical: >80)
   - ✅ Security false positives: <0.1% (critical: >0.5%)
   - ✅ Prompt generation: <30s (critical: >60s)
   - ✅ Token efficiency: >85% (critical: <70%)
   - ✅ 95th percentile latency: <200ms (critical: >500ms)

### 4. **Quality & Consistency**
   - ✅ Generates brand-accurate prompts from research
   - ✅ Maintains consistent voice across regenerations
   - ✅ Improves conversation metrics vs. manual prompts
   - ✅ GPT-4 prefix caching optimization

### 5. **Usability & Operations**
   - ✅ Simple CLI interface (`generate_system_prompt.py`)
   - ✅ Clear error messages and validation feedback
   - ✅ Integration with existing workflow scripts
   - ✅ Comprehensive monitoring and alerting

### 6. **Maintainability & Extensibility**
   - ✅ Modular design with clear separation of concerns
   - ✅ AST-based modification system for future enhancements
   - ✅ Good test coverage including integration tests
   - ✅ Version control integration with PR workflow
   - ✅ Extensible template system for new brands

This comprehensive v2.0 plan provides a production-ready system that bridges brand research and AI personas while adding critical runtime security and performance monitoring capabilities for voice commerce applications.