# Research Pipeline Refactoring Plan

## Overview

This document provides a step-by-step plan to refactor the Core Research Pipeline architecture to address design pattern issues while maintaining system functionality throughout the process.

## Current Architecture Issues

### Primary Problems
1. **Single Responsibility Violation**: BaseResearcher handles research execution, quality evaluation, file I/O, progress tracking, and prompt management
2. **Code Duplication**: Web search logic repeated across multiple researcher classes
3. **Inconsistent Inheritance**: Each researcher implements template methods differently
4. **Manual Factory Pattern**: Non-extensible dictionary-based researcher management
5. **Mixed Concerns**: Quality evaluation embedded in research logic instead of being cross-cutting

### Quality Issues
- BaseResearcher constructor has 8 parameters with complex initialization
- `_execute_core_research()` method is 187 lines (too complex)
- Adding new research phases requires modifying core orchestration code
- Quality evaluation logic duplicated across classes

## Refactoring Strategy

**Approach**: Incremental changes with backward compatibility
**Risk Level**: Progressive from low to medium risk
**Validation**: Test after each phase to ensure functionality preserved

---

## Phase 1: Extract Data Source Strategies (LOW RISK)
**Duration**: 2-3 days  
**Goal**: Eliminate code duplication in data gathering without changing core architecture

### Step 1.1: Create Data Source Abstractions
**Files to Create**:
- `src/research/data_sources/__init__.py`
- `src/research/data_sources/base.py`
- `src/research/data_sources/web_search.py`
- `src/research/data_sources/product_catalog.py`

**Implementation**:
```python
# src/research/data_sources/base.py
class DataSource(ABC):
    @abstractmethod
    async def gather(self, queries: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        pass

# src/research/data_sources/web_search.py  
class WebSearchDataSource(DataSource):
    def __init__(self, search_provider: TavilySearchProvider):
        self.search_provider = search_provider
    
    async def gather(self, queries: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        # Extract common web search logic from researchers
        pass

# src/research/data_sources/product_catalog.py
class ProductCatalogDataSource(DataSource):
    async def gather(self, queries: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        # Extract product catalog logic from ProductStyleResearcher
        pass
```

### Step 1.2: Refactor Foundation Researcher
**Files to Modify**: `src/research/foundation_research.py`

**Changes**:
- Extract web search logic from `FoundationResearcher._gather_data()`
- Update to use `WebSearchDataSource`
- Keep interface unchanged - no breaking changes

**Validation Command**: 
```bash
python brand_researcher.py --brand specialized.com --phase foundation_research
```

### Step 1.3: Apply to Remaining Researchers
**Files to Modify**:
- `src/research/market_positioning_research.py`
- `src/research/product_style_research.py`
- `src/research/customer_cultural_research.py`
- `src/research/voice_messaging_research.py`

**Approach**: One researcher per commit, test after each

**Success Criteria**:
- Web search logic centralized
- 60%+ reduction in duplicated code
- All existing tests pass
- Research phases produce identical results

---

## Phase 2: Extract Quality Evaluation (MEDIUM RISK)
**Duration**: 3-4 days  
**Goal**: Remove quality concerns from BaseResearcher while maintaining functionality

### Step 2.1: Create Quality Service
**Files to Create**:
- `src/research/quality/quality_service.py`

**Implementation**:
```python
class QualityService:
    def __init__(self, evaluator: QualityEvaluator):
        self.evaluator = evaluator
    
    async def ensure_quality(self, content: str, phase_name: str, threshold: float, 
                           max_attempts: int = 3, improvement_feedback: Optional[List[str]] = None) -> QualityResult:
        # Extract quality logic from BaseResearcher._research_with_quality_wrapper()
        pass
        
    async def evaluate_quality(self, content: str, phase_name: str) -> QualityEvaluation:
        # Extract from BaseResearcher._evaluate_quality()
        pass
```

### Step 2.2: Inject Quality Service into BaseResearcher
**Files to Modify**: `src/research/base_researcher.py`

**Changes**:
- Add `quality_service: QualityService` parameter to constructor  
- Keep existing quality methods as thin wrappers
- No behavior changes yet

### Step 2.3: Migrate Quality Logic
**Files to Modify**: `src/research/base_researcher.py`

**Changes**:
- Move quality evaluation logic from BaseResearcher to QualityService
- Update BaseResearcher to delegate to service
- Remove quality-related methods: `_evaluate_quality()`, `_research_with_quality_wrapper()`

**Validation Commands**:
```bash
python demo_quality_evaluation.py
./scripts/brand_manager.sh status specialized.com
```

**Success Criteria**:
- Quality logic reusable and testable in isolation
- Quality evaluation behavior unchanged
- Quality metadata still saved correctly

---

## Phase 3: Simplify BaseResearcher (MEDIUM RISK)
**Duration**: 2-3 days  
**Goal**: Reduce BaseResearcher responsibilities while maintaining template method pattern

### Step 3.1: Extract Storage Operations
**Files to Create**:
- `src/research/storage/research_storage.py`

**Implementation**:
```python
class ResearchStorage:
    def __init__(self, storage_manager):
        self.storage_manager = storage_manager
    
    def save_research_result(self, brand_domain: str, phase_name: str, result: ResearchResult) -> None:
        pass
    
    def load_cached_result(self, brand_domain: str, phase_name: str, cache_duration_days: int) -> Optional[ResearchResult]:
        pass
        
    def is_cache_valid(self, brand_domain: str, phase_name: str, cache_duration_days: int) -> bool:
        pass
```

### Step 3.2: Extract Progress Tracking
**Files to Create**:
- `src/research/progress/research_progress.py`

**Implementation**:
```python
class ResearchProgressTracker:
    def __init__(self, progress_tracker: ProgressTracker):
        self.progress_tracker = progress_tracker
    
    def start_phase(self, phase_name: str) -> None:
        pass
    
    def complete_phase(self, phase_name: str, result: ResearchResult) -> None:
        pass
        
    def update_progress(self, phase_name: str, step: str, progress: float) -> None:
        pass
```

### Step 3.3: Simplify BaseResearcher Constructor
**Files to Modify**: `src/research/base_researcher.py`

**Changes**:
- Inject services instead of creating them: `ResearchStorage`, `ResearchProgressTracker`
- Reduce from 8 parameters to 4 core dependencies
- Create builder pattern for complex initialization

**New Constructor**:
```python
def __init__(self, brand_domain: str, researcher_name: str, 
             storage: ResearchStorage, progress: ResearchProgressTracker):
    self.brand_domain = brand_domain
    self.researcher_name = researcher_name
    self.storage = storage
    self.progress = progress
```

**Success Criteria**:
- BaseResearcher constructor simplified
- Storage and progress logic reusable
- All factory functions updated to use new pattern

---

## Phase 4: Create Research Engine (MEDIUM RISK)
**Duration**: 4-5 days  
**Goal**: Separate research execution from research logic

### Step 4.1: Create ResearchEngine
**Files to Create**:
- `src/research/engine/research_engine.py`

**Implementation**:
```python
class ResearchEngine:
    def __init__(self, storage: ResearchStorage, progress: ResearchProgressTracker, 
                 quality: QualityService):
        self.storage = storage
        self.progress = progress  
        self.quality = quality
    
    async def execute_research(self, researcher: BaseResearcher, 
                             force_refresh: bool = False, 
                             improvement_feedback: Optional[List[str]] = None) -> ResearchResult:
        # Extract execution logic from BaseResearcher._execute_core_research()
        pass
        
    async def _execute_with_caching(self, researcher: BaseResearcher, force_refresh: bool) -> ResearchResult:
        # Extract caching logic
        pass
        
    async def _execute_with_quality_assurance(self, researcher: BaseResearcher, result: ResearchResult) -> ResearchResult:
        # Extract quality assurance orchestration
        pass
```

### Step 4.2: Update BaseResearcher to Use Engine
**Files to Modify**: `src/research/base_researcher.py`

**Changes**:
- Inject ResearchEngine into BaseResearcher (via factory functions)
- Delegate execution to engine: `return await self.engine.execute_research(self, force_refresh)`
- Keep public interface unchanged

### Step 4.3: Move Complex Logic to Engine
**Files to Modify**: 
- `src/research/base_researcher.py`
- `src/research/engine/research_engine.py`

**Changes**:
- Move 187-line `_execute_core_research()` logic to engine
- Move caching logic to engine
- Move quality evaluation orchestration to engine
- BaseResearcher becomes pure research logic (template methods only)

**Final BaseResearcher Interface**:
```python
class BaseResearcher:
    async def research(self, force_refresh: bool = False, improvement_feedback: Optional[List[str]] = None) -> ResearchResult:
        return await self.engine.execute_research(self, force_refresh, improvement_feedback)
    
    @abstractmethod
    async def _gather_data(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def _analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def _synthesize_results(self, analysis: Dict[str, Any]) -> str:
        pass
```

**Success Criteria**:
- BaseResearcher reduced from 200+ lines to <100 lines
- Clear separation between "what to research" and "how to execute research"
- Research execution logic reusable across different contexts

---

## Phase 5: Improve Factory and Registry (LOW RISK)
**Duration**: 2-3 days  
**Goal**: Make research phase management more extensible

### Step 5.1: Create Research Phase Registry
**Files to Create**:
- `src/research/registry/phase_registry.py`

**Implementation**:
```python
@dataclass
class PhaseDefinition:
    name: str
    factory_func: Callable[..., BaseResearcher]
    dependencies: List[str] = field(default_factory=list)
    quality_threshold: float = 8.0
    cache_duration_days: int = 7

class ResearchPhaseRegistry:
    def __init__(self):
        self._phases: Dict[str, PhaseDefinition] = {}
    
    def register_phase(self, definition: PhaseDefinition) -> None:
        self._phases[definition.name] = definition
    
    def get_phase(self, name: str, **kwargs) -> BaseResearcher:
        if name not in self._phases:
            raise ValueError(f"Unknown research phase: {name}")
        return self._phases[name].factory_func(**kwargs)
    
    def get_execution_order(self) -> List[str]:
        # Topological sort based on dependencies
        pass
    
    def list_phases(self) -> List[str]:
        return list(self._phases.keys())
```

### Step 5.2: Update EnhancedBrandResearcher
**Files to Modify**: `brand_researcher.py`

**Changes**:
- Replace manual dictionary with registry
- Keep existing phase names and behavior
- Add registration calls for existing phases

**New Implementation**:
```python
class EnhancedBrandResearcher:
    def __init__(self, brand_domain: str):
        self.registry = ResearchPhaseRegistry()
        self._register_standard_phases()
        
    def _register_standard_phases(self):
        self.registry.register_phase(PhaseDefinition(
            name="foundation",
            factory_func=lambda: get_foundation_researcher(brand_domain=self.brand_domain),
            quality_threshold=8.0
        ))
        # ... register other phases
```

### Step 5.3: Add New Phase Registration
**Files to Create**: `src/research/registry/standard_phases.py`

**Implementation**:
```python
def register_standard_phases(registry: ResearchPhaseRegistry, brand_domain: str) -> None:
    """Register all standard research phases"""
    
    registry.register_phase(PhaseDefinition(
        name="foundation",
        factory_func=lambda: get_foundation_researcher(brand_domain=brand_domain),
        dependencies=[],
        quality_threshold=8.0
    ))
    
    registry.register_phase(PhaseDefinition(
        name="market_positioning", 
        factory_func=lambda: get_market_positioning_researcher(brand_domain=brand_domain),
        dependencies=["foundation"],
        quality_threshold=8.0
    ))
    # ... etc
```

**Success Criteria**:
- New research phases can be added via registration only
- Phase dependencies automatically enforced
- No changes required to core orchestration for new phases

---

## Phase 6: Advanced Patterns (OPTIONAL)
**Duration**: TBD  
**Goal**: Further improvements for complex scenarios

### Step 6.1: Add Pipeline Configuration
**Files to Create**: `src/research/config/pipeline_configuration.py`

**Purpose**: Enable different research pipelines for different brand types

### Step 6.2: Add Error Boundaries  
**Files to Create**: `src/research/engine/error_boundary.py`

**Purpose**: Implement retry logic, fallbacks, circuit breakers

### Step 6.3: Add Dependency Injection Container
**Files to Create**: `src/research/container/research_container.py`

**Purpose**: Centralized service configuration and dependency management

---

## Implementation Guidelines

### Risk Mitigation Strategy
1. **One change per commit** - Easy to revert if issues arise
2. **Maintain interfaces** - Existing code continues to work during transition
3. **Add new alongside old** - Gradual migration, not replacement
4. **Test at each step** - Ensure functionality preserved

### Validation Points
After each phase, run these commands to verify system integrity:
```bash
# Test individual research phase
python brand_researcher.py --brand specialized.com --phase foundation_research

# Test complete pipeline
python brand_researcher.py --brand specialized.com --auto-continue

# Test quality evaluation
python demo_quality_evaluation.py

# Test workflow management
./scripts/brand_manager.sh status specialized.com
./scripts/brand_manager.sh next-step specialized.com

# Run existing tests
python test_specialized.py
pytest tests/
```

### Success Metrics
- **Code Quality**: BaseResearcher complexity reduced from 200+ lines to <100 lines
- **Maintainability**: Code duplication reduced by 60%+
- **Extensibility**: New research phases addable with 0 changes to core orchestration
- **Testability**: Quality evaluation logic reusable across different contexts
- **Functionality**: All existing features work identically

### Rollback Strategy
Each phase is designed with clear rollback points:
- **Phase 1**: Revert data source extraction, restore original _gather_data methods
- **Phase 2**: Restore quality logic to BaseResearcher, remove QualityService
- **Phase 3**: Restore original BaseResearcher constructor and methods
- **Phase 4**: Remove ResearchEngine, restore _execute_core_research logic
- **Phase 5**: Restore manual dictionary in EnhancedBrandResearcher

### Documentation Updates
After completion, update:
- `CLAUDE.md` - Reflect new architecture patterns
- `PROJECT_FILEMAP.md` - Update core files and architecture description  
- `COPILOT_NOTES.md` - Document refactoring decisions and lessons learned

---

## Current Status: Planning Complete
**Next Step**: Begin Phase 1 - Extract Data Source Strategies
**Estimated Timeline**: 15-20 days total implementation
**Risk Assessment**: Low to Medium, with clear rollback points