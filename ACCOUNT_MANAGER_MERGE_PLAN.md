# AccountManager Merge Plan

## Current State

We have two separate AccountManager implementations:

### 1. `liddy.AccountManager` (Core Package)
**Purpose**: Search configuration and catalog intelligence
- Search backend configuration (Pinecone indexes)
- Product catalog intelligence loading
- Query enhancement prompts
- Brand-specific configurations
- Variant types management

### 2. `liddy_voice.AccountManager` (Voice Package)
**Purpose**: Voice assistant configuration
- TTS (Text-to-Speech) provider settings
- Voice personality and greeting management
- Dynamic configuration loading from GCP
- RAG index configuration for voice search
- Multi-agent support

## Merge Strategy

### Option 1: Complete Merge (Recommended)
Merge both AccountManagers into a single unified class in the liddy package.

**Advantages**:
- Single source of truth for account configuration
- Eliminates cross-package dependencies
- Simpler to maintain and extend
- Consistent configuration loading pattern

**Implementation Steps**:
1. Move voice-specific data classes to liddy package
2. Merge the two AccountManager classes
3. Use optional loading for voice-specific features
4. Update all imports across the codebase

### Option 2: Inheritance Pattern
Keep core AccountManager in liddy, make voice AccountManager inherit from it.

**Advantages**:
- Maintains separation of concerns
- Voice package can extend without modifying core
- Backward compatibility easier

**Disadvantages**:
- Still have two classes to maintain
- Potential for confusion about which to use
- Inheritance can become complex

### Option 3: Composition Pattern
Keep both but have voice AccountManager compose/delegate to core.

**Advantages**:
- Clear separation of responsibilities
- Each package maintains its own domain

**Disadvantages**:
- More complex architecture
- Still need to manage two classes

## Recommended Approach: Complete Merge

### 1. New Structure in `liddy.account_manager`

```python
# liddy/account_manager/__init__.py

class AccountManager:
    """Unified account configuration manager."""
    
    def __init__(self, account: str):
        self.account = self._normalize_account(account)
        self._search_config = None
        self._catalog_intelligence = None
        self._voice_config = None  # Optional voice configuration
        self._is_voice_enabled = False
    
    # Core methods (existing)
    def get_search_config(self) -> SearchConfig
    def get_catalog_intelligence(self) -> Optional[Dict[str, Any]]
    def get_search_enhancement_prompt(self, context_type: str) -> Optional[str]
    def get_variant_types(self) -> List[Dict[str, str]]
    
    # Voice methods (merged from voice package)
    def get_tts_settings(self) -> Optional[AccountTTSSettings]
    def get_default_greeting(self) -> str
    def get_personality_sliders(self) -> Dict[str, float]
    def get_rag_details(self) -> Tuple[Optional[str], Optional[str]]
    
    # Configuration loading
    async def _load_config(self) -> None:
        """Load all configurations for this account."""
        # Load core config (search, catalog)
        await self._load_core_config()
        
        # Try to load voice config if available
        try:
            await self._load_voice_config()
            self._is_voice_enabled = True
        except Exception:
            # Voice config is optional
            self._is_voice_enabled = False
```

### 2. Data Classes Organization

```
liddy/
├── account_manager/
│   ├── __init__.py          # Main AccountManager class
│   ├── models.py            # All data models
│   │   ├── SearchConfig
│   │   ├── TtsProviderSettings
│   │   ├── ElevenLabsTtsProviderSettings
│   │   └── AccountTTSSettings
│   └── config_loaders.py    # Configuration loading utilities
```

### 3. Migration Steps

#### Phase 1: Preparation
1. Create new models.py file with all data classes
2. Create config_loaders.py for configuration loading logic
3. Add voice-specific methods to liddy.AccountManager as optional features

#### Phase 2: Migration
1. Update liddy.AccountManager to include voice functionality
2. Add backward compatibility layer in liddy_voice.account_manager
3. Update imports to use new structure
4. Test thoroughly

#### Phase 3: Cleanup
1. Deprecate liddy_voice.AccountManager
2. Remove legacy account_manager files in src/ and voice_assistant/
3. Update documentation

### 4. Backward Compatibility

During transition, liddy_voice.account_manager becomes a thin wrapper:

```python
# liddy_voice/account_manager.py
from liddy.account_manager import (
    AccountManager as BaseAccountManager,
    ElevenLabsTtsProviderSettings,
    AccountTTSSettings,
    get_account_manager
)

# Deprecated - for backward compatibility only
AccountManager = BaseAccountManager

__all__ = [
    'AccountManager',
    'ElevenLabsTtsProviderSettings', 
    'AccountTTSSettings',
    'get_account_manager'
]
```

### 5. Benefits of Unified AccountManager

1. **Single Source of Truth**: All account configuration in one place
2. **Optional Features**: Voice features only loaded when needed
3. **Better Performance**: Single cache, single loading mechanism
4. **Easier Testing**: One class to test instead of two
5. **Cleaner Architecture**: No cross-package dependencies
6. **Future Extensibility**: Easy to add new features (e.g., email settings)

### 6. Potential Challenges

1. **Package Dependencies**: Need to ensure liddy doesn't depend on voice-specific packages
2. **Optional Loading**: Must handle cases where voice config doesn't exist
3. **Testing**: Need comprehensive tests for both core and voice features
4. **Migration Risk**: Many files to update, potential for breakage

### 7. Implementation Timeline

- **Week 1**: Create new structure, add voice methods to core
- **Week 2**: Migrate and test all imports
- **Week 3**: Clean up and remove deprecated code
- **Week 4**: Documentation and final testing

## Decision Required

Before proceeding, we need to decide:

1. **Merge Approach**: Complete merge (recommended) vs inheritance vs composition?
2. **Migration Strategy**: Big bang vs gradual with compatibility layer?
3. **Voice Dependencies**: How to handle voice-specific dependencies in core package?
4. **Configuration Storage**: Unify configuration storage approach?

## Next Steps

1. Review and approve merge strategy
2. Create detailed technical design
3. Set up feature branch for implementation
4. Begin phased implementation
5. Comprehensive testing
6. Documentation update
7. Deploy and monitor