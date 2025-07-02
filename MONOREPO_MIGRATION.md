# Monorepo Migration Plan

## Overview
Migrating from separate `catalog-maintenance` and `voice-service` repositories to a unified `python-liddy` monorepo.

## Migration Status

### Phase 1: Repository Setup ✅
- [x] Create package directory structure
- [x] Create package __init__ files
- [x] Create root pyproject.toml
- [ ] Update .gitignore for monorepo
- [ ] Create development setup script

### Phase 2: Core Package Extraction (Next)
- [ ] Move shared models to packages/liddy/models/
- [ ] Move storage to packages/liddy/storage/
- [ ] Move search to packages/liddy/search/
- [ ] Move account_manager to packages/liddy/account_manager/
- [ ] Move product_manager to packages/liddy/product_manager/
- [ ] Update imports in moved files

### Phase 3: Intelligence Package Setup
- [ ] Move research modules to packages/liddy_intelligence/research/
- [ ] Move ingestion modules to packages/liddy_intelligence/ingestion/
- [ ] Move agents to packages/liddy_intelligence/agents/
- [ ] Move workflow to packages/liddy_intelligence/workflow/
- [ ] Update imports

### Phase 4: Voice Package Integration
- [ ] Copy voice-service/spence to packages/liddy_voice/
- [ ] Remove duplicate files (use core instead)
- [ ] Update imports to use liddy core
- [ ] Integrate main.py

### Phase 5: Testing & Validation
- [ ] Set up unified test structure
- [ ] Ensure all tests pass
- [ ] Test editable installs
- [ ] Test deployment configs

### Phase 6: Repository Rename & Cleanup
- [ ] Rename GitHub repository to python-liddy
- [ ] Update remote URLs
- [ ] Archive voice-service repo
- [ ] Update documentation

## File Mapping

### Core Package Files
```
src/models/* → packages/liddy/models/
src/storage.py → packages/liddy/storage/
src/search/* → packages/liddy/search/
src/account_manager.py → packages/liddy/account_manager/
src/models/product_manager.py → packages/liddy/product_manager/
src/llm/prompt_manager.py → packages/liddy/prompt_manager/
```

### Intelligence Package Files
```
src/research/* → packages/liddy_intelligence/research/
src/ingestion/* → packages/liddy_intelligence/ingestion/
src/agents/* → packages/liddy_intelligence/agents/
src/workflow/* → packages/liddy_intelligence/workflow/
src/llm/* → packages/liddy_intelligence/llm/
src/catalog/* → packages/liddy_intelligence/catalog/
```

### Voice Package Files
```
../voice-service/spence/* → packages/liddy_voice/
../voice-service/main.py → packages/liddy_voice/main.py
```

## Notes
- Keeping llm_service.py in voice package (LiveKit-specific)
- Using catalog's SearchService as the unified search
- Gradual migration to minimize disruption