# Migration Notice

This `src/` directory contains legacy code that has been migrated to the monorepo structure.

## Migration Map

### Core Functionality → `packages/liddy/`
- `storage.py` → `liddy.storage`
- `models/` → `liddy.models`
- `search/` → `liddy.search`
- `account_manager.py` → `liddy.account_manager`

### Intelligence Engine → `packages/liddy_intelligence/`
- `research/` → `liddy_intelligence.research`
- `llm/` → `liddy_intelligence.llm`
- `agents/` → `liddy_intelligence.agents`
- `workflow/` → `liddy_intelligence.workflow`
- `web_search.py` → `liddy_intelligence.web_search`
- `progress_tracker.py` → `liddy_intelligence.progress_tracker`

### Catalog Processing → `packages/liddy_intelligence/`
- `catalog/` → `liddy_intelligence.catalog`
- `ingestion/` → `liddy_intelligence.ingestion`

### Voice Components → `packages/liddy_voice/`
- Voice assistant functionality has been moved to the liddy_voice package

## Important Notes

1. **DO NOT** add new code to this directory
2. **DO NOT** modify existing files (except for import redirects)
3. **DO** update your imports to use the new package structure
4. This directory will be removed in a future update

## Quick Migration Guide

Replace your imports:

```python
# Old
from src.storage import get_account_storage_provider
from src.research.base_researcher import BaseResearcher

# New
from liddy.storage import get_account_storage_provider
from liddy_intelligence.research.base_researcher import BaseResearcher
```

For CLI scripts, use the runner scripts in `run/`:
```bash
# Old
python src/research/brand_researcher.py specialized.com

# New
python run/brand_research.py specialized.com
```