# Cleanup Plan for Monorepo Migration

## Files to Keep

### Core Documentation
- README.md
- CLAUDE.md (project instructions)
- pyproject.toml
- MONOREPO_MIGRATION.md

### Main Entry Points (to organize)
1. **Brand Research**: `brand_researcher.py`
2. **Catalog Ingestion**: `pre_generate_descriptors.py`, `ingest_product_catalog.py`
3. **Voice Testing**: `voice_search_comparison.py`

### Essential Guides (to consolidate)
- RUN_INGESTION_GUIDE.md (keep, update for monorepo)

## Files to Remove

### Old Documentation (37 files)
- All ROADMAP/*.md files
- All implementation guides (*_GUIDE.md, *_PLAN.md, *_SUMMARY.md)
- All voice_assistant/*.md files
- PROJECT_FILEMAP.md, COPILOT_NOTES.md, etc.

### Test/Demo Files to Reorganize
- Move useful tests to `tests/` directory with proper structure
- Remove all demo_*.py files
- Remove example_*.py files
- Keep only actively used test files

### Files to Archive
- Archive old tests and demos (already in archive/)
- Remove test files from root directory

## New Documentation Structure
```
docs/
├── README.md              # Main documentation
├── intelligence/
│   ├── brand_research.md
│   ├── catalog_ingestion.md
│   └── api_reference.md
├── voice/
│   ├── getting_started.md
│   └── api_reference.md
└── development/
    ├── setup.md
    └── testing.md
```