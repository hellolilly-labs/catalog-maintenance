# Manual Runners for Liddy Intelligence

This directory contains simple runner scripts for manually executing and debugging the core Liddy Intelligence functions.

## Available Runners

### 1. Brand Research Pipeline
```bash
python run/brand_research.py specialized.com
python run/brand_research.py specialized.com --phase foundation
python run/brand_research.py specialized.com --auto-continue
python run/brand_research.py specialized.com --status
```

### 2. Product Descriptor Generation
```bash
python run/generate_descriptors.py specialized.com
python run/generate_descriptors.py specialized.com --force
python run/generate_descriptors.py specialized.com --evaluate-models
```

### 3. Product Catalog Ingestion (to Pinecone)
```bash
python run/ingest_catalog.py specialized.com
python run/ingest_catalog.py specialized.com --force-update
```

### 4. Voice Search Testing
```bash
python run/voice_search.py flexfits.com
python run/voice_search.py flexfits.com --setup-baseline
```

## Purpose

These runners are for local development and debugging. They:
- Set up the proper Python path
- Import the actual implementation from the package structure
- Pass through all command-line arguments
- Handle async execution

## Note

Always run these from the repository root:
```bash
cd /path/to/catalog-maintenance
python run/brand_research.py specialized.com
```