# Tests Directory

This directory contains all test files organized by category.

## Structure

- `core/` - Core package tests
- `intelligence/` - Intelligence package tests  
- `voice/` - Voice package tests
- `integration/` - Integration tests
- `performance/` - Performance benchmarks and tests
- `stt/` - Speech-to-Text (STT) extractor tests
- `ingestion/` - Catalog ingestion tests

## Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/performance/

# Run with coverage
pytest --cov=packages
```