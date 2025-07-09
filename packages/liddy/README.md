# Liddy Core Package

Base functionality and shared utilities for the Liddy AI platform.

## Overview

The `liddy` package provides core functionality that is shared across all Liddy services:

- **Storage Abstraction**: Unified interface for file storage (local/GCS)
- **Data Models**: Core data structures (Product, Brand, etc.)
- **Configuration**: Centralized settings management
- **Utilities**: Common helper functions

## Installation

```bash
pip install -e packages/liddy
```

## Key Components

### Storage System
```python
from liddy.storage import get_account_storage_provider

# Get storage provider (auto-detects local vs GCS)
storage = get_account_storage_provider()

# Read/write files
content = await storage.read_file("specialized.com", "products.json")
await storage.write_file("specialized.com", "analysis.json", json_data)
```

### Product Manager
```python
from liddy.models.product_manager import get_product_manager

# Load products for a brand
manager = await get_product_manager("specialized.com")
products = await manager.get_products()
```

### Configuration
```python
from liddy.config import get_settings

settings = get_settings()
print(settings.openai_api_key)
```

## Architecture

```
liddy/
├── __init__.py
├── config.py          # Settings management
├── storage.py         # Storage abstraction
├── models/
│   ├── product.py     # Product data model
│   └── product_manager.py  # Product catalog management
└── utils/
    └── ...           # Helper utilities
```

## Dependencies

- pydantic: Data validation and settings
- google-cloud-storage: GCS support (optional)
- aiofiles: Async file operations

## Development

This is a foundational package - changes here affect all other Liddy packages.
Ensure backward compatibility when making modifications.