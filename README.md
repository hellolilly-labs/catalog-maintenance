# Liddy AI Platform

Monorepo for the Liddy AI Platform, containing:
- **Liddy Core** - Shared infrastructure and utilities
- **Liddy Intelligence Engine** - Brand research and catalog management
- **Liddy Voice Assistant** - Real-time conversational AI

## Structure

```
python-liddy/
├── packages/
│   ├── liddy/              # Core shared functionality
│   ├── liddy_intelligence/ # Brand Intelligence Engine
│   └── liddy_voice/        # Voice AI Assistant
├── deployments/            # Deployment configurations
├── tests/                  # Test suites
└── docs/                   # Documentation
```

## Quick Start

### Development Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the packages
./scripts/install_packages.sh

# Install in development mode
./scripts/setup_dev.sh
```

### Running Services

#### Brand Intelligence Engine

```bash
# Run brand research
python run/brand_research.py specialized.com

# Ingest product catalog  
python run/ingest_catalog.py specialized.com

# Generate descriptors
python run/generate_descriptors.py specialized.com
```

#### Voice Assistant

```bash
# Start voice assistant
python run/voice_main.py
```

## Key Components

### Liddy Core (`packages/liddy/`)
- **Models**: Product, User, Account data models
- **Storage**: Unified storage abstraction (GCS, local)
- **Search**: RAG search infrastructure (Pinecone)
- **Account Manager**: Account configuration and intelligence
- **Product Manager**: In-memory product catalog management

### Liddy Intelligence (`packages/liddy_intelligence/`)
- **Brand Research**: 8-phase AI-powered brand analysis
- **Catalog Ingestion**: Product catalog processing and enrichment
- **Voice Testing**: Search quality evaluation tools

### Liddy Voice (`packages/liddy_voice/`)
- **Assistants**: LiveKit-based voice agents
- **Conversation**: Dialog management and tracking
- **Session**: User session state management

## Documentation

- [Development Guide](docs/development/setup.md)
- [Intelligence Engine Guide](docs/intelligence/README.md)
- [Voice Assistant Guide](docs/voice/README.md)

## License

Copyright (c) 2024 Liddy AI. All rights reserved.