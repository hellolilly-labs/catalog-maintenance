# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2025-07-14

### Added
- System prompt generation framework for AI personas
- Generic AI persona template with variable placeholders
- Research-to-template mapping guide
- Comprehensive generation plan with token optimization
- Enhanced voice agent with product state tracking
- ProductManager class with smart URL extraction
- Redis-based product state persistence
- Temperature control and model configuration for voice assistant
- LLM prewarm capability to cache system prompts
- Enhanced price descriptor validation for sale prices
- Product history tracking in UserState

### Changed
- Improved session state management with better error handling
- Enhanced unified descriptor generation with category normalization
- Optimized voice agent initialization for faster time-to-first-utterance
- Updated Redis product loader to store products permanently (no TTL)
- Improved price descriptor updater to detect outdated sale prices

### Fixed
- Fixed product URL caching and validation in session state
- Fixed recent_product_ids backward compatibility parsing
- Fixed browsing history tracking with proper timestamps

## [0.2.0] - 2024-01-09

### Added
- Comprehensive price-based search system with natural language query support
- Industry terminology research system (`IndustryTerminologyResearcher`)
- Price statistics analyzer with multi-modal distribution detection
- Price descriptor updater for automatic price information in descriptors
- Content classification system for terminology (safe/understand_only/offensive)
- Enhanced search service with price extraction and filtering
- Voice assistant price integration with contextual summaries
- Runner scripts for terminology research and price updates
- Automation scripts for workflow management
- Comprehensive documentation for price-based search

### Changed
- Enhanced `SearchService` to support price queries
- Updated `VoiceSearchWrapper` to include price context
- Modified `UnifiedDescriptorGenerator` to integrate with terminology research
- Improved `WebSearch` to capture Tavily AI-generated answers

### Fixed
- Mid-tier terminology not being captured in reports
- Parser not detecting brand-specific mid-tier indicators
- Sample size calculation returning 0 for small catalogs

## [0.1.0] - 2024-01-01

### Added
- Initial monorepo structure with namespace packages
- Core liddy package with storage abstraction
- Brand intelligence pipeline with 8-phase research
- Product catalog ingestion system
- Voice assistant with real-time capabilities
- Basic search functionality
- Docker deployment configurations

### Changed
- Migrated from single repository to monorepo structure
- Reorganized code into namespace packages

### Deprecated
- Legacy src/ directory structure (being migrated)

### Security
- Removed hardcoded API keys from code
- Added Google Cloud authentication for all environments

[Unreleased]: https://github.com/hellolilly-labs/catalog-maintenance/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/hellolilly-labs/catalog-maintenance/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/hellolilly-labs/catalog-maintenance/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hellolilly-labs/catalog-maintenance/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hellolilly-labs/catalog-maintenance/releases/tag/v0.1.0