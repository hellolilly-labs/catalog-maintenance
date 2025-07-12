# Tools Directory

Utility scripts and tools for maintaining the catalog-maintenance system.

## Tools

### Descriptor Management (`descriptors/`)
- `check_descriptor_progress.py` - Check descriptor generation progress for brands
- `fix_missing_descriptors.py` - Generate missing descriptors for products
- `generate_missing_descriptors.py` - Batch descriptor generation
- `verify_descriptor_quality.py` - Verify quality scores of generated descriptors

### Research Tools
- `verify_research_paths.py` - Verify research file paths and structure

## Usage Examples

```bash
# Check descriptor progress
python tools/descriptors/check_descriptor_progress.py

# Fix missing descriptors for a brand
python tools/descriptors/fix_missing_descriptors.py

# Verify research paths
python tools/verify_research_paths.py
```