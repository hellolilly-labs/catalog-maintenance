"""
DEPRECATED: This src/ directory is deprecated.

All code has been migrated to the monorepo structure:
- Core functionality → packages/liddy/
- Intelligence engine → packages/liddy_intelligence/
- Voice components → packages/liddy_voice/

Please update your imports to use the new package structure.
See MIGRATION_NOTICE.md for details.
"""

import warnings

warnings.warn(
    "The src/ directory is deprecated. Please update imports to use packages/liddy*, packages/liddy_intelligence*, or packages/liddy_voice*",
    DeprecationWarning,
    stacklevel=2
)