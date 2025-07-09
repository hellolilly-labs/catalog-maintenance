#!/bin/bash
# Version bumping script for catalog-maintenance

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the bump type (major, minor, patch)
BUMP_TYPE=${1:-patch}

# Validate bump type
if [[ ! "$BUMP_TYPE" =~ ^(major|minor|patch)$ ]]; then
    echo -e "${RED}Error: Invalid bump type. Use 'major', 'minor', or 'patch'${NC}"
    echo "Usage: $0 [major|minor|patch]"
    exit 1
fi

# Read current version
if [ ! -f VERSION ]; then
    echo -e "${RED}Error: VERSION file not found${NC}"
    exit 1
fi

CURRENT_VERSION=$(cat VERSION)
echo -e "${YELLOW}Current version: $CURRENT_VERSION${NC}"

# Parse version components
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"

# Bump version based on type
case $BUMP_TYPE in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
esac

# Create new version
NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo -e "${GREEN}New version: $NEW_VERSION${NC}"

# Update VERSION file
echo "$NEW_VERSION" > VERSION

# Update CHANGELOG.md
TODAY=$(date +%Y-%m-%d)
CHANGELOG_ENTRY="## [$NEW_VERSION] - $TODAY"

# Add new version section after [Unreleased]
sed -i.bak "/## \[Unreleased\]/a\\
\\
$CHANGELOG_ENTRY\\
\\
### Added\\
\\
### Changed\\
\\
### Fixed\\
" CHANGELOG.md

# Update comparison links
sed -i.bak "s|\[Unreleased\]:.*|\[Unreleased\]: https://github.com/hellolilly-labs/catalog-maintenance/compare/v$NEW_VERSION...HEAD|" CHANGELOG.md

# Add new comparison link
COMPARISON_LINE="[$NEW_VERSION]: https://github.com/hellolilly-labs/catalog-maintenance/compare/v$CURRENT_VERSION...v$NEW_VERSION"
echo "$COMPARISON_LINE" >> CHANGELOG.md

# Clean up backup files
rm -f CHANGELOG.md.bak

echo -e "${GREEN}✅ Version bumped to $NEW_VERSION${NC}"
echo ""
echo "Next steps:"
echo "1. Update CHANGELOG.md with your changes"
echo "2. Commit: git add VERSION CHANGELOG.md && git commit -m \"chore: bump version to $NEW_VERSION\""
echo "3. Tag: git tag -a v$NEW_VERSION -m \"Release version $NEW_VERSION\""
echo "4. Push: git push origin main --tags"