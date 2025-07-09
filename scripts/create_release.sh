#!/bin/bash
# Create a new release for catalog-maintenance

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're on main/dev branch
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "main" && "$CURRENT_BRANCH" != "dev" ]]; then
    echo -e "${YELLOW}Warning: You're on branch '$CURRENT_BRANCH', not main or dev${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: You have uncommitted changes${NC}"
    git status --short
    exit 1
fi

# Read current version
if [ ! -f VERSION ]; then
    echo -e "${RED}Error: VERSION file not found${NC}"
    exit 1
fi

VERSION=$(cat VERSION)
TAG="v$VERSION"

echo -e "${GREEN}Creating release for version: $VERSION${NC}"

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo -e "${RED}Error: Tag $TAG already exists${NC}"
    exit 1
fi

# Extract release notes from CHANGELOG
echo "Extracting release notes from CHANGELOG.md..."
RELEASE_NOTES=""
IN_VERSION_SECTION=0
while IFS= read -r line; do
    if [[ "$line" =~ ^##[[:space:]]\[$VERSION\] ]]; then
        IN_VERSION_SECTION=1
        continue
    elif [[ "$line" =~ ^##[[:space:]]\[ ]] && [ $IN_VERSION_SECTION -eq 1 ]; then
        break
    elif [ $IN_VERSION_SECTION -eq 1 ]; then
        RELEASE_NOTES+="$line"$'\n'
    fi
done < CHANGELOG.md

# Create git tag
echo -e "${YELLOW}Creating git tag $TAG...${NC}"
git tag -a "$TAG" -m "Release version $VERSION

$RELEASE_NOTES"

# Push tag
echo -e "${YELLOW}Pushing tag to origin...${NC}"
git push origin "$TAG"

# Create GitHub release
echo -e "${YELLOW}Creating GitHub release...${NC}"
gh release create "$TAG" \
    --title "Release $VERSION" \
    --notes "$RELEASE_NOTES" \
    --draft

echo -e "${GREEN}✅ Release created successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Review the draft release on GitHub"
echo "2. Add any additional release notes"
echo "3. Publish the release when ready"
echo ""
echo "Release URL: https://github.com/hellolilly-labs/catalog-maintenance/releases/tag/$TAG"