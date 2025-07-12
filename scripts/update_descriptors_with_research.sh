#!/bin/bash
# Update Product Descriptors with Terminology Research
#
# This script ensures that industry terminology research is run before
# generating or updating product descriptors, enabling better semantic
# price categorization.
#
# Usage: ./scripts/update_descriptors_with_research.sh specialized.com

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the account from command line
ACCOUNT=$1

if [ -z "$ACCOUNT" ]; then
    echo -e "${RED}Error: Account/brand domain required${NC}"
    echo "Usage: $0 specialized.com"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Updating Descriptors for $ACCOUNT${NC}"
echo -e "${BLUE}========================================${NC}"

# Step 1: Check if terminology research exists
echo -e "\n${YELLOW}Step 1: Checking for terminology research...${NC}"
RESEARCH_FILE="accounts/${ACCOUNT}/research/industry_terminology/research.md"

if [ -f "$RESEARCH_FILE" ]; then
    echo -e "${GREEN}✓ Terminology research found${NC}"
    
    # Check age of research
    if [ "$(uname)" = "Darwin" ]; then
        # macOS
        RESEARCH_AGE=$(( ($(date +%s) - $(stat -f %m "$RESEARCH_FILE")) / 86400 ))
    else
        # Linux
        RESEARCH_AGE=$(( ($(date +%s) - $(stat -c %Y "$RESEARCH_FILE")) / 86400 ))
    fi
    
    if [ "$RESEARCH_AGE" -gt 30 ]; then
        echo -e "${YELLOW}⚠ Research is ${RESEARCH_AGE} days old. Consider refreshing.${NC}"
        read -p "Refresh research? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Running terminology research...${NC}"
            python run/research_industry_terminology.py "$ACCOUNT" --force
        fi
    fi
else
    echo -e "${YELLOW}⚠ No terminology research found${NC}"
    echo -e "${BLUE}Running terminology research...${NC}"
    python run/research_industry_terminology.py "$ACCOUNT"
fi

# Step 2: Analyze price distribution
echo -e "\n${YELLOW}Step 2: Analyzing price distribution...${NC}"
python run/analyze_price_distribution.py "$ACCOUNT"

# Step 3: Check current descriptor status
echo -e "\n${YELLOW}Step 3: Checking descriptor status...${NC}"
DESCRIPTOR_STATUS=$(python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'packages')
import asyncio
from liddy.storage import get_account_storage_provider

async def check():
    storage = get_account_storage_provider()
    products = await storage.get_product_catalog('$ACCOUNT')
    total = len(products)
    with_desc = sum(1 for p in products if p.get('descriptor'))
    return f'{with_desc}/{total}'

print(asyncio.run(check()))
" 2>/dev/null || echo "0/0")

echo -e "Current descriptor coverage: ${DESCRIPTOR_STATUS}"

# Step 4: Update descriptors
echo -e "\n${YELLOW}Step 4: Updating descriptors with price information...${NC}"
read -p "Update all descriptors? This will include terminology-aware pricing. (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # First update existing descriptors with price
    echo -e "${BLUE}Updating existing descriptors...${NC}"
    python run/update_descriptor_prices.py "$ACCOUNT"
    
    # Then generate missing descriptors
    echo -e "${BLUE}Generating missing descriptors...${NC}"
    python run/generate_descriptors.py "$ACCOUNT"
    
    # Final price analysis with updated descriptors
    echo -e "\n${BLUE}Final analysis with updated descriptors...${NC}"
    python run/analyze_price_distribution.py "$ACCOUNT"
fi

# Step 5: Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Update Complete for $ACCOUNT${NC}"
echo -e "${GREEN}========================================${NC}"

# Show key files
echo -e "\nKey files updated:"
echo -e "  • ${BLUE}accounts/${ACCOUNT}/research/industry_terminology/research.md${NC}"
echo -e "  • ${BLUE}accounts/${ACCOUNT}/products.json${NC}"
echo -e "  • ${BLUE}accounts/${ACCOUNT}/analysis/price_distribution_${ACCOUNT}.json${NC}"

# Suggest next steps
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Test price-based search queries:"
echo -e "     ${BLUE}python run/test_search.py \"$ACCOUNT\" \"bikes under 3000\"${NC}"
echo -e "     ${BLUE}python run/test_search.py \"$ACCOUNT\" \"premium gravel bikes\"${NC}"
echo -e "  2. Review semantic phrases in the analysis output"
echo -e "  3. Consider category-specific pricing if recommended"