#!/bin/bash
# Ensure Research Before Descriptors
#
# This script ensures all necessary research phases are complete before
# generating product descriptors. It checks for:
# 1. Brand research phases (8 phases)
# 2. Industry terminology research
# 3. Product catalog research synthesis
#
# Usage: ./scripts/ensure_research_before_descriptors.sh specialized.com [--force]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get parameters
ACCOUNT=$1
FORCE_FLAG=$2

if [ -z "$ACCOUNT" ]; then
    echo -e "${RED}Error: Account/brand domain required${NC}"
    echo "Usage: $0 specialized.com [--force]"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Ensuring Research for $ACCOUNT${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check research age
check_research_age() {
    local file=$1
    local max_age_days=$2
    
    if [ -f "$file" ]; then
        if [ "$(uname)" = "Darwin" ]; then
            # macOS
            age_days=$(( ($(date +%s) - $(stat -f %m "$file")) / 86400 ))
        else
            # Linux
            age_days=$(( ($(date +%s) - $(stat -c %Y "$file")) / 86400 ))
        fi
        
        if [ "$age_days" -gt "$max_age_days" ]; then
            echo "$age_days"
            return 1
        else
            echo "$age_days"
            return 0
        fi
    else
        echo "-1"
        return 2
    fi
}

# Step 1: Check brand research phases
echo -e "\n${YELLOW}Step 1: Checking brand research phases...${NC}"
RESEARCH_PHASES=(
    "foundation_research"
    "market_positioning_research"
    "product_style_research"
    "customer_cultural_research"
    "voice_messaging_research"
    "interview_synthesis_research"
    "linearity_analysis_research"
    "industry_terminology"
    "research_integration"
)

missing_phases=()
stale_phases=()

for phase in "${RESEARCH_PHASES[@]}"; do
    research_file="accounts/${ACCOUNT}/research/${phase}/research.md"
    age=$(check_research_age "$research_file" 60)
    status=$?
    
    if [ $status -eq 2 ]; then
        missing_phases+=("$phase")
        echo -e "  ${RED}✗ $phase - Missing${NC}"
    elif [ $status -eq 1 ]; then
        stale_phases+=("$phase")
        echo -e "  ${YELLOW}⚠ $phase - ${age} days old (stale)${NC}"
    else
        echo -e "  ${GREEN}✓ $phase - ${age} days old${NC}"
    fi
done

# Step 2: Check industry terminology research
echo -e "\n${YELLOW}Step 2: Checking industry terminology research...${NC}"
TERMINOLOGY_FILE="accounts/${ACCOUNT}/research/industry_terminology/research.md"
term_age=$(check_research_age "$TERMINOLOGY_FILE" 30)
term_status=$?

if [ $term_status -eq 2 ]; then
    echo -e "  ${RED}✗ Industry terminology - Missing${NC}"
    terminology_missing=true
elif [ $term_status -eq 1 ]; then
    echo -e "  ${YELLOW}⚠ Industry terminology - ${term_age} days old (stale)${NC}"
    terminology_stale=true
else
    echo -e "  ${GREEN}✓ Industry terminology - ${term_age} days old${NC}"
    terminology_missing=false
    terminology_stale=false
fi

# Step 3: Check product catalog research
echo -e "\n${YELLOW}Step 3: Checking product catalog research synthesis...${NC}"
CATALOG_FILE="accounts/${ACCOUNT}/research/product_catalog_research/research.md"
catalog_age=$(check_research_age "$CATALOG_FILE" 30)
catalog_status=$?

if [ $catalog_status -eq 2 ]; then
    echo -e "  ${RED}✗ Product catalog research - Missing${NC}"
    catalog_missing=true
elif [ $catalog_status -eq 1 ]; then
    echo -e "  ${YELLOW}⚠ Product catalog research - ${catalog_age} days old (stale)${NC}"
    catalog_stale=true
else
    echo -e "  ${GREEN}✓ Product catalog research - ${catalog_age} days old${NC}"
    catalog_missing=false
    catalog_stale=false
fi

# Step 4: Run missing research
echo -e "\n${YELLOW}Step 4: Running missing/stale research...${NC}"

# Handle missing brand research phases
if [ ${#missing_phases[@]} -gt 0 ] || [ "$FORCE_FLAG" = "--force" ]; then
    if [ ${#missing_phases[@]} -gt 0 ]; then
        echo -e "${RED}Missing ${#missing_phases[@]} research phases${NC}"
    fi
    
    if [ "$FORCE_FLAG" = "--force" ]; then
        echo -e "${BLUE}Force flag set - running all research phases${NC}"
        read -p "Run all brand research phases? (y/N) " -n 1 -r
    else
        read -p "Run missing brand research phases? (y/N) " -n 1 -r
    fi
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Running brand research pipeline...${NC}"
        python run/brand_research.py "$ACCOUNT" --phase all
    fi
elif [ ${#stale_phases[@]} -gt 0 ]; then
    echo -e "${YELLOW}${#stale_phases[@]} research phases are stale${NC}"
    read -p "Refresh stale phases? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for phase in "${stale_phases[@]}"; do
            echo -e "${BLUE}Refreshing $phase...${NC}"
            python run/brand_research.py "$ACCOUNT" --phase "$phase" --force
        done
    fi
fi

# Handle industry terminology research
if [ "$terminology_missing" = true ] || [ "$FORCE_FLAG" = "--force" ]; then
    echo -e "\n${BLUE}Running industry terminology research...${NC}"
    python run/research_industry_terminology.py "$ACCOUNT" $([ "$FORCE_FLAG" = "--force" ] && echo "--force")
elif [ "$terminology_stale" = true ]; then
    read -p "Refresh industry terminology research? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Refreshing terminology research...${NC}"
        python run/research_industry_terminology.py "$ACCOUNT" --force
    fi
fi

# Handle product catalog research
if [ "$catalog_missing" = true ] || [ "$FORCE_FLAG" = "--force" ]; then
    echo -e "\n${BLUE}Running product catalog research synthesis...${NC}"
    # This synthesizes all the brand research phases
    python run/brand_research.py "$ACCOUNT" --phase product_catalog $([ "$FORCE_FLAG" = "--force" ] && echo "--force")
elif [ "$catalog_stale" = true ]; then
    read -p "Refresh product catalog research? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Refreshing catalog research...${NC}"
        python run/brand_research.py "$ACCOUNT" --phase product_catalog --force
    fi
fi

# Step 5: Final verification
echo -e "\n${YELLOW}Step 5: Final verification...${NC}"

all_good=true

# Check all phases again
for phase in "${RESEARCH_PHASES[@]}"; do
    research_file="accounts/${ACCOUNT}/research/${phase}/research.md"
    if [ ! -f "$research_file" ]; then
        echo -e "  ${RED}✗ $phase still missing${NC}"
        all_good=false
    fi
done

# Check terminology
if [ ! -f "$TERMINOLOGY_FILE" ]; then
    echo -e "  ${RED}✗ Industry terminology still missing${NC}"
    all_good=false
fi

# Check catalog
if [ ! -f "$CATALOG_FILE" ]; then
    echo -e "  ${RED}✗ Product catalog research still missing${NC}"
    all_good=false
fi

# Summary
echo -e "\n${GREEN}========================================${NC}"
if [ "$all_good" = true ]; then
    echo -e "${GREEN}✅ All research complete for $ACCOUNT${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    echo -e "\n${YELLOW}Ready to generate descriptors!${NC}"
    echo -e "Next steps:"
    echo -e "  1. Generate descriptors with research:"
    echo -e "     ${BLUE}python run/generate_descriptors.py $ACCOUNT${NC}"
    echo -e "  2. Or update existing descriptors with pricing:"
    echo -e "     ${BLUE}python run/update_descriptor_prices.py $ACCOUNT${NC}"
    echo -e "  3. Test price-based search:"
    echo -e "     ${BLUE}python run/test_price_enhancement.py $ACCOUNT --test-search${NC}"
else
    echo -e "${RED}❌ Some research is still missing${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "\nPlease run the missing research before generating descriptors."
fi