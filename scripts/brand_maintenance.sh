#!/bin/bash

# ============================================
# Brand Maintenance Script
# Regular Updates & Refresh Workflows
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
BRAND_URL=""
MAINTENANCE_TYPE=""
SKIP_CONFIRMATION=false
DRY_RUN=false

# Help function
show_help() {
    echo "Brand Maintenance Script - Regular Updates & Refresh Workflows"
    echo ""
    echo "Usage: $0 --brand <brand_url> --type <maintenance_type> [options]"
    echo ""
    echo "Required:"
    echo "  --brand <url>           Brand URL to maintain"
    echo "  --type <type>           Type of maintenance to perform"
    echo ""
    echo "Maintenance Types:"
    echo "  refresh-stale           Refresh only stale research phases"
    echo "  refresh-all             Force refresh all research phases"
    echo "  refresh-catalog         Update product catalog only"
    echo "  refresh-knowledge       Update knowledge base only"
    echo "  refresh-rag             Update RAG optimization only"
    echo "  refresh-persona         Generate new AI persona recommendations"
    echo "  smart-refresh           Intelligent refresh based on cache status"
    echo "  weekly-maintenance      Full weekly maintenance routine"
    echo "  monthly-maintenance     Full monthly maintenance routine"
    echo ""
    echo "Specific Refresh Options:"
    echo "  --phases <list>         Specific research phases (comma-separated)"
    echo "                         Options: foundation,market_positioning,product_style,"
    echo "                                 customer_cultural,voice_messaging,rag_optimization"
    echo ""
    echo "Options:"
    echo "  --skip-confirmation     Skip confirmation prompts"
    echo "  --dry-run              Show what would be done without executing"
    echo "  --force                Force refresh even if not needed"
    echo "  --quality-check        Run quality evaluation after refresh"
    echo ""
    echo "Examples:"
    echo "  $0 --brand specialized.com --type refresh-stale"
    echo "  $0 --brand nike.com --type smart-refresh --quality-check"
    echo "  $0 --brand adidas.com --phases 'product_style,voice_messaging'"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --brand)
            BRAND_URL="$2"
            shift 2
            ;;
        --type)
            MAINTENANCE_TYPE="$2"
            shift 2
            ;;
        --phases)
            SPECIFIC_PHASES="$2"
            shift 2
            ;;
        --skip-confirmation)
            SKIP_CONFIRMATION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE_REFRESH=true
            shift
            ;;
        --quality-check)
            QUALITY_CHECK=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$BRAND_URL" ]; then
    echo -e "${RED}Error: --brand parameter is required${NC}"
    show_help
    exit 1
fi

if [ -z "$MAINTENANCE_TYPE" ] && [ -z "$SPECIFIC_PHASES" ]; then
    echo -e "${RED}Error: --type parameter or --phases is required${NC}"
    show_help
    exit 1
fi

# Logging functions
log_step() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Dry run logging
dry_run_action() {
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would execute: $1"
    else
        log_step "Executing: $1"
        eval "$1"
    fi
}

# Check maintenance prerequisites
check_prerequisites() {
    log_step "Checking maintenance prerequisites..."
    
    # Check if brand exists
    if ! python3 src/status/pipeline_status.py --brand "$BRAND_URL" --health-only >/dev/null 2>&1; then
        log_error "Brand $BRAND_URL not found in system"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Show maintenance plan
show_maintenance_plan() {
    local maintenance_type=$1
    
    echo ""
    echo "🔧 BRAND MAINTENANCE PLAN"
    echo "═══════════════════════════"
    echo "Brand: $BRAND_URL"
    echo "Type: $maintenance_type"
    echo "Timestamp: $(date)"
    
    if [ "$DRY_RUN" = true ]; then
        echo "Mode: DRY RUN (no changes will be made)"
    fi
    
    echo ""
    
    case $maintenance_type in
        "refresh-stale")
            echo "📋 Stale Refresh Plan:"
            echo "• Check all research phases for staleness"
            echo "• Refresh only phases past their cache expiry"
            echo "• Skip fresh phases to save costs"
            ;;
        "refresh-all")
            echo "📋 Full Refresh Plan:"
            echo "• Force refresh ALL research phases"
            echo "• Regenerate brand intelligence completely"
            echo "• High cost but ensures latest information"
            ;;
        "refresh-catalog")
            echo "📋 Catalog Refresh Plan:"
            echo "• Sync latest product data"
            echo "• Update product descriptors"
            echo "• Refresh vector embeddings"
            ;;
        "refresh-knowledge")
            echo "📋 Knowledge Base Refresh Plan:"
            echo "• Re-ingest brand intelligence"
            echo "• Update knowledge chunks"
            echo "• Optimize vector indexes"
            ;;
        "smart-refresh")
            echo "📋 Smart Refresh Plan:"
            echo "• Analyze current status"
            echo "• Refresh based on cache expiry and quality scores"
            echo "• Optimize cost vs freshness balance"
            ;;
        "weekly-maintenance")
            echo "📋 Weekly Maintenance Plan:"
            echo "• Refresh high-frequency phases (voice_messaging)"
            echo "• Update product catalog"
            echo "• Check for performance degradation"
            ;;
        "monthly-maintenance")
            echo "📋 Monthly Maintenance Plan:"
            echo "• Refresh medium-frequency phases"
            echo "• Deep quality evaluation"
            echo "• Update RAG optimization"
            ;;
    esac
    
    if [ "$SKIP_CONFIRMATION" = false ] && [ "$DRY_RUN" = false ]; then
        echo ""
        read -p "Proceed with maintenance? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Maintenance cancelled."
            exit 0
        fi
    fi
}

# Refresh stale research phases
refresh_stale_phases() {
    log_step "Checking for stale research phases..."
    
    # Get list of stale phases
    stale_phases=$(python3 src/status/pipeline_status.py --brand "$BRAND_URL" --stale-phases --json | jq -r '.stale_phases[]' 2>/dev/null || echo "")
    
    if [ -z "$stale_phases" ]; then
        log_success "No stale phases found - all research is fresh!"
        return 0
    fi
    
    log_info "Found stale phases: $stale_phases"
    
    # Convert to comma-separated list
    phase_list=$(echo "$stale_phases" | tr '\n' ',' | sed 's/,$//')
    
    # Refresh stale phases
    dry_run_action "python3 src/research/brand_researcher.py --brand '$BRAND_URL' --phases '$phase_list' --verbose"
    
    log_success "Stale phases refreshed"
}

# Smart refresh based on status analysis
smart_refresh() {
    log_step "Analyzing brand status for smart refresh..."
    
    # Get comprehensive status
    status_output=$(python3 src/status/pipeline_status.py --brand "$BRAND_URL" --smart-analysis --json 2>/dev/null || echo "{}")
    
    # Extract recommendations
    refresh_recommendations=$(echo "$status_output" | jq -r '.refresh_recommendations[]' 2>/dev/null || echo "")
    
    if [ -z "$refresh_recommendations" ]; then
        log_success "No refresh needed - brand is optimally maintained!"
        return 0
    fi
    
    log_info "Smart refresh recommendations:"
    echo "$refresh_recommendations"
    
    # Execute recommendations
    while IFS= read -r recommendation; do
        if [ -n "$recommendation" ]; then
            dry_run_action "$recommendation"
        fi
    done <<< "$refresh_recommendations"
    
    log_success "Smart refresh completed"
}

# Refresh specific phases
refresh_specific_phases() {
    local phases=$1
    
    log_step "Refreshing specific phases: $phases"
    
    dry_run_action "python3 src/research/brand_researcher.py --brand '$BRAND_URL' --phases '$phases' --verbose"
    
    log_success "Specific phases refreshed"
}

# Weekly maintenance routine
weekly_maintenance() {
    log_step "Starting weekly maintenance routine..."
    
    # 1. Refresh high-frequency phases
    dry_run_action "python3 src/research/brand_researcher.py --brand '$BRAND_URL' --phases 'voice_messaging' --verbose"
    
    # 2. Update product catalog
    dry_run_action "python3 src/product_ingestor.py --brand '$BRAND_URL' --incremental-sync --use-brand-intelligence"
    
    # 3. Check for alerts
    alerts_found=$(python3 src/status/pipeline_status.py --brand "$BRAND_URL" --alerts-count 2>/dev/null || echo "0")
    
    if [ "$alerts_found" -gt 0 ]; then
        log_warning "Found $alerts_found alert(s) during weekly maintenance"
        python3 src/status/pipeline_status.py --brand "$BRAND_URL" --alerts-only
    fi
    
    log_success "Weekly maintenance completed"
}

# Monthly maintenance routine
monthly_maintenance() {
    log_step "Starting monthly maintenance routine..."
    
    # 1. Refresh medium-frequency phases
    dry_run_action "python3 src/research/brand_researcher.py --brand '$BRAND_URL' --phases 'product_style,customer_cultural' --verbose"
    
    # 2. Deep quality evaluation
    dry_run_action "python3 src/quality/quality_evaluator.py --brand '$BRAND_URL' --comprehensive-evaluation"
    
    # 3. Update RAG optimization
    dry_run_action "python3 src/research/brand_researcher.py --brand '$BRAND_URL' --phases 'rag_optimization' --verbose"
    
    # 4. Performance analysis
    dry_run_action "python3 src/status/pipeline_status.py --brand '$BRAND_URL' --performance-analysis --save-report"
    
    log_success "Monthly maintenance completed"
}

# Quality check after maintenance
run_quality_check() {
    if [ "$QUALITY_CHECK" = true ]; then
        log_step "Running post-maintenance quality check..."
        
        dry_run_action "python3 src/quality/quality_evaluator.py --brand '$BRAND_URL' --post-maintenance-check"
        
        log_success "Quality check completed"
    fi
}

# Show maintenance summary
show_maintenance_summary() {
    echo ""
    echo "📊 MAINTENANCE SUMMARY"
    echo "═══════════════════════"
    
    if [ "$DRY_RUN" = false ]; then
        # Show updated status
        python3 src/status/pipeline_status.py --brand "$BRAND_URL" --health-only
        
        echo ""
        echo "💡 Next Steps:"
        echo "  • Monitor performance: ./scripts/brand_status.sh --brand $BRAND_URL --performance"
        echo "  • Check for new alerts: ./scripts/brand_status.sh --brand $BRAND_URL --alerts"
    else
        echo "DRY RUN completed - no changes were made"
        echo ""
        echo "💡 To execute these changes:"
        echo "  • Remove --dry-run flag and run again"
    fi
}

# Main execution
main() {
    echo "🔧 Brand Maintenance Starting..."
    echo "Brand: $BRAND_URL"
    echo "Timestamp: $(date)"
    echo ""
    
    check_prerequisites
    
    # Handle specific phases
    if [ -n "$SPECIFIC_PHASES" ]; then
        show_maintenance_plan "refresh-phases"
        refresh_specific_phases "$SPECIFIC_PHASES"
        run_quality_check
        show_maintenance_summary
        return 0
    fi
    
    # Handle maintenance types
    show_maintenance_plan "$MAINTENANCE_TYPE"
    
    # Record start time
    START_TIME=$(date +%s)
    
    case $MAINTENANCE_TYPE in
        "refresh-stale")
            refresh_stale_phases
            ;;
        "refresh-all")
            dry_run_action "python3 src/research/brand_researcher.py --brand '$BRAND_URL' --full-research --force"
            ;;
        "refresh-catalog")
            dry_run_action "python3 src/product_ingestor.py --brand '$BRAND_URL' --full-sync --use-brand-intelligence"
            ;;
        "refresh-knowledge")
            dry_run_action "python3 src/knowledge_ingestor.py --brand '$BRAND_URL' --full-refresh --include-brand-intelligence"
            ;;
        "refresh-rag")
            dry_run_action "python3 src/research/brand_researcher.py --brand '$BRAND_URL' --phases 'rag_optimization' --force"
            ;;
        "refresh-persona")
            dry_run_action "python3 src/research/brand_researcher.py --brand '$BRAND_URL' --phases 'ai_persona_generation'"
            ;;
        "smart-refresh")
            smart_refresh
            ;;
        "weekly-maintenance")
            weekly_maintenance
            ;;
        "monthly-maintenance")
            monthly_maintenance
            ;;
        *)
            log_error "Unknown maintenance type: $MAINTENANCE_TYPE"
            exit 1
            ;;
    esac
    
    run_quality_check
    
    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    log_success "Maintenance completed in ${MINUTES}m ${SECONDS}s"
    
    show_maintenance_summary
}

# Run main function
main "$@" 