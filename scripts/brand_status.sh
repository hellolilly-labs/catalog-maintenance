#!/bin/bash

# ============================================
# Brand Status & Monitoring Script
# Comprehensive Pipeline Health Monitoring
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
MODE="overview"
WATCH_MODE=false
WATCH_INTERVAL=30

# Help function
show_help() {
    echo "Brand Status & Monitoring Script"
    echo ""
    echo "Usage: $0 --brand <brand_url> [options]"
    echo ""
    echo "Required:"
    echo "  --brand <url>           Brand URL to check status"
    echo ""
    echo "Status Modes:"
    echo "  --overview              Complete pipeline overview (default)"
    echo "  --research              Research phases status only"
    echo "  --catalog               Product catalog status only"
    echo "  --knowledge             Knowledge base status only"
    echo "  --rag                   RAG optimization status only"
    echo "  --persona               AI persona status only"
    echo "  --alerts                Show alerts and issues only"
    echo "  --health                Quick health score only"
    echo "  --performance           Performance metrics only"
    echo ""
    echo "Monitoring Options:"
    echo "  --watch                 Continuously monitor (updates every 30s)"
    echo "  --interval <seconds>    Custom watch interval (default: 30)"
    echo "  --batch <brands>        Check multiple brands (comma-separated)"
    echo ""
    echo "Examples:"
    echo "  $0 --brand specialized.com"
    echo "  $0 --brand nike.com --research --watch"
    echo "  $0 --batch 'specialized.com,nike.com,adidas.com' --alerts"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --brand)
            BRAND_URL="$2"
            shift 2
            ;;
        --overview)
            MODE="overview"
            shift
            ;;
        --research)
            MODE="research"
            shift
            ;;
        --catalog)
            MODE="catalog"
            shift
            ;;
        --knowledge)
            MODE="knowledge"
            shift
            ;;
        --rag)
            MODE="rag"
            shift
            ;;
        --persona)
            MODE="persona"
            shift
            ;;
        --alerts)
            MODE="alerts"
            shift
            ;;
        --health)
            MODE="health"
            shift
            ;;
        --performance)
            MODE="performance"
            shift
            ;;
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --interval)
            WATCH_INTERVAL="$2"
            shift 2
            ;;
        --batch)
            BATCH_BRANDS="$2"
            shift 2
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

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
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

# Check single brand status
check_brand_status() {
    local brand=$1
    local mode=$2
    
    case $mode in
        "overview")
            python3 src/status/pipeline_status.py --brand "$brand"
            ;;
        "research")
            python3 src/status/pipeline_status.py --brand "$brand" --research-only
            ;;
        "catalog")
            python3 src/status/pipeline_status.py --brand "$brand" --catalog-only
            ;;
        "knowledge")
            python3 src/status/pipeline_status.py --brand "$brand" --knowledge-only
            ;;
        "rag")
            python3 src/status/pipeline_status.py --brand "$brand" --rag-only
            ;;
        "persona")
            python3 src/status/pipeline_status.py --brand "$brand" --persona-only
            ;;
        "alerts")
            python3 src/status/pipeline_status.py --brand "$brand" --alerts-only
            ;;
        "health")
            python3 src/status/pipeline_status.py --brand "$brand" --health-only
            ;;
        "performance")
            python3 src/status/pipeline_status.py --brand "$brand" --performance
            ;;
    esac
}

# Batch status check
check_batch_status() {
    local brands=$1
    local mode=$2
    
    log_info "Checking status for multiple brands..."
    
    # Convert comma-separated string to array
    IFS=',' read -ra BRAND_ARRAY <<< "$brands"
    
    for brand in "${BRAND_ARRAY[@]}"; do
        echo ""
        echo "═══════════════════════════════════════"
        echo "📊 Status for: $brand"
        echo "═══════════════════════════════════════"
        check_brand_status "$brand" "$mode"
    done
}

# Watch mode with continuous monitoring
watch_status() {
    local brand=$1
    local mode=$2
    local interval=$3
    
    log_info "Starting continuous monitoring for $brand (interval: ${interval}s)"
    log_info "Press Ctrl+C to stop monitoring"
    
    while true; do
        clear
        echo "🔄 Live Brand Status Monitor"
        echo "Brand: $brand | Mode: $mode | Interval: ${interval}s"
        echo "Last Update: $(date)"
        echo "═══════════════════════════════════════════════════════════"
        
        check_brand_status "$brand" "$mode"
        
        echo ""
        echo "Next update in ${interval} seconds... (Ctrl+C to stop)"
        sleep "$interval"
    done
}

# Emergency alert check
check_emergency_alerts() {
    local brand=$1
    
    log_info "Checking for critical alerts..."
    
    # Get alerts only
    alerts_output=$(python3 src/status/pipeline_status.py --brand "$brand" --alerts-only 2>&1)
    alerts_count=$(echo "$alerts_output" | grep -c "🚨\|❌\|⚠️" || true)
    
    if [ "$alerts_count" -gt 0 ]; then
        log_error "Found $alerts_count alert(s) for $brand"
        echo "$alerts_output"
        return 1
    else
        log_success "No critical alerts found for $brand"
        return 0
    fi
}

# Performance summary
show_performance_summary() {
    local brand=$1
    
    echo ""
    echo "📈 PERFORMANCE SUMMARY: $brand"
    echo "═══════════════════════════════════════"
    
    # Get performance metrics
    python3 src/status/pipeline_status.py --brand "$brand" --performance
    
    # Show recent trends
    echo ""
    echo "📊 Recent Health Trends (7 days):"
    python3 src/status/pipeline_status.py --brand "$brand" --health-trend --days 7
}

# Quick health check for monitoring systems
quick_health_check() {
    local brand=$1
    
    # Get just the health score
    health_score=$(python3 src/status/pipeline_status.py --brand "$brand" --health-only --json | jq -r '.overall_health' 2>/dev/null || echo "unknown")
    
    if [ "$health_score" != "unknown" ]; then
        if (( $(echo "$health_score >= 8.0" | bc -l) )); then
            echo -e "${GREEN}🟢 HEALTHY ($health_score/10)${NC}"
        elif (( $(echo "$health_score >= 6.0" | bc -l) )); then
            echo -e "${YELLOW}🟡 WARNING ($health_score/10)${NC}"
        else
            echo -e "${RED}🔴 CRITICAL ($health_score/10)${NC}"
        fi
    else
        echo -e "${RED}❓ UNKNOWN${NC}"
    fi
}

# Main execution
main() {
    echo "📊 Brand Status Monitor"
    echo "Timestamp: $(date)"
    echo ""
    
    # Handle batch mode
    if [ -n "$BATCH_BRANDS" ]; then
        check_batch_status "$BATCH_BRANDS" "$MODE"
        exit 0
    fi
    
    # Validate brand URL
    if [ -z "$BRAND_URL" ]; then
        echo -e "${RED}Error: --brand parameter is required${NC}"
        show_help
        exit 1
    fi
    
    # Handle watch mode
    if [ "$WATCH_MODE" = true ]; then
        watch_status "$BRAND_URL" "$MODE" "$WATCH_INTERVAL"
        exit 0
    fi
    
    # Regular status check
    echo "Checking status for: $BRAND_URL"
    echo "Mode: $MODE"
    echo ""
    
    check_brand_status "$BRAND_URL" "$MODE"
    
    # If overview mode, show additional summary
    if [ "$MODE" = "overview" ]; then
        echo ""
        echo "💡 Quick Actions:"
        echo "  • Refresh stale research: ./scripts/brand_maintenance.sh --brand $BRAND_URL --refresh-stale"
        echo "  • Monitor continuously: $0 --brand $BRAND_URL --watch"
        echo "  • Check alerts only: $0 --brand $BRAND_URL --alerts"
    fi
}

# Handle Ctrl+C gracefully in watch mode
trap 'echo -e "\n${YELLOW}Monitoring stopped.${NC}"; exit 0' INT

# Run main function
main "$@" 