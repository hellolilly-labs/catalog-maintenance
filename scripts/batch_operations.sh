#!/bin/bash

# ============================================
# Batch Operations Script
# Multi-Brand Processing & Management
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
BATCH_FILE=""
BRAND_LIST=""
OPERATION=""
MAX_PARALLEL=3
CONTINUE_ON_ERROR=false
SAVE_LOGS=true

# Help function
show_help() {
    echo "Batch Operations Script - Multi-Brand Processing & Management"
    echo ""
    echo "Usage: $0 --operation <operation> --brands <brands> [options]"
    echo ""
    echo "Required:"
    echo "  --operation <op>        Batch operation to perform"
    echo ""
    echo "Brand Selection (choose one):"
    echo "  --brands <list>         Comma-separated brand URLs"
    echo "  --file <path>           File with brand URLs (one per line)"
    echo "  --discover              Auto-discover all brands in system"
    echo ""
    echo "Batch Operations:"
    echo "  status-check            Quick health check for all brands"
    echo "  full-status             Complete status report for all brands"
    echo "  onboard-new             Onboard multiple new brands"
    echo "  refresh-stale           Refresh stale content across brands"
    echo "  smart-maintenance       Intelligent maintenance for all brands"
    echo "  weekly-maintenance      Weekly maintenance routine"
    echo "  monthly-maintenance     Monthly maintenance routine"
    echo "  emergency-check         Check for critical alerts across brands"
    echo "  performance-report      Generate performance report for all brands"
    echo "  cost-analysis           Analyze and report API costs by brand"
    echo "  quality-audit           Comprehensive quality audit"
    echo ""
    echo "Options:"
    echo "  --parallel <count>      Max parallel operations (default: 3)"
    echo "  --continue-on-error     Continue processing even if some brands fail"
    echo "  --no-logs              Don't save detailed logs"
    echo "  --dry-run              Show what would be done without executing"
    echo "  --output-format <fmt>   Output format: table, json, csv (default: table)"
    echo ""
    echo "Examples:"
    echo "  $0 --operation status-check --discover"
    echo "  $0 --operation onboard-new --brands 'nike.com,adidas.com,puma.com'"
    echo "  $0 --operation refresh-stale --file brands.txt --parallel 5"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --operation)
            OPERATION="$2"
            shift 2
            ;;
        --brands)
            BRAND_LIST="$2"
            shift 2
            ;;
        --file)
            BATCH_FILE="$2"
            shift 2
            ;;
        --discover)
            DISCOVER_BRANDS=true
            shift
            ;;
        --parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --no-logs)
            SAVE_LOGS=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --output-format)
            OUTPUT_FORMAT="$2"
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

# Validate required parameters
if [ -z "$OPERATION" ]; then
    echo -e "${RED}Error: --operation parameter is required${NC}"
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

# Setup logging
setup_logging() {
    if [ "$SAVE_LOGS" = true ]; then
        LOG_DIR="logs/batch_operations"
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"
        log_info "Batch logs will be saved to: $LOG_FILE"
    fi
}

# Discover brands in system
discover_brands() {
    log_step "Discovering brands in system..."
    
    # Use Python to discover brands from storage
    discovered_brands=$(python -c "
import os
from pathlib import Path

accounts_dir = Path('accounts')
if accounts_dir.exists():
    brands = [d.name for d in accounts_dir.iterdir() if d.is_dir()]
    print(','.join(brands))
else:
    print('')
" 2>/dev/null || echo "")
    
    if [ -n "$discovered_brands" ]; then
        log_success "Discovered $(echo "$discovered_brands" | tr ',' '\n' | wc -l) brands"
        echo "$discovered_brands"
    else
        log_warning "No brands discovered in system"
        echo ""
    fi
}

# Load brands from file
load_brands_from_file() {
    local file_path=$1
    
    if [ ! -f "$file_path" ]; then
        log_error "Brand file not found: $file_path"
        exit 1
    fi
    
    # Read brands from file (one per line, skip comments and empty lines)
    brands=$(grep -v '^#' "$file_path" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
    
    if [ -n "$brands" ]; then
        log_success "Loaded $(echo "$brands" | tr ',' '\n' | wc -l) brands from file"
        echo "$brands"
    else
        log_error "No valid brands found in file: $file_path"
        exit 1
    fi
}

# Get brand list based on input method
get_brand_list() {
    if [ "$DISCOVER_BRANDS" = true ]; then
        discover_brands
    elif [ -n "$BATCH_FILE" ]; then
        load_brands_from_file "$BATCH_FILE"
    elif [ -n "$BRAND_LIST" ]; then
        echo "$BRAND_LIST"
    else
        log_error "No brand selection method specified"
        show_help
        exit 1
    fi
}

# Execute operation for single brand
execute_brand_operation() {
    local brand=$1
    local operation=$2
    local brand_log=""
    
    if [ "$SAVE_LOGS" = true ]; then
        brand_log="$LOG_DIR/${brand}_$(date +%H%M%S).log"
    fi
    
    case $operation in
        "status-check")
            if [ -n "$brand_log" ]; then
                python3 src/status/pipeline_status.py --brand "$brand" --health-only > "$brand_log" 2>&1
            else
                python3 src/status/pipeline_status.py --brand "$brand" --health-only
            fi
            ;;
        "full-status")
            if [ -n "$brand_log" ]; then
                python3 src/status/pipeline_status.py --brand "$brand" > "$brand_log" 2>&1
            else
                python3 src/status/pipeline_status.py --brand "$brand"
            fi
            ;;
        "onboard-new")
            if [ -n "$brand_log" ]; then
                ./scripts/brand_onboarding.sh --brand "$brand" --skip-confirmation > "$brand_log" 2>&1
            else
                ./scripts/brand_onboarding.sh --brand "$brand" --skip-confirmation
            fi
            ;;
        "refresh-stale")
            if [ -n "$brand_log" ]; then
                ./scripts/brand_maintenance.sh --brand "$brand" --type refresh-stale --skip-confirmation > "$brand_log" 2>&1
            else
                ./scripts/brand_maintenance.sh --brand "$brand" --type refresh-stale --skip-confirmation
            fi
            ;;
        "smart-maintenance")
            if [ -n "$brand_log" ]; then
                ./scripts/brand_maintenance.sh --brand "$brand" --type smart-refresh --skip-confirmation > "$brand_log" 2>&1
            else
                ./scripts/brand_maintenance.sh --brand "$brand" --type smart-refresh --skip-confirmation
            fi
            ;;
        "weekly-maintenance")
            if [ -n "$brand_log" ]; then
                ./scripts/brand_maintenance.sh --brand "$brand" --type weekly-maintenance --skip-confirmation > "$brand_log" 2>&1
            else
                ./scripts/brand_maintenance.sh --brand "$brand" --type weekly-maintenance --skip-confirmation
            fi
            ;;
        "monthly-maintenance")
            if [ -n "$brand_log" ]; then
                ./scripts/brand_maintenance.sh --brand "$brand" --type monthly-maintenance --skip-confirmation > "$brand_log" 2>&1
            else
                ./scripts/brand_maintenance.sh --brand "$brand" --type monthly-maintenance --skip-confirmation
            fi
            ;;
        "emergency-check")
            if [ -n "$brand_log" ]; then
                python3 src/status/pipeline_status.py --brand "$brand" --alerts-only > "$brand_log" 2>&1
            else
                python3 src/status/pipeline_status.py --brand "$brand" --alerts-only
            fi
            ;;
        *)
            log_error "Unknown operation: $operation"
            return 1
            ;;
    esac
}

# Parallel execution manager
execute_parallel_operations() {
    local brands=$1
    local operation=$2
    local max_parallel=$3
    
    # Convert comma-separated brands to array
    IFS=',' read -ra BRAND_ARRAY <<< "$brands"
    
    local total_brands=${#BRAND_ARRAY[@]}
    local completed=0
    local failed=0
    local pids=()
    local current_parallel=0
    
    log_info "Processing $total_brands brands with max $max_parallel parallel operations"
    
    for brand in "${BRAND_ARRAY[@]}"; do
        # Wait if we've reached max parallel limit
        while [ $current_parallel -ge $max_parallel ]; do
            # Check for completed jobs
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}"
                    exit_code=$?
                    
                    if [ $exit_code -eq 0 ]; then
                        log_success "Completed: ${BRAND_ARRAY[$i]}"
                        ((completed++))
                    else
                        log_error "Failed: ${BRAND_ARRAY[$i]} (exit code: $exit_code)"
                        ((failed++))
                        
                        if [ "$CONTINUE_ON_ERROR" = false ]; then
                            log_error "Stopping batch due to failure (use --continue-on-error to continue)"
                            exit 1
                        fi
                    fi
                    
                    unset pids[$i]
                    ((current_parallel--))
                fi
            done
            
            sleep 1
        done
        
        # Start new job
        log_step "Starting: $brand"
        if [ "$DRY_RUN" = true ]; then
            log_info "[DRY RUN] Would execute $operation for $brand"
            sleep 1  # Simulate processing time
        else
            execute_brand_operation "$brand" "$operation" &
            pids[$completed]=$!
            ((current_parallel++))
        fi
    done
    
    # Wait for remaining jobs
    for pid in "${pids[@]}"; do
        if [ -n "$pid" ]; then
            wait "$pid"
            exit_code=$?
            
            if [ $exit_code -eq 0 ]; then
                ((completed++))
            else
                ((failed++))
            fi
        fi
    done
    
    echo ""
    log_info "Batch operation completed: $completed successful, $failed failed"
    
    if [ $failed -gt 0 ] && [ "$CONTINUE_ON_ERROR" = false ]; then
        exit 1
    fi
}

# Generate batch report
generate_batch_report() {
    local brands=$1
    local operation=$2
    
    echo ""
    echo "📊 BATCH OPERATION REPORT"
    echo "═══════════════════════════"
    echo "Operation: $operation"
    echo "Timestamp: $(date)"
    echo "Total Brands: $(echo "$brands" | tr ',' '\n' | wc -l)"
    echo ""
    
    case $OUTPUT_FORMAT in
        "json")
            generate_json_report "$brands" "$operation"
            ;;
        "csv")
            generate_csv_report "$brands" "$operation"
            ;;
        *)
            generate_table_report "$brands" "$operation"
            ;;
    esac
}

# Generate table report
generate_table_report() {
    local brands=$1
    local operation=$2
    
    printf "%-25s %-15s %-20s %-30s\n" "BRAND" "STATUS" "HEALTH SCORE" "ALERTS"
    printf "%-25s %-15s %-20s %-30s\n" "-----" "------" "------------" "------"
    
    IFS=',' read -ra BRAND_ARRAY <<< "$brands"
    
    for brand in "${BRAND_ARRAY[@]}"; do
        # Get brand status
        health_score=$(python3 src/status/pipeline_status.py --brand "$brand" --health-only --json 2>/dev/null | jq -r '.overall_health' || echo "N/A")
        alerts_count=$(python3 src/status/pipeline_status.py --brand "$brand" --alerts-count 2>/dev/null || echo "0")
        
        # Determine status
        if [ "$health_score" != "N/A" ]; then
            if (( $(echo "$health_score >= 8.0" | bc -l) )); then
                status="🟢 HEALTHY"
            elif (( $(echo "$health_score >= 6.0" | bc -l) )); then
                status="🟡 WARNING"
            else
                status="🔴 CRITICAL"
            fi
        else
            status="❓ UNKNOWN"
        fi
        
        printf "%-25s %-15s %-20s %-30s\n" "$brand" "$status" "$health_score/10" "$alerts_count alert(s)"
    done
}

# Main execution
main() {
    echo "🚀 Batch Operations Starting..."
    echo "Operation: $OPERATION"
    echo "Timestamp: $(date)"
    echo ""
    
    setup_logging
    
    # Get brand list
    brands=$(get_brand_list)
    
    if [ -z "$brands" ]; then
        log_error "No brands to process"
        exit 1
    fi
    
    brand_count=$(echo "$brands" | tr ',' '\n' | wc -l)
    log_info "Processing $brand_count brand(s): $brands"
    
    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Execute batch operation
    case $OPERATION in
        "status-check"|"full-status"|"onboard-new"|"refresh-stale"|"smart-maintenance"|"weekly-maintenance"|"monthly-maintenance"|"emergency-check")
            execute_parallel_operations "$brands" "$OPERATION" "$MAX_PARALLEL"
            ;;
        "performance-report"|"cost-analysis"|"quality-audit")
            log_step "Generating $OPERATION for all brands..."
            # These operations need special handling
            python3 src/analytics/batch_analyzer.py --operation "$OPERATION" --brands "$brands"
            ;;
        *)
            log_error "Unknown batch operation: $OPERATION"
            exit 1
            ;;
    esac
    
    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    log_success "Batch operation completed in ${MINUTES}m ${SECONDS}s"
    
    # Generate report
    generate_batch_report "$brands" "$OPERATION"
    
    if [ "$SAVE_LOGS" = true ]; then
        log_info "Detailed logs saved to: $LOG_DIR"
    fi
}

# Run main function
main "$@" 