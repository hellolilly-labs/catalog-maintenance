#!/bin/bash

# ============================================
# Brand Manager - Master Control Script
# Easy Access to All Brand Management Functions
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Help function
show_help() {
    echo "🏢 Brand Manager - Master Control Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "📋 Available Commands:"
    echo ""
    echo "🚀 ONBOARDING & SETUP:"
    echo "  onboard <brand>              Complete zero-to-AI-agent onboarding"
    echo "  onboard-batch <brands>       Onboard multiple brands"
    echo ""
    echo "📊 STATUS & MONITORING:"
    echo "  status <brand>               Complete pipeline status"
    echo "  health <brand>               Quick health check"
    echo "  alerts <brand>               Show alerts and issues"
    echo "  watch <brand>                Live status monitoring"
    echo "  dashboard                    Multi-brand dashboard"
    echo ""
    echo "🔧 MAINTENANCE & UPDATES:"
    echo "  refresh <brand>              Smart refresh (recommended)"
    echo "  refresh-stale <brand>        Refresh only stale content"
    echo "  refresh-all <brand>          Force refresh everything"
    echo "  weekly <brand>               Weekly maintenance"
    echo "  monthly <brand>              Monthly maintenance"
    echo ""
    echo "🔄 WORKFLOW MANAGEMENT:"
    echo "  next-step <brand>            Show next recommended step"
    echo "  resume <brand>               Execute next step automatically"
    echo "  workflow <brand>             Show detailed workflow status"
    echo "  history <brand>              Show workflow step history"
    echo ""
    echo "🔍 BATCH OPERATIONS:"
    echo "  batch-status                 Check status of all brands"
    echo "  batch-refresh                Smart refresh all brands"
    echo "  batch-alerts                 Check alerts across all brands"
    echo "  batch-next-steps             Show next steps for all brands"
    echo ""
    echo "💡 QUICK ACTIONS:"
    echo "  discover                     Find all brands in system"
    echo "  emergency                    Emergency health check"
    echo "  logs                         View recent batch logs"
    echo ""
    echo "Examples:"
    echo "  $0 onboard specialized.com"
    echo "  $0 status nike.com"
    echo "  $0 next-step specialized.com"
    echo "  $0 resume specialized.com"
    echo "  $0 batch-next-steps"
    echo "  $0 dashboard"
    echo ""
    echo "For detailed help on specific commands, use:"
    echo "  $0 <command> --help"
}

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

# Check if scripts are executable
check_script_permissions() {
    local scripts=("brand_onboarding.sh" "brand_status.sh" "brand_maintenance.sh" "batch_operations.sh")
    
    for script in "${scripts[@]}"; do
        if [ ! -x "$SCRIPT_DIR/$script" ]; then
            log_warning "Making $script executable..."
            chmod +x "$SCRIPT_DIR/$script"
        fi
    done
}

# Parse command and route to appropriate script
main() {
    # Check script permissions
    check_script_permissions
    
    # Handle no arguments
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    local command=$1
    shift  # Remove command from arguments
    
    case $command in
        # Onboarding commands
        "onboard")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for onboarding"
                echo "Usage: $0 onboard <brand_url>"
                exit 1
            fi
            log_info "Starting brand onboarding for: $1"
            "$SCRIPT_DIR/brand_onboarding.sh" --brand "$1" "${@:2}"
            ;;
        
        "onboard-batch")
            if [ $# -eq 0 ]; then
                log_error "Brand list required for batch onboarding"
                echo "Usage: $0 onboard-batch <brand1,brand2,brand3>"
                exit 1
            fi
            log_info "Starting batch onboarding for: $1"
            "$SCRIPT_DIR/batch_operations.sh" --operation onboard-new --brands "$1" "${@:2}"
            ;;
        
        # Status commands
        "status")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for status check"
                echo "Usage: $0 status <brand_url>"
                exit 1
            fi
            "$SCRIPT_DIR/brand_status.sh" --brand "$1" --overview "${@:2}"
            ;;
        
        "health")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for health check"
                echo "Usage: $0 health <brand_url>"
                exit 1
            fi
            "$SCRIPT_DIR/brand_status.sh" --brand "$1" --health "${@:2}"
            ;;
        
        "alerts")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for alerts check"
                echo "Usage: $0 alerts <brand_url>"
                exit 1
            fi
            "$SCRIPT_DIR/brand_status.sh" --brand "$1" --alerts "${@:2}"
            ;;
        
        "watch")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for watch mode"
                echo "Usage: $0 watch <brand_url>"
                exit 1
            fi
            log_info "Starting live monitoring for: $1"
            "$SCRIPT_DIR/brand_status.sh" --brand "$1" --overview --watch "${@:2}"
            ;;
        
        "dashboard")
            log_info "Opening multi-brand dashboard..."
            "$SCRIPT_DIR/batch_operations.sh" --operation status-check --discover --output-format table "$@"
            ;;
        
        # Maintenance commands
        "refresh")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for refresh"
                echo "Usage: $0 refresh <brand_url>"
                exit 1
            fi
            log_info "Starting smart refresh for: $1"
            "$SCRIPT_DIR/brand_maintenance.sh" --brand "$1" --type smart-refresh "${@:2}"
            ;;
        
        "refresh-stale")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for stale refresh"
                echo "Usage: $0 refresh-stale <brand_url>"
                exit 1
            fi
            log_info "Refreshing stale content for: $1"
            "$SCRIPT_DIR/brand_maintenance.sh" --brand "$1" --type refresh-stale "${@:2}"
            ;;
        
        "refresh-all")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for full refresh"
                echo "Usage: $0 refresh-all <brand_url>"
                exit 1
            fi
            log_warning "Starting FULL refresh for: $1 (this may be expensive)"
            "$SCRIPT_DIR/brand_maintenance.sh" --brand "$1" --type refresh-all "${@:2}"
            ;;
        
        "weekly")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for weekly maintenance"
                echo "Usage: $0 weekly <brand_url>"
                exit 1
            fi
            log_info "Running weekly maintenance for: $1"
            "$SCRIPT_DIR/brand_maintenance.sh" --brand "$1" --type weekly-maintenance "${@:2}"
            ;;
        
        "monthly")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for monthly maintenance"
                echo "Usage: $0 monthly <brand_url>"
                exit 1
            fi
            log_info "Running monthly maintenance for: $1"
            "$SCRIPT_DIR/brand_maintenance.sh" --brand "$1" --type monthly-maintenance "${@:2}"
            ;;
        
        # Workflow management commands
        "next-step")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for next-step"
                echo "Usage: $0 next-step <brand_url>"
                exit 1
            fi
            log_info "Getting next step for: $1"
            python3 src/workflow/workflow_state_manager.py --brand "$1" --action status
            echo ""
            echo "💡 To execute the next step automatically:"
            echo "  $0 resume $1"
            ;;
        
        "resume")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for resume"
                echo "Usage: $0 resume <brand_url>"
                exit 1
            fi
            log_info "Resuming workflow for: $1"
            
            # Get the next step command
            NEXT_COMMAND=$(python3 src/workflow/workflow_state_manager.py --brand "$1" --action next-step)
            
            if [ -n "$NEXT_COMMAND" ]; then
                echo "🔄 Executing next step: $NEXT_COMMAND"
                echo ""
                
                # Execute the command
                eval "$NEXT_COMMAND"
                
                # Update workflow state based on success/failure
                if [ $? -eq 0 ]; then
                    log_success "Step completed successfully for $1"
                    python3 src/workflow/workflow_state_manager.py --brand "$1" --action update --step-completed "resume_step"
                else
                    log_error "Step failed for $1"
                    python3 src/workflow/workflow_state_manager.py --brand "$1" --action update --error "Resume step failed"
                fi
            else
                log_warning "No next step available for $1"
            fi
            ;;
        
        "workflow")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for workflow status"
                echo "Usage: $0 workflow <brand_url>"
                exit 1
            fi
            python3 src/workflow/workflow_state_manager.py --brand "$1" --action status "${@:2}"
            ;;
        
        "history")
            if [ $# -eq 0 ]; then
                log_error "Brand URL required for workflow history"
                echo "Usage: $0 history <brand_url>"
                exit 1
            fi
            python3 src/workflow/workflow_state_manager.py --brand "$1" --action history "${@:2}"
            ;;
        
        # Batch operations
        "batch-status")
            log_info "Checking status across all brands..."
            "$SCRIPT_DIR/batch_operations.sh" --operation status-check --discover "$@"
            ;;
        
        "batch-refresh")
            log_info "Smart refreshing all brands..."
            "$SCRIPT_DIR/batch_operations.sh" --operation smart-maintenance --discover "$@"
            ;;
        
        "batch-alerts")
            log_info "Checking alerts across all brands..."
            "$SCRIPT_DIR/batch_operations.sh" --operation emergency-check --discover "$@"
            ;;
        
        # Update existing batch-status command to include next steps
        "batch-next-steps")
            log_info "Checking next steps for all brands..."
            python3 -c "
import asyncio
import sys
sys.path.append('src')
from workflow.workflow_state_manager import WorkflowStateManager

async def show_all_next_steps():
    manager = WorkflowStateManager()
    brand_states = await manager.list_all_brand_states()
    
    if not brand_states:
        print('No brands found in system.')
        return
    
    print('\\n📋 NEXT STEPS FOR ALL BRANDS')
    print('═' * 50)
    
    for brand_url, progress in brand_states.items():
        print(f'\\n🏢 {brand_url}')
        print(f'   State: {progress.current_state.value} ({progress.total_progress_percent:.1f}% complete)')
        
        if progress.next_step:
            print(f'   Next: {progress.next_step.action}')
            print(f'   Command: {progress.next_step.command}')
            print(f'   Priority: {progress.next_step.priority.value} | Duration: {progress.next_step.estimated_duration} | Cost: {progress.next_step.estimated_cost}')
        else:
            print('   Next: No action needed')

asyncio.run(show_all_next_steps())
"
            ;;
        
        # Utility commands
        "discover")
            log_info "Discovering brands in system..."
            python3 -c "
import os
from pathlib import Path

accounts_dir = Path('accounts')
if accounts_dir.exists():
    brands = [d.name for d in accounts_dir.iterdir() if d.is_dir()]
    if brands:
        print(f'Found {len(brands)} brands:')
        for brand in sorted(brands):
            print(f'  • {brand}')
    else:
        print('No brands found in system.')
else:
    print('No accounts directory found.')
"
            ;;
        
        "emergency")
            log_warning "Running emergency health check across all brands..."
            "$SCRIPT_DIR/batch_operations.sh" --operation emergency-check --discover --continue-on-error "$@"
            ;;
        
        "logs")
            log_info "Recent batch operation logs:"
            if [ -d "logs/batch_operations" ]; then
                ls -lt logs/batch_operations/ | head -10
            else
                echo "No batch logs found."
            fi
            ;;
        
        # Help and unknown commands
        "help"|"--help"|"-h")
            show_help
            ;;
        
        *)
            log_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Welcome message
echo "🏢 Brand Manager v1.0"
echo "═══════════════════════"

# Run main function
main "$@" 