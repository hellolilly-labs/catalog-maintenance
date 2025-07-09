#!/bin/bash

# ============================================
# New Brand Onboarding Script
# Complete Zero-to-AI-Agent Pipeline
# ============================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BRAND_URL=""
SKIP_CONFIRMATION=false
PARALLEL_PROCESSING=false

# Help function
show_help() {
    echo "Brand Onboarding Script - Complete Zero-to-AI-Agent Pipeline"
    echo ""
    echo "Usage: $0 --brand <brand_url> [options]"
    echo ""
    echo "Required:"
    echo "  --brand <url>           Brand URL to onboard (e.g., specialized.com)"
    echo ""
    echo "Options:"
    echo "  --skip-confirmation     Skip confirmation prompts"
    echo "  --parallel              Run some steps in parallel (faster)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --brand specialized.com"
    echo "  $0 --brand nike.com --parallel --skip-confirmation"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --brand)
            BRAND_URL="$2"
            shift 2
            ;;
        --skip-confirmation)
            SKIP_CONFIRMATION=true
            shift
            ;;
        --parallel)
            PARALLEL_PROCESSING=true
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

# Logging function
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

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check if Python environment is ready
    if ! python --version >/dev/null 2>&1; then
        log_error "Python is not available"
        exit 1
    fi
    
    # Check if required environment variables are set
    if [ -z "$OPENAI_API_KEY" ] || [ -z "$ANTHROPIC_API_KEY" ]; then
        log_error "Required API keys not set in environment"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Show pipeline overview
show_pipeline_overview() {
    echo ""
    echo "🚀 BRAND ONBOARDING PIPELINE"
    echo "═══════════════════════════════"
    echo "Brand: $BRAND_URL"
    echo ""
    echo "Pipeline Steps:"
    echo "1. 🔬 Brand Intelligence Research (15-20 min)"
    echo "2. 📦 Product Catalog Ingestion"
    echo "3. 📚 Knowledge Base Creation"
    echo "4. 🔍 RAG Query Optimization"
    echo "5. 🤖 AI Sales Agent Persona Generation"
    echo "6. 🎨 Avatar Generation"
    echo "7. ✅ Pipeline Validation"
    echo ""
    
    if [ "$SKIP_CONFIRMATION" = false ]; then
        read -p "Continue with onboarding? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Onboarding cancelled."
            exit 0
        fi
    fi
}

# Step 1: Brand Intelligence Research
run_brand_research() {
    log_step "Step 1: Starting Brand Intelligence Research..."
    
    # Update workflow state
    python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state research_in_progress --step-completed "started_research" 2>/dev/null || true
    
    # Run full brand research with all phases
    python3 src/research/brand_researcher.py \
        --brand "$BRAND_URL" \
        --full-research \
        --verbose
    
    if [ $? -eq 0 ]; then
        log_success "Brand research completed"
        # Update workflow state
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state research_complete --step-completed "brand_research" 2>/dev/null || true
    else
        log_error "Brand research failed"
        # Update workflow state with error
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state error_state --error "Brand research failed" 2>/dev/null || true
        exit 1
    fi
}

# Step 2: Product Catalog Ingestion
run_catalog_ingestion() {
    log_step "Step 2: Starting Product Catalog Ingestion..."
    
    # Update workflow state
    python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state catalog_in_progress --step-completed "started_catalog" 2>/dev/null || true
    
    # Full sync with brand-aware processing
    python3 src/product_ingestor.py \
        --brand "$BRAND_URL" \
        --full-sync \
        --use-brand-intelligence \
        --verbose
    
    if [ $? -eq 0 ]; then
        log_success "Product catalog ingestion completed"
        # Update workflow state
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state catalog_complete --step-completed "catalog_ingestion" 2>/dev/null || true
    else
        log_error "Product catalog ingestion failed"
        # Update workflow state with error
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state error_state --error "Catalog ingestion failed" 2>/dev/null || true
        exit 1
    fi
}

# Step 3: Knowledge Base Creation
run_knowledge_ingestion() {
    log_step "Step 3: Starting Knowledge Base Creation..."
    
    # Update workflow state
    python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state knowledge_in_progress --step-completed "started_knowledge" 2>/dev/null || true
    
    # Create knowledge base with brand intelligence integration
    python3 src/knowledge_ingestor.py \
        --brand "$BRAND_URL" \
        --include-brand-intelligence \
        --linearity-aware \
        --verbose
    
    if [ $? -eq 0 ]; then
        log_success "Knowledge base creation completed"
        # Update workflow state
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state knowledge_complete --step-completed "knowledge_base" 2>/dev/null || true
    else
        log_error "Knowledge base creation failed"
        # Update workflow state with error
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state error_state --error "Knowledge base creation failed" 2>/dev/null || true
        exit 1
    fi
}

# Step 4: RAG Optimization
run_rag_optimization() {
    log_step "Step 4: Starting RAG Query Optimization..."
    
    # Update workflow state
    python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state rag_in_progress --step-completed "started_rag" 2>/dev/null || true
    
    # Generate brand-specific query transformations
    python3 src/research/brand_researcher.py \
        --brand "$BRAND_URL" \
        --phases rag_optimization \
        --verbose
    
    if [ $? -eq 0 ]; then
        log_success "RAG optimization completed"
        # Update workflow state
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state rag_complete --step-completed "rag_optimization" 2>/dev/null || true
    else
        log_warning "RAG optimization failed (continuing with standard search)"
        # Don't fail the pipeline for RAG optimization failure
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state rag_complete --step-completed "rag_optimization_skipped" 2>/dev/null || true
    fi
}

# Step 5: AI Persona Generation
run_persona_generation() {
    log_step "Step 5: Starting AI Sales Agent Persona Generation..."
    
    # Update workflow state
    python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state persona_in_progress --step-completed "started_persona" 2>/dev/null || true
    
    # Generate AI personas with avatars
    python3 src/research/brand_researcher.py \
        --brand "$BRAND_URL" \
        --phases ai_persona_generation \
        --verbose
    
    if [ $? -eq 0 ]; then
        # Check if live persona needs initialization
        python3 src/persona/persona_manager.py \
            --brand "$BRAND_URL" \
            --initialize-if-missing \
            --verbose
        
        log_success "AI persona generation completed"
        # Update workflow state
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state persona_complete --step-completed "persona_generation" 2>/dev/null || true
    else
        log_error "AI persona generation failed"
        # Update workflow state with error
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state error_state --error "Persona generation failed" 2>/dev/null || true
        exit 1
    fi
}

# Step 6: Pipeline Validation
validate_pipeline() {
    log_step "Step 6: Validating Complete Pipeline..."
    
    # Run comprehensive status check
    python3 src/status/pipeline_status.py \
        --brand "$BRAND_URL" \
        --validate-complete
    
    if [ $? -eq 0 ]; then
        log_success "Pipeline validation passed"
        # Update workflow state to complete
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state pipeline_complete --step-completed "pipeline_validation" 2>/dev/null || true
    else
        log_error "Pipeline validation failed"
        # Update workflow state with error
        python3 src/workflow/workflow_state_manager.py --brand "$BRAND_URL" --action update --new-state partial_failure --error "Pipeline validation failed" 2>/dev/null || true
        exit 1
    fi
}

# Parallel processing option
run_parallel_steps() {
    log_step "Running steps 2-3 in parallel..."
    
    # Run catalog and knowledge ingestion in parallel
    python3 src/product_ingestor.py --brand "$BRAND_URL" --full-sync --use-brand-intelligence &
    CATALOG_PID=$!
    
    python3 src/knowledge_ingestor.py --brand "$BRAND_URL" --include-brand-intelligence &
    KNOWLEDGE_PID=$!
    
    # Wait for both to complete
    wait $CATALOG_PID
    CATALOG_STATUS=$?
    
    wait $KNOWLEDGE_PID
    KNOWLEDGE_STATUS=$?
    
    if [ $CATALOG_STATUS -eq 0 ] && [ $KNOWLEDGE_STATUS -eq 0 ]; then
        log_success "Parallel processing completed"
    else
        log_error "Parallel processing failed"
        exit 1
    fi
}

# Show final status
show_final_status() {
    echo ""
    echo "🎉 BRAND ONBOARDING COMPLETE!"
    echo "══════════════════════════════"
    
    # Show final pipeline status
    python3 src/status/pipeline_status.py --brand "$BRAND_URL"
    
    echo ""
    echo "Next Steps:"
    echo "• Check AI persona: python3 src/persona/persona_manager.py --brand $BRAND_URL --status"
    echo "• Test RAG search: python3 src/rag/test_search.py --brand $BRAND_URL --query 'test query'"
    echo "• Deploy AI agent: [Deploy to your AI agent platform]"
    echo ""
}

# Main execution
main() {
    echo "🚀 Brand Onboarding Pipeline Starting..."
    echo "Brand: $BRAND_URL"
    echo "Timestamp: $(date)"
    echo ""
    
    check_prerequisites
    show_pipeline_overview
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Execute pipeline steps
    run_brand_research
    
    if [ "$PARALLEL_PROCESSING" = true ]; then
        run_parallel_steps
    else
        run_catalog_ingestion
        run_knowledge_ingestion
    fi
    
    run_rag_optimization
    run_persona_generation
    validate_pipeline
    
    # Calculate total time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    log_success "Pipeline completed in ${MINUTES}m ${SECONDS}s"
    
    show_final_status
}

# Run main function
main "$@" 