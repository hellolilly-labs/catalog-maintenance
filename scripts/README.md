# Brand Management Shell Scripts
## Complete Workflow Automation for Zero-to-AI-Agent Pipeline

This directory contains comprehensive shell scripts that wrap all CLI commands into easy-to-use workflows for managing your brand intelligence pipeline.

---

## 🚀 **Quick Start**

### **Master Control Script**
Use the main `brand_manager.sh` script for all operations:

```bash
# Complete brand onboarding
./scripts/brand_manager.sh onboard specialized.com

# Check brand status
./scripts/brand_manager.sh status specialized.com

# Show next recommended step
./scripts/brand_manager.sh next-step specialized.com

# Resume from where you left off
./scripts/brand_manager.sh resume specialized.com

# Smart refresh maintenance
./scripts/brand_manager.sh refresh specialized.com

# Live monitoring
./scripts/brand_manager.sh watch specialized.com

# Multi-brand dashboard
./scripts/brand_manager.sh dashboard
```

---

## 📋 **Available Scripts**

### **1. brand_manager.sh** - Master Control Script
**Purpose**: Single entry point for all brand management operations
**Features**: 
- Intuitive command interface
- Auto-discovery of brands
- Built-in help system
- Permission management

**Common Commands**:
```bash
# Onboarding
./scripts/brand_manager.sh onboard <brand>
./scripts/brand_manager.sh onboard-batch "brand1,brand2,brand3"

# Status & Monitoring
./scripts/brand_manager.sh status <brand>
./scripts/brand_manager.sh health <brand>
./scripts/brand_manager.sh alerts <brand>
./scripts/brand_manager.sh watch <brand>
./scripts/brand_manager.sh dashboard

# Maintenance
./scripts/brand_manager.sh refresh <brand>
./scripts/brand_manager.sh refresh-stale <brand>
./scripts/brand_manager.sh weekly <brand>
./scripts/brand_manager.sh monthly <brand>

# Batch Operations
./scripts/brand_manager.sh batch-status
./scripts/brand_manager.sh batch-refresh
./scripts/brand_manager.sh batch-alerts

# Workflow Management
./scripts/brand_manager.sh next-step <brand>
./scripts/brand_manager.sh resume <brand>
./scripts/brand_manager.sh workflow <brand>
./scripts/brand_manager.sh history <brand>

# Batch Operations
./scripts/brand_manager.sh batch-status
./scripts/brand_manager.sh batch-refresh
./scripts/brand_manager.sh batch-alerts
./scripts/brand_manager.sh batch-next-steps

# Utilities
./scripts/brand_manager.sh discover
./scripts/brand_manager.sh emergency
./scripts/brand_manager.sh logs
```

### **2. brand_onboarding.sh** - Complete Pipeline Setup
**Purpose**: Full zero-to-AI-agent onboarding for new brands
**Duration**: 15-25 minutes per brand
**Cost**: $8-15 per brand

**Features**:
- ✅ Complete brand intelligence research (all 7 phases)
- ✅ Product catalog ingestion with brand awareness
- ✅ Knowledge base creation with linearity analysis
- ✅ RAG optimization for brand-specific search
- ✅ AI persona generation with avatars
- ✅ Pipeline validation and health checks

**Usage Examples**:
```bash
# Basic onboarding
./scripts/brand_onboarding.sh --brand specialized.com

# Fast onboarding with parallel processing
./scripts/brand_onboarding.sh --brand nike.com --parallel

# Automated onboarding (no prompts)
./scripts/brand_onboarding.sh --brand adidas.com --skip-confirmation --parallel
```

### **3. brand_status.sh** - Comprehensive Monitoring
**Purpose**: Real-time pipeline status monitoring and alerts
**Features**:
- 📊 Complete pipeline health scoring
- 🔍 Detailed component status (research, catalog, knowledge, RAG, persona)
- 🚨 Alert management with actionable recommendations
- 📈 Performance metrics and trends
- 🔄 Live monitoring with auto-refresh

**Usage Examples**:
```bash
# Complete status overview
./scripts/brand_status.sh --brand specialized.com

# Specific component status
./scripts/brand_status.sh --brand nike.com --research
./scripts/brand_status.sh --brand adidas.com --catalog
./scripts/brand_status.sh --brand puma.com --alerts

# Live monitoring (updates every 30 seconds)
./scripts/brand_status.sh --brand specialized.com --watch

# Custom monitoring interval
./scripts/brand_status.sh --brand nike.com --watch --interval 10

# Batch status check
./scripts/brand_status.sh --batch "specialized.com,nike.com,adidas.com" --health
```

### **4. brand_maintenance.sh** - Regular Updates & Refresh
**Purpose**: Scheduled maintenance and intelligent refresh workflows
**Features**:
- 🧠 Smart refresh based on cache expiry and quality scores
- ⚡ Stale-only refresh to minimize costs
- 🔄 Full refresh for comprehensive updates
- 📅 Weekly/monthly maintenance routines
- 🎯 Specific phase targeting

**Maintenance Types**:
- **Smart Refresh** (Recommended): Analyzes status and refreshes optimally
- **Stale Refresh**: Only refreshes content past cache expiry
- **Full Refresh**: Forces complete regeneration (expensive)
- **Component Refresh**: Updates specific pipeline components
- **Scheduled Maintenance**: Weekly/monthly routines

**Usage Examples**:
```bash
# Smart refresh (recommended)
./scripts/brand_maintenance.sh --brand specialized.com --type smart-refresh

# Refresh only stale content
./scripts/brand_maintenance.sh --brand nike.com --type refresh-stale

# Force full refresh (expensive)
./scripts/brand_maintenance.sh --brand adidas.com --type refresh-all --force

# Specific research phases
./scripts/brand_maintenance.sh --brand puma.com --phases "product_style,voice_messaging"

# Scheduled maintenance
./scripts/brand_maintenance.sh --brand specialized.com --type weekly-maintenance
./scripts/brand_maintenance.sh --brand nike.com --type monthly-maintenance

# Dry run (see what would be done)
./scripts/brand_maintenance.sh --brand adidas.com --type smart-refresh --dry-run
```

### **5. batch_operations.sh** - Multi-Brand Processing
**Purpose**: Efficient processing of multiple brands simultaneously
**Features**:
- 🔄 Parallel processing with configurable limits
- 📊 Comprehensive reporting (table, JSON, CSV formats)
- 🚨 Error handling with continue-on-error option
- 📝 Detailed logging for audit trails
- 🎯 Brand discovery and filtering

**Batch Operations**:
- **Status Checks**: Health monitoring across all brands
- **Maintenance**: Smart refresh and scheduled maintenance
- **Onboarding**: Bulk brand setup
- **Analytics**: Performance and cost analysis
- **Emergency**: Critical alert monitoring

**Usage Examples**:
```bash
# Status check all brands
./scripts/batch_operations.sh --operation status-check --discover

# Onboard multiple brands
./scripts/batch_operations.sh --operation onboard-new --brands "nike.com,adidas.com,puma.com"

# Smart maintenance for all brands
./scripts/batch_operations.sh --operation smart-maintenance --discover --parallel 5

# From file with brand list
./scripts/batch_operations.sh --operation refresh-stale --file brands.txt

# Emergency alert check
./scripts/batch_operations.sh --operation emergency-check --discover --continue-on-error

# Performance report generation
./scripts/batch_operations.sh --operation performance-report --discover --output-format json
```

---

## 🔄 **Workflow State Management System**

### **Intelligent "Next Step" Tracking**
The system automatically tracks where each brand is in the pipeline and suggests the next logical step. This enables you to:

- **Resume from any point**: Pick up exactly where you left off after interruptions
- **Smart recommendations**: Get AI-powered suggestions for optimal next actions
- **Progress tracking**: See detailed completion percentages and time estimates
- **Error recovery**: Gracefully handle failures and retry from the right point
- **API-ready**: Future API integration with complete state management

### **Workflow States**
```
🔄 NOT_STARTED           → Brand not onboarded yet
🔄 RESEARCH_IN_PROGRESS   → Brand research underway  
✅ RESEARCH_COMPLETE      → Brand research finished
🔄 CATALOG_IN_PROGRESS    → Product catalog ingestion underway
✅ CATALOG_COMPLETE       → Product catalog finished
🔄 KNOWLEDGE_IN_PROGRESS  → Knowledge base creation underway
✅ KNOWLEDGE_COMPLETE     → Knowledge base finished
🔄 RAG_IN_PROGRESS        → RAG optimization underway
✅ RAG_COMPLETE           → RAG optimization finished
🔄 PERSONA_IN_PROGRESS    → AI persona generation underway
✅ PERSONA_COMPLETE       → AI persona finished
🎉 PIPELINE_COMPLETE      → Full pipeline operational
🔧 MAINTENANCE_MODE       → Regular maintenance operations
❌ ERROR_STATE           → Needs intervention
⚠️ PARTIAL_FAILURE       → Some components failed
```

### **Next Step Commands**
```bash
# See what to do next
./scripts/brand_manager.sh next-step specialized.com

# Execute next step automatically
./scripts/brand_manager.sh resume specialized.com

# Detailed workflow status
./scripts/brand_manager.sh workflow specialized.com

# Show step history
./scripts/brand_manager.sh history specialized.com

# Next steps for all brands
./scripts/brand_manager.sh batch-next-steps
```

### **Example Workflow Output**
```
🔄 WORKFLOW STATUS: specialized.com
Current State: catalog_complete
Progress: 60.0%
Last Updated: 2024-12-20 14:30:00

📋 NEXT STEP:
Action: Create knowledge base
Command: python src/knowledge_ingestor.py --brand specialized.com --include-brand-intelligence
Priority: high
Duration: 5-10 minutes
Cost: $1-3
Reason: Product catalog is ready, create knowledge base

💡 To execute the next step automatically:
  ./scripts/brand_manager.sh resume specialized.com
```

### **Resume Capability**
Perfect for interrupted workflows:
```bash
# Start onboarding (gets interrupted after research phase)
./scripts/brand_manager.sh onboard newbrand.com

# Later, check where you left off
./scripts/brand_manager.sh next-step newbrand.com

# Resume exactly where you stopped
./scripts/brand_manager.sh resume newbrand.com
```

### **API-Ready State Management**
The workflow system stores complete state in JSON format, making it perfect for future API integration:

```json
{
  "brand_url": "specialized.com",
  "current_state": "catalog_complete", 
  "total_progress_percent": 60.0,
  "next_step": {
    "action": "Create knowledge base",
    "command": "python src/knowledge_ingestor.py --brand specialized.com --include-brand-intelligence",
    "priority": "high",
    "estimated_duration": "5-10 minutes",
    "estimated_cost": "$1-3"
  },
  "step_history": [...],
  "errors": [...]
}
```

---

## 📊 **Status Tracking System**

### **Health Scores (0-10 Scale)**
- **🟢 8.0-10.0**: Excellent - All systems healthy
- **🟡 6.0-7.9**: Warning - Some attention needed
- **🔴 0.0-5.9**: Critical - Immediate action required

### **Component Status Types**
- **✅ COMPLETE**: Fresh and high quality
- **⚠️ STALE**: Needs refresh due to cache expiry
- **❌ MISSING**: Not generated yet
- **🔄 IN_PROGRESS**: Currently being generated
- **💥 ERROR**: Failed generation (needs intervention)
- **⏳ DEPENDENCY_WAIT**: Waiting for dependencies

### **Alert System**
Scripts automatically generate actionable alerts with specific commands to resolve issues:

```bash
# Example alert output
🚨 ALERTS for specialized.com:
  ⚠️  Phase 'product_style' needs refresh
      Action: ./scripts/brand_maintenance.sh --brand specialized.com --phases product_style
  
  ❌ 47 products have stale descriptors  
      Action: python src/product_ingestor.py --refresh-stale-descriptors
```

---

## 🔧 **Workflow Examples**

### **Daily Operations**
```bash
# Quick health check
./scripts/brand_manager.sh health specialized.com

# Check for alerts
./scripts/brand_manager.sh alerts specialized.com

# See what needs to be done next
./scripts/brand_manager.sh batch-next-steps

# View multi-brand dashboard
./scripts/brand_manager.sh dashboard
```

### **Weekly Maintenance**
```bash
# Smart refresh for key brands
./scripts/brand_manager.sh refresh specialized.com
./scripts/brand_manager.sh refresh nike.com

# Batch weekly maintenance
./scripts/batch_operations.sh --operation weekly-maintenance --discover
```

### **Monthly Deep Maintenance**
```bash
# Comprehensive monthly maintenance
./scripts/brand_manager.sh monthly specialized.com

# Quality audit across all brands
./scripts/batch_operations.sh --operation quality-audit --discover

# Performance report generation
./scripts/batch_operations.sh --operation performance-report --discover
```

### **New Brand Onboarding**
```bash
# Single brand onboarding
./scripts/brand_manager.sh onboard newbrand.com

# Batch onboarding
./scripts/brand_manager.sh onboard-batch "brand1.com,brand2.com,brand3.com"
```

### **Interrupted Workflow Recovery**
```bash
# Check where you left off
./scripts/brand_manager.sh next-step interrupted-brand.com

# Resume from the exact point of interruption
./scripts/brand_manager.sh resume interrupted-brand.com

# View complete workflow history
./scripts/brand_manager.sh history interrupted-brand.com

# Check progress of partial onboarding
./scripts/brand_manager.sh workflow interrupted-brand.com
```

### **Emergency Response**
```bash
# Emergency health check across all brands
./scripts/brand_manager.sh emergency

# Check critical alerts
./scripts/brand_manager.sh batch-alerts

# Detailed troubleshooting
./scripts/brand_manager.sh status problematic-brand.com
```

---

## 📈 **Performance & Cost Optimization**

### **Cost-Efficient Refresh Strategies**
1. **Smart Refresh** ($1-4): Analyzes need and refreshes optimally
2. **Stale Refresh** ($2-6): Only refreshes expired content  
3. **Selective Phases** ($1-3): Target specific research phases
4. **Scheduled Maintenance** ($1-5): Regular low-cost updates

### **Parallel Processing**
- Default: 3 parallel operations
- Configurable: `--parallel 5` for faster processing
- Monitoring: Real-time progress tracking
- Error Handling: Continue-on-error options

### **Monitoring & Alerts**
- **Real-time**: Live status monitoring with auto-refresh
- **Batch**: Multi-brand health dashboards
- **Historical**: Trend analysis and performance tracking
- **Proactive**: Alert-driven maintenance recommendations

---

## 🚀 **Getting Started**

### **1. First Time Setup**
```bash
# Make scripts executable (automatic)
chmod +x scripts/*.sh

# Discover existing brands
./scripts/brand_manager.sh discover

# Check system health
./scripts/brand_manager.sh dashboard
```

### **2. Onboard Your First Brand**
```bash
# Complete onboarding
./scripts/brand_manager.sh onboard your-brand.com

# Monitor progress
./scripts/brand_manager.sh watch your-brand.com
```

### **3. Set Up Regular Maintenance**
```bash
# Weekly smart refresh
./scripts/brand_manager.sh refresh your-brand.com

# Check for issues
./scripts/brand_manager.sh alerts your-brand.com
```

---

## 💡 **Pro Tips**

### **For Single Brands**
- Use `./scripts/brand_manager.sh` for all operations
- Monitor with `--watch` during onboarding
- Use `refresh` (smart) for regular maintenance
- Check `alerts` before major operations

### **For Multiple Brands**
- Use `batch-*` commands for efficiency  
- Set `--parallel 5` for faster processing
- Use `--continue-on-error` for robustness
- Save logs with `--output-format json`

### **For Production**
- Schedule weekly `smart-refresh` maintenance
- Run monthly `quality-audit` reviews
- Monitor `dashboard` daily
- Set up `emergency` alert monitoring

### **For Development**
- Use `--dry-run` to test changes
- Use `--verbose` for debugging
- Check `logs` for detailed analysis
- Use specific component flags for focused testing

### **For Workflow Management**
- Use `next-step` to understand pipeline state
- Use `resume` for hands-off automation
- Use `workflow` for detailed progress tracking
- Use `history` for debugging failed steps
- Use `batch-next-steps` for multi-brand oversight

---

This workflow automation system transforms complex CLI operations into simple, reliable commands that can be used by anyone on your team! 🎯 