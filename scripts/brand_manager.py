#!/usr/bin/env python3

"""
Brand Manager CLI
Provides workflow management commands for the catalog maintenance system
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow.workflow_state_manager import get_workflow_manager, WorkflowState

def format_time_ago(dt: datetime) -> str:
    """Format datetime as time ago string"""
    now = datetime.now()
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hours ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minutes ago"
    else:
        return "Just now"

def print_brand_status(brand_domain: str, workflow_manager, detailed: bool = False):
    """Print status for a single brand"""
    workflow_info = workflow_manager.get_brand_info(brand_domain)
    next_step = workflow_manager.get_next_step(brand_domain)
    
    # Status icon based on state
    status_icons = {
        WorkflowState.NOT_STARTED: "⚪",
        WorkflowState.RESEARCH_IN_PROGRESS: "🔄",
        WorkflowState.RESEARCH_COMPLETE: "✅",
        WorkflowState.CATALOG_IN_PROGRESS: "🔄",
        WorkflowState.CATALOG_COMPLETE: "✅",
        WorkflowState.KNOWLEDGE_IN_PROGRESS: "🔄",
        WorkflowState.KNOWLEDGE_COMPLETE: "✅",
        WorkflowState.RAG_IN_PROGRESS: "🔄",
        WorkflowState.RAG_COMPLETE: "✅",
        WorkflowState.PERSONA_IN_PROGRESS: "🔄",
        WorkflowState.PERSONA_COMPLETE: "✅",
        WorkflowState.PIPELINE_COMPLETE: "🎉",
        WorkflowState.FAILED: "❌",
        WorkflowState.MAINTENANCE_REQUIRED: "🔧"
    }
    
    icon = status_icons.get(workflow_info.current_state, "❓")
    
    print(f"\n{icon} {brand_domain}")
    print(f"   State: {workflow_info.current_state.value}")
    print(f"   Updated: {format_time_ago(workflow_info.last_updated)}")
    
    if workflow_info.current_phase:
        print(f"   Current Phase: {workflow_info.current_phase}")
    
    if detailed:
        print(f"   Completed Phases: {len(workflow_info.completed_phases)}")
        if workflow_info.failed_phases:
            print(f"   Failed Phases: {workflow_info.failed_phases}")
        if workflow_info.total_research_time > 0:
            print(f"   Research Time: {workflow_info.total_research_time:.1f} minutes")
        if workflow_info.quality_scores:
            avg_quality = sum(workflow_info.quality_scores.values()) / len(workflow_info.quality_scores)
            print(f"   Avg Quality Score: {avg_quality:.2f}")
    
    print(f"   Next Step: {next_step}")

def cmd_list_brands(args):
    """List all tracked brands with their workflow states"""
    workflow_manager = get_workflow_manager()
    brands = workflow_manager.get_all_brands()
    
    if not brands:
        print("📭 No brands found in the system.")
        print("\n💡 Start by running: python src/research/brand_researcher.py --brand [domain] --foundation")
        return
    
    print(f"\n📊 Brand Workflow Status ({len(brands)} brands)")
    print("=" * 60)
    
    # Group by state for better organization
    states_to_show = [
        WorkflowState.FAILED,
        WorkflowState.MAINTENANCE_REQUIRED,
        WorkflowState.RESEARCH_IN_PROGRESS,
        WorkflowState.CATALOG_IN_PROGRESS,
        WorkflowState.KNOWLEDGE_IN_PROGRESS,
        WorkflowState.RAG_IN_PROGRESS,
        WorkflowState.PERSONA_IN_PROGRESS,
        WorkflowState.NOT_STARTED,
        WorkflowState.RESEARCH_COMPLETE,
        WorkflowState.CATALOG_COMPLETE,
        WorkflowState.KNOWLEDGE_COMPLETE,
        WorkflowState.RAG_COMPLETE,
        WorkflowState.PERSONA_COMPLETE,
        WorkflowState.PIPELINE_COMPLETE,
    ]
    
    for state in states_to_show:
        brands_in_state = workflow_manager.get_brands_by_state(state)
        if brands_in_state:
            for brand in sorted(brands_in_state):
                print_brand_status(brand, workflow_manager, detailed=args.detailed)

def cmd_brand_status(args):
    """Show detailed status for a specific brand"""
    workflow_manager = get_workflow_manager()
    brand_domain = args.brand
    
    print(f"\n🔍 DETAILED STATUS: {brand_domain}")
    print("=" * 60)
    
    print_brand_status(brand_domain, workflow_manager, detailed=True)
    
    # Show maintenance check
    maintenance_info = workflow_manager.check_maintenance_needed(brand_domain)
    if maintenance_info['needs_maintenance']:
        print(f"\n🔧 MAINTENANCE REQUIRED")
        print(f"   Stale Phases: {maintenance_info['stale_phases']}")
        print(f"   Days Since Research: {maintenance_info['days_since_research']}")

def cmd_next_step(args):
    """Get next step command for a brand"""
    workflow_manager = get_workflow_manager()
    next_step = workflow_manager.get_next_step(args.brand)
    
    print(f"📋 Next Step for {args.brand}:")
    print(f"   {next_step}")

def cmd_workflow_summary(args):
    """Show summary of all workflows"""
    workflow_manager = get_workflow_manager()
    summary = workflow_manager.get_workflow_summary()
    
    print(f"\n📊 WORKFLOW SUMMARY")
    print("=" * 40)
    print(f"Total Brands: {summary['total_brands']}")
    print(f"Completed: {summary['completed_brands']} ({summary['completion_rate']:.1%})")
    print(f"Maintenance Needed: {summary['brands_needing_maintenance']}")
    
    print(f"\n📈 STATE DISTRIBUTION:")
    for state, count in summary['state_distribution'].items():
        if count > 0:
            print(f"   {state}: {count}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Brand Manager - Workflow management for catalog maintenance system"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all brands and their workflow states')
    list_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show detailed status for a brand')
    status_parser.add_argument('--brand', required=True, help='Brand domain (e.g., specialized.com)')
    
    # Next step command
    next_parser = subparsers.add_parser('next', help='Get next step for a brand')
    next_parser.add_argument('--brand', required=True, help='Brand domain')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show workflow summary for all brands')
    
    # Research progress command
    research_parser = subparsers.add_parser('research', help='Show detailed research progress for a brand')
    research_parser.add_argument('--brand', required=True, help='Brand domain')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            cmd_list_brands(args)
        elif args.command == 'status':
            cmd_brand_status(args)
        elif args.command == 'next':
            cmd_next_step(args)
        elif args.command == 'summary':
            cmd_workflow_summary(args)
        elif args.command == 'research':
            # Implementation for research command
            pass
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 