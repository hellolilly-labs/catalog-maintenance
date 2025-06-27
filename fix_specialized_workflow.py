#!/usr/bin/env python3
"""
Fix Specialized.com Workflow State

This script repairs the workflow state for specialized.com by:
1. Detecting completed research phases from filesystem
2. Updating the workflow state to reflect actual completion status
3. Setting the correct next step for the pipeline
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from workflow.workflow_state_manager import WorkflowStateManager, WorkflowState
from storage import get_account_storage_provider

def detect_completed_research_phases(brand_domain: str) -> dict:
    """Detect which research phases are actually completed by checking for files"""
    
    research_dir = f"local/account_storage/accounts/{brand_domain}/research_phases"
    
    if not os.path.exists(research_dir):
        print(f"❌ Research directory not found: {research_dir}")
        return {}
    
    # Map phase file patterns to phase keys
    phase_patterns = {
        "foundation_research": "foundation_research",
        "market_positioning": "market_positioning",
        "product_style_research": "product_style",
        "customer_cultural_research": "customer_cultural", 
        "voice_messaging_research": "voice_messaging",
        "interview_synthesis_research": "interview_synthesis",
        "linearity_analysis_research": "linearity_analysis",
        "research_integration": "research_integration"
    }
    
    completed_phases = {}
    
    for file_pattern, phase_key in phase_patterns.items():
        # Check for the three required files
        md_file = os.path.join(research_dir, f"{file_pattern}.md")
        metadata_file = os.path.join(research_dir, f"{file_pattern}_metadata.json")
        sources_file = os.path.join(research_dir, f"{file_pattern}_sources.json")
        
        if all(os.path.exists(f) for f in [md_file, metadata_file, sources_file]):
            # Get quality score from metadata if available
            quality_score = None
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    quality_score = metadata.get('quality_score', 0.8)
            except Exception as e:
                print(f"⚠️ Could not read metadata for {phase_key}: {e}")
                quality_score = 0.8  # Default
            
            # Get file modification time for completion date
            try:
                completion_time = datetime.fromtimestamp(os.path.getmtime(md_file))
            except:
                completion_time = datetime.now()
            
            completed_phases[phase_key] = {
                "quality_score": quality_score,
                "completed_at": completion_time,
                "files": [md_file, metadata_file, sources_file]
            }
            
            print(f"✅ {phase_key}: quality={quality_score:.2f}, completed={completion_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            missing_files = [f for f in [md_file, metadata_file, sources_file] if not os.path.exists(f)]
            print(f"❌ {phase_key}: missing files {missing_files}")
    
    return completed_phases

def fix_workflow_state_direct(brand_domain: str):
    """Fix the workflow state directly by writing to the JSON file"""
    
    print(f"🔧 Direct repair for {brand_domain}")
    print("=" * 60)
    
    # Detect completed phases
    completed_phases = detect_completed_research_phases(brand_domain)
    
    if not completed_phases:
        print("❌ No completed research phases detected!")
        return False
    
    print(f"\n📊 Found {len(completed_phases)} completed research phases")
    
    # Create the corrected workflow state
    now = datetime.now()
    latest_completion = max(phase_info["completed_at"] for phase_info in completed_phases.values())
    
    # Determine if all phases are complete
    required_phases = {
        "foundation_research", "market_positioning", "product_style", "customer_cultural",
        "voice_messaging", "interview_synthesis", "linearity_analysis", "research_integration"
    }
    completed_phase_set = set(completed_phases.keys())
    
    if completed_phase_set == required_phases:
        current_state = "research_complete"
        current_phase = None
        print("✅ All research phases complete - setting state to RESEARCH_COMPLETE")
    else:
        missing_phases = required_phases - completed_phase_set
        next_phase = sorted(missing_phases)[0]
        current_state = f"{next_phase}_in_progress"
        current_phase = next_phase
        print(f"⏳ Missing phases: {sorted(missing_phases)} - setting state to {current_state}")
    
    # Build quality scores dict
    quality_scores = {
        phase_key: phase_info["quality_score"] 
        for phase_key, phase_info in completed_phases.items()
    }
    
    # Create the workflow state data
    workflow_data = {
        "brand_domain": brand_domain,
        "current_state": current_state,
        "last_updated": now.isoformat(),
        "created_at": "2025-06-25T10:00:00.000000",  # Use original creation date
        "completed_phases": list(completed_phases.keys()),
        "failed_phases": [],
        "current_phase": current_phase,
        "total_research_time": len(completed_phases) * 3.0,  # Estimate 3 min per phase
        "total_cost": 0.0,
        "quality_scores": quality_scores,
        "last_error": None,
        "retry_count": 0,
        "last_research_date": latest_completion.isoformat(),
        "needs_refresh": False,
        "stale_phases": []
    }
    
    # Write directly to the file
    workflow_file_path = f"local/account_storage/accounts/{brand_domain}/workflow_state.json"
    os.makedirs(os.path.dirname(workflow_file_path), exist_ok=True)
    
    with open(workflow_file_path, 'w') as f:
        json.dump(workflow_data, f, indent=2)
    
    print(f"\n✅ DIRECTLY wrote corrected workflow state:")
    print(f"   State: {current_state}")
    print(f"   Completed phases: {len(completed_phases)}")
    print(f"   Quality scores: {len(quality_scores)}")
    print(f"   File: {workflow_file_path}")
    
    return True

def fix_workflow_state(brand_domain: str):
    """Fix the workflow state for a brand"""
    
    print(f"🔧 Analyzing workflow state for {brand_domain}")
    print("=" * 60)
    
    # Detect completed phases
    completed_phases = detect_completed_research_phases(brand_domain)
    
    if not completed_phases:
        print("❌ No completed research phases detected!")
        return False
    
    print(f"\n📊 Found {len(completed_phases)} completed research phases")
    
    # Initialize workflow manager
    storage_provider = get_account_storage_provider()
    workflow_manager = WorkflowStateManager(storage_provider)
    
    # Get current state
    current_info = workflow_manager.get_brand_info(brand_domain)
    print(f"\n📋 Current State: {current_info.current_state.value}")
    print(f"📋 Current Completed Phases: {current_info.completed_phases}")
    print(f"📋 Current Failed Phases: {current_info.failed_phases}")
    
    # Update workflow state with completed phases
    print(f"\n🔄 Updating workflow state...")
    
    # Clear failed phases and update completed phases
    current_info.failed_phases = []
    current_info.completed_phases = list(completed_phases.keys())
    current_info.last_error = None
    current_info.retry_count = 0
    
    # Update quality scores
    for phase_key, phase_info in completed_phases.items():
        current_info.quality_scores[phase_key] = phase_info["quality_score"]
    
    # Set the most recent completion date
    if completed_phases:
        latest_completion = max(phase_info["completed_at"] for phase_info in completed_phases.values())
        current_info.last_research_date = latest_completion
    
    # Determine correct state based on completed phases
    all_required_phases = set(workflow_manager.get_required_research_phases().keys())
    completed_phase_set = set(completed_phases.keys())
    
    if completed_phase_set == all_required_phases:
        # All research phases complete
        current_info.current_state = WorkflowState.RESEARCH_COMPLETE
        current_info.current_phase = None
        print("✅ All research phases complete - setting state to RESEARCH_COMPLETE")
    else:
        # Some phases still missing
        missing_phases = all_required_phases - completed_phase_set
        next_phase = sorted(missing_phases)[0]  # Get first missing phase
        
        phase_config = workflow_manager.get_required_research_phases()[next_phase]
        current_info.current_state = phase_config["states"]["in_progress"]
        current_info.current_phase = next_phase
        print(f"⏳ Missing phases: {sorted(missing_phases)} - setting state to {current_info.current_state.value}")
    
    # Update timestamp
    current_info.last_updated = datetime.now()
    
    # Save the corrected state
    workflow_manager._save_brand_state(brand_domain, current_info)
    
    print(f"\n✅ Updated workflow state:")
    print(f"   State: {current_info.current_state.value}")
    print(f"   Completed phases: {len(current_info.completed_phases)}")
    print(f"   Quality scores: {len(current_info.quality_scores)}")
    print(f"   Failed phases: {len(current_info.failed_phases)}")
    
    # Show next step
    next_step = workflow_manager.get_next_step(brand_domain)
    print(f"\n🎯 Next step: {next_step}")
    
    return True

def main():
    """Main function"""
    brand_domain = "specialized.com"
    
    print("🛠️ Specialized.com Workflow State Repair Tool")
    print("=" * 60)
    
    # Check if specialized.com directory exists
    brand_dir = f"local/account_storage/accounts/{brand_domain}"
    if not os.path.exists(brand_dir):
        print(f"❌ Brand directory not found: {brand_dir}")
        return False
    
    # Try direct repair first (bypass WorkflowStateManager issues)
    print("\n🔧 Attempting DIRECT repair (bypassing WorkflowStateManager)...")
    success = fix_workflow_state_direct(brand_domain)
    
    if success:
        print(f"\n🎉 Successfully repaired workflow state for {brand_domain}!")
        print("\n💡 You can now resume the pipeline with:")
        print(f"    python brand_researcher.py --brand {brand_domain} --status")
        print(f"    python brand_researcher.py --brand {brand_domain} --auto-continue")
    else:
        print(f"\n❌ Failed to repair workflow state for {brand_domain}")
    
    return success

if __name__ == "__main__":
    main() 