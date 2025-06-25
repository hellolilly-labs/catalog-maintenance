#!/usr/bin/env python3

"""
Workflow State Manager
Tracks brand pipeline progress and manages workflow continuity
"""

import json
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class WorkflowState(Enum):
    """Brand pipeline workflow states"""
    NOT_STARTED = "not_started"
    RESEARCH_IN_PROGRESS = "research_in_progress"
    RESEARCH_COMPLETE = "research_complete"
    CATALOG_IN_PROGRESS = "catalog_in_progress"
    CATALOG_COMPLETE = "catalog_complete"
    KNOWLEDGE_IN_PROGRESS = "knowledge_in_progress"
    KNOWLEDGE_COMPLETE = "knowledge_complete"
    RAG_IN_PROGRESS = "rag_in_progress"
    RAG_COMPLETE = "rag_complete"
    PERSONA_IN_PROGRESS = "persona_in_progress"
    PERSONA_COMPLETE = "persona_complete"
    PIPELINE_COMPLETE = "pipeline_complete"
    MAINTENANCE_MODE = "maintenance_mode"
    ERROR_STATE = "error_state"
    PARTIAL_FAILURE = "partial_failure"

class NextStepPriority(Enum):
    """Priority levels for next steps"""
    CRITICAL = "critical"      # Must be done immediately
    HIGH = "high"             # Should be done soon
    MEDIUM = "medium"         # Normal priority
    LOW = "low"              # Can be deferred
    MAINTENANCE = "maintenance"  # Regular maintenance

@dataclass
class NextStep:
    """Represents the next recommended action for a brand"""
    action: str                    # Human-readable action description
    command: str                   # Specific CLI command to execute
    priority: NextStepPriority     # Priority level
    estimated_duration: str        # Estimated time to complete
    estimated_cost: str           # Estimated API cost
    prerequisites: List[str]       # Any prerequisites needed
    reason: str                   # Why this step is needed
    automation_ready: bool         # Can this be automated?

@dataclass
class WorkflowProgress:
    """Detailed progress tracking for workflow components"""
    brand_url: str
    current_state: WorkflowState
    last_updated: datetime
    next_step: Optional[NextStep]
    
    # Component completion status
    research_phases: Dict[str, str]  # phase_name -> status
    catalog_status: str
    knowledge_status: str
    rag_status: str
    persona_status: str
    
    # Progress tracking
    total_progress_percent: float
    estimated_completion: Optional[datetime]
    
    # Error tracking
    errors: List[Dict[str, str]]
    warnings: List[Dict[str, str]]
    
    # Workflow metadata
    onboarding_started: Optional[datetime]
    last_successful_step: Optional[str]
    step_history: List[Dict[str, str]]

class WorkflowStateManager:
    """Manages workflow state and next step recommendations"""
    
    def __init__(self, storage_path: str = "accounts"):
        self.storage_path = Path(storage_path)
        
    async def get_workflow_state(self, brand_url: str) -> WorkflowProgress:
        """Get current workflow state for a brand"""
        
        # Load existing state or create new
        state_file = self.storage_path / brand_url / "workflow_state.json"
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)
                
            # Convert datetime strings back to datetime objects
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
            if data.get('onboarding_started'):
                data['onboarding_started'] = datetime.fromisoformat(data['onboarding_started'])
            if data.get('estimated_completion'):
                data['estimated_completion'] = datetime.fromisoformat(data['estimated_completion'])
                
            # Reconstruct NextStep if exists
            if data.get('next_step'):
                next_step_data = data['next_step']
                next_step_data['priority'] = NextStepPriority(next_step_data['priority'])
                data['next_step'] = NextStep(**next_step_data)
            
            data['current_state'] = WorkflowState(data['current_state'])
            return WorkflowProgress(**data)
        else:
            # Create new workflow state
            return await self._create_initial_state(brand_url)
    
    async def _create_initial_state(self, brand_url: str) -> WorkflowProgress:
        """Create initial workflow state for a new brand"""
        
        # Analyze current brand status to determine starting state
        current_state, progress_data = await self._analyze_current_status(brand_url)
        
        next_step = await self._determine_next_step(brand_url, current_state, progress_data)
        
        workflow_progress = WorkflowProgress(
            brand_url=brand_url,
            current_state=current_state,
            last_updated=datetime.now(),
            next_step=next_step,
            research_phases=progress_data.get('research_phases', {}),
            catalog_status=progress_data.get('catalog_status', 'not_started'),
            knowledge_status=progress_data.get('knowledge_status', 'not_started'),
            rag_status=progress_data.get('rag_status', 'not_started'),
            persona_status=progress_data.get('persona_status', 'not_started'),
            total_progress_percent=progress_data.get('total_progress', 0.0),
            estimated_completion=None,
            errors=[],
            warnings=[],
            onboarding_started=datetime.now() if current_state != WorkflowState.NOT_STARTED else None,
            last_successful_step=None,
            step_history=[]
        )
        
        await self._save_workflow_state(workflow_progress)
        return workflow_progress
    
    async def _analyze_current_status(self, brand_url: str) -> Tuple[WorkflowState, Dict]:
        """Analyze current brand status to determine workflow state"""
        
        progress_data = {}
        
        # Check if brand directory exists
        brand_dir = self.storage_path / brand_url
        if not brand_dir.exists():
            return WorkflowState.NOT_STARTED, progress_data
        
        # Analyze research phases
        research_phases = {}
        research_dir = brand_dir / "research_phases"
        if research_dir.exists():
            phase_files = list(research_dir.glob("*.json"))
            for phase_file in phase_files:
                phase_name = phase_file.stem
                try:
                    with open(phase_file, 'r') as f:
                        phase_data = json.load(f)
                    
                    if phase_data.get('quality_score', 0) >= 7.0:
                        research_phases[phase_name] = 'complete'
                    else:
                        research_phases[phase_name] = 'needs_improvement'
                except:
                    research_phases[phase_name] = 'error'
        
        progress_data['research_phases'] = research_phases
        
        # Analyze catalog status
        if (brand_dir / "products").exists():
            progress_data['catalog_status'] = 'complete'
        else:
            progress_data['catalog_status'] = 'not_started'
        
        # Analyze knowledge status
        if (brand_dir / "knowledge_chunks").exists():
            progress_data['knowledge_status'] = 'complete'
        else:
            progress_data['knowledge_status'] = 'not_started'
        
        # Analyze RAG status
        if research_phases.get('rag_optimization') == 'complete':
            progress_data['rag_status'] = 'complete'
        else:
            progress_data['rag_status'] = 'not_started'
        
        # Analyze persona status
        if (brand_dir / "ai_personas").exists():
            progress_data['persona_status'] = 'complete'
        else:
            progress_data['persona_status'] = 'not_started'
        
        # Determine overall state
        research_complete = len([p for p in research_phases.values() if p == 'complete']) >= 6
        catalog_complete = progress_data['catalog_status'] == 'complete'
        knowledge_complete = progress_data['knowledge_status'] == 'complete'
        rag_complete = progress_data['rag_status'] == 'complete'
        persona_complete = progress_data['persona_status'] == 'complete'
        
        # Calculate progress percentage
        total_components = 5  # research, catalog, knowledge, rag, persona
        completed_components = sum([
            1 if research_complete else 0,
            1 if catalog_complete else 0,
            1 if knowledge_complete else 0,
            1 if rag_complete else 0,
            1 if persona_complete else 0
        ])
        progress_data['total_progress'] = (completed_components / total_components) * 100
        
        # Determine state
        if completed_components == 5:
            return WorkflowState.PIPELINE_COMPLETE, progress_data
        elif persona_complete:
            return WorkflowState.PERSONA_COMPLETE, progress_data
        elif rag_complete:
            return WorkflowState.RAG_COMPLETE, progress_data
        elif knowledge_complete:
            return WorkflowState.KNOWLEDGE_COMPLETE, progress_data
        elif catalog_complete:
            return WorkflowState.CATALOG_COMPLETE, progress_data
        elif research_complete:
            return WorkflowState.RESEARCH_COMPLETE, progress_data
        elif len(research_phases) > 0:
            return WorkflowState.RESEARCH_IN_PROGRESS, progress_data
        else:
            return WorkflowState.NOT_STARTED, progress_data
    
    async def _determine_next_step(self, brand_url: str, state: WorkflowState, progress_data: Dict) -> NextStep:
        """Determine the next recommended step based on current state"""
        
        next_steps = {
            WorkflowState.NOT_STARTED: NextStep(
                action="Start brand onboarding",
                command=f"./scripts/brand_manager.sh onboard {brand_url}",
                priority=NextStepPriority.HIGH,
                estimated_duration="15-25 minutes",
                estimated_cost="$8-15",
                prerequisites=[],
                reason="Brand has not been onboarded yet",
                automation_ready=True
            ),
            
            WorkflowState.RESEARCH_IN_PROGRESS: NextStep(
                action="Continue brand research",
                command=f"./scripts/brand_maintenance.sh --brand {brand_url} --type refresh-stale",
                priority=NextStepPriority.HIGH,
                estimated_duration="5-15 minutes",
                estimated_cost="$2-8",
                prerequisites=[],
                reason="Brand research is partially complete",
                automation_ready=True
            ),
            
            WorkflowState.RESEARCH_COMPLETE: NextStep(
                action="Start product catalog ingestion",
                command=f"python src/product_ingestor.py --brand {brand_url} --full-sync --use-brand-intelligence",
                priority=NextStepPriority.HIGH,
                estimated_duration="10-20 minutes",
                estimated_cost="$3-6",
                prerequisites=["Brand research complete"],
                reason="Research is complete, ready for catalog ingestion",
                automation_ready=True
            ),
            
            WorkflowState.CATALOG_COMPLETE: NextStep(
                action="Create knowledge base",
                command=f"python src/knowledge_ingestor.py --brand {brand_url} --include-brand-intelligence",
                priority=NextStepPriority.HIGH,
                estimated_duration="5-10 minutes",
                estimated_cost="$1-3",
                prerequisites=["Product catalog complete"],
                reason="Product catalog is ready, create knowledge base",
                automation_ready=True
            ),
            
            WorkflowState.KNOWLEDGE_COMPLETE: NextStep(
                action="Optimize RAG search",
                command=f"python src/research/brand_researcher.py --brand {brand_url} --phases rag_optimization",
                priority=NextStepPriority.MEDIUM,
                estimated_duration="5-10 minutes",
                estimated_cost="$2-4",
                prerequisites=["Knowledge base complete"],
                reason="Knowledge base ready, optimize search capabilities",
                automation_ready=True
            ),
            
            WorkflowState.RAG_COMPLETE: NextStep(
                action="Generate AI sales persona",
                command=f"python src/research/brand_researcher.py --brand {brand_url} --phases ai_persona_generation",
                priority=NextStepPriority.MEDIUM,
                estimated_duration="5-10 minutes",
                estimated_cost="$3-5",
                prerequisites=["RAG optimization complete"],
                reason="RAG optimized, ready for AI persona generation",
                automation_ready=True
            ),
            
            WorkflowState.PERSONA_COMPLETE: NextStep(
                action="Validate pipeline completion",
                command=f"python src/status/pipeline_status.py --brand {brand_url} --validate-complete",
                priority=NextStepPriority.LOW,
                estimated_duration="2-5 minutes",
                estimated_cost="$0",
                prerequisites=["AI persona complete"],
                reason="Validate complete pipeline functionality",
                automation_ready=True
            ),
            
            WorkflowState.PIPELINE_COMPLETE: NextStep(
                action="Smart maintenance check",
                command=f"./scripts/brand_manager.sh refresh {brand_url}",
                priority=NextStepPriority.MAINTENANCE,
                estimated_duration="2-10 minutes",
                estimated_cost="$1-4",
                prerequisites=[],
                reason="Pipeline complete, check for maintenance needs",
                automation_ready=True
            ),
            
            WorkflowState.MAINTENANCE_MODE: NextStep(
                action="Monitor and maintain",
                command=f"./scripts/brand_manager.sh status {brand_url}",
                priority=NextStepPriority.LOW,
                estimated_duration="1-2 minutes",
                estimated_cost="$0",
                prerequisites=[],
                reason="Regular monitoring and maintenance",
                automation_ready=True
            )
        }
        
        return next_steps.get(state, NextStep(
            action="Check brand status",
            command=f"./scripts/brand_manager.sh status {brand_url}",
            priority=NextStepPriority.MEDIUM,
            estimated_duration="1-2 minutes",
            estimated_cost="$0",
            prerequisites=[],
            reason="Determine current state and next steps",
            automation_ready=True
        ))
    
    async def update_workflow_state(self, brand_url: str, new_state: WorkflowState, 
                                  step_completed: str = None, error: str = None) -> WorkflowProgress:
        """Update workflow state after a step completion or error"""
        
        workflow_progress = await self.get_workflow_state(brand_url)
        
        # Update state
        old_state = workflow_progress.current_state
        workflow_progress.current_state = new_state
        workflow_progress.last_updated = datetime.now()
        
        # Add to step history
        step_entry = {
            "timestamp": datetime.now().isoformat(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "step_completed": step_completed,
            "error": error
        }
        workflow_progress.step_history.append(step_entry)
        
        # Update last successful step
        if step_completed and not error:
            workflow_progress.last_successful_step = step_completed
        
        # Add error if present
        if error:
            workflow_progress.errors.append({
                "timestamp": datetime.now().isoformat(),
                "step": step_completed or "unknown",
                "error": error
            })
        
        # Re-analyze current status and determine next step
        _, progress_data = await self._analyze_current_status(brand_url)
        workflow_progress.total_progress_percent = progress_data.get('total_progress', 0.0)
        workflow_progress.research_phases = progress_data.get('research_phases', {})
        workflow_progress.catalog_status = progress_data.get('catalog_status', 'not_started')
        workflow_progress.knowledge_status = progress_data.get('knowledge_status', 'not_started')
        workflow_progress.rag_status = progress_data.get('rag_status', 'not_started')
        workflow_progress.persona_status = progress_data.get('persona_status', 'not_started')
        
        # Determine next step
        workflow_progress.next_step = await self._determine_next_step(brand_url, new_state, progress_data)
        
        # Estimate completion time
        if workflow_progress.total_progress_percent < 100:
            # Rough estimate based on remaining work
            remaining_percent = 100 - workflow_progress.total_progress_percent
            estimated_minutes = (remaining_percent / 100) * 30  # Assume 30 minutes total
            workflow_progress.estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)
        
        await self._save_workflow_state(workflow_progress)
        return workflow_progress
    
    async def _save_workflow_state(self, workflow_progress: WorkflowProgress):
        """Save workflow state to storage"""
        
        brand_dir = self.storage_path / workflow_progress.brand_url
        brand_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = brand_dir / "workflow_state.json"
        
        # Convert to dict and handle datetime serialization
        data = asdict(workflow_progress)
        data['last_updated'] = workflow_progress.last_updated.isoformat()
        
        if workflow_progress.onboarding_started:
            data['onboarding_started'] = workflow_progress.onboarding_started.isoformat()
        
        if workflow_progress.estimated_completion:
            data['estimated_completion'] = workflow_progress.estimated_completion.isoformat()
        
        # Handle NextStep serialization
        if workflow_progress.next_step:
            data['next_step']['priority'] = workflow_progress.next_step.priority.value
        
        # Handle enum serialization
        data['current_state'] = workflow_progress.current_state.value
        
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def get_next_step_command(self, brand_url: str) -> str:
        """Get the CLI command for the next recommended step"""
        
        workflow_progress = await self.get_workflow_state(brand_url)
        
        if workflow_progress.next_step:
            return workflow_progress.next_step.command
        else:
            return f"./scripts/brand_manager.sh status {brand_url}"
    
    async def list_all_brand_states(self) -> Dict[str, WorkflowProgress]:
        """Get workflow states for all brands in the system"""
        
        brand_states = {}
        
        if self.storage_path.exists():
            for brand_dir in self.storage_path.iterdir():
                if brand_dir.is_dir():
                    brand_url = brand_dir.name
                    try:
                        brand_states[brand_url] = await self.get_workflow_state(brand_url)
                    except Exception as e:
                        print(f"Error loading state for {brand_url}: {e}")
        
        return brand_states

# CLI interface for workflow management
async def main():
    """CLI interface for workflow state management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Workflow State Manager")
    parser.add_argument("--brand", required=True, help="Brand URL")
    parser.add_argument("--action", choices=['status', 'next-step', 'update', 'history'], 
                       default='status', help="Action to perform")
    parser.add_argument("--new-state", help="New state for update action")
    parser.add_argument("--step-completed", help="Step that was completed")
    parser.add_argument("--error", help="Error message if step failed")
    parser.add_argument("--json", action='store_true', help="Output as JSON")
    
    args = parser.parse_args()
    
    manager = WorkflowStateManager()
    
    if args.action == 'status':
        workflow_progress = await manager.get_workflow_state(args.brand)
        
        if args.json:
            # Convert to JSON-serializable format
            data = asdict(workflow_progress)
            data['current_state'] = workflow_progress.current_state.value
            data['last_updated'] = workflow_progress.last_updated.isoformat()
            if workflow_progress.next_step:
                data['next_step']['priority'] = workflow_progress.next_step.priority.value
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"\n🔄 WORKFLOW STATUS: {args.brand}")
            print(f"Current State: {workflow_progress.current_state.value}")
            print(f"Progress: {workflow_progress.total_progress_percent:.1f}%")
            print(f"Last Updated: {workflow_progress.last_updated}")
            
            if workflow_progress.next_step:
                print(f"\n📋 NEXT STEP:")
                print(f"Action: {workflow_progress.next_step.action}")
                print(f"Command: {workflow_progress.next_step.command}")
                print(f"Priority: {workflow_progress.next_step.priority.value}")
                print(f"Duration: {workflow_progress.next_step.estimated_duration}")
                print(f"Cost: {workflow_progress.next_step.estimated_cost}")
                print(f"Reason: {workflow_progress.next_step.reason}")
    
    elif args.action == 'next-step':
        command = await manager.get_next_step_command(args.brand)
        print(command)
    
    elif args.action == 'update':
        if not args.new_state:
            print("Error: --new-state required for update action")
            return
        
        new_state = WorkflowState(args.new_state)
        workflow_progress = await manager.update_workflow_state(
            args.brand, new_state, args.step_completed, args.error
        )
        
        print(f"Updated {args.brand} to state: {workflow_progress.current_state.value}")
        if workflow_progress.next_step:
            print(f"Next step: {workflow_progress.next_step.action}")
    
    elif args.action == 'history':
        workflow_progress = await manager.get_workflow_state(args.brand)
        
        print(f"\n📋 WORKFLOW HISTORY: {args.brand}")
        for entry in workflow_progress.step_history[-10:]:  # Last 10 entries
            print(f"{entry['timestamp']}: {entry['from_state']} → {entry['to_state']}")
            if entry.get('step_completed'):
                print(f"  ✅ Completed: {entry['step_completed']}")
            if entry.get('error'):
                print(f"  ❌ Error: {entry['error']}")

if __name__ == "__main__":
    asyncio.run(main()) 