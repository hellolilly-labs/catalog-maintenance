#!/usr/bin/env python3

"""
Workflow State Manager

Manages the overall brand pipeline workflow state using file-system-as-state approach.
The research phase files themselves are the source of truth - no separate JSON files needed.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class WorkflowState(Enum):
    """Brand pipeline workflow states"""
    NOT_STARTED = "not_started"
    
    # Research phases (8 phases total)
    FOUNDATION_RESEARCH_IN_PROGRESS = "foundation_research_in_progress"
    FOUNDATION_RESEARCH_COMPLETE = "foundation_research_complete"
    MARKET_POSITIONING_IN_PROGRESS = "market_positioning_in_progress"
    MARKET_POSITIONING_COMPLETE = "market_positioning_complete"
    PRODUCT_STYLE_IN_PROGRESS = "product_style_in_progress"
    PRODUCT_STYLE_COMPLETE = "product_style_complete"
    CUSTOMER_CULTURAL_IN_PROGRESS = "customer_cultural_in_progress"
    CUSTOMER_CULTURAL_COMPLETE = "customer_cultural_complete"
    VOICE_MESSAGING_IN_PROGRESS = "voice_messaging_in_progress"
    VOICE_MESSAGING_COMPLETE = "voice_messaging_complete"
    INTERVIEW_SYNTHESIS_IN_PROGRESS = "interview_synthesis_in_progress"
    INTERVIEW_SYNTHESIS_COMPLETE = "interview_synthesis_complete"
    LINEARITY_ANALYSIS_IN_PROGRESS = "linearity_analysis_in_progress"
    LINEARITY_ANALYSIS_COMPLETE = "linearity_analysis_complete"
    RESEARCH_INTEGRATION_IN_PROGRESS = "research_integration_in_progress"
    RESEARCH_INTEGRATION_COMPLETE = "research_integration_complete"
    
    # All research complete (transition state)
    RESEARCH_COMPLETE = "research_complete"
    
    # Post-research phases
    CATALOG_IN_PROGRESS = "catalog_in_progress"
    CATALOG_COMPLETE = "catalog_complete"
    KNOWLEDGE_IN_PROGRESS = "knowledge_in_progress"
    KNOWLEDGE_COMPLETE = "knowledge_complete"
    RAG_IN_PROGRESS = "rag_in_progress"
    RAG_COMPLETE = "rag_complete"
    PERSONA_IN_PROGRESS = "persona_in_progress"
    PERSONA_COMPLETE = "persona_complete"
    PIPELINE_COMPLETE = "pipeline_complete"
    
    # Error and maintenance states
    FAILED = "failed"
    MAINTENANCE_REQUIRED = "maintenance_required"

@dataclass
class BrandWorkflowInfo:
    """Complete workflow information for a brand derived from file system"""
    brand_domain: str
    current_state: WorkflowState
    last_updated: datetime
    created_at: datetime
    
    # Progress tracking (derived from files)
    completed_phases: List[str]
    failed_phases: List[str]
    current_phase: Optional[str] = None
    
    # State metadata (aggregated from individual phase metadata)
    total_research_time: float = 0.0
    total_cost: float = 0.0
    quality_scores: Dict[str, float] = None
    
    # Error tracking
    last_error: Optional[str] = None
    retry_count: int = 0
    
    # Maintenance info
    last_research_date: Optional[datetime] = None
    needs_refresh: bool = False
    stale_phases: List[str] = None
    
    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = {}
        if self.stale_phases is None:
            self.stale_phases = []

class WorkflowStateManager:
    """
    File-System-as-State Workflow Manager
    
    Instead of maintaining workflow_state.json files, this manager derives all state
    information directly from the research phase files that exist in the file system.
    """
    
    def __init__(self, storage_manager=None):
        self.storage_manager = storage_manager
        # No longer maintain in-memory state cache - derive everything from files
    
    # def _get_research_directory(self, brand_domain: str) -> str:
    #     """Get the research phases directory path for a brand"""
    #     return f"local/account_storage/accounts/{brand_domain}/research"
    
    def _get_phase_file_patterns(self) -> Dict[str, str]:
        """Get mapping of file patterns to phase keys"""
        return {
            "foundation": "foundation",
            "market_positioning": "market_positioning",
            "product_style": "product_style",
            "customer_cultural": "customer_cultural", 
            "voice_messaging": "voice_messaging",
            "interview_synthesis": "interview_synthesis",
            "linearity_analysis": "linearity_analysis",
            "research_integration": "research_integration"
        }
    
    async def _scan_completed_phases(self, brand_domain: str) -> Dict[str, Dict[str, Any]]:
        """
        Scan file system to determine completed research phases
        Returns dict of phase_key -> metadata
        """
        if not self.storage_manager:
            # Fallback to direct file system access for local development
            return self._scan_completed_phases_direct(brand_domain)
        
        completed_phases = {}
        phase_patterns = self._get_phase_file_patterns()
        
        try:
            for file_pattern, phase_key in phase_patterns.items():
                # Check for the three required files using storage provider
                md_file = f"research/{file_pattern}/research.md"
                metadata_file = f"research/{file_pattern}/research_metadata.json"
                sources_file = f"research/{file_pattern}/research_sources.json"
                
                # Check if all three files exist
                files_exist = await asyncio.gather(
                    self.storage_manager.file_exists(brand_domain, md_file),
                    self.storage_manager.file_exists(brand_domain, metadata_file),
                    self.storage_manager.file_exists(brand_domain, sources_file),
                    return_exceptions=True
                )
                
                if all(isinstance(exists, bool) and exists for exists in files_exist):
                    # Phase is complete - extract metadata
                    try:
                        metadata_content = await self.storage_manager.read_file(brand_domain, metadata_file)
                        if metadata_content:
                            metadata = json.loads(metadata_content)
                        else:
                            metadata = {}
                        
                        # Get file metadata for timestamps
                        file_metadata = await self.storage_manager.get_file_metadata(brand_domain, md_file)
                        completion_time = datetime.now()
                        if file_metadata and 'modified_time' in file_metadata:
                            completion_time = datetime.fromisoformat(file_metadata['modified_time'].replace('Z', ''))
                        
                        completed_phases[phase_key] = {
                            'quality_score': metadata.get('quality_score', 0.8),
                            'completion_time': completion_time,
                            'research_time_minutes': metadata.get('research_time_minutes', 0),
                            'cost': metadata.get('cost', 0.0),
                            'metadata': metadata,
                            'files': {
                                'research': md_file,
                                'metadata': metadata_file,
                                'sources': sources_file
                            }
                        }
                        
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {phase_key}: {e}")
                        # Phase files exist but metadata is corrupted - mark as complete with defaults
                        file_metadata = await self.storage_manager.get_file_metadata(brand_domain, md_file)
                        completion_time = datetime.now()
                        if file_metadata and 'modified_time' in file_metadata:
                            completion_time = datetime.fromisoformat(file_metadata['modified_time'].replace('Z', ''))
                        
                        completed_phases[phase_key] = {
                            'quality_score': 0.8,
                            'completion_time': completion_time,
                            'research_time_minutes': 0,
                            'cost': 0.0,
                            'metadata': {},
                            'files': {
                                'research': md_file,
                                'metadata': metadata_file,
                                'sources': sources_file
                            }
                        }
        
        except Exception as e:
            logger.error(f"Error scanning completed phases for {brand_domain}: {e}")
            # Fallback to direct file system access
            return self._scan_completed_phases_direct(brand_domain)
        
        return completed_phases
    
    def _scan_completed_phases_direct(self, brand_domain: str) -> Dict[str, Dict[str, Any]]:
        """
        Direct file system scan fallback (for local development)
        """
        research_dir = f"local/account_storage/accounts/{brand_domain}/research"
        
        if not os.path.exists(research_dir):
            return {}
        
        completed_phases = {}
        phase_patterns = self._get_phase_file_patterns()
        
        for file_pattern, phase_key in phase_patterns.items():
            # Check for the three required files
            md_file = os.path.join(research_dir, f"{file_pattern}", "research.md")
            metadata_file = os.path.join(research_dir, f"{file_pattern}", "research_metadata.json")
            sources_file = os.path.join(research_dir, f"{file_pattern}", "research_sources.json")
            
            if all(os.path.exists(f) for f in [md_file, metadata_file, sources_file]):
                # Phase is complete - extract metadata
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Get file timestamps
                    md_mtime = os.path.getmtime(md_file)
                    completion_time = datetime.fromtimestamp(md_mtime)
                    
                    completed_phases[phase_key] = {
                        'quality_score': metadata.get('quality_score', 0.8),
                        'completion_time': completion_time,
                        'research_time_minutes': metadata.get('research_time_minutes', 0),
                        'cost': metadata.get('cost', 0.0),
                        'metadata': metadata,
                        'files': {
                            'research': md_file,
                            'metadata': metadata_file,
                            'sources': sources_file
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"Error reading metadata for {phase_key}: {e}")
                    # Phase files exist but metadata is corrupted - mark as complete with defaults
                    completion_time = datetime.fromtimestamp(os.path.getmtime(md_file))
                    completed_phases[phase_key] = {
                        'quality_score': 0.8,
                        'completion_time': completion_time,
                        'research_time_minutes': 0,
                        'cost': 0.0,
                        'metadata': {},
                        'files': {
                            'research': md_file,
                            'metadata': metadata_file,
                            'sources': sources_file
                        }
                    }
        
        return completed_phases
    
    def _derive_workflow_state(self, completed_phases: Dict[str, Dict[str, Any]]) -> WorkflowState:
        """Derive current workflow state from completed phases"""
        required_phases = set(self.get_required_research_phases().keys())
        completed_phase_set = set(completed_phases.keys())
        
        if completed_phase_set == required_phases:
            # All research phases complete
            return WorkflowState.RESEARCH_COMPLETE
        elif not completed_phases:
            # No phases started
            return WorkflowState.NOT_STARTED
        else:
            # Some phases complete, determine current phase
            missing_phases = required_phases - completed_phase_set
            if missing_phases:
                # Find the first missing phase (next to work on)
                phase_order = list(self.get_required_research_phases().keys())
                for phase in phase_order:
                    if phase in missing_phases:
                        # Return in-progress state for next phase
                        phase_config = self.get_required_research_phases()[phase]
                        return phase_config["states"]["in_progress"]
            
            # Fallback
            return WorkflowState.RESEARCH_COMPLETE
    
    async def get_brand_info(self, brand_domain: str) -> BrandWorkflowInfo:
        """Get complete workflow information for a brand by scanning file system"""
        
        # Scan completed phases from file system
        completed_phases_data = await self._scan_completed_phases(brand_domain)
        completed_phases = list(completed_phases_data.keys())
        
        # Derive workflow state
        current_state = self._derive_workflow_state(completed_phases_data)
        
        # Calculate current phase
        required_phases = list(self.get_required_research_phases().keys())
        missing_phases = [p for p in required_phases if p not in completed_phases]
        current_phase = missing_phases[0] if missing_phases else None
        
        # Aggregate metadata
        total_research_time = sum(
            phase_data.get('research_time_minutes', 0) 
            for phase_data in completed_phases_data.values()
        )
        total_cost = sum(
            phase_data.get('cost', 0.0) 
            for phase_data in completed_phases_data.values()
        )
        quality_scores = {
            phase: phase_data.get('quality_score', 0.8)
            for phase, phase_data in completed_phases_data.items()
        }
        
        # Get latest completion time
        last_research_date = None
        if completed_phases_data:
            completion_times = [
                phase_data['completion_time'] 
                for phase_data in completed_phases_data.values()
                if 'completion_time' in phase_data
            ]
            if completion_times:
                last_research_date = max(completion_times)
        
        # Create timestamp - use latest file modification or now
        last_updated = last_research_date or datetime.now()
        created_at = last_updated  # Approximation
        
        return BrandWorkflowInfo(
            brand_domain=brand_domain,
            current_state=current_state,
            last_updated=last_updated,
            created_at=created_at,
            completed_phases=completed_phases,
            failed_phases=[],  # Could derive from error files if needed
            current_phase=current_phase,
            total_research_time=total_research_time,
            total_cost=total_cost,
            quality_scores=quality_scores,
            last_research_date=last_research_date
        )
    
    def get_brand_info_sync(self, brand_domain: str) -> BrandWorkflowInfo:
        """Sync wrapper for get_brand_info"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_brand_info(brand_domain))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.get_brand_info(brand_domain))
    
    async def get_brand_state(self, brand_domain: str) -> WorkflowState:
        """Get current workflow state for a brand"""
        brand_info = await self.get_brand_info(brand_domain)
        return brand_info.current_state
    
    def get_brand_state_sync(self, brand_domain: str) -> WorkflowState:
        """Sync wrapper for get_brand_state"""
        return self.get_brand_info_sync(brand_domain).current_state
    
    def get_required_research_phases(self) -> Dict[str, Dict[str, Any]]:
        """Get the complete list of required research phases per ROADMAP Section 4.2"""
        return {
            "foundation": {
                "name": "Foundation Research",
                "cache_duration_days": 180,  # 6 months
                "research_time_minutes": "3-5",
                "quality_threshold": 8.0,
                "description": "Core brand identity that rarely changes",
                "states": {
                    "in_progress": WorkflowState.FOUNDATION_RESEARCH_IN_PROGRESS,
                    "complete": WorkflowState.FOUNDATION_RESEARCH_COMPLETE
                },
                "optional": False
            },
            "market_positioning": {
                "name": "Market Positioning Research", 
                "cache_duration_days": 120,  # 4 months
                "research_time_minutes": "2-4",
                "quality_threshold": 7.5,
                "description": "Competitive landscape and market position",
                "states": {
                    "in_progress": WorkflowState.MARKET_POSITIONING_IN_PROGRESS,
                    "complete": WorkflowState.MARKET_POSITIONING_COMPLETE
                },
                "optional": False
            },
            "product_style": {
                "name": "Product & Style Intelligence",
                "cache_duration_days": 90,   # 3 months
                "research_time_minutes": "2-3", 
                "quality_threshold": 7.5,
                "description": "Design philosophy and product aesthetics",
                "states": {
                    "in_progress": WorkflowState.PRODUCT_STYLE_IN_PROGRESS,
                    "complete": WorkflowState.PRODUCT_STYLE_COMPLETE
                },
                "optional": False
            },
            "customer_cultural": {
                "name": "Customer & Cultural Intelligence",
                "cache_duration_days": 105,   # 3.5 months
                "research_time_minutes": "2-3",
                "quality_threshold": 7.5,
                "description": "Target audience and cultural relevance",
                "states": {
                    "in_progress": WorkflowState.CUSTOMER_CULTURAL_IN_PROGRESS,
                    "complete": WorkflowState.CUSTOMER_CULTURAL_COMPLETE
                },
                "optional": False
            },
            "voice_messaging": {
                "name": "Voice & Messaging Analysis",
                "cache_duration_days": 75,   # 2.5 months
                "research_time_minutes": "1-2",
                "quality_threshold": 7.0,
                "description": "Brand voice and messaging analysis",
                "states": {
                    "in_progress": WorkflowState.VOICE_MESSAGING_IN_PROGRESS,
                    "complete": WorkflowState.VOICE_MESSAGING_COMPLETE
                },
                "optional": False
            },
            "interview_synthesis": {
                "name": "AI Brand Ethos Voice Interview Synthesis",
                "cache_duration_days": 150,   # 5 months
                "research_time_minutes": "3-5",
                "quality_threshold": 8.0,
                "description": "AI brand ethos voice interview synthesis",
                "states": {
                    "in_progress": WorkflowState.INTERVIEW_SYNTHESIS_IN_PROGRESS,
                    "complete": WorkflowState.INTERVIEW_SYNTHESIS_COMPLETE
                },
                "optional": True
            },
            "linearity_analysis": {
                "name": "Linearity Analysis",
                "cache_duration_days": 45,  # 1.5 months
                "research_time_minutes": "2-4",
                "quality_threshold": 7.5,
                "description": "Linearity and consistency analysis",
                "states": {
                    "in_progress": WorkflowState.LINEARITY_ANALYSIS_IN_PROGRESS,
                    "complete": WorkflowState.LINEARITY_ANALYSIS_COMPLETE
                },
                "optional": False
            },
            "research_integration": {
                "name": "Research Integration", 
                "cache_duration_days": 30,   # 1 month
                "research_time_minutes": "1-2",
                "quality_threshold": 8.0,
                "description": "Cross-validate and unify all research phases",
                "states": {
                    "in_progress": WorkflowState.RESEARCH_INTEGRATION_IN_PROGRESS,
                    "complete": WorkflowState.RESEARCH_INTEGRATION_COMPLETE
                },
                "optional": False
            }
        }
    
    async def are_all_research_phases_complete(self, brand_domain: str) -> bool:
        """Check if all required research phases are complete"""
        completed_phases_data = await self._scan_completed_phases(brand_domain)
        required_phases = set(self.get_required_research_phases().keys())
        completed_phases = set(completed_phases_data.keys())
        return required_phases.issubset(completed_phases)
    
    async def get_research_progress_summary(self, brand_domain: str) -> Dict[str, Any]:
        """Get detailed research progress summary"""
        completed_phases_data = await self._scan_completed_phases(brand_domain)
        required_phases = self.get_required_research_phases()
        
        completed_phases = list(completed_phases_data.keys())
        missing_phases = [
            phase for phase in required_phases.keys() 
            if phase not in completed_phases
        ]
        
        # next_phase is first missing phase, not current active phase
        next_phase = missing_phases[0] if missing_phases else None
        
        total_phases = len(required_phases)
        completed_count = len(completed_phases)
        completion_percentage = (completed_count / total_phases) * 100
        
        return {
            "total_phases": total_phases,
            "completed_count": completed_count,
            "completion_percentage": completion_percentage,
            "completed_phases": completed_phases,
            "next_phase": next_phase,
            "all_complete": await self.are_all_research_phases_complete(brand_domain),
            "missing_phases": missing_phases
        }
    
    async def get_next_step(self, brand_domain: str) -> str:
        """Get the next recommended step for a brand"""
        current_state = await self.get_brand_state(brand_domain)
        
        # Check if we're in any research phase (in progress or just starting)
        research_in_progress_states = [
            WorkflowState.NOT_STARTED,
            WorkflowState.FOUNDATION_RESEARCH_IN_PROGRESS,
            WorkflowState.MARKET_POSITIONING_IN_PROGRESS,
            WorkflowState.PRODUCT_STYLE_IN_PROGRESS,
            WorkflowState.CUSTOMER_CULTURAL_IN_PROGRESS,
            WorkflowState.VOICE_MESSAGING_IN_PROGRESS,
            WorkflowState.INTERVIEW_SYNTHESIS_IN_PROGRESS,
            WorkflowState.LINEARITY_ANALYSIS_IN_PROGRESS,
            WorkflowState.RESEARCH_INTEGRATION_IN_PROGRESS
        ]
        
        if current_state in research_in_progress_states:
            # Use research phase tracker for granular research recommendations
            from src.workflow.research_phase_tracker import get_research_phase_tracker
            
            research_tracker = get_research_phase_tracker(self)
            return research_tracker.get_next_step_command(brand_domain)
        
        next_steps = {
            WorkflowState.RESEARCH_COMPLETE: f"Start product catalog ingestion: python src/product_ingestor.py --full-sync --brand {brand_domain}",
            WorkflowState.CATALOG_IN_PROGRESS: "Resume catalog ingestion (check logs for progress)",
            WorkflowState.CATALOG_COMPLETE: f"Start knowledge base ingestion: python src/knowledge_ingestor.py --brand {brand_domain}",
            WorkflowState.KNOWLEDGE_IN_PROGRESS: "Resume knowledge base ingestion (check logs for progress)",
            WorkflowState.KNOWLEDGE_COMPLETE: "Configure RAG system integration",
            WorkflowState.RAG_COMPLETE: f"Generate AI persona: python src/persona_generator.py --brand {brand_domain}",
            WorkflowState.PERSONA_COMPLETE: "✅ Pipeline complete! Brand ready for AI sales agent",
            WorkflowState.PIPELINE_COMPLETE: "✅ Pipeline complete! Run maintenance check if needed",
            WorkflowState.FAILED: "Review errors and retry failed phase",
            WorkflowState.MAINTENANCE_REQUIRED: f"Run maintenance: python src/research/brand_researcher.py --brand {brand_domain} --auto-refresh"
        }
        
        return next_steps.get(current_state, "Unknown state - check workflow manually")
    
    async def check_maintenance_needed(self, brand_domain: str) -> Dict[str, Any]:
        """Check if brand needs maintenance (stale research phases)"""
        completed_phases_data = await self._scan_completed_phases(brand_domain)
        
        if not completed_phases_data:
            return {"needs_maintenance": False, "reason": "No completed phases"}
        
        # Use the required research phases for staleness thresholds
        required_phases = self.get_required_research_phases()
        
        stale_phases = []
        for phase_key, phase_data in completed_phases_data.items():
            if phase_key in required_phases:
                threshold_days = required_phases[phase_key]["cache_duration_days"]
                completion_time = phase_data.get('completion_time')
                if completion_time:
                    days_since_completion = (datetime.now() - completion_time).days
                    if days_since_completion > threshold_days:
                        stale_phases.append(phase_key)
        
        needs_maintenance = len(stale_phases) > 0
        
        # Get latest completion time
        completion_times = [
            phase_data['completion_time'] 
            for phase_data in completed_phases_data.values()
            if 'completion_time' in phase_data
        ]
        latest_completion = max(completion_times) if completion_times else None
        days_since_research = (datetime.now() - latest_completion).days if latest_completion else None
        
        return {
            "needs_maintenance": needs_maintenance,
            "stale_phases": stale_phases,
            "days_since_research": days_since_research,
            "total_phases": len(completed_phases_data)
        }

# Global workflow manager instance
_global_workflow_manager: Optional[WorkflowStateManager] = None

def get_workflow_manager() -> WorkflowStateManager:
    """Get global workflow state manager instance"""
    global _global_workflow_manager
    if _global_workflow_manager is None:
        from src.storage import get_account_storage_provider
        storage_manager = get_account_storage_provider()
        _global_workflow_manager = WorkflowStateManager(storage_manager)
    return _global_workflow_manager

# CLI interface for workflow management
async def main():
    """CLI interface for workflow state management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="File-System-as-State Workflow Manager")
    parser.add_argument("--brand", required=True, help="Brand URL")
    parser.add_argument("--action", choices=['status', 'next-step', 'summary'], 
                       default='status', help="Action to perform")
    parser.add_argument("--json", action='store_true', help="Output as JSON")
    
    args = parser.parse_args()
    
    manager = get_workflow_manager()
    
    if args.action == 'status':
        workflow_info = await manager.get_brand_info(args.brand)
        
        if args.json:
            # Convert to JSON-serializable format
            data = asdict(workflow_info)
            data['current_state'] = workflow_info.current_state.value
            data['last_updated'] = workflow_info.last_updated.isoformat()
            if workflow_info.last_research_date:
                data['last_research_date'] = workflow_info.last_research_date.isoformat()
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"\n🔄 WORKFLOW STATUS: {args.brand}")
            print(f"Current State: {workflow_info.current_state.value}")
            print(f"Completed Phases: {len(workflow_info.completed_phases)}/8")
            print(f"Total Research Time: {workflow_info.total_research_time:.1f} minutes")
            print(f"Last Updated: {workflow_info.last_updated}")
            
            if workflow_info.current_phase:
                print(f"\n📋 NEXT PHASE: {workflow_info.current_phase}")
    
    elif args.action == 'next-step':
        command = await manager.get_next_step(args.brand)
        print(command)
    
    elif args.action == 'summary':
        progress = await manager.get_research_progress_summary(args.brand)
        
        print(f"\n📊 RESEARCH PROGRESS: {args.brand}")
        print(f"Completion: {progress['completion_percentage']:.1f}% ({progress['completed_count']}/{progress['total_phases']})")
        print(f"Next Phase: {progress['next_phase'] or 'None (research complete)'}")
        
        if progress['missing_phases']:
            print(f"Missing: {', '.join(progress['missing_phases'])}")

if __name__ == "__main__":
    asyncio.run(main()) 