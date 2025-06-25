#!/usr/bin/env python3

"""
Workflow State Manager
Tracks brand pipeline progress and manages workflow continuity
"""

import json
import os
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

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
    """Complete workflow information for a brand"""
    brand_domain: str
    current_state: WorkflowState
    last_updated: datetime
    created_at: datetime
    
    # Progress tracking
    completed_phases: List[str]
    failed_phases: List[str]
    current_phase: Optional[str] = None
    
    # State metadata
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
    """Manage workflow states for multiple brands with filesystem persistence"""
    
    def __init__(self, storage_manager=None):
        self.storage_manager = storage_manager
        self.workflow_states: Dict[str, BrandWorkflowInfo] = {}
        self._load_all_brand_states()
    
    def _get_workflow_file_path(self, brand_domain: str) -> str:
        """Get filesystem path for brand workflow state file"""
        if hasattr(self.storage_manager, 'get_account_file_path'):
            return self.storage_manager.get_account_file_path(brand_domain, "workflow_state.json")
        else:
            # Local fallback
            return f"local/account_storage/accounts/{brand_domain}/workflow_state.json"
    
    def _load_all_brand_states(self):
        """Discover and load all brand workflow states from filesystem"""
        
        # Discover brands from filesystem
        discovered_brands = self._discover_brands_from_filesystem()
        
        for brand_domain in discovered_brands:
            try:
                workflow_info = self._load_brand_state(brand_domain)
                if workflow_info:
                    self.workflow_states[brand_domain] = workflow_info
                else:
                    # Create new workflow state for discovered brand
                    self.workflow_states[brand_domain] = self._create_new_workflow_state(brand_domain)
                    
            except Exception as e:
                logger.warning(f"Failed to load workflow state for {brand_domain}: {e}")
                
        logger.info(f"Loaded workflow states for {len(self.workflow_states)} brands")
    
    def _discover_brands_from_filesystem(self) -> List[str]:
        """Discover brand domains from filesystem structure"""
        brands = []
        
        # Check both GCP and local storage paths
        storage_paths = [
            "local/account_storage/accounts",  # Local storage
        ]
        
        for storage_path in storage_paths:
            if os.path.exists(storage_path):
                try:
                    for item in os.listdir(storage_path):
                        item_path = os.path.join(storage_path, item)
                        if os.path.isdir(item_path) and '.' in item:  # Likely a domain
                            brands.append(item)
                except Exception as e:
                    logger.warning(f"Failed to scan {storage_path}: {e}")
        
        # Remove duplicates and sort
        brands = sorted(list(set(brands)))
        logger.info(f"Discovered {len(brands)} brands from filesystem: {brands}")
        return brands
    
    def _load_brand_state(self, brand_domain: str) -> Optional[BrandWorkflowInfo]:
        """Load workflow state for a specific brand"""
        try:
            workflow_file_path = self._get_workflow_file_path(brand_domain)
            
            if hasattr(self.storage_manager, 'load_account_file') and self.storage_manager:
                # Use storage manager
                content = self.storage_manager.load_account_file(brand_domain, "workflow_state.json")
                if content:
                    data = json.loads(content)
                else:
                    return None
            else:
                # Local fallback
                if not os.path.exists(workflow_file_path):
                    return None
                    
                with open(workflow_file_path, 'r') as f:
                    data = json.load(f)
            
            # Parse datetime fields
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            if data.get('last_research_date'):
                data['last_research_date'] = datetime.fromisoformat(data['last_research_date'])
            
            # Convert state string to enum
            data['current_state'] = WorkflowState(data['current_state'])
            
            return BrandWorkflowInfo(**data)
            
        except Exception as e:
            logger.warning(f"Failed to load workflow state for {brand_domain}: {e}")
            return None
    
    def _save_brand_state(self, brand_domain: str, workflow_info: BrandWorkflowInfo):
        """Save workflow state for a specific brand"""
        try:
            # Convert to serializable format
            data = asdict(workflow_info)
            data['last_updated'] = workflow_info.last_updated.isoformat()
            data['created_at'] = workflow_info.created_at.isoformat()
            if workflow_info.last_research_date:
                data['last_research_date'] = workflow_info.last_research_date.isoformat()
            data['current_state'] = workflow_info.current_state.value
            
            if hasattr(self.storage_manager, 'save_account_file') and self.storage_manager:
                # Use storage manager
                self.storage_manager.save_account_file(
                    brand_domain, 
                    "workflow_state.json", 
                    json.dumps(data, indent=2)
                )
            else:
                # Local fallback
                workflow_file_path = self._get_workflow_file_path(brand_domain)
                os.makedirs(os.path.dirname(workflow_file_path), exist_ok=True)
                
                with open(workflow_file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            logger.info(f"Saved workflow state for {brand_domain}: {workflow_info.current_state.value}")
            
        except Exception as e:
            logger.error(f"Failed to save workflow state for {brand_domain}: {e}")
            raise
    
    def _create_new_workflow_state(self, brand_domain: str) -> BrandWorkflowInfo:
        """Create new workflow state for a brand"""
        now = datetime.now()
        
        workflow_info = BrandWorkflowInfo(
            brand_domain=brand_domain,
            current_state=WorkflowState.NOT_STARTED,
            last_updated=now,
            created_at=now,
            completed_phases=[],
            failed_phases=[]
        )
        
        # Save immediately
        self._save_brand_state(brand_domain, workflow_info)
        
        return workflow_info
    
    # Public API methods
    
    def get_brand_state(self, brand_domain: str) -> WorkflowState:
        """Get current workflow state for a brand"""
        if brand_domain not in self.workflow_states:
            # Create new workflow state for unknown brand
            self.workflow_states[brand_domain] = self._create_new_workflow_state(brand_domain)
        
        return self.workflow_states[brand_domain].current_state
    
    def get_brand_info(self, brand_domain: str) -> BrandWorkflowInfo:
        """Get complete workflow information for a brand"""
        if brand_domain not in self.workflow_states:
            self.workflow_states[brand_domain] = self._create_new_workflow_state(brand_domain)
        
        return self.workflow_states[brand_domain]
    
    def update_brand_state(self, brand_domain: str, new_state: WorkflowState, 
                          phase_name: Optional[str] = None, error: Optional[str] = None,
                          quality_score: Optional[float] = None, duration: Optional[float] = None,
                          cost: Optional[float] = None):
        """Update workflow state for a brand with optional metadata"""
        
        if brand_domain not in self.workflow_states:
            self.workflow_states[brand_domain] = self._create_new_workflow_state(brand_domain)
        
        workflow_info = self.workflow_states[brand_domain]
        old_state = workflow_info.current_state
        
        # Update core state
        workflow_info.current_state = new_state
        workflow_info.last_updated = datetime.now()
        workflow_info.current_phase = phase_name
        
        # Handle phase completion
        if phase_name and new_state in [WorkflowState.RESEARCH_COMPLETE, WorkflowState.CATALOG_COMPLETE, 
                                       WorkflowState.KNOWLEDGE_COMPLETE, WorkflowState.RAG_COMPLETE,
                                       WorkflowState.PERSONA_COMPLETE, WorkflowState.PIPELINE_COMPLETE]:
            if phase_name not in workflow_info.completed_phases:
                workflow_info.completed_phases.append(phase_name)
            
            # Remove from failed phases if it was there
            if phase_name in workflow_info.failed_phases:
                workflow_info.failed_phases.remove(phase_name)
        
        # Handle failures
        if new_state == WorkflowState.FAILED:
            workflow_info.last_error = error
            workflow_info.retry_count += 1
            if phase_name and phase_name not in workflow_info.failed_phases:
                workflow_info.failed_phases.append(phase_name)
        else:
            # Clear error on successful transition
            workflow_info.last_error = None
            workflow_info.retry_count = 0
        
        # Update metrics
        if quality_score is not None and phase_name:
            workflow_info.quality_scores[phase_name] = quality_score
        
        if duration is not None:
            workflow_info.total_research_time += duration
        
        if cost is not None:
            workflow_info.total_cost += cost
        
        # Update research date for research phases
        if phase_name and 'research' in phase_name.lower():
            workflow_info.last_research_date = datetime.now()
        
        # Save state
        self._save_brand_state(brand_domain, workflow_info)
        
        logger.info(f"Updated {brand_domain}: {old_state.value} → {new_state.value}" + 
                   (f" (phase: {phase_name})" if phase_name else ""))
    
    def get_next_step(self, brand_domain: str) -> str:
        """Get the next recommended step for a brand"""
        current_state = self.get_brand_state(brand_domain)
        
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
    
    def can_resume(self, brand_domain: str) -> bool:
        """Check if a brand workflow can be resumed"""
        current_state = self.get_brand_state(brand_domain)
        
        resumable_states = [
            WorkflowState.FOUNDATION_RESEARCH_IN_PROGRESS,
            WorkflowState.MARKET_POSITIONING_IN_PROGRESS,
            WorkflowState.PRODUCT_STYLE_IN_PROGRESS,
            WorkflowState.CUSTOMER_CULTURAL_IN_PROGRESS,
            WorkflowState.VOICE_MESSAGING_IN_PROGRESS,
            WorkflowState.INTERVIEW_SYNTHESIS_IN_PROGRESS,
            WorkflowState.LINEARITY_ANALYSIS_IN_PROGRESS,
            WorkflowState.RESEARCH_INTEGRATION_IN_PROGRESS,
            WorkflowState.CATALOG_IN_PROGRESS,
            WorkflowState.KNOWLEDGE_IN_PROGRESS,
            WorkflowState.RAG_IN_PROGRESS,
            WorkflowState.PERSONA_IN_PROGRESS,
            WorkflowState.FAILED
        ]
        
        return current_state in resumable_states
    
    def get_all_brands(self) -> List[str]:
        """Get list of all tracked brands"""
        return list(self.workflow_states.keys())
    
    def get_brands_by_state(self, state: WorkflowState) -> List[str]:
        """Get all brands in a specific workflow state"""
        return [
            brand for brand, info in self.workflow_states.items()
            if info.current_state == state
        ]
    
    def get_required_research_phases(self) -> Dict[str, Dict[str, Any]]:
        """Get the complete list of required research phases per ROADMAP Section 4.2"""
        return {
            "foundation_research": {
                "name": "Foundation Research",
                "cache_duration_days": 180,  # 6 months
                "research_time_minutes": "3-5",
                "quality_threshold": 8.0,
                "description": "Core brand identity that rarely changes",
                "states": {
                    "in_progress": WorkflowState.FOUNDATION_RESEARCH_IN_PROGRESS,
                    "complete": WorkflowState.FOUNDATION_RESEARCH_COMPLETE
                }
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
                }
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
                }
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
                }
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
                }
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
                }
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
                }
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
                }
            }
        }
    
    def start_research_phase(self, brand_domain: str, phase_key: str):
        """Start a specific research phase"""
        required_phases = self.get_required_research_phases()
        if phase_key not in required_phases:
            raise ValueError(f"Unknown research phase: {phase_key}")
        
        phase_config = required_phases[phase_key]
        in_progress_state = phase_config["states"]["in_progress"]
        
        self.update_brand_state(
            brand_domain=brand_domain,
            new_state=in_progress_state,
            phase_name=phase_key
        )
        
        logger.info(f"Started {phase_config['name']} for {brand_domain}")
    
    def complete_research_phase(self, brand_domain: str, phase_key: str, 
                               quality_score: Optional[float] = None, 
                               duration: Optional[float] = None):
        """Complete a specific research phase"""
        required_phases = self.get_required_research_phases()
        if phase_key not in required_phases:
            raise ValueError(f"Unknown research phase: {phase_key}")
        
        phase_config = required_phases[phase_key]
        complete_state = phase_config["states"]["complete"]
        
        self.update_brand_state(
            brand_domain=brand_domain,
            new_state=complete_state,
            phase_name=phase_key,
            quality_score=quality_score,
            duration=duration
        )
        
        logger.info(f"Completed {phase_config['name']} for {brand_domain}")
        
        # Check if all research phases are complete
        if self.are_all_research_phases_complete(brand_domain):
            logger.info(f"🎉 All research phases complete for {brand_domain}! Ready for catalog ingestion.")
    
    def get_current_research_phase(self, brand_domain: str) -> Optional[str]:
        """Get the current research phase in progress"""
        workflow_info = self.get_brand_info(brand_domain)
        current_state = workflow_info.current_state
        
        # Map states to phase keys
        state_to_phase = {}
        for phase_key, phase_config in self.get_required_research_phases().items():
            state_to_phase[phase_config["states"]["in_progress"]] = phase_key
            state_to_phase[phase_config["states"]["complete"]] = phase_key
        
        return state_to_phase.get(current_state)
    
    def get_research_phase_status(self, brand_domain: str, phase_key: str) -> str:
        """Get status of a specific research phase: 'not_started', 'in_progress', 'complete'"""
        workflow_info = self.get_brand_info(brand_domain)
        required_phases = self.get_required_research_phases()
        
        if phase_key not in required_phases:
            return "unknown"
        
        phase_config = required_phases[phase_key]
        current_state = workflow_info.current_state
        
        if current_state == phase_config["states"]["complete"]:
            return "complete"
        elif current_state == phase_config["states"]["in_progress"]:
            return "in_progress"
        else:
            # Check if this phase should be available (previous phases complete)
            phase_order = list(required_phases.keys())
            current_phase_index = phase_order.index(phase_key)
            
            # Check if all previous phases are complete
            for i in range(current_phase_index):
                prev_phase = phase_order[i]
                prev_phase_status = self.get_research_phase_status(brand_domain, prev_phase)
                if prev_phase_status != "complete":
                    return "blocked"  # Cannot start until previous phases complete
            
            return "not_started"
    
    def are_all_research_phases_complete(self, brand_domain: str) -> bool:
        """Check if all required research phases are complete"""
        return self.get_current_research_phase(brand_domain) is None
    
    def get_research_progress_summary(self, brand_domain: str) -> Dict[str, Any]:
        """Get detailed research progress summary"""
        workflow_info = self.get_brand_info(brand_domain)
        required_phases = self.get_required_research_phases()
        completed_phases = [
            phase for phase in required_phases.keys() 
            if phase in workflow_info.completed_phases
        ]
        next_phase = self.get_current_research_phase(brand_domain)
        
        total_phases = len(required_phases)
        completed_count = len(completed_phases)
        completion_percentage = (completed_count / total_phases) * 100
        
        return {
            "total_phases": total_phases,
            "completed_count": completed_count,
            "completion_percentage": completion_percentage,
            "completed_phases": completed_phases,
            "next_phase": next_phase,
            "all_complete": self.are_all_research_phases_complete(brand_domain),
            "missing_phases": [
                phase for phase in required_phases.keys() 
                if phase not in workflow_info.completed_phases
            ]
        }

    def check_maintenance_needed(self, brand_domain: str) -> Dict[str, Any]:
        """Check if brand needs maintenance (stale research phases)"""
        workflow_info = self.get_brand_info(brand_domain)
        
        if not workflow_info.last_research_date:
            return {"needs_maintenance": False, "reason": "No research date available"}
        
        # Use the required research phases for staleness thresholds
        required_phases = self.get_required_research_phases()
        
        stale_phases = []
        for phase_key, phase_config in required_phases.items():
            if phase_key in workflow_info.completed_phases:
                threshold_days = phase_config["cache_duration_days"]
                days_since_research = (datetime.now() - workflow_info.last_research_date).days
                if days_since_research > threshold_days:
                    stale_phases.append(phase_key)
        
        needs_maintenance = len(stale_phases) > 0
        
        if needs_maintenance and workflow_info.current_state == WorkflowState.PIPELINE_COMPLETE:
            # Update state to maintenance required
            workflow_info.current_state = WorkflowState.MAINTENANCE_REQUIRED
            workflow_info.stale_phases = stale_phases
            self._save_brand_state(brand_domain, workflow_info)
        
        return {
            "needs_maintenance": needs_maintenance,
            "stale_phases": stale_phases,
            "days_since_research": (datetime.now() - workflow_info.last_research_date).days if workflow_info.last_research_date else None,
            "total_phases": len(workflow_info.completed_phases)
        }
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of all brand workflows"""
        
        state_counts = {}
        for state in WorkflowState:
            state_counts[state.value] = len(self.get_brands_by_state(state))
        
        total_brands = len(self.workflow_states)
        completed_brands = len(self.get_brands_by_state(WorkflowState.PIPELINE_COMPLETE))
        
        return {
            "total_brands": total_brands,
            "completed_brands": completed_brands,
            "completion_rate": completed_brands / total_brands if total_brands > 0 else 0,
            "state_distribution": state_counts,
            "brands_needing_maintenance": len(self.get_brands_by_state(WorkflowState.MAINTENANCE_REQUIRED))
        }

    def mark_phase_complete(self, brand_domain: str, phase_name: str, 
                           quality_score: Optional[float] = None, duration: Optional[float] = None):
        """Mark a specific phase as complete regardless of overall workflow state"""
        
        if brand_domain not in self.workflow_states:
            self.workflow_states[brand_domain] = self._create_new_workflow_state(brand_domain)
        
        workflow_info = self.workflow_states[brand_domain]
        
        # Add to completed phases if not already there
        if phase_name not in workflow_info.completed_phases:
            workflow_info.completed_phases.append(phase_name)
            logger.info(f"✅ Marked phase complete: {brand_domain} → {phase_name}")
        
        # Remove from failed phases if it was there
        if phase_name in workflow_info.failed_phases:
            workflow_info.failed_phases.remove(phase_name)
        
        # Update metadata
        if quality_score is not None:
            workflow_info.quality_scores[phase_name] = quality_score
        
        if duration is not None:
            workflow_info.total_research_time += duration
        
        # Update research date for research phases
        if 'research' in phase_name.lower():
            workflow_info.last_research_date = datetime.now()
        
        # Update last updated timestamp
        workflow_info.last_updated = datetime.now()
        
        # Save state
        self._save_brand_state(brand_domain, workflow_info)

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
    
    parser = argparse.ArgumentParser(description="Workflow State Manager")
    parser.add_argument("--brand", required=True, help="Brand URL")
    parser.add_argument("--action", choices=['status', 'next-step', 'update', 'history'], 
                       default='status', help="Action to perform")
    parser.add_argument("--new-state", help="New state for update action")
    parser.add_argument("--step-completed", help="Step that was completed")
    parser.add_argument("--error", help="Error message if step failed")
    parser.add_argument("--json", action='store_true', help="Output as JSON")
    
    args = parser.parse_args()
    
    manager = get_workflow_manager()
    
    if args.action == 'status':
        workflow_progress = manager.get_brand_info(args.brand)
        
        if args.json:
            # Convert to JSON-serializable format
            data = asdict(workflow_progress)
            data['current_state'] = workflow_progress.current_state.value
            data['last_updated'] = workflow_progress.last_updated.isoformat()
            if workflow_progress.current_phase:
                data['current_phase'] = workflow_progress.current_phase
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"\n🔄 WORKFLOW STATUS: {args.brand}")
            print(f"Current State: {workflow_progress.current_state.value}")
            print(f"Progress: {workflow_progress.total_research_time:.1f} minutes")
            print(f"Last Updated: {workflow_progress.last_updated}")
            
            if workflow_progress.current_phase:
                print(f"\n📋 CURRENT PHASE: {workflow_progress.current_phase}")
    
    elif args.action == 'next-step':
        command = manager.get_next_step(args.brand)
        print(command)
    
    elif args.action == 'update':
        if not args.new_state:
            print("Error: --new-state required for update action")
            return
        
        new_state = WorkflowState(args.new_state)
        manager.update_brand_state(
            args.brand, new_state, args.step_completed, args.error
        )
        
        print(f"Updated {args.brand} to state: {new_state.value}")
        if args.step_completed:
            print(f"Current phase: {args.step_completed}")
    
    elif args.action == 'history':
        workflow_progress = manager.get_brand_info(args.brand)
        
        print(f"\n📋 WORKFLOW HISTORY: {args.brand}")
        for phase in workflow_progress.completed_phases[-10:]:  # Last 10 completed phases
            print(f"{phase}: ✅ Completed")
        for phase in workflow_progress.failed_phases[-10:]:  # Last 10 failed phases
            print(f"{phase}: ❌ Failed")

if __name__ == "__main__":
    asyncio.run(main()) 