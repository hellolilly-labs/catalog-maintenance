#!/usr/bin/env python3

"""
Research Phase Tracker
Tracks the 6-8 research phases defined in ROADMAP Section 4.2
"""

from typing import Dict, List, Optional, Any

class ResearchPhaseTracker:
    """Tracks detailed research phases per ROADMAP Section 4.2"""
    
    def __init__(self, workflow_manager=None):
        self.workflow_manager = workflow_manager
        
    def get_required_research_phases(self) -> Dict[str, Dict[str, Any]]:
        """Get the complete list of required research phases per ROADMAP Section 4.2"""
        return {
            "foundation": {
                "name": "Foundation Research",
                "cache_duration_days": 180,  # 6 months
                "research_time_minutes": "3-5",
                "quality_threshold": 8.0,
                "description": "Core brand identity that rarely changes"
            },
            "market_positioning": {
                "name": "Market Positioning Research", 
                "cache_duration_days": 120,  # 4 months
                "research_time_minutes": "2-4",
                "quality_threshold": 7.5,
                "description": "Competitive landscape and market position"
            },
            "product_style": {
                "name": "Product & Style Intelligence",
                "cache_duration_days": 60,   # 2 months
                "research_time_minutes": "2-3", 
                "quality_threshold": 8.0,
                "description": "Current products, collections, and style evolution"
            },
            "customer_cultural": {
                "name": "Customer & Cultural Intelligence",
                "cache_duration_days": 90,   # 3 months
                "research_time_minutes": "2-3",
                "quality_threshold": 7.5,
                "description": "Target audience and cultural relevance"
            },
            "voice_messaging": {
                "name": "Voice & Messaging Analysis",
                "cache_duration_days": 30,   # 1 month
                "research_time_minutes": "1-2",
                "quality_threshold": 8.5,
                "description": "Current brand voice and communication style"
            },
            "interview_synthesis": {
                "name": "AI Brand Ethos Voice Interview Synthesis",
                "cache_duration_days": 90,   # Auto-updates when new interviews added
                "research_time_minutes": "3-5",
                "quality_threshold": 9.0,
                "description": "Process direct brand voice from interview transcripts"
            },
            "linearity_analysis": {
                "name": "Linearity Analysis",
                "cache_duration_days": 120,  # 4 months
                "research_time_minutes": "2-4",
                "quality_threshold": 8.0,
                "description": "Brand positioning on linear vs non-linear shopping behavior spectrum"
            },
            "research_integration": {
                "name": "Research Integration", 
                "cache_duration_days": 30,   # Runs when any phase updates
                "research_time_minutes": "1-2",
                "quality_threshold": 8.0,
                "description": "Cross-validate and unify all research phases"
            }
        }
    
    def get_next_research_phase(self, brand_domain: str) -> Optional[str]:
        """Get the next research phase that needs to be completed"""
        if not self.workflow_manager:
            return "foundation"  # Default start
            
        workflow_info = self.workflow_manager.get_brand_info_sync(brand_domain)
        
        # Check each required phase in order
        phase_order = [
            "foundation",
            "market_positioning", 
            "product_style",
            "customer_cultural",
            "voice_messaging",
            "interview_synthesis",
            "linearity_analysis",
            "research_integration"
        ]
        
        for phase_key in phase_order:
            if phase_key not in workflow_info.completed_phases:
                return phase_key
        
        return None  # All phases complete
    
    def are_all_research_phases_complete(self, brand_domain: str) -> bool:
        """Check if all required research phases are complete"""
        return self.get_next_research_phase(brand_domain) is None
    
    def get_research_progress_summary(self, brand_domain: str) -> Dict[str, Any]:
        """Get detailed research progress summary"""
        if not self.workflow_manager:
            return {"error": "No workflow manager available"}
            
        workflow_info = self.workflow_manager.get_brand_info_sync(brand_domain)
        required_phases = self.get_required_research_phases()
        completed_phases = [
            phase for phase in required_phases.keys() 
            if phase in workflow_info.completed_phases
        ]
        next_phase = self.get_next_research_phase(brand_domain)
        
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
    
    def get_next_step_command(self, brand_domain: str) -> str:
        """Get the specific command for the next research phase"""
        next_phase = self.get_next_research_phase(brand_domain)
        
        if not next_phase:
            # All research complete - ready for catalog
            return f"✅ All research complete! Start catalog: python src/product_ingestor.py --full-sync --brand {brand_domain}"
        
        # Map phases to their CLI commands
        phase_commands = {
            "foundation": f"python liddy_intelligence/brand_research/brand_intelligence_pipeline.py --brand {brand_domain} --phase foundation_research",
            "market_positioning": f"python liddy_intelligence/brand_research/brand_intelligence_pipeline.py --brand {brand_domain} --phase market_positioning",
            "product_style": f"python liddy_intelligence/brand_research/brand_intelligence_pipeline.py --brand {brand_domain} --phase product_style",
            "customer_cultural": f"python liddy_intelligence/brand_research/brand_intelligence_pipeline.py --brand {brand_domain} --phase customer_cultural",
            "voice_messaging": f"python liddy_intelligence/brand_research/brand_intelligence_pipeline.py --brand {brand_domain} --phase voice_messaging",
            "interview_synthesis": f"python liddy_intelligence/brand_research/brand_intelligence_pipeline.py --brand {brand_domain} --phase interview_synthesis",
            "linearity_analysis": f"python liddy_intelligence/brand_research/brand_intelligence_pipeline.py --brand {brand_domain} --phase linearity_analysis",
            "research_integration": f"python liddy_intelligence/brand_research/brand_intelligence_pipeline.py --brand {brand_domain} --phase research_integration"
        }
        
        phase_info = self.get_required_research_phases()[next_phase]
        command = phase_commands.get(next_phase, f"⚠️ Research phase '{next_phase}' not implemented yet")
        
        return f"📋 Next: {phase_info['name']} ({phase_info['research_time_minutes']} min) - {command}"

# Global instance
_global_research_tracker: Optional[ResearchPhaseTracker] = None

def get_research_phase_tracker(workflow_manager=None) -> ResearchPhaseTracker:
    """Get global research phase tracker instance"""
    global _global_research_tracker
    if _global_research_tracker is None:
        _global_research_tracker = ResearchPhaseTracker(workflow_manager)
    return _global_research_tracker 