#!/usr/bin/env python3
"""
Enhanced Brand Research CLI
Orchestrates the complete 8-phase brand intelligence pipeline per ROADMAP Section 4.2

Usage:
    python scripts/brand_researcher.py --brand specialized.com --phase foundation_research
    python scripts/brand_researcher.py --brand specialized.com --phase all  
    python scripts/brand_researcher.py --brand specialized.com --auto-continue
    python scripts/brand_researcher.py --brand flexfits.com --auto-continue
    python scripts/brand_researcher.py --brand darakayejewelry.com --auto-continue
    python scripts/brand_researcher.py --brand specialized.com --force-regenerate
"""

import asyncio
import argparse
import logging
import sys
import time
from typing import Dict, Any, Optional

# Import all research phase implementations
from src.research.base_researcher import BaseResearcher
from src.research.foundation_research import get_foundation_researcher
from src.research.market_positioning_research import get_market_positioning_researcher
from src.research.product_style_research import get_product_style_researcher
from src.research.customer_cultural_research import get_customer_cultural_researcher
from src.research.voice_messaging_research import get_voice_messaging_researcher
from src.research.interview_synthesis_research import get_interview_synthesis_researcher
from src.research.linearity_analysis_research import get_linearity_analysis_researcher
from src.research.research_integration import get_research_integration_processor

# Workflow management - import WorkflowState directly
from src.workflow.workflow_state_manager import get_workflow_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedBrandResearcher:
    """Enhanced brand researcher supporting all 8 research phases"""
    
    def __init__(self, brand_domain: str):
        self.workflow_manager = get_workflow_manager()
        self.brand_domain = brand_domain.lower()
        
        # Map phase names to researcher instances
        self.researchers: Dict[str, BaseResearcher] = {
            "foundation": get_foundation_researcher(brand_domain=self.brand_domain),
            "market_positioning": get_market_positioning_researcher(brand_domain=self.brand_domain),
            "product_style": get_product_style_researcher(brand_domain=self.brand_domain),
            "customer_cultural": get_customer_cultural_researcher(brand_domain=self.brand_domain),
            "voice_messaging": get_voice_messaging_researcher(brand_domain=self.brand_domain),
            "interview_synthesis": get_interview_synthesis_researcher(brand_domain=self.brand_domain),
            "linearity_analysis": get_linearity_analysis_researcher(brand_domain=self.brand_domain),
            "research_integration": get_research_integration_processor(brand_domain=self.brand_domain)
        }
        
        # # Map phase names to researcher methods
        # self.research_methods = {
        #     "foundation": "research",
        #     "market_positioning": "research",
        #     "product_style": "research",
        #     "customer_cultural": "research",
        #     "voice_messaging": "research",
        #     "interview_synthesis": "research",
        #     "linearity_analysis": "research",
        #     "research_integration": "research"
        # }
    
    async def run_research_phase(self, brand_domain: str, phase_key: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Run a specific research phase"""
        
        if phase_key not in self.researchers:
            raise ValueError(f"Unknown research phase: {phase_key}. Available phases: {list(self.researchers.keys())}")
        
        logger.info(f"🔬 Starting {phase_key} research for {brand_domain}")
        
        try:
            # Get researcher and method
            researcher = self.researchers[phase_key]
            # method_name = self.research_methods[phase_key]
            # research_method = getattr(researcher, method_name)
            
            # Run the research
            start_time = time.time()
            # result = await research_method(force_refresh=force_refresh)
            result = await researcher.research(force_refresh=force_refresh)
            duration = time.time() - start_time
            
            # Update workflow state to complete
            quality_score = result.get("quality_score", 0.75)
            
            logger.info(f"✅ Completed {phase_key} research for {brand_domain} in {duration:.1f}s (quality: {quality_score:.2f})")
            
            return {
                "phase": phase_key,
                "brand": brand_domain,
                "success": True,
                "duration": duration,
                "quality_score": quality_score,
                "files": result.get("files", []),
                "data_sources": result.get("data_sources", 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed {phase_key} research for {brand_domain}: {e}")
            raise
    
    async def run_all_research_phases(self, brand_domain: str, force_refresh: bool = False, 
                                    continue_from_current: bool = False) -> Dict[str, Any]:
        """Run all research phases in sequence"""
        
        logger.info(f"🚀 Starting complete brand research pipeline for {brand_domain}")
        
        # Get phase order
        required_phases = await self.workflow_manager.get_required_research_phases()
        phase_order = list(required_phases.keys())
        
        results = {}
        start_time = time.time()
        
        for phase_key in phase_order:
            try:
                # Check if phase is already complete (unless force refresh)
                if not force_refresh and continue_from_current:
                    phase_status = await self.workflow_manager.get_research_phase_status(brand_domain, phase_key)
                    if phase_status == "complete":
                        logger.info(f"⏭️ Skipping {phase_key} - already complete")
                        results[phase_key] = {"phase": phase_key, "success": True, "skipped": True}
                        continue
                
                # Run the research phase
                phase_result = await self.run_research_phase(brand_domain, phase_key, force_refresh)
                results[phase_key] = phase_result
                
                # Small delay between phases
                await asyncio.sleep(1)
                
            except Exception as e:
                # if the phase is optional, skip it
                if required_phases[phase_key].get("optional"):
                    logger.info(f"❌ Pipeline failed at {phase_key}: {e}")
                    logger.info(f"⏭️ Skipping {phase_key} - optional phase")
                    results[phase_key] = {"phase": phase_key, "success": False, "skipped": True, "error": str(e)}
                    continue
                logger.error(f"❌ Pipeline failed at {phase_key}: {e}")
                results[phase_key] = {"phase": phase_key, "success": False, "error": str(e)}
                raise
        
        total_duration = time.time() - start_time
        
        # Get final research progress summary
        progress_summary = await self.workflow_manager.get_research_progress_summary(brand_domain)
        
        logger.info(f"🎉 Complete brand research pipeline finished in {total_duration:.1f}s")
        logger.info(f"📊 Research Progress: {progress_summary['completed_count']}/{progress_summary['total_phases']} phases complete ({progress_summary['completion_percentage']:.1f}%)")
        
        return {
            "brand": brand_domain,
            "success": True,
            "total_duration": total_duration,
            "phase_results": results,
            "progress_summary": progress_summary,
            "all_phases_complete": progress_summary["all_complete"]
        }
    
    async def continue_research_pipeline(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Continue research pipeline from current state"""
        
        # Check current research progress
        progress_summary = await self.workflow_manager.get_research_progress_summary(brand_domain)
        
        if progress_summary["all_complete"] and not force_refresh:
            logger.info(f"✅ All research phases already complete for {brand_domain}")
            return {
                "brand": brand_domain,
                "success": True,
                "message": "All research phases already complete",
                "progress_summary": progress_summary
            }
        
        # Get next phase to run
        next_phase = progress_summary.get("next_phase")
        if not next_phase:
            logger.info(f"✅ No more research phases needed for {brand_domain}")
            return {
                "brand": brand_domain,
                "success": True,
                "message": "No more research phases needed",
                "progress_summary": progress_summary
            }
        
        logger.info(f"🔄 Continuing research pipeline from {next_phase} for {brand_domain}")
        
        # Run remaining phases
        remaining_phases = progress_summary["missing_phases"]
        required_phases = self.workflow_manager.get_required_research_phases()
        results = {}
        start_time = time.time()
        
        for phase_key in remaining_phases:
            try:
                phase_result = await self.run_research_phase(brand_domain, phase_key, force_refresh)
                results[phase_key] = phase_result
                await asyncio.sleep(1)
            except Exception as e:
                # if the phase is optional, skip it
                if required_phases[phase_key].get("optional"):
                    logger.info(f"⏭️ Skipping {phase_key} - optional phase")
                    results[phase_key] = {"phase": phase_key, "success": False, "skipped": True, "error": str(e)}
                    continue
                logger.error(f"❌ Pipeline continuation failed at {phase_key}: {e}")
                results[phase_key] = {"phase": phase_key, "success": False, "error": str(e)}
                raise
        
        total_duration = time.time() - start_time
        final_progress = await self.workflow_manager.get_research_progress_summary(brand_domain)
        
        logger.info(f"🎉 Research pipeline continuation completed in {total_duration:.1f}s")
        
        return {
            "brand": brand_domain,
            "success": True,
            "total_duration": total_duration,
            "phase_results": results,
            "progress_summary": final_progress,
            "all_phases_complete": final_progress["all_complete"]
        }
    
    def get_available_phases(self) -> Dict[str, str]:
        """Get available research phases"""
        required_phases = self.workflow_manager.get_required_research_phases()
        return {phase_key: phase_config["name"] for phase_key, phase_config in required_phases.items()}
    
    async def get_brand_status(self, brand_domain: str) -> Dict[str, Any]:
        """Get comprehensive brand research status"""
        progress_summary = await self.workflow_manager.get_research_progress_summary(brand_domain)
        current_state = await self.workflow_manager.get_brand_state(brand_domain)
        
        # Get phase-by-phase status
        phase_statuses = {}
        for phase_key in self.workflow_manager.get_required_research_phases().keys():
            phase_statuses[phase_key] = await self.workflow_manager.get_research_phase_status(brand_domain, phase_key)
        
        return {
            "brand": brand_domain,
            "current_state": current_state.value,
            "progress_summary": progress_summary,
            "phase_statuses": phase_statuses,
            "next_step": await self.workflow_manager.get_next_step(brand_domain)
        }


async def main():
    parser = argparse.ArgumentParser(description="Enhanced Brand Research CLI")
    parser.add_argument("--brand", required=True, help="Brand domain (e.g., specialized.com)")
    parser.add_argument("--phase", help="Specific research phase to run (use 'all' for complete pipeline)")
    parser.add_argument("--auto-continue", action="store_true", help="Continue from current research state")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate even if cached")
    parser.add_argument("--status", action="store_true", help="Show research status for brand")
    parser.add_argument("--list-phases", action="store_true", help="List available research phases")
    
    args = parser.parse_args()
    
    researcher = EnhancedBrandResearcher(brand_domain=args.brand)
    
    try:
        if args.list_phases:
            # List available phases
            phases = researcher.get_available_phases()
            print("\n📋 Available Research Phases:")
            for phase_key, phase_name in phases.items():
                print(f"  • {phase_key}: {phase_name}")
            print(f"\n🔄 Special commands:")
            print(f"  • all: Run complete 8-phase pipeline")
            print(f"  • --auto-continue: Continue from current state")
            return
        
        if args.status:
            # Show brand status
            status = await researcher.get_brand_status(args.brand)
            print(f"\n📊 Brand Research Status: {args.brand}")
            print(f"Current State: {status['current_state']}")
            print(f"Progress: {status['progress_summary']['completed_count']}/{status['progress_summary']['total_phases']} phases ({status['progress_summary']['completion_percentage']:.1f}%)")
            print(f"Next Step: {status['next_step']}")
            
            print(f"\n📋 Phase Status:")
            for phase_key, phase_status in status['phase_statuses'].items():
                status_icon = {"complete": "✅", "in_progress": "🔄", "not_started": "⏸️", "blocked": "🚫"}.get(phase_status, "❓")
                print(f"  {status_icon} {phase_key}: {phase_status}")
            return
        
        if args.auto_continue:
            # Continue from current state
            result = await researcher.continue_research_pipeline(args.brand, args.force_regenerate)
            if result["success"]:
                print(f"\n✅ Research pipeline continuation completed!")
                if result.get("all_phases_complete"):
                    print(f"🎉 All research phases complete for {args.brand}! Ready for catalog ingestion.")
            else:
                print(f"\n❌ Research pipeline continuation failed!")
            sys.exit(0)
            
        elif args.phase == "all":
            # Run complete pipeline
            result = await researcher.run_all_research_phases(args.brand, args.force_regenerate, continue_from_current=True)
            if result["success"]:
                print(f"\n✅ Complete research pipeline finished!")
                if result.get("all_phases_complete"):
                    print(f"🎉 All research phases complete for {args.brand}! Ready for catalog ingestion.")
            else:
                print(f"\n❌ Research pipeline failed!")
            sys.exit(1)
            
        elif args.phase:
            # Run specific phase
            if args.phase not in researcher.get_available_phases():
                print(f"❌ Unknown phase: {args.phase}")
                print(f"Available phases: {list(researcher.get_available_phases().keys())}")
                sys.exit(1)
            
            result = await researcher.run_research_phase(args.brand, args.phase, args.force_regenerate)
            if result["success"]:
                print(f"\n✅ {args.phase} research completed for {args.brand}!")
                print(f"Duration: {result['duration']:.1f}s, Quality: {result['quality_score']:.2f}")
            else:
                print(f"\n❌ {args.phase} research failed!")
            sys.exit(1)
        
        else:
            print("❌ Please specify --phase, --auto-continue, --status, or --list-phases")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        logger.error(f"Research failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 