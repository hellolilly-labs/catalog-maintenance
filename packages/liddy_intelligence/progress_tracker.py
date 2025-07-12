# src/progress_tracker.py

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import os
import logging

logger = logging.getLogger(__name__)

class StepStatus(Enum):
    """Execution status for research steps"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"

class StepType(Enum):
    """Types of research steps for all 8 research phases"""
    # Main research phases (8 phases total per ROADMAP)
    FOUNDATION_RESEARCH = "foundation"
    MARKET_POSITIONING = "market_positioning" 
    MARKET_POSITIONING_RESEARCH = "market_positioning"
    PRODUCT_INTELLIGENCE = "product_intelligence"
    BRAND_STYLE = "brand_style"
    PRODUCT_STYLE = "product_style"
    CUSTOMER_CULTURAL = "customer_cultural"
    VOICE_MESSAGING = "voice_messaging"
    INTERVIEW_INTEGRATION = "interview_integration"
    INTERVIEW_SYNTHESIS = "interview_synthesis"
    INDUSTRY_TERMINOLOGY = "industry_terminology"
    LINEARITY_ANALYSIS = "linearity_analysis"
    PRODUCT_CATALOG = "product_catalog"
    RESEARCH_INTEGRATION = "research_integration"
    SYNTHESIS = "synthesis"
    
    # Sub-steps for detailed tracking
    DATA_GATHERING = "data_gathering"
    LLM_ANALYSIS = "llm_analysis"
    SYNTHESIS_GENERATION = "synthesis_generation"
    STORAGE_SAVE = "storage_save"
    QUALITY_VALIDATION = "quality_validation"

@dataclass
class ProgressMetrics:
    """Metrics for step execution"""
    total_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return (self.completed_operations / self.total_operations) * 100
    
    @property
    def duration_seconds(self) -> float:
        if not self.start_time:
            return 0.0
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()
    
    @property
    def estimated_time_remaining(self) -> Optional[timedelta]:
        if not self.start_time or self.completed_operations == 0:
            return None
        
        elapsed = datetime.utcnow() - self.start_time
        ops_per_second = self.completed_operations / elapsed.total_seconds()
        remaining_ops = self.total_operations - self.completed_operations
        
        if ops_per_second > 0:
            return timedelta(seconds=remaining_ops / ops_per_second)
        return None

@dataclass
class Checkpoint:
    """Individual progress checkpoint for audit trail"""
    checkpoint_id: str
    timestamp: datetime
    step_id: str
    step_type: str
    status: str
    progress_percentage: float
    duration_seconds: float
    current_operation: str
    quality_score: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for JSON serialization"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp.isoformat() + "Z",
            "step_id": self.step_id,
            "step_type": self.step_type,
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "duration_seconds": self.duration_seconds,
            "current_operation": self.current_operation,
            "quality_score": self.quality_score,
            "error_message": self.error_message,
            "warnings": self.warnings
        }

@dataclass
class StepProgress:
    """Progress tracking for individual research step"""
    step_id: str
    step_type: StepType
    status: StepStatus
    brand: str
    phase_name: str
    
    # Progress metrics
    metrics: ProgressMetrics
    
    # Status details
    current_operation: str = ""
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    # Results
    output_files: List[str] = None
    cache_hit: bool = False
    quality_score: Optional[float] = None
    
    # Metadata
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.output_files is None:
            self.output_files = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

class ProgressTracker:
    """Real-time progress tracking for research pipeline with persistent checkpoints"""
    
    def __init__(self, storage_manager=None, enable_checkpoints: bool = True):
        self.storage_manager = storage_manager
        self.enable_checkpoints = enable_checkpoints
        self._steps: Dict[str, StepProgress] = {}
        self._listeners: List[Callable[[StepProgress], None]] = []
        self._lock = threading.Lock()
        self._active_step_id: Optional[str] = None
        self._research_session_id: str = str(uuid.uuid4())
        self._checkpoints: List[Checkpoint] = []
    
    async def _create_checkpoint(self, step: StepProgress, operation: str):
        """Create a progress checkpoint for audit trail and immediately persist to disk"""
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            step_id=step.step_id,
            step_type=step.step_type.value,
            status=step.status.value,
            progress_percentage=step.metrics.progress_percentage,
            duration_seconds=step.metrics.duration_seconds,
            current_operation=operation,
            quality_score=step.quality_score,
            error_message=step.error_message,
            warnings=step.warnings.copy()
        )
        
        with self._lock:
            self._checkpoints.append(checkpoint)
        
        logger.debug(f"📊 Checkpoint: {step.phase_name} - {operation} ({step.metrics.progress_percentage:.1f}%)")
        
        # 🔥 IMMEDIATELY PERSIST TO DISK - File-System-as-State!
        await self._save_progress_log(step.brand, step.step_type.value)
    
    async def create_step(
        self, 
        step_type: StepType, 
        brand: str, 
        phase_name: str,
        total_operations: int = 1
    ) -> str:
        """Create a new progress tracking step"""
        step_id = str(uuid.uuid4())
        
        with self._lock:
            step = StepProgress(
                step_id=step_id,
                step_type=step_type,
                status=StepStatus.PENDING,
                brand=brand,
                phase_name=phase_name,
                metrics=ProgressMetrics(total_operations=total_operations)
            )
            self._steps[step_id] = step
            self._active_step_id = step_id
        
        # Create initial checkpoint (now persisted immediately)
        if self.enable_checkpoints:
            await self._create_checkpoint(step, "Step created")
        
        self._notify_listeners(step)
        return step_id
    
    async def start_step(self, step_id: str, current_operation: str = ""):
        """Mark step as started"""
        with self._lock:
            if step_id not in self._steps:
                raise ValueError(f"Step {step_id} not found")
            
            step = self._steps[step_id]
            step.status = StepStatus.RUNNING
            step.current_operation = current_operation
            step.metrics.start_time = datetime.utcnow()
            step.updated_at = datetime.utcnow()
            self._active_step_id = step_id
        
        # Create checkpoint for step start (now persisted immediately)
        if self.enable_checkpoints:
            await self._create_checkpoint(step, "Step started")
        
        self._notify_listeners(step)
    
    async def update_progress(
        self, 
        step_id: str, 
        completed_operations: int = None,
        current_operation: str = None,
        increment: int = 1
    ):
        """Update step progress"""
        with self._lock:
            if step_id not in self._steps:
                raise ValueError(f"Step {step_id} not found")
            
            step = self._steps[step_id]
            
            if completed_operations is not None:
                step.metrics.completed_operations = completed_operations
            else:
                step.metrics.completed_operations += increment
            
            if current_operation:
                step.current_operation = current_operation
            
            step.updated_at = datetime.utcnow()
            
            # Update estimated completion
            if step.metrics.estimated_time_remaining:
                step.metrics.estimated_completion = (
                    datetime.utcnow() + step.metrics.estimated_time_remaining
                )
        
        # Create checkpoint for progress updates (every 10% or significant operations)
        # NOW PERSISTED IMMEDIATELY TO DISK!
        if self.enable_checkpoints and (
            step.metrics.progress_percentage % 10 < 5 or  # Every ~10%
            current_operation  # Significant operation changes
        ):
            await self._create_checkpoint(step, "Progress update")
        
        self._notify_listeners(step)
    
    async def complete_step(
        self, 
        step_id: str, 
        output_files: List[str] = None,
        quality_score: float = None,
        cache_hit: bool = False
    ):
        """Mark step as completed"""
        with self._lock:
            if step_id not in self._steps:
                raise ValueError(f"Step {step_id} not found")
            
            step = self._steps[step_id]
            step.status = StepStatus.CACHED if cache_hit else StepStatus.COMPLETED
            step.metrics.end_time = datetime.utcnow()
            step.metrics.completed_operations = step.metrics.total_operations
            step.current_operation = "Completed"
            step.cache_hit = cache_hit
            step.updated_at = datetime.utcnow()
            
            if output_files:
                step.output_files.extend(output_files)
            if quality_score is not None:
                step.quality_score = quality_score
        
        # Create completion checkpoint (already persisted via _create_checkpoint)
        if self.enable_checkpoints:
            await self._create_checkpoint(step, "Step completed")
        
        self._notify_listeners(step)
    
    async def fail_step(self, step_id: str, error_message: str):
        """Mark step as failed"""
        with self._lock:
            if step_id not in self._steps:
                raise ValueError(f"Step {step_id} not found")
            
            step = self._steps[step_id]
            step.status = StepStatus.FAILED
            step.error_message = error_message
            step.metrics.end_time = datetime.utcnow()
            step.updated_at = datetime.utcnow()
        
        # Create failure checkpoint (already persisted via _create_checkpoint)
        if self.enable_checkpoints:
            await self._create_checkpoint(step, "Step failed")
        
        self._notify_listeners(step)
    
    async def add_warning(self, step_id: str, warning: str):
        """Add warning to step"""
        with self._lock:
            if step_id not in self._steps:
                raise ValueError(f"Step {step_id} not found")
            
            step = self._steps[step_id]
            step.warnings.append(warning)
            step.updated_at = datetime.utcnow()
        
        # Create checkpoint for warnings (important for debugging) - NOW PERSISTED IMMEDIATELY
        if self.enable_checkpoints:
            await self._create_checkpoint(step, f"Warning: {warning}")
        
        self._notify_listeners(step)
    
    def load_progress_log(self, brand: str, phase: str) -> Optional[Dict[str, Any]]:
        """Load progress checkpoints from persistent storage for analysis"""
        try:
            if hasattr(self.storage_manager, 'load_account_file') and self.storage_manager:
                # Use storage manager if available
                file_path = f"research/{phase}/progress.json"
                content = self.storage_manager.load_account_file(brand, file_path)
                if content:
                    return json.loads(content)
            else:
                # Fallback to local storage
                local_file = f"local/account_storage/accounts/{brand}/research/{phase}/progress.json"
                if os.path.exists(local_file):
                    with open(local_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load progress log for {brand}/{phase}: {e}")
        
        return None
    
    def get_step(self, step_id: str) -> Optional[StepProgress]:
        """Get step progress"""
        with self._lock:
            return self._steps.get(step_id)
    
    def get_active_step(self) -> Optional[StepProgress]:
        """Get currently active step"""
        with self._lock:
            if self._active_step_id:
                return self._steps.get(self._active_step_id)
            return None
    
    def get_brand_steps(self, brand: str) -> List[StepProgress]:
        """Get all steps for a brand"""
        with self._lock:
            return [step for step in self._steps.values() if step.brand == brand]
    
    def get_phase_steps(self, brand: str, phase_name: str) -> List[StepProgress]:
        """Get all steps for a specific phase"""
        with self._lock:
            return [
                step for step in self._steps.values() 
                if step.brand == brand and step.phase_name == phase_name
            ]
    
    def get_progress_checkpoints(self, brand: str = None, phase: str = None) -> List[Checkpoint]:
        """Get progress checkpoints, optionally filtered by brand/phase"""
        with self._lock:
            if not brand and not phase:
                return self._checkpoints.copy()
            
            # Filter checkpoints
            filtered = []
            for checkpoint in self._checkpoints:
                step = self._steps.get(checkpoint.step_id)
                if step:
                    if brand and step.brand != brand:
                        continue
                    if phase and step.step_type.value != phase:
                        continue
                    filtered.append(checkpoint)
            
            return filtered
    
    def add_progress_listener(self, listener: Callable[[StepProgress], None]):
        """Add progress update listener"""
        self._listeners.append(listener)
    
    def remove_progress_listener(self, listener: Callable[[StepProgress], None]):
        """Remove progress update listener"""
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    def _notify_listeners(self, step: StepProgress):
        """Notify all listeners of progress update"""
        for listener in self._listeners:
            try:
                listener(step)
            except Exception as e:
                print(f"Error in progress listener: {e}")
    
    def get_summary_report(self, brand: str) -> Dict[str, Any]:
        """Generate summary report for brand research"""
        steps = self.get_brand_steps(brand)
        
        if not steps:
            return {"brand": brand, "status": "no_steps", "steps": []}
        
        total_duration = sum(step.metrics.duration_seconds for step in steps)
        completed_steps = [s for s in steps if s.status == StepStatus.COMPLETED]
        failed_steps = [s for s in steps if s.status == StepStatus.FAILED]
        cached_steps = [s for s in steps if s.status == StepStatus.CACHED]
        
        return {
            "brand": brand,
            "research_session_id": self._research_session_id,
            "total_steps": len(steps),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "cached_steps": len(cached_steps),
            "total_duration_seconds": total_duration,
            "average_quality_score": self._calculate_average_quality(completed_steps),
            "total_checkpoints": len(self.get_progress_checkpoints(brand=brand)),
            "steps": [
                {
                    "step_type": step.step_type.value,
                    "phase_name": step.phase_name,
                    "status": step.status.value,
                    "duration_seconds": step.metrics.duration_seconds,
                    "progress_percentage": step.metrics.progress_percentage,
                    "quality_score": step.quality_score,
                    "cache_hit": step.cache_hit,
                    "output_files": step.output_files,
                    "error_message": step.error_message,
                    "warnings": step.warnings
                }
                for step in sorted(steps, key=lambda x: x.created_at)
            ]
        }
    
    def _calculate_average_quality(self, steps: List[StepProgress]) -> Optional[float]:
        """Calculate average quality score"""
        scores = [s.quality_score for s in steps if s.quality_score is not None]
        return sum(scores) / len(scores) if scores else None
    
    def print_live_status(self, brand: str = None):
        """Print current live status to console"""
        if brand:
            steps = self.get_brand_steps(brand)
        else:
            steps = list(self._steps.values())
        
        print(f"\n🔄 **LIVE RESEARCH STATUS** {'- ' + brand if brand else ''}")
        print("=" * 60)
        
        active_step = self.get_active_step()
        if active_step:
            print(f"🟢 **ACTIVE**: {active_step.phase_name}")
            print(f"   Operation: {active_step.current_operation}")
            print(f"   Progress: {active_step.metrics.progress_percentage:.1f}%")
            if active_step.metrics.estimated_time_remaining:
                eta = active_step.metrics.estimated_time_remaining
                print(f"   ETA: {eta.total_seconds():.0f}s remaining")
            print()
        
        phase_groups = {}
        for step in steps:
            if step.phase_name not in phase_groups:
                phase_groups[step.phase_name] = []
            phase_groups[step.phase_name].append(step)
        
        for phase_name, phase_steps in phase_groups.items():
            latest_step = max(phase_steps, key=lambda x: x.updated_at)
            status_icon = {
                StepStatus.PENDING: "⏳",
                StepStatus.RUNNING: "🟢",
                StepStatus.COMPLETED: "✅",
                StepStatus.CACHED: "💾",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }.get(latest_step.status, "❓")
            
            duration = f"({latest_step.metrics.duration_seconds:.1f}s)" if latest_step.metrics.duration_seconds > 0 else ""
            
            print(f"{status_icon} {phase_name} {duration}")
            if latest_step.status == StepStatus.RUNNING:
                print(f"    {latest_step.current_operation}")
            elif latest_step.status == StepStatus.FAILED:
                print(f"    Error: {latest_step.error_message}")
            elif latest_step.quality_score:
                print(f"    Quality: {latest_step.quality_score:.2f}")
        
        # Show checkpoint summary if enabled
        if self.enable_checkpoints and brand:
            checkpoints = self.get_progress_checkpoints(brand=brand)
            if checkpoints:
                print(f"\n📊 **CHECKPOINT SUMMARY**: {len(checkpoints)} checkpoints recorded")
    
    async def _save_progress_log(self, brand: str, phase: str):
        """Save progress checkpoints to persistent storage"""
        if not self.enable_checkpoints or not self.storage_manager:
            return
            
        try:
            # Filter checkpoints for this brand/phase
            relevant_checkpoints = [
                cp for cp in self._checkpoints 
                if any(step.brand == brand and step.step_type.value == phase 
                      for step in self._steps.values() 
                      if step.step_id == cp.step_id)
            ]
            
            if not relevant_checkpoints:
                return
            
            # Create progress log document
            progress_log = {
                "research_session_id": self._research_session_id,
                "brand_domain": brand,
                "phase": phase,
                "session_start": min(cp.timestamp for cp in relevant_checkpoints).isoformat() + "Z",
                "session_end": max(cp.timestamp for cp in relevant_checkpoints).isoformat() + "Z",
                "total_checkpoints": len(relevant_checkpoints),
                "final_status": relevant_checkpoints[-1].status if relevant_checkpoints else "unknown",
                "total_duration_seconds": max(cp.duration_seconds for cp in relevant_checkpoints) if relevant_checkpoints else 0,
                "quality_score": relevant_checkpoints[-1].quality_score if relevant_checkpoints else None,
                "checkpoints": [cp.to_dict() for cp in relevant_checkpoints]
            }
            
            # Save to storage using storage manager
            success = await self.storage_manager.write_file(
                account=brand,
                file_path=f"research/{phase}/progress.json",
                content=json.dumps(progress_log, indent=2),
                content_type="application/json"
            )
            
            if success:
                logger.debug(f"💾 Saved progress log: {brand}/research/{phase}/progress.json")
            else:
                logger.warning(f"⚠️ Failed to save progress log for {brand}/{phase}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save progress log for {brand}/{phase}: {e}")

# Global progress tracker instance
_global_tracker: Optional[ProgressTracker] = None

def get_progress_tracker() -> ProgressTracker:
    """Get global progress tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker

def create_console_listener() -> Callable[[StepProgress], None]:
    """Create a console progress listener"""
    def console_listener(step: StepProgress):
        timestamp = step.updated_at.strftime("%H:%M:%S")
        status_icon = {
            StepStatus.PENDING: "⏳",
            StepStatus.RUNNING: "🟢", 
            StepStatus.COMPLETED: "✅",
            StepStatus.CACHED: "💾",
            StepStatus.FAILED: "❌",
            StepStatus.SKIPPED: "⏭️"
        }.get(step.status, "❓")
        
        if step.status == StepStatus.RUNNING:
            progress = f"({step.metrics.progress_percentage:.1f}%)"
            print(f"[{timestamp}] {status_icon} {step.phase_name} {progress} - {step.current_operation}")
        elif step.status == StepStatus.COMPLETED:
            duration = f"({step.metrics.duration_seconds:.1f}s)"
            quality = f"Q:{step.quality_score:.2f}" if step.quality_score else ""
            cache_note = "CACHED" if step.cache_hit else ""
            print(f"[{timestamp}] {status_icon} {step.phase_name} {duration} {quality} {cache_note}")
        elif step.status == StepStatus.FAILED:
            print(f"[{timestamp}] {status_icon} {step.phase_name} FAILED - {step.error_message}")
    
    return console_listener 