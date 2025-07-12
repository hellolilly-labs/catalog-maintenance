"""
Quality Evaluation Storage Manager
Handles persistent storage of quality evaluations and feedback
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from liddy.storage import get_account_storage_provider

logger = logging.getLogger(__name__)

class QualityStorageManager:
    """Manages storage of quality evaluation results and feedback"""
    
    def __init__(self, storage_manager=None):
        self.storage_manager = storage_manager or get_account_storage_provider()
    
    async def store_quality_evaluation(self, evaluation) -> str:
        """Store quality evaluation results"""
        
        try:
            # Convert evaluation to dict
            evaluation_data = evaluation.to_dict()
            
            # Create timestamp-based filename
            timestamp = evaluation.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{evaluation.phase_name}_quality_{timestamp}.json"
            
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                quality_dir = os.path.join(
                    self.storage_manager.base_dir, 
                    "accounts", 
                    evaluation.brand_domain, 
                    "research_quality"
                )
                os.makedirs(quality_dir, exist_ok=True)
                
                file_path = os.path.join(quality_dir, filename)
                with open(file_path, "w") as f:
                    json.dump(evaluation_data, f, indent=2)
                
                # Update quality history index
                await self._update_quality_index(evaluation, file_path)
                
                logger.info(f"✅ Stored quality evaluation: {file_path}")
                return file_path
                
            else:
                # GCP storage
                quality_blob_path = f"accounts/{evaluation.brand_domain}/research_quality/{filename}"
                blob = self.storage_manager.bucket.blob(quality_blob_path)
                blob.upload_from_string(json.dumps(evaluation_data, indent=2))
                
                # Update quality history index
                await self._update_quality_index(evaluation, quality_blob_path)
                
                logger.info(f"✅ Stored quality evaluation: {quality_blob_path}")
                return quality_blob_path
                
        except Exception as e:
            logger.error(f"❌ Error storing quality evaluation: {e}")
            raise
    
    async def _update_quality_index(self, evaluation, file_path: str):
        """Update the quality evaluation index for efficient retrieval"""
        
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                index_path = os.path.join(
                    self.storage_manager.base_dir,
                    "accounts", 
                    evaluation.brand_domain,
                    "research_quality",
                    "quality_index.json"
                )
                
                # Load existing index
                if os.path.exists(index_path):
                    with open(index_path, "r") as f:
                        index_data = json.load(f)
                else:
                    index_data = {"evaluations": [], "last_updated": None}
                
                # Add new evaluation to index
                index_entry = {
                    "phase_name": evaluation.phase_name,
                    "timestamp": evaluation.timestamp.isoformat(),
                    "quality_score": evaluation.quality_score,
                    "passes_threshold": evaluation.passes_threshold,
                    "file_path": file_path,
                    "evaluator_model": evaluation.evaluator_model
                }
                
                index_data["evaluations"].append(index_entry)
                index_data["last_updated"] = datetime.now().isoformat()
                
                # Keep only last 50 evaluations per brand
                index_data["evaluations"] = sorted(
                    index_data["evaluations"], 
                    key=lambda x: x["timestamp"], 
                    reverse=True
                )[:50]
                
                # Save updated index
                with open(index_path, "w") as f:
                    json.dump(index_data, f, indent=2)
                
            else:
                # GCP storage
                index_blob_path = f"accounts/{evaluation.brand_domain}/research_quality/quality_index.json"
                index_blob = self.storage_manager.bucket.blob(index_blob_path)
                
                # Load existing index
                if index_blob.exists():
                    index_content = index_blob.download_as_text()
                    index_data = json.loads(index_content)
                else:
                    index_data = {"evaluations": [], "last_updated": None}
                
                # Add new evaluation to index
                index_entry = {
                    "phase_name": evaluation.phase_name,
                    "timestamp": evaluation.timestamp.isoformat(),
                    "quality_score": evaluation.quality_score,
                    "passes_threshold": evaluation.passes_threshold,
                    "file_path": file_path,
                    "evaluator_model": evaluation.evaluator_model
                }
                
                index_data["evaluations"].append(index_entry)
                index_data["last_updated"] = datetime.now().isoformat()
                
                # Keep only last 50 evaluations per brand
                index_data["evaluations"] = sorted(
                    index_data["evaluations"],
                    key=lambda x: x["timestamp"],
                    reverse=True
                )[:50]
                
                # Save updated index
                index_blob.upload_from_string(json.dumps(index_data, indent=2))
                
        except Exception as e:
            logger.warning(f"Failed to update quality index: {e}")
    
    async def get_quality_history(
        self, 
        brand_domain: str, 
        phase_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get quality evaluation history for a brand/phase"""
        
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                index_path = os.path.join(
                    self.storage_manager.base_dir,
                    "accounts",
                    brand_domain,
                    "research_quality", 
                    "quality_index.json"
                )
                
                if not os.path.exists(index_path):
                    return []
                
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                
            else:
                # GCP storage
                index_blob_path = f"accounts/{brand_domain}/research_quality/quality_index.json"
                index_blob = self.storage_manager.bucket.blob(index_blob_path)
                
                if not index_blob.exists():
                    return []
                
                index_content = index_blob.download_as_text()
                index_data = json.loads(index_content)
            
            # Filter by phase if specified
            evaluations = index_data.get("evaluations", [])
            if phase_name:
                evaluations = [e for e in evaluations if e["phase_name"] == phase_name]
            
            # Return limited results
            return evaluations[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving quality history: {e}")
            return []
    
    async def get_latest_evaluation(
        self, 
        brand_domain: str, 
        phase_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent quality evaluation for a phase"""
        
        history = await self.get_quality_history(brand_domain, phase_name, limit=1)
        return history[0] if history else None
    
    async def store_improvement_context(
        self, 
        brand_domain: str, 
        phase_name: str, 
        improvement_feedback: List[str],
        quality_score: float
    ) -> str:
        """Store improvement context for next research attempt"""
        
        try:
            improvement_data = {
                "phase_name": phase_name,
                "brand_domain": brand_domain,
                "improvement_feedback": improvement_feedback,
                "quality_score": quality_score,
                "created_at": datetime.now().isoformat(),
                "used": False
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{phase_name}_improvement_context_{timestamp}.json"
            
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                context_dir = os.path.join(
                    self.storage_manager.base_dir,
                    "accounts",
                    brand_domain,
                    "research_quality",
                    "improvement_context"
                )
                os.makedirs(context_dir, exist_ok=True)
                
                file_path = os.path.join(context_dir, filename)
                with open(file_path, "w") as f:
                    json.dump(improvement_data, f, indent=2)
                
                logger.info(f"✅ Stored improvement context: {file_path}")
                return file_path
                
            else:
                # GCP storage
                context_blob_path = f"accounts/{brand_domain}/research_quality/improvement_context/{filename}"
                blob = self.storage_manager.bucket.blob(context_blob_path)
                blob.upload_from_string(json.dumps(improvement_data, indent=2))
                
                logger.info(f"✅ Stored improvement context: {context_blob_path}")
                return context_blob_path
                
        except Exception as e:
            logger.error(f"❌ Error storing improvement context: {e}")
            raise
    
    async def get_improvement_context(
        self, 
        brand_domain: str, 
        phase_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get unused improvement context for a phase"""
        
        try:
            if hasattr(self.storage_manager, 'base_dir'):
                # Local storage
                context_dir = os.path.join(
                    self.storage_manager.base_dir,
                    "accounts",
                    brand_domain,
                    "research_quality",
                    "improvement_context"
                )
                
                if not os.path.exists(context_dir):
                    return None
                
                # Find latest unused improvement context
                context_files = [
                    f for f in os.listdir(context_dir) 
                    if f.startswith(f"{phase_name}_improvement_context_") and f.endswith(".json")
                ]
                
                if not context_files:
                    return None
                
                # Sort by timestamp (newest first)
                context_files.sort(reverse=True)
                
                for context_file in context_files:
                    context_path = os.path.join(context_dir, context_file)
                    with open(context_path, "r") as f:
                        context_data = json.load(f)
                    
                    if not context_data.get("used", True):
                        # Mark as used
                        context_data["used"] = True
                        context_data["used_at"] = datetime.now().isoformat()
                        
                        with open(context_path, "w") as f:
                            json.dump(context_data, f, indent=2)
                        
                        return context_data
                
                return None
                
            else:
                # GCP storage - simplified for now
                # In production, would implement blob listing and filtering
                logger.warning("GCP improvement context retrieval not fully implemented")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving improvement context: {e}")
            return None
    
    async def get_quality_analytics(self, brand_domain: str) -> Dict[str, Any]:
        """Get quality analytics for a brand"""
        
        try:
            history = await self.get_quality_history(brand_domain, limit=100)
            
            if not history:
                return {"message": "No quality evaluations found"}
            
            # Calculate analytics
            total_evaluations = len(history)
            passed_evaluations = len([e for e in history if e["passes_threshold"]])
            avg_quality_score = sum(e["quality_score"] for e in history) / total_evaluations
            
            # Phase-specific analytics
            phase_analytics = {}
            for evaluation in history:
                phase = evaluation["phase_name"]
                if phase not in phase_analytics:
                    phase_analytics[phase] = {
                        "count": 0,
                        "avg_score": 0.0,
                        "pass_rate": 0.0,
                        "scores": []
                    }
                
                phase_analytics[phase]["count"] += 1
                phase_analytics[phase]["scores"].append(evaluation["quality_score"])
            
            # Calculate phase-specific metrics
            for phase, data in phase_analytics.items():
                scores = data["scores"]
                data["avg_score"] = sum(scores) / len(scores)
                data["pass_rate"] = len([s for s in scores if s >= 8.0]) / len(scores)
                del data["scores"]  # Remove raw scores from output
            
            return {
                "brand_domain": brand_domain,
                "total_evaluations": total_evaluations,
                "overall_pass_rate": passed_evaluations / total_evaluations,
                "avg_quality_score": avg_quality_score,
                "phase_analytics": phase_analytics,
                "latest_evaluation": history[0] if history else None
            }
            
        except Exception as e:
            logger.error(f"Error generating quality analytics: {e}")
            return {"error": str(e)} 