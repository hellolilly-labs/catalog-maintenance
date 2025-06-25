"""
Fallback Prompt Manager for testing without Langfuse
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("prompt-manager-fallback")

class MockPromptClient:
    """Mock prompt client for testing"""
    def __init__(self, name: str, prompt: str):
        self.name = name
        self.prompt = prompt

class PromptManager:
    """Fallback prompt manager that works without Langfuse"""
    _instance = None
    
    @staticmethod
    def get_prompt_manager() -> "PromptManager":
        """Get the PromptManager instance"""
        if not PromptManager._instance:
            PromptManager._instance = PromptManager()
        return PromptManager._instance
    
    def __init__(self):
        """Initialize the fallback PromptManager"""
        self.prompts = {}
        logger.info("Using fallback prompt manager (no Langfuse)")
    
    async def get_prompt(self, prompt_name: str, default_prompt: str = None) -> Optional[MockPromptClient]:
        """Get a prompt (creates mock version with default_prompt)"""
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        
        if default_prompt:
            mock_prompt = MockPromptClient(prompt_name, default_prompt)
            self.prompts[prompt_name] = mock_prompt
            logger.info(f"Created mock prompt: {prompt_name}")
            return mock_prompt
        
        return None
