import logging
from typing import Dict, Any, Optional
from langfuse import get_client
from langfuse.types import PromptClient
 
 
logger = logging.getLogger("prompt-manager")

class MockPromptClient:
    """Mock prompt client for when Langfuse is unavailable"""
    def __init__(self, name: str, prompt: str):
        self.name = name
        self.prompt = prompt
        self.version = 1
        self.type = "text"

class PromptManager:
    _instance = None
    """
    Manages system prompts using Langfuse.
    """
    
    @staticmethod
    def get_prompt_manager() -> "PromptManager":
        """Get the PromptManager instance"""
        if not PromptManager._instance:
            PromptManager._instance = PromptManager()
        return PromptManager._instance
    
    def __init__(self):
        """
        Initialize the PromptManager.
        
        """
        try:
            self.langfuse = get_client()
            logger.info("PromptManager initialized with Langfuse client")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse client: {e}")
            self.langfuse = None

        # Load base prompts
        self._load_base_prompts()

    def _load_base_prompts(self):
        """Load base prompts - implementation for initialization"""
        # This method is called during initialization
        # For now, it's a placeholder that allows initialization to complete
        logger.info("PromptManager initialized successfully")

    async def get_prompt(self, prompt_name: str, default_prompt: str=None) -> Optional[PromptClient]:
        """Get a prompt from Langfuse, creating it if it doesn't exist"""
        
        # if the prompt_name does not contain a /, add it to the internal namespace
        if "/" not in prompt_name:
            prompt_name = f"internal/{prompt_name}"
        
        if not self.langfuse:
            logger.info(f"Langfuse unavailable, using fallback prompt for: {prompt_name}")
            return MockPromptClient(prompt_name, default_prompt)
        
        try:
            # Try to get existing prompt
            prompt = self.langfuse.get_prompt(prompt_name)
            
            if prompt:
                logger.info(f"Retrieved existing prompt from Langfuse: {prompt_name}")
                return prompt
        except Exception as e:
            logger.warning(f"Error retrieving prompt from Langfuse: {e}")
            
        # Create new prompt if it doesn't exist and we have default content
        if default_prompt:
            try:
                logger.info(f"Creating new prompt in Langfuse: {prompt_name}")
                prompt = self.langfuse.create_prompt(
                    name=prompt_name, 
                    prompt=default_prompt, 
                    type="text", 
                    labels=["brand_research", "auto_generated", "production"]
                )
                
                if prompt:
                    logger.info(f"Successfully created prompt in Langfuse: {prompt_name}")
                    return prompt
                else:
                    logger.warning(f"Failed to create prompt in Langfuse, using fallback: {prompt_name}")
                    return MockPromptClient(prompt_name, default_prompt)
                    
            except Exception as e:
                logger.warning(f"Error creating prompt in Langfuse ({e}), using fallback: {prompt_name}")
                return MockPromptClient(prompt_name, default_prompt)
        
        # Always return fallback if no default prompt
        else:
            logger.error(f"No default prompt provided for {prompt_name}")
            return None
            
    def get_langfuse_status(self) -> Dict[str, Any]:
        """Get the current status of Langfuse integration"""
        return {
            "available": self.langfuse is not None,
            "client_initialized": self.langfuse is not None,
            "mode": "langfuse" if self.langfuse is not None else "fallback"
        }
        