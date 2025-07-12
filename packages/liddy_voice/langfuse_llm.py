"""
Simple Langfuse observability wrappers using @observe() decorator
"""

import time
from typing import Any, Dict, Optional
import logging

from langfuse import observe, get_client

logger = logging.getLogger(__name__)

class LangfuseObservability:
    """
    Simple observability helper using Langfuse @observe decorators
    This follows the recommended Langfuse approach with minimal LiveKit interference
    """
    
    def __init__(self, model_name: str, account: str, user_hash: str = None, session_id: str = None):
        self.model_name = model_name
        self.account = account
        self.user_hash = user_hash
        self.session_id = session_id

    @observe(name="voice_llm_call")
    async def track_llm_call(self, chat_ctx, llm_callable, *args, **kwargs):
        """
        Track any LLM call with automatic Langfuse observability
        """
        
        # Extract input context for observability
        messages = []
        try:
            for item in chat_ctx.items:
                if hasattr(item, 'role'):
                    content = getattr(item, 'text_content', str(item.content))
                    messages.append({"role": item.role, "content": content[:200]})  # Truncate for privacy
        except Exception as e:
            logger.debug(f"Could not extract messages for observability: {e}")
        
        # Metadata for this call
        metadata = {
            "model": self.model_name,
            "account": self.account,
            "session_id": self.session_id,
            "user_hash": self.user_hash,
            "input_messages": len(messages),
            "service": "livekit_voice"
        }
        
        # The @observe decorator will automatically:
        # - Track timing
        # - Log input/output
        # - Capture exceptions
        # - Send to Langfuse
        
        start_time = time.time()
        
        try:
            # Call the actual LLM function
            result = await llm_callable(*args, **kwargs)
            
            # Log success metrics
            processing_time = time.time() - start_time
            logger.debug(f"LLM call completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            # @observe will automatically capture this exception
            logger.error(f"Error in LLM call: {e}")
            raise

    @observe(name="voice_tts_call")
    async def track_tts_call(self, text_input, tts_callable, *args, **kwargs):
        """
        Track any TTS call with automatic Langfuse observability
        """
        
        # Extract text for observability (privacy-safe)
        text_preview = ""
        text_length = 0
        
        try:
            if isinstance(text_input, str):
                text_preview = text_input[:100]  # First 100 chars
                text_length = len(text_input)
            else:
                # Handle AsyncIterable text input
                text_preview = "streaming_text"
                text_length = 0
        except Exception as e:
            logger.debug(f"Could not extract text for observability: {e}")
        
        metadata = {
            "account": self.account,
            "session_id": self.session_id,
            "user_hash": self.user_hash,
            "text_length": text_length,
            "text_preview": text_preview,
            "service": "livekit_voice_tts"
        }
        
        start_time = time.time()
        
        try:
            # Call the actual TTS function
            result = await tts_callable(*args, **kwargs)
            
            # Log success metrics
            processing_time = time.time() - start_time
            logger.debug(f"TTS call completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            # @observe will automatically capture this exception
            logger.error(f"Error in TTS call: {e}")
            raise

    @observe(name="voice_function_tool")
    async def track_function_tool(self, tool_name: str, tool_callable, *args, **kwargs):
        """
        Track function tool calls with automatic Langfuse observability
        """
        
        metadata = {
            "tool_name": tool_name,
            "account": self.account,
            "session_id": self.session_id,
            "user_hash": self.user_hash,
            "service": "livekit_voice_tools"
        }
        
        start_time = time.time()
        
        try:
            # Call the actual function tool
            result = await tool_callable(*args, **kwargs)
            
            # Log success metrics
            processing_time = time.time() - start_time
            logger.debug(f"Function tool '{tool_name}' completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            # @observe will automatically capture this exception
            logger.error(f"Error in function tool '{tool_name}': {e}")
            raise

# Placeholder classes for backward compatibility
class LangfuseLLM:
    """Placeholder - use LangfuseObservability.track_llm_call instead"""
    pass

class LangfuseTTS:
    """Placeholder - use LangfuseObservability.track_tts_call instead"""
    pass

"""
LiveKit LLM with Langfuse OpenAI integration
"""

import os
import hashlib
import time
from typing import Optional
from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI
from livekit.plugins import openai

class LangfuseLKOpenAILLM(openai.LLM):
    """
    LiveKit LLM with Langfuse observability using recommended approach
    """
    
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        account: Optional[str] = None,
        **kwargs
    ):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_host = os.getenv("LANGFUSE_HOST")
        if not langfuse_public_key or not langfuse_secret_key or not langfuse_host:
            raise ValueError("LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY or LANGFUSE_HOST is not set")
        # Create Langfuse-wrapped OpenAI client
        langfuse_client = LangfuseAsyncOpenAI(
            api_key=openai_api_key,
            max_retries=0,
            # Langfuse automatically captures:
            # - All requests/responses  
            # - Timing & token usage
            # - Errors & performance
        )
        
        # Create privacy-safe user hash
        if session_id:
            self.session_id = session_id
        else:
            if user_id and account:
                salt = f"observability_{account}"
                hash_input = f"{user_id}:{account}:{salt}"
                self.user_hash = f"usr_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"
                self.session_id = f"{int(time.time())}_{user_id}"
            else:
                self.user_hash = None
                self.session_id = None
            
        self.account = account
        
        # 🔑 KEY: Pass Langfuse client to LiveKit LLM
        super().__init__(
            model=model,
            client=langfuse_client,  # This makes ALL calls observable
            **kwargs
        )


"""
LiveKit Google LLM with Langfuse integration
"""

from livekit.plugins import google
from livekit.agents import llm
from langfuse import observe, get_client
import asyncio
from typing import AsyncIterator

class LangfuseLKGoogleLLM(google.llm.LLM):
    """
    LiveKit Google LLM with Langfuse observability using decorator approach
    
    Uses the recommended Langfuse @observe() decorator pattern for cleaner 
    instrumentation compared to manual stream wrapping. This approach:
    - Automatically captures input/output
    - Tracks timing and performance
    - Reports usage metrics
    - Handles errors gracefully
    """
    
    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash-001",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        account: Optional[str] = None,
        **kwargs
    ):
        # Validate Langfuse environment variables
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_host = os.getenv("LANGFUSE_HOST")
        if not langfuse_public_key or not langfuse_secret_key or not langfuse_host:
            raise ValueError("LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY or LANGFUSE_HOST is not set")
        
        # Initialize Langfuse client
        self.langfuse = get_client()
        
        # Create privacy-safe user hash
        if session_id:
            self.session_id = session_id
        else:
            if user_id and account:
                salt = f"observability_{account}"
                hash_input = f"{user_id}:{account}:{salt}"
                self.user_hash = f"usr_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"
                self.session_id = f"{int(time.time())}_{user_id}"
            else:
                self.user_hash = None
                self.session_id = None
            
        self.account = account
        self.model_name = model
        
        # Create a clean Google LLM instance for delegation
        self._google_llm = google.llm.LLM(model=model, **kwargs)
        
        # Initialize parent with same parameters but don't use it directly
        super().__init__(model=model, **kwargs)
    
    @observe()
    def chat(self, **kwargs):
        """Override chat method to return instrumented stream with observability"""
        
        # Extract input for observability
        input_messages = []
        try:
            if 'chat_ctx' in kwargs:
                chat_ctx = kwargs['chat_ctx']
                for item in chat_ctx.items:
                    if hasattr(item, 'role'):
                        content = getattr(item, 'text_content', str(item.content))
                        input_messages.append({
                            "role": item.role, 
                            "content": content[:500]  # Truncate for privacy
                        })
        except Exception as e:
            logger.debug(f"Could not extract messages for observability: {e}")
        
        # Set metadata for the generation
        try:
            self.langfuse.update_current_generation(
                input=input_messages,
                model=self.model_name,
                metadata={
                    "account": self.account,
                    "user_hash": self.user_hash,
                    "session_id": self.session_id,
                    "service": "livekit_voice_google",
                    "provider": "google"
                }
            )
        except Exception as e:
            logger.warning(f"Failed to set Langfuse generation metadata: {e}")
        
        # Create the stream using the clean Google LLM instance
        original_stream = self._google_llm.chat(**kwargs)
        
        # Return wrapped stream that will collect final data
        return LangfuseGoogleLLMStream(
            original_stream=original_stream,
            langfuse_client=self.langfuse,
            model_name=self.model_name,
            account=self.account,
            user_hash=self.user_hash,
            session_id=self.session_id
        )


class LangfuseGoogleLLMStream:
    """
    Simple wrapper that delegates everything to original stream
    """
    
    def __init__(
        self,
        original_stream,
        langfuse_client,
        model_name: str,
        account: str,
        user_hash: Optional[str],
        session_id: Optional[str]
    ):
        self.original_stream = original_stream
        self.langfuse = langfuse_client
        self.model_name = model_name
        self.account = account
        self.user_hash = user_hash
        self.session_id = session_id
        self.full_response = ""
        self.usage_data = None
        
    def __aiter__(self):
        return self._instrumented_iteration()
    
    async def __aenter__(self):
        """Support async context manager protocol"""
        if hasattr(self.original_stream, '__aenter__'):
            await self.original_stream.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager protocol"""
        if hasattr(self.original_stream, '__aexit__'):
            return await self.original_stream.__aexit__(exc_type, exc_val, exc_tb)
        return None
    
    @observe(as_type="generation")
    async def _instrumented_iteration(self):
        """Iterate and collect data for final Langfuse update"""
        try:
            async for chunk in self.original_stream:
                # Collect response text
                if hasattr(chunk, 'delta') and chunk.delta and chunk.delta.content:
                    self.full_response += chunk.delta.content
                
                # Collect usage data
                if hasattr(chunk, 'usage') and chunk.usage:
                    self.usage_data = chunk.usage
                
                # Yield chunk immediately (preserves streaming)
                yield chunk
            
            # Update Langfuse with final data
            try:
                self.langfuse.update_current_generation(
                    output=self.full_response,
                    usage_details={
                        "input": self.usage_data.prompt_tokens if self.usage_data else 0,
                        "output": self.usage_data.completion_tokens if self.usage_data else 0,
                        "total": self.usage_data.total_tokens if self.usage_data else 0
                    } if self.usage_data else None
                )
            except Exception as e:
                logger.warning(f"Failed to update Langfuse generation with final data: {e}")
                
        except Exception as e:
            logger.error(f"Error in Google LLM stream: {e}")
            raise
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original stream"""
        return getattr(self.original_stream, name) 