import json
import os
import logging
from livekit.agents import llm
from livekit.plugins import google, openai, groq
from typing import Optional, Any
from liddy_voice.spence.langfuse_llm import LangfuseLKOpenAILLM

logger = logging.getLogger(__name__)

class LlmService:
  
  @staticmethod
  def fetch_model_service_from_model(
    model_name: str,
    account: str,
    user: Optional[str] = None,
    model_use: Optional[str] = None,
  ) -> llm.LLM:
    
    metadata = {
      "account": account,
    }
    if model_use:
      metadata = {"model_use": model_use}
    
    # if model name includes 'gemini', then instantiate google.llm
    if 'gemini' in model_name or model_name == "meta/llama-4-scout-17b-16e-instruct-maas":
      # Check if Langfuse observability should be enabled for Google models
      if (os.getenv("LANGFUSE_SECRET_KEY") and 
          os.getenv("LANGFUSE_PUBLIC_KEY") and 
          user and account):
        try:
          # Use Langfuse-enhanced Google LLM for automatic observability
          logger.info(f"Using Langfuse-enhanced Google LLM for model: {model_name}")
          from liddy_voice.spence.langfuse_llm import LangfuseLKGoogleLLM
          if "2.5" in model_name:
            return LangfuseLKGoogleLLM(
              model=model_name, 
              user_id=user,
              account=account,
              # thinking_config={"thinking_budget": 1250}
            )
          else:
            return LangfuseLKGoogleLLM(
              model=model_name, 
              user_id=user,
              account=account
            )
        except Exception as e:
          logger.warning(f"Failed to initialize Langfuse Google LLM, falling back to regular Google LLM: {e}")
          import traceback
          logger.warning(f"Full traceback: {traceback.format_exc()}")
          # Fallback to regular Google LLM
          if "2.5" in model_name:
            return google.llm.LLM(model=model_name, thinking_config={"thinking_budget": 1250})
          else:
            return google.llm.LLM(model=model_name)
      else:
        # Use regular Google LLM (no observability or missing credentials)
        if "2.5" in model_name:
          return google.llm.LLM(model=model_name, thinking_config={"thinking_budget": 1250})
        else:
          return google.llm.LLM(model=model_name)
    # else if model name starts with 'gpt' or 'o', then instantiate openai.llm with Langfuse if enabled
    elif model_name.__contains__('gpt') or model_name.startswith('o') or model_name.startswith('ft:'):
      # Check if Langfuse observability should be enabled
      if (os.getenv("LANGFUSE_SECRET_KEY") and 
          os.getenv("LANGFUSE_PUBLIC_KEY") and 
          user and account):
        try:
          # Use Langfuse-enhanced LLM for automatic observability
          logger.info(f"Using Langfuse-enhanced LLM for model: {model_name}")
          return LangfuseLKOpenAILLM(
            model=model_name, 
            user_id=user,
            account=account,
            store=model_name=="gpt-4.1" or True, 
            metadata=metadata
          )
        except Exception as e:
          logger.warning(f"Failed to initialize Langfuse LLM, falling back to regular OpenAI: {e}")
          # Fallback to regular OpenAI LLM
          return openai.llm.LLM(model=model_name, store=model_name=="gpt-4.1" or True, user=user, metadata=metadata)
      else:
        # Use regular OpenAI LLM (no observability or missing credentials)
        return openai.llm.LLM(model=model_name, store=model_name=="gpt-4.1" or True, user=user, metadata=metadata)
    elif model_name.startswith('llama'):
      return openai.llm.LLM.with_cerebras(model=model_name, user=user)
    elif model_name.startswith('meta'):
      return groq.LLM(model=model_name, user=user)
    else:
      return google.llm.LLM(model="gemini-2.5-flash")
      # raise ValueError
  
  @staticmethod
  def max_input_tokens(model_name:str) -> int:
    if 'gemini' in model_name:
      # 128k tokens
      return 128000
    elif model_name.startswith('gpt') or model_name.startswith('o'):
      # 8196 tokens
      return 8196
  
  @staticmethod
  async def chat_wrapper(llm_service: llm.LLM, chat_ctx: llm.ChatContext) -> str:
      """
      This function wraps the LLM service to handle streaming responses and
      return the final response as a dictionary.
      """
      analysis_stream = llm_service.chat(
          chat_ctx=chat_ctx,
          # max_tokens=max_tokens,
          # temperature=temperature,
      )
      analysis_content = ""
      try:
          async for chunk in analysis_stream:
              if isinstance(chunk, llm.ChatChunk):
                  if chunk.delta and chunk.delta.content:
                      analysis_content += chunk.delta.content
              elif hasattr(chunk, "choices") and chunk.choices[0].delta.content:
                  analysis_content += chunk.choices[0].delta.content
      finally:
          await analysis_stream.aclose()

      return analysis_content
  
  @staticmethod
  def parse_json_response(analysis_content: str) -> dict:
      """
      This function parses the JSON response from the LLM service.
      It extracts the JSON object from the response string.
      """
      # Find the first and last curly braces in the response string
      # and extract the substring between them
      if not isinstance(analysis_content, str):
          return analysis_content
      
      # find the first and last curly braces or brackets in the response string
      first_index = min(max(0, analysis_content.find('{')), max(0, analysis_content.find('[')))
      last_index = max(min(analysis_content.__len__()-1, analysis_content.rfind('}')), min(analysis_content.__len__()-1, analysis_content.rfind(']')))
        
      response_obj = json.loads(analysis_content[first_index:last_index + 1])
      return response_obj
