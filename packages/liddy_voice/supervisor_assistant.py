import asyncio
import json
import logging
import os
import re
import random
import time
import hashlib

from langfuse import observe, get_client

from cerebras.cloud.sdk import Cerebras, AsyncCerebras

from livekit.rtc import (
    Participant
)

from livekit.agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    JobContext,
    RunContext,
    ToolError,
    function_tool,
    get_job_context,
    llm,
    tokenize,
    metrics
)
from livekit.protocol.models import RpcError
from livekit.agents.voice import UserStateChangedEvent, AgentStateChangedEvent
from livekit.plugins import deepgram, groq, openai, silero
from typing import Optional, AsyncIterable, List
from liddy_voice.conversation.analyzer import ConversationAnalyzer

from liddy.models.product import Product
from liddy.models.product_manager import ProductManager
from liddy_voice.session_state_manager import SessionStateManager
from liddy_voice.rag_unified import PineconeRAG
from liddy.model import UserState, BasicChatMessage
from liddy_voice.llm_service import LlmService
from liddy_voice.user_manager import UserManager
from liddy.storage import get_account_storage_provider as get_storage_provider
from liddy_voice.account_manager import get_account_manager
# Conditional import to avoid circular dependency
try:
    from liddy_voice.voice_search_wrapper import VoiceSearchService as SearchService
except ImportError:
    SearchService = None
from livekit.rtc import RemoteParticipant
from liddy_voice.account_prompt_manager import AccountPromptManager

logger = logging.getLogger(__name__)

class SupervisorResponse:
    """Structured response format for ChatAgent integration"""
    def __init__(self):
        self.suggested_message: str = ""
        self.context_updates: dict = {}
        self.follow_up_suggestions: list = []
        self.tool_results: dict = {}
        self.confidence: float = 1.0

class SupervisorAssistant(Agent):
    """
    Full-capability assistant for handling complex requests from ChatAgent.
    This is a copy of Assistant.py functionality with structured response capability.
    """
    
    _current_context: Optional[tuple[str, str]] = None
    
    def __init__(self, ctx: JobContext, primary_model: str, user_id: str = None, 
                 chat_ctx: Optional[llm.ChatContext | None] = None, account: str = None):
        prompts_dir = os.getenv("PROMPTS_DIR", None)
        self._prompt_manager = AccountPromptManager(account=account, prompts_dir=prompts_dir)
        super().__init__(
            instructions=self._prompt_manager.build_system_instruction_prompt(account=account), 
            chat_ctx=chat_ctx
        )
        self._ctx = ctx
        self._primary_model = primary_model
        self._user_id = user_id
        self._session_id = f"{int(time.time())}_{user_id}" if user_id else None
        self._current_context = None
        self._usage_collector = metrics.UsageCollector()
        self._account = account
        self.last_stt_message = None
        self._session = None  # Will be set by ChatAgent
        self._product_manager: ProductManager = None
        
        # Create privacy-safe user identifier
        self.user_hash = self.create_user_hash(user_id, account) if user_id else None
                        
        # Fast LLM for state analysis
        self._fast_llm = openai.LLM(model="gpt-4.1-nano")
        
        # Use AsyncCerebras for non-blocking API calls
        self.async_cerebras_client = AsyncCerebras(
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )
        # Keep the sync client for any remaining sync usage
        self.cerebras_client = Cerebras(
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )
        
        # prewarm the LLM
        asyncio.create_task(self._prewarm_llm())
        
        self.inactivity_task: asyncio.Task | None = None
        self._closing_task: asyncio.Task[None] | None = None
        self._is_closing_session = False
        
        self._planned_reconnect: bool = False
        self._initital_close_on_disconnect: bool | None = None
        self._reconnect_task: asyncio.Task[None] | None = None

        self.wait_phrases = [
            "hang on",
            "one sec",
            "just a moment",
            "yep, give me a second",
            "let me check that for you",
            "hold on a second",
            "let me look that up for you",
            "ok",
            "ok, let me see",
        ]

    def create_user_hash(self, user_id: str, account: str) -> str:
        """Create privacy-safe user identifier for correlation without PII exposure"""
        if not user_id or not account:
            return None
        salt = f"observability_{account}"
        hash_input = f"{user_id}:{account}:{salt}"
        full_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        return f"usr_{full_hash[:12]}" 
    
    async def get_product_manager(self) -> ProductManager:
        """Get the product manager, using Redis if enabled."""
        if not self._product_manager:
            # Check if we should use Redis
            if os.getenv('USE_REDIS_PRODUCTS', 'true').lower() == 'true':
                from liddy.models.redis_product_manager import get_redis_product_manager
                self._product_manager = await get_redis_product_manager(account=self._account)
                logger.info(f"Using Redis-backed ProductManager for {self._account}")
            else:
                # Original in-memory approach
                from liddy.models.product_manager import get_product_manager
                self._product_manager = await get_product_manager(account=self._account)
        return self._product_manager

    async def on_exit(self) -> None:
        print("on_exit")
        await super().on_exit()
        self._ctx.delete_room()
        # self._ctx.shutdown()

    async def _prewarm_llm(self) -> None:
        if self.instructions and len(self.instructions) > 0:
            try:
                # get the model from the llm_service
                llm_service = LlmService.fetch_model_service_from_model(model_name=self._primary_model, account=self._account, model_use="prewarm")
                if llm_service:
                    chat_ctx = llm.ChatContext(
                        items=[
                            llm.ChatMessage(role="system", content=[self.instructions]),
                            llm.ChatMessage(role="user", content=["ping"])
                        ]
                    )
                    response = await LlmService.chat_wrapper(
                        llm_service=llm_service,
                        chat_ctx=chat_ctx,
                    )
                    print(f"Prewarmed {self._primary_model} with response: {response}")
            except Exception as e:
                logger.error(f"Error prewarming LLM: {e}")
           
    def set_session(self, session) -> None:
        """Set the LiveKit session for proper RPC access"""
        self._session = session
        logger.debug("SupervisorAssistant: Session context set for RPC calls")

    def set_user_id(self, user_id: str) -> None:
        if self._user_id == user_id:
            return
        self._user_id = user_id
        self._session_id = f"{int(time.time())}_{user_id}" if user_id else None
    
    def get_user_id(self) -> str:
        return self._user_id

    def set_account(self, account: str) -> None:
        self._account = account
    
    def get_account(self) -> str:
        return self._account
    
    def set_current_context(self, context: tuple[str, str]) -> None:
        """Set the current context for the agent"""
        # if context has changed, then update the prompt
        if self._current_context != context:
            self._current_context = context
            # add a system message to the context
            self.chat_ctx.add_message(
                role="user",
                content=f"{context[1]}"
            )
        else:
            # log the context
            logger.debug(f"Context has not changed: {context}")

    # NEW: State analysis engine for ChatAgent integration
    async def analyze_conversation_state(self, chat_ctx: llm.ChatContext, current_state: dict) -> dict:
        """
        Analyze conversation and return updated state for ChatAgent.
        Uses fast LLM for efficient state analysis.
        """
        try:
            # Get recent messages for analysis
            recent_messages = []
            for message in chat_ctx.items[-5:]:  # Last 5 messages
                if hasattr(message, 'role') and hasattr(message, 'text_content'):
                    recent_messages.append({
                        "role": message.role,
                        "content": message.text_content or str(message.content)
                    })
            
            # State analysis prompt
            state_analysis_prompt = f"""
Analyze this conversation and update the state for the ChatAgent.

Current State: {json.dumps(current_state, indent=2)}
Recent Conversation: {json.dumps(recent_messages, indent=2)}

Return JSON with updated state:
{{
  "conversation_phase": "greeting|discovery|focus|support|consideration|closing",
  "active_products": ["product_id_1", "product_id_2"],
  "user_preferences": {{"budget": "range", "use_case": "type"}},
  "recent_actions": ["action_1", "action_2"],
  "context_summary": "Brief current context",
  "can_handle_locally": ["pattern_1", "pattern_2"],
  "suggested_responses": ["response_1", "response_2"]
}}

Focus on: products discussed, preferences revealed, what ChatAgent can handle next.
"""

            chat_ctx_for_analysis = llm.ChatContext(items=[
                llm.ChatMessage(role="system", content=[state_analysis_prompt]),
                llm.ChatMessage(role="user", content=["Analyze and return updated state as JSON"])
            ])
            
            response = await LlmService.chat_wrapper(
                llm_service=self._fast_llm,
                chat_ctx=chat_ctx_for_analysis,
            )
            
            # Parse JSON response
            try:
                updated_state = json.loads(response)
                return updated_state
            except json.JSONDecodeError:
                logger.error(f"Failed to parse state analysis response: {response}")
                return current_state
                
        except Exception as e:
            logger.error(f"Error in analyze_conversation_state: {e}")
            return current_state

    # NEW: Handle delegation from ChatAgent with streaming response
    async def handle_delegation_streaming(self, user_message: str, reason: str, context: dict):
        """
        Handle a delegated request from ChatAgent with streaming response for immediate speaking.
        
        This streams response chunks back to ChatAgent so it can start speaking immediately
        rather than waiting for the complete response.
        
        Args:
            user_message: The user's original message
            reason: Why ChatAgent delegated this request
            context: Current conversation context and state including full chat_context
            
        Yields:
            Response chunks for immediate speaking by ChatAgent
        """
        try:
            logger.info(f"SupervisorAssistant handling delegation (streaming): {reason}")
            
            # Extract the full chat context for SupervisorAssistant to use
            chat_context_items = context.get("chat_context", [])
            conversation_state = context.get("conversation_state", {})
            
            # Build a temporary chat context for SupervisorAssistant to work with
            if chat_context_items:
                # Convert the chat context items back to ChatContext for SupervisorAssistant to use
                temp_chat_ctx = llm.ChatContext()
                for item in chat_context_items:
                    if hasattr(item, 'role') and hasattr(item, 'content'):
                        temp_chat_ctx.items.append(item)
                    elif isinstance(item, dict) and 'role' in item and 'content' in item:
                        temp_chat_ctx.items.append(llm.ChatMessage(role=item['role'], content=item['content']))
                
                # Temporarily update SupervisorAssistant's chat context
                original_chat_ctx = self.chat_ctx
                self.chat_ctx = temp_chat_ctx
            
            # Execute appropriate tools based on delegation reason and stream results
            if "product" in reason.lower():
                # Handle product-related requests with streaming
                async for chunk in self._handle_product_delegation_streaming(user_message, context):
                    yield chunk
                    
            elif "knowledge" in reason.lower():
                # Handle knowledge-related requests with streaming
                async for chunk in self._handle_knowledge_delegation_streaming(user_message, context):
                    yield chunk
                    
            else:
                # Generic handling with context awareness
                contextual_response = f"Let me help you with that. {self._get_contextual_acknowledgment(conversation_state)}"
                yield contextual_response
                
            # Restore original chat context
            if chat_context_items:
                self.chat_ctx = original_chat_ctx
            
        except Exception as e:
            logger.error(f"Error in handle_delegation_streaming: {e}")
            yield "I'm having trouble with that request right now."

    # NEW: Handle delegation from ChatAgent with structured response
    async def handle_delegation(self, user_message: str, reason: str, context: dict) -> SupervisorResponse:
        """
        Handle a delegated request from ChatAgent and return structured response.
        
        Args:
            user_message: The user's original message
            reason: Why ChatAgent delegated this request
            context: Current conversation context and state including full chat_context
            
        Returns:
            SupervisorResponse with suggested message, context updates, etc.
        """
        try:
            logger.info(f"SupervisorAssistant handling delegation: {reason}")
            
            response = SupervisorResponse()
            
            # Extract the full chat context for SupervisorAssistant to use
            chat_context_items = context.get("chat_context", [])
            conversation_state = context.get("conversation_state", {})
            
            # Build a temporary chat context for SupervisorAssistant to work with
            if chat_context_items:
                # Convert the chat context items back to ChatContext for SupervisorAssistant to use
                temp_chat_ctx = llm.ChatContext()
                for item in chat_context_items:
                    if hasattr(item, 'role') and hasattr(item, 'content'):
                        temp_chat_ctx.items.append(item)
                    elif isinstance(item, dict) and 'role' in item and 'content' in item:
                        temp_chat_ctx.items.append(llm.ChatMessage(role=item['role'], content=item['content']))
                
                # Temporarily update SupervisorAssistant's chat context
                original_chat_ctx = self.chat_ctx
                self.chat_ctx = temp_chat_ctx
            
            # Execute appropriate tools based on delegation reason
            if "product" in reason.lower():
                # Handle product-related requests with full context
                tool_results = await self._handle_product_delegation(user_message, context)
                response.tool_results = tool_results
                response.suggested_message = self._craft_product_response(tool_results, context, conversation_state)
                response.context_updates = self._extract_product_context(tool_results)
                response.follow_up_suggestions = ["Ask about sizing", "Show colors", "Compare models"]
                
            elif "knowledge" in reason.lower():
                # Handle knowledge-related requests with full context
                tool_results = await self._handle_knowledge_delegation(user_message, context)
                response.tool_results = tool_results
                response.suggested_message = self._craft_knowledge_response(tool_results, context, conversation_state)
                response.context_updates = self._extract_knowledge_context(tool_results)
                response.follow_up_suggestions = ["Tell me more", "What else", "Any questions"]
                
            else:
                # Generic handling with context awareness
                response.suggested_message = f"Let me help you with that. {self._get_contextual_acknowledgment(conversation_state)}"
                response.confidence = 0.7
                
            # Restore original chat context
            if chat_context_items:
                self.chat_ctx = original_chat_ctx
            
            return response
            
        except Exception as e:
            logger.error(f"Error in handle_delegation: {e}")
            # Return fallback response
            fallback_response = SupervisorResponse()
            fallback_response.suggested_message = "I'm having trouble with that request right now."
            fallback_response.confidence = 0.1
            return fallback_response

    async def _handle_product_delegation_streaming(self, user_message: str, context: dict):
        """Handle product-related delegation with streaming response"""
        try:
            conversation_state = context.get("conversation_state", {})
            
            # Yield immediate acknowledgment based on request type
            if any(trigger in user_message.lower() for trigger in ["find", "search", "look for"]):
                yield self._craft_immediate_acknowledgment("search", conversation_state)
            elif any(trigger in user_message.lower() for trigger in ["show", "display", "pull up"]):
                yield self._craft_immediate_acknowledgment("display", conversation_state)
            else:
                yield self._craft_immediate_acknowledgment("product", conversation_state)
            
            # Execute tools and stream results
            results = {}
            
            # Check if this is a search request
            search_triggers = ["find", "search", "look for", "show me", "get me"]
            if any(trigger in user_message.lower() for trigger in search_triggers):
                # Stream search results
                search_results = await self._execute_product_search(user_message)
                if search_results.get("success"):
                    results["search_results"] = search_results
                    # Stream search completion
                    yield self._craft_search_completion_message(search_results, conversation_state)
                    
            # Check if this is a display request
            display_triggers = ["show", "display", "pull up", "bring up"]
            if any(trigger in user_message.lower() for trigger in display_triggers):
                # Try to extract product ID from context or search results
                product_id = self._extract_product_id(user_message, context, results.get("search_results"))
                if product_id:
                    display_results = await self._execute_product_display(product_id)
                    if display_results.get("success"):
                        results["display_results"] = display_results
                        # Stream display completion
                        yield self._craft_display_completion_message(display_results, conversation_state)
                        
        except Exception as e:
            logger.error(f"Error in _handle_product_delegation_streaming: {e}")
            yield f"I encountered an issue: {str(e)}"

    async def _handle_knowledge_delegation_streaming(self, user_message: str, context: dict):
        """Handle knowledge-related delegation with streaming response"""
        try:
            conversation_state = context.get("conversation_state", {})
            
            # Yield immediate acknowledgment
            yield self._craft_immediate_acknowledgment("knowledge", conversation_state)
            
            # Execute knowledge search and stream results
            knowledge_results = await self._execute_knowledge_search(user_message)
            if knowledge_results.get("success"):
                # Stream knowledge completion
                yield self._craft_knowledge_completion_message(knowledge_results, conversation_state)
            else:
                yield "I couldn't find specific information about that, but let me help you in another way."
                
        except Exception as e:
            logger.error(f"Error in _handle_knowledge_delegation_streaming: {e}")
            yield f"I encountered an issue: {str(e)}"

    def _craft_immediate_acknowledgment(self, request_type: str, conversation_state: dict) -> str:
        """Craft immediate acknowledgment for streaming response"""
        phase = conversation_state.get("conversation_phase", "discovery")
        
        acknowledgments = {
            "search": ["I found some great options for you!", "Here are some perfect matches!", "Let me show you what I found!"],
            "display": ["Here's exactly what you're looking for!", "Perfect choice! Here are the details:", "Great selection! Let me show you:"],
            "knowledge": ["Here's what you need to know:", "I can help with that!", "Let me explain:"],
            "product": ["I can help you with that!", "Let me get that information:", "Here's what I found:"]
        }
        
        return random.choice(acknowledgments.get(request_type, acknowledgments["product"]))

    def _craft_search_completion_message(self, search_results: dict, conversation_state: dict) -> str:
        """Craft completion message for search results"""
        if search_results.get("result"):
            return search_results["result"]
        else:
            return "I found several options that should work perfectly for you."

    def _craft_display_completion_message(self, display_results: dict, conversation_state: dict) -> str:
        """Craft completion message for display results"""
        if display_results.get("result"):
            return display_results["result"]
        else:
            return "Here are all the details you need!"

    def _craft_knowledge_completion_message(self, knowledge_results: dict, conversation_state: dict) -> str:
        """Craft completion message for knowledge results"""
        if knowledge_results.get("result"):
            return str(knowledge_results["result"])
        else:
            return "I hope that helps clarify things for you!"

    async def _handle_product_delegation(self, user_message: str, context: dict) -> dict:
        """Handle product-related delegation - search and/or display products"""
        try:
            results = {}
            
            # Check if this is a search request
            search_triggers = ["find", "search", "look for", "show me", "get me"]
            if any(trigger in user_message.lower() for trigger in search_triggers):
                # Use the existing product_search functionality
                search_results = await self._execute_product_search(user_message)
                results["search_results"] = search_results
                
            # Check if this is a display request
            display_triggers = ["show", "display", "pull up", "bring up"]
            if any(trigger in user_message.lower() for trigger in display_triggers):
                # Try to extract product ID from context or search results
                product_id = self._extract_product_id(user_message, context, results.get("search_results"))
                if product_id:
                    display_results = await self._execute_product_display(product_id)
                    results["display_results"] = display_results
                    
            return results
            
        except Exception as e:
            logger.error(f"Error in _handle_product_delegation: {e}")
            return {"error": str(e)}

    async def _handle_knowledge_delegation(self, user_message: str, context: dict) -> dict:
        """Handle knowledge-related delegation"""
        try:
            # Use the existing knowledge_search functionality
            knowledge_results = await self._execute_knowledge_search(user_message)
            return {"knowledge_results": knowledge_results}
            
        except Exception as e:
            logger.error(f"Error in _handle_knowledge_delegation: {e}")
            return {"error": str(e)}

    def _craft_product_response(self, tool_results: dict, context: dict, conversation_state: dict) -> str:
        """Craft a branded response for product-related results"""
        try:
            conversation_phase = conversation_state.get("conversation_phase", "discovery")
            
            if "search_results" in tool_results:
                if conversation_phase == "greeting":
                    return "I found some great options to get you started! Let me show you what stands out."
                else:
                    return "I found some great options for you! Let me show you what stands out."
            elif "display_results" in tool_results:
                if conversation_phase == "consideration":
                    return "Here's what you're looking for! This is a fantastic choice that should work perfectly."
                else:
                    return "Here's what you're looking for! This is a fantastic choice."
            else:
                return "Great choice, let me pull that up!"
        except Exception as e:
            logger.error(f"Error crafting product response: {e}")
            return "Let me check that for you."

    def _craft_knowledge_response(self, tool_results: dict, context: dict, conversation_state: dict) -> str:
        """Craft a branded response for knowledge-related results"""
        try:
            conversation_phase = conversation_state.get("conversation_phase", "discovery")
            
            if "knowledge_results" in tool_results:
                if conversation_phase == "support":
                    return "Great follow-up question! Here's what you need to know."
                else:
                    return "Great question! Here's what you need to know."
            else:
                return "Let me look that up for you."
        except Exception as e:
            logger.error(f"Error crafting knowledge response: {e}")
            return "Let me check that for you."

    def _get_contextual_acknowledgment(self, conversation_state: dict) -> str:
        """Get contextual acknowledgment based on conversation state"""
        try:
            conversation_phase = conversation_state.get("conversation_phase", "discovery")
            active_products = conversation_state.get("active_products", [])
            
            if conversation_phase == "consideration" and active_products:
                return "I can see you're weighing your options."
            elif conversation_phase == "focus":
                return "Let me dive deeper into the details for you."
            elif conversation_phase == "support":
                return "I'm here to help with any questions."
            else:
                return "I'm on it!"
        except Exception as e:
            logger.error(f"Error getting contextual acknowledgment: {e}")
            return "I'm on it!"

    def _extract_product_context(self, tool_results: dict) -> dict:
        """Extract context updates from product tool results"""
        context_updates = {}
        
        if "search_results" in tool_results:
            context_updates["conversation_phase"] = "focus"
            # Extract product IDs from search results if available
            # This would be implemented based on the actual search result format
            
        if "display_results" in tool_results:
            context_updates["conversation_phase"] = "consideration"
            context_updates["product_shown"] = True
            
        return context_updates

    def _extract_knowledge_context(self, tool_results: dict) -> dict:
        """Extract context updates from knowledge tool results"""
        context_updates = {}
        
        if "knowledge_results" in tool_results:
            context_updates["conversation_phase"] = "support"
            context_updates["knowledge_provided"] = True
            
        return context_updates

    def _extract_product_id(self, user_message: str, context: dict, search_results: dict = None) -> str:
        """Extract product ID from message, context, or search results"""
        # This would implement logic to extract product IDs
        # For now, return None - will be implemented with actual tool integration
        return None

    # Placeholder methods for tool execution - will be replaced with actual tool methods
    @observe(name="product_search")
    @function_tool()
    async def product_search(
        self,
        context: RunContext[UserState],
        query: str,
    ):
        """Use this to search for products that the user may be interested in. Include as much depth and detail in the query as possible to get the best results. Generally, the only way for you to know about individual products is through this tool call, so feel free to liberally call this tool call.
        
        Args:
            query: The query to use for the search. Be as descriptive and detailed as possible based on the context of the entire conversation. For example, the query "laptop" is not very descriptive, but "mid-range laptop with 16GB RAM, SSD storage, and dedicated graphics for video editing and gaming" is much more descriptive. The more descriptive the query, the better the results will be. You can also use this to search for specific products by name or ID, but be sure to include as much context as possible to help narrow down the results.
        """
        try:
            # Input validation
            if not query:
                raise ToolError("`query` is required")
        
            # Get products through ProductManager
            products = await (await self.get_product_manager()).get_products()
            products_results = []
            
            account_manager = await get_account_manager(account=self._account)
            index_name, embedding_model = account_manager.get_rag_details()
            
            # Choose search approach based on catalog size
            if not index_name:
                # For small catalogs: Use LLM-based search with standardized status updates
                if SearchService:
                    products_results = await self.perform_search_with_status(
                        search_fn=lambda q, **params: SearchService.search_products_llm(
                            q,
                            products=params.get("products", []),
                            user_state=self.session.userdata
                        ),
                        query=query,
                        search_params={
                            "products": products,
                            "account": self._account,
                            "enhancer_params": {
                                "chat_ctx": self.chat_ctx,
                                "product_knowledge": self._prompt_manager.product_search_knowledge
                            }
                        }
                    )
                else:
                    # Fallback when SearchService is not available
                    products_results = products[:5]  # Simple fallback for testing
            else:
                # For large catalogs: Use RAG search with standardized status updates
                if SearchService:
                    rag_results = await self.perform_search_with_status(
                        search_fn=lambda q, **params: SearchService.perform_search(
                            query=q,
                            search_function=SearchService.search_products_rag,
                            search_params=params,
                            user_state=self.session.userdata,
                            query_enhancer=SearchService.enhance_product_query
                        ),
                        query=query,
                        search_params={
                            "account": self._account,
                            "enhancer_params": {
                                "chat_ctx": self.chat_ctx,
                                "product_knowledge": self._prompt_manager.product_search_knowledge
                            }
                        }
                    )
                else:
                    # Fallback when SearchService is not available
                    rag_results = []
                
                # Process RAG results into product objects
                if rag_results and not isinstance(rag_results, dict) or not rag_results.get("error"):
                    for result in rag_results:
                        if result.get('metadata'):
                            products_results.append(Product.from_metadata(result['metadata']))
            
            # Format results as markdown
            markdown_results = "# Products (in descending order):\n\nRemember that you are an AI sales agent speaking to a human, so be sure to use natural language to respond and not just a list of products. For example, never say the product URL unless the user specifically asks for it.\n\n"
            for i, result in enumerate(products_results):
                markdown_results += f"{Product.to_markdown(result, depth=1)}\n\n"
                
            markdown_results += f"\n\n**CRITICAL**: Be sure to call the `display_product` tool call to display the product that you talk about. You can use the product ID from the results above to do this.\n\n"

            return markdown_results
            
        except Exception as e:
            logger.error(f"Error in product_search: {e}")
            raise ToolError("Unable to search for products")

    # Update the placeholder method to use the actual tool
    async def _execute_product_search(self, query: str) -> dict:
        """Execute product search using the actual tool method"""
        try:
            # Create a mock RunContext for the tool call
            # In a real implementation, this would be passed from the ChatAgent
            mock_context = type('MockContext', (), {'userdata': self.session.userdata if hasattr(self, 'session') else None})()
            
            result = await self.product_search(mock_context, query)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error executing product search: {e}")
            return {"success": False, "error": str(e)}
        
    @observe(name="display_product")
    @function_tool()
    async def display_product(
        self,
        product_id: str,
        variant: Optional[dict[str, str]]=None,
        resumption_message: str="",
    ):
        """Use this to display or show exactly one products. The tool call itself will handle actually doing this. Never try to display or show products on your own, ALWAYS use this tool call. It is not possible to show multiple products at once, so you must narrow it down to a single product.
        
        You typically would first use `product_search` to find the product ID, and then use this function to display the product.
        
        NOTE: If you know the user is already looking at that product (because they are viewing that page or something like that) then you can silently call this function without necessarily saying anything to the user. This is useful for showing the product details without interrupting the conversation. Otherwise, it is generally a good idea to say something to the user to let them know you are displaying the product.
        
        Args:
            product_id: The ID of the product to display. Required.
            variant: The ID or name of the product variant to display. This is optional and can be used to show a specific variant of the product. If not provided, the default variant will be shown. Include the name of the variant, such as "color" or "size", and the value of the variant, such as "red" or "large". For example, "color=red" or "size=large". If you don't know the variant ID, you can use the `product_search` function to find it.
            resumption_message: The message to say back to the user after displaying the product. This is required and should be a natural language message that is appropriate for the context of the conversation.
        """
        if not isinstance(product_id, str):
            raise ToolError("`product_id` is required")
        try:
            logger.info(f"display_product: {product_id}")
            
            product = await (await self.get_product_manager()).find_product_by_id(product_id=product_id)
            products = []
            
            if not product:
                products = await (await self.get_product_manager()).get_products()
                products_names_urls_ids = [(p.name, p.productUrl, p.id) for p in products]
                raise ToolError(f"The product ID's provided are not in the product catalog. Please provide valid product ID's. Use the `product_search` function to find a valid product ID. If at all possible, don't ask the user for the product ID, instead use the `product_search` function to find it silently and use that.\n\nValid product ID's: {json.dumps(products_names_urls_ids)}")
            
            print(f"Product URL: {product.productUrl}")
            product_results = {
                "name": product.name,     
                "productUrl": product.productUrl, 
                "id": product.id, 
                "imageUrls": [product.imageUrls[0]] if product.imageUrls else [],
            }
            if variant:
                for k, v in variant.items():
                    variant_query_param = None
                    variant_name = k
                    variant_value = v
                    # get the account variants
                    account_manager = await get_account_manager(account=self._account)
                    variant_types = account_manager.get_variant_types()
                    if variant_types and len(variant_types) > 0:
                        if len(variant_types) == 1:
                            variant_type = variant_types[0]
                        else:
                            for vt in variant_types:
                                if variant_name and variant_name.lower() == vt.lower():
                                    variant_type = vt
                                    break

                    if variant_type:
                        if variant_name == "color":
                            # find the color variant in the product
                            for color in product.colors:
                                if isinstance(color, str):
                                    if variant_value.lower() == color.lower():
                                        variant_query_param = f"color={color}"
                                        break
                                elif isinstance(color, dict):
                                    if "id" in color and color.get("id").lower() == variant_value.lower():
                                        variant_query_param = f"color={color['id']}"
                                        break
                                    elif "name" in color and color.get("name").lower() == variant_value.lower():
                                        variant_query_param = f"color={color['id']}"
                                        break
                            
                        
                    if variant_query_param:
                        product_results["variantId"] = variant_query_param
                        # if the variant_id is in the form "<property_name>=<value>", then append it as a query parameter to the product URL
                        if "?" in product_results["productUrl"]:
                            product_results["productUrl"] += f"&{variant_query_param}"
                        else:
                            product_results["productUrl"] += f"?{variant_query_param}"
            
            user_state = self.session.userdata if hasattr(self, 'session') and self.session else None
            history = SessionStateManager.get_user_recent_history(user_id=user_state.user_id) if user_state else None

            current_product = None
            if history and len(history) > 0:
                most_recent_history = history[0]
                if most_recent_history.url:
                    product_url = most_recent_history.url
                    current_product = await Product.find_by_url(product_url, account=user_state.account) if user_state else None
            
            if hasattr(self, 'session') and self.session and self.session.current_speech:
                await self.session.current_speech.wait_for_playout()
            
            # Try to get participant info from session first, fallback to job context
            participant_identity = None
            local_participant = None
            
            if self._session and hasattr(self._session, 'room'):
                # Use session context (preferred for ChatAgent)
                if hasattr(self._session.room, 'remote_participants'):
                    participant_identity = next(iter(self._session.room.remote_participants)) if self._session.room.remote_participants else None
                if hasattr(self._session.room, 'local_participant'):
                    local_participant = self._session.room.local_participant
                room_name = self._session.room.name if hasattr(self._session.room, 'name') else None
            else:
                # Fallback to job context (for original Assistant)
                participant_identity = next(iter(get_job_context().room.remote_participants)) if get_job_context().room.remote_participants else None
                local_participant = get_job_context().room.local_participant if get_job_context().room else None
                room_name = get_job_context().room.name if get_job_context().room else None
            
            # Because this potentially causes the UI to do a hard nav to a new page, update the should_reconnect_until timeout for this account/user_id
            if user_state:
                UserManager.update_user_room_reconnect_time(
                    account=self._account,
                    user_id=user_state.user_id,
                    room_id=room_name,
                    auto_reconnect_until=time.time() + 60 * 5,  # 5 minutes
                    resumption_message=resumption_message
                )
            
            try:
                if hasattr(self, '_setup_planned_reconnect'):
                    self._setup_planned_reconnect()
                    
                if participant_identity and local_participant:
                    logger.debug(f"SupervisorAssistant: Performing RPC call using {'session' if self._session else 'job context'} context")
                    rpc_result = await local_participant.perform_rpc(
                        destination_identity=participant_identity,
                        method="display_products",
                        payload=json.dumps({"products": [product_results], "resumptionMessage": resumption_message}),
                        response_timeout=10.0,
                    )
                else:
                    logger.warning(f"SupervisorAssistant: No participant or local_participant available for RPC call")
            except Exception as e:
                if isinstance(e, RpcError) or str(e) == "Response timeout":
                    logger.error(f"RpcError in display_product: {e}")
                    # Ignore for now
                elif isinstance(e, ConnectionError):
                    logger.error(f"ConnectionError in display_product: {e}")
                    # Ignore for now
            
            result = ""
            if current_product and current_product.id == product_id:
                result = "The user is already looking at this product, so no need to display it again--and you might even very kindly and positively mention that, or just positively reenforce their selection. But here are the details for your reference:\n\n"
                
            result += "# Product Details\n\n"
            result += f"{Product.to_markdown(product, depth=1)}\n\n"
            
            return result
        except ToolError as e:
            logger.error(f"ToolError in display_product: {e}")
            raise e
        except ConnectionError as e:
            logger.error(f"ConnectionError in display_product: {e}")
            # Ignore for now
        except Exception as e:
            if isinstance(e, RpcError) or str(e) == "Response timeout":
                logger.error(f"RpcError in display_product: {e}")
                # Ignore for now
            elif isinstance(e, ConnectionError):
                logger.error(f"ConnectionError in display_product: {e}")
                # Ignore for now
            else:
                logger.error(f"Error displaying product: {e}")
                raise ToolError(f"Unable to show product: {e}")

    # Update the placeholder method to use the actual tool
    async def _execute_product_display(self, product_id: str) -> dict:
        """Execute product display using the actual tool method"""
        try:
            result = await self.display_product(product_id, None, "")
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error executing product display: {e}")
            return {"success": False, "error": str(e)}
        
    async def _execute_knowledge_search(self, query: str) -> dict:
        """Execute knowledge search - placeholder for actual tool method"""
        # This will be replaced with the actual knowledge_search tool method
        return {"placeholder": "knowledge_search_results"}

    async def generate_and_say_wait_phrase_if_needed(self, use_fast_llm: bool = False, say_filler: bool = True, force: bool=False) -> Optional[str]:
        try:
            if not use_fast_llm:
                wait_phrase = random.choice(self.wait_phrases)
                # DON'T use session directly - ChatAgent will handle speaking
                # if say_filler and hasattr(self, 'session') and self.session:
                #     await self.session.say(wait_phrase, add_to_chat_ctx=False)
                return wait_phrase
            
            chat_ctx = self.chat_ctx.copy(
                exclude_function_call=True, exclude_instructions=True, tools=[]
            ).truncate(max_items=5)
            
            if chat_ctx.items and len(chat_ctx.items) > 0:
                most_recent_message = chat_ctx.items[-1]
                if force or hasattr(most_recent_message, "role") and most_recent_message.role == "user":
                    fast_response = random.choice(self.wait_phrases)  # Simplified for now
                    logger.debug(f"Fast response: {fast_response}")
                    # DON'T use session directly - ChatAgent will handle speaking
                    # if say_filler and hasattr(self, 'session') and self.session:
                    #     await self.session.say(fast_response, add_to_chat_ctx=True)
                    return fast_response
                else:
                    logger.debug("Last message is from assistant, no filler needed")
        except Exception as e:
            logger.error(f"Error in generate_and_say_wait_phrase_if_needed: {e}")
            return None

    async def perform_search_with_status(
        self, 
        search_fn, 
        query, 
        search_params=None, 
        timeout=25.0,
        status_delay=0.5
    ):
        """Unified search with status updates and proper timeout handling"""
        status_task = None
        try:
            # Create a delayed status update task that only runs after status_delay
            async def delayed_status_update():
                await asyncio.sleep(status_delay)
                await self.generate_and_say_wait_phrase_if_needed(use_fast_llm=False, say_filler=True)
            
            status_task = asyncio.create_task(delayed_status_update())
            
            # Perform search with timeout
            search_results = await asyncio.wait_for(
                search_fn(query, **(search_params or {})),
                timeout=timeout
            )
            
            return search_results
            
        except asyncio.TimeoutError:
            logger.warning(f"Search timed out for query: {query}")
            return {"error": "Search timed out", "results": []}
        except Exception as e:
            logger.error(f"Error in perform_search_with_status: {e}")
            return {"error": str(e), "results": []}
        finally:
            # Always ensure the status task is properly canceled
            if status_task:
                if not status_task.done():
                    status_task.cancel()
                    try:
                        # Give a short timeout to clean up the task
                        await asyncio.wait_for(asyncio.shield(status_task), timeout=0.1)
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass  # Task was canceled or timed out, which is expected 

    @function_tool(
        description="Search for knowledge base articles from the 'information' namespace. Provide a detailed query (and optionally a topic name) to get relevant knowledge. Returns articles with titles and summaries from the knowledge base."
    )
    async def knowledge_search(
        self,
        context: RunContext,
        query: str,
        topic_name: str = "",
        top_k: int = 10,
        top_n: int = 5,
        min_score: float = 0.15,
        min_n: int = 0
    ) -> dict:
        """
        Search the knowledge base (restricted to the "information" namespace) using the provided query.
        
        Args:
            context: RunContext from LiveKit.
            query: The search query string (required).
            topic_name: Optional topic name to refine the query.
            top_k: Maximum number of documents to return (default: 10).
            top_n: Number of top results for re-ranking (default: 5).
            min_score: Minimum similarity score threshold (default: 0.15).
            min_n: Minimum number of results to consider (default: 0).
        
        Returns:
            A dictionary with an "articles" key holding the list of matching knowledge base entries.
        """
        if not query:
            return {"error": "The 'query' parameter is required."}

        # Combine topic_name with query if provided
        search_query = f"Topic: {topic_name}\n\n{query}" if topic_name else query

        try:
            # Use the standardized search function with status updates
            if SearchService:
                results = await self.perform_search_with_status(
                    search_fn=lambda q, **params: SearchService.perform_search(
                        query=q,
                        search_function=SearchService.search_knowledge,
                        search_params=params,
                        user_state=self.session.userdata if hasattr(self, 'session') and self.session else None,
                        query_enhancer=SearchService.enhance_knowledge_query
                    ),
                    query=search_query,
                    search_params={
                        "account": self._account,
                        "top_k": top_k,
                        "top_n": top_n,
                        "min_score": min_score,
                        "min_n": min_n,
                        "enhancer_params": {
                            "chat_ctx": self.chat_ctx,
                            "knowledge_base": self._prompt_manager.knowledge_search_guidance or ""
                        }
                    }
                )
            else:
                # Fallback when SearchService is not available
                results = [{"metadata": {"title": "Test knowledge", "content": f"Information about {search_query}"}}]

            # Process results
            articles = []
            if results and not isinstance(results, dict) or not results.get("error"):
                for res in results:
                    if "metadata" in res and res["metadata"]:
                        articles.append(res["metadata"])
                    elif "text" in res:
                        articles.append(res["text"])
                    else:
                        articles.append(res)
                        
            return {"articles": articles}
            
        except Exception as e:
            logger.error(f"Error in knowledge_search function: {e}")
            return {"error": str(e)}

    # Update the placeholder method to use the actual tool
    async def _execute_knowledge_search(self, query: str) -> dict:
        """Execute knowledge search using the actual tool method"""
        try:
            # Create a mock RunContext for the tool call
            mock_context = type('MockContext', (), {})()
            
            result = await self.knowledge_search(mock_context, query)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error executing knowledge search: {e}")
            return {"success": False, "error": str(e)} 

    @function_tool()
    async def end_conversation(
        self,
        context: RunContext,
        reason: str,
    ) -> str:
        """
        Called when user want to leave the conversation
        
        Args:
            reason: The reason for ending the conversation. This is optional and can be used to help identify the product. For example, if you are unsure of the exact product_url, you can use the product_name to search for it. This is not required, but it is a good idea to include it if you have it.
        """

        logger.debug("Ending conversation from function tool")

        try:
            if hasattr(self, 'session') and self.session:
                SessionStateManager.store_conversation_exit_reason(user_state=self.session.userdata, reason=reason)
                handle = self.session.current_speech
                if handle:
                    await handle.wait_for_playout()
                await self.session.generate_reply(instructions="say goodbye to the user")
                
                # Try to get participant info from session first, fallback to job context
                participant_identity = None
                local_participant = None
                
                if self._session and hasattr(self._session, 'room'):
                    # Use session context (preferred for ChatAgent)
                    if hasattr(self._session.room, 'remote_participants'):
                        participant_identity = next(iter(self._session.room.remote_participants)) if self._session.room.remote_participants else None
                    if hasattr(self._session.room, 'local_participant'):
                        local_participant = self._session.room.local_participant
                else:
                    # Fallback to job context (for original Assistant)
                    participant_identity = next(iter(get_job_context().room.remote_participants)) if get_job_context().room.remote_participants else None
                    local_participant = get_job_context().room.local_participant if get_job_context().room else None
                
                if participant_identity and local_participant:
                    logger.debug(f"SupervisorAssistant: Performing end_conversation RPC call using {'session' if self._session else 'job context'} context")
                    end_result = await local_participant.perform_rpc(
                        destination_identity=participant_identity, 
                        method="end_conversation",
                        payload=json.dumps({
                            "reason": reason,
                        }),
                        response_timeout=30.0,
                    )
                    logger.debug(f"End conversation result: {end_result}")
                else:
                    logger.warning(f"SupervisorAssistant: No participant or local_participant available for end_conversation RPC call")

            if hasattr(self, '_closing_task'):
                self._closing_task = asyncio.create_task(self.session.aclose()) if hasattr(self, 'session') and self.session else None
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")

        return "Conversation ended" 