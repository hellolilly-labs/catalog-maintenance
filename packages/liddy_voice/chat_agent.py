import asyncio
import json
import logging
import os
import time
from typing import Optional, AsyncIterable, List, Dict

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
)
from livekit.protocol.models import RpcError
from livekit.agents.voice import UserStateChangedEvent, AgentStateChangedEvent
from livekit.plugins import openai
from livekit.rtc import RemoteParticipant

from liddy.models.product import Product
from liddy_voice.session_state_manager import SessionStateManager
from liddy.model import UserState, BasicChatMessage
from liddy_voice.llm_service import LlmService
from liddy_voice.user_manager import UserManager
from liddy.storage import get_account_storage_provider as get_storage_provider
from liddy_voice.account_manager import AccountManager
# Conditional import to avoid circular dependency
try:
    from liddy_voice.voice_search_wrapper import VoiceSearchService as SearchService
except ImportError:
    SearchService = None
from liddy_voice.account_prompt_manager import AccountPromptManager
from liddy_voice.supervisor_assistant import SupervisorAssistant, SupervisorResponse

logger = logging.getLogger(__name__)

class ChatAgent(Agent):
    """
    Fast, knowledge-enhanced conversational agent using gpt-4.1-nano.
    
    This agent handles most conversations locally using rich knowledge injection,
    and delegates to SupervisorAssistant only when tools/actions are needed.
    
    Key Features:
    - Fast response with gpt-4.1-nano for immediate feedback
    - Rich knowledge base injection (base_knowledge + brand_knowledge + product_search_knowledge)
    - Smart SLM-based delegation via @function_tool() pattern
    - Natural acknowledgment when calling delegation tool
    - Structured integration with SupervisorAssistant
    """
    
    # ChatAgent is designed to use the fast gpt-4.1-nano model
    CHAT_AGENT_MODEL = "gpt-4.1-nano"
    
    def __init__(self, ctx: JobContext, primary_model: str, user_id: str = None, 
                 chat_ctx: Optional[llm.ChatContext | None] = None, account: str = None):
        
        # Initialize PromptManager for knowledge injection
        prompts_dir = os.getenv("PROMPTS_DIR", None)
        self._prompt_manager = AccountPromptManager(account=account, prompts_dir=prompts_dir)
        
        # Store account and initialize core attributes first
        self._account = account
        self._ctx = ctx
        self._primary_model = primary_model
        self._user_id = user_id
        self._session_id = f"{int(time.time())}_{user_id}" if user_id else None
        self.last_stt_message = None
        
        # Initialize knowledge base access early
        self._base_knowledge = getattr(self._prompt_manager, 'base_knowledge', '')
        self._brand_knowledge = getattr(self._prompt_manager, 'brand_knowledge', '')
        self._product_search_knowledge = getattr(self._prompt_manager, 'product_search_knowledge', '')
        
        # Build ChatAgent-specific instructions with knowledge injection
        chat_instructions = self._build_chat_agent_instructions_with_knowledge()
        
        super().__init__(
            instructions=chat_instructions,
            chat_ctx=chat_ctx
        )
        
        # Conversation state management
        self._conversation_state = {
            "conversation_phase": "greeting",
            "active_products": [],
            "user_preferences": {},
            "recent_actions": [],
            "context_summary": "",
            "can_handle_locally": [],
            "suggested_responses": []
        }
        
        # Fast model for ChatAgent responses
        self._chat_model = openai.LLM(model=self.CHAT_AGENT_MODEL)
        
        # SupervisorAssistant for tool delegation
        self._supervisor = SupervisorAssistant(ctx, primary_model, user_id, chat_ctx, account)
        
        # Session reference will be set after session.start() in main_chat_agent.py
        self._session = None
        
        # Streaming delegation state
        self._pending_delegation = None
        
        # Pre-warm the LLM in background thread (non-blocking)
        import threading
        def prewarm_in_thread():
            try:
                asyncio.run(self._prewarm_llm())
            except Exception as e:
                logger.debug(f"ChatAgent: Pre-warming failed in thread: {e}")
        
        threading.Thread(target=prewarm_in_thread, daemon=True).start()
        
        logger.info(f"ChatAgent initialized with knowledge bases: base={len(self._base_knowledge)}, brand={len(self._brand_knowledge)}, product={len(self._product_search_knowledge)}")

    async def _prewarm_llm(self) -> None:
        """
        Pre-warm the LLM to cache the base system prompt.
        
        This makes an initial LLM call with the base system prompt + simple message
        so that the first real user interaction can benefit from cached tokens.
        
        This method runs in the background and won't block the main conversation flow.
        """
        if not self.instructions or len(self.instructions) == 0:
            logger.debug("ChatAgent: No instructions available for pre-warming")
            return
            
        try:
            logger.debug(f"ChatAgent: Starting pre-warm for {self.CHAT_AGENT_MODEL}")
            
            # Get the LLM service for ChatAgent's model
            llm_service = LlmService.fetch_model_service_from_model(
                model_name=self.CHAT_AGENT_MODEL,  # Use same model as ChatAgent will use
                account=self._account, 
                model_use="prewarm"
            )
            
            if not llm_service:
                logger.debug("ChatAgent: No LLM service available for pre-warming")
                return
                
            # Create warming context with base system prompt + simple user message
            prewarm_ctx = llm.ChatContext(
                items=[
                    llm.ChatMessage(role="system", content=[self.instructions]),
                    llm.ChatMessage(role="user", content=["ping"])
                ]
            )
            
            # Make the warming call to cache the system prompt
            response = await LlmService.chat_wrapper(
                llm_service=llm_service,
                chat_ctx=prewarm_ctx,
            )
            
            logger.info(f"✅ ChatAgent pre-warmed {self.CHAT_AGENT_MODEL} successfully")
            logger.debug(f"Pre-warm response: {response}")
                
        except Exception as e:
            logger.warning(f"ChatAgent pre-warming failed (non-critical): {e}")
            # Non-critical error - continue without pre-warming

    def _build_chat_agent_instructions_with_knowledge(self) -> str:
        """
        Build ChatAgent-specific instructions with rich knowledge injection.
        This is where the ChatAgent gets its enhanced capabilities.
        """
        # Get base instructions and use account-specific prompts for all content
        base_instructions = self._prompt_manager.build_system_instruction_prompt(account=self._account)
        
        # Create ChatAgent-specific enhanced instructions
        chat_agent_instructions = f"""
# AI Sales Assistant – ChatAgent Mode

## Core Role & Capabilities
{base_instructions}

## Enhanced ChatAgent Architecture

### **Knowledge Base Integration:**
You have access to comprehensive knowledge from the business:

**Base Knowledge (Technical & Industry Information):**
{self._base_knowledge[:2000] if self._base_knowledge else 'Loading from knowledge base...'}

**Brand Knowledge (Company & Product Expertise):**
{self._brand_knowledge[:2000] if self._brand_knowledge else 'Loading from knowledge base...'}

**Product Search Knowledge (Catalog & Categories):**
{self._product_search_knowledge[:1000] if self._product_search_knowledge else 'Loading from knowledge base...'}

### **Handle Locally - No Delegation Needed:**
* **Product Comparisons**: "What's the difference between [Product A] and [Product B]?" → Use brand_knowledge
* **Technical Questions**: "How does [technology/material] work?" → Use base_knowledge  
* **Sizing/Fit Advice**: "What size for [customer profile]?" → Use base_knowledge guidance
* **Brand Heritage**: "Tell me about [company] history" → Use brand_knowledge
* **Product Categories**: "What's [category] good for?" → Use product_search_knowledge
* **Material/Technical Info**: "Why is [material] expensive?" → Use base_knowledge
* **Feature Questions**: "How does [feature] work?" → Use base_knowledge

### **Delegate to Supervisor - Actions Required:**
* **Product Search**: "Find me [product type] under $[amount]" → Requires product_search tool
* **Product Display**: "Show me [specific product]" → Requires display_product tool  
* **Inventory Queries**: "What's available in [specification]?" → Requires real-time search
* **Specific Product Details**: "Pull up [product] specs" → Requires display_product tool

## Modified Conversation Flow

| Step | ChatAgent Action | Approach |
|------|------------------|----------|
| **1 Clarify** | Handle locally | Use knowledge bases for advisory questions |
| **2 Request Tools** | Immediate acknowledgment + delegate | "Great choice, let me pull that up!" + delegate_to_supervisor() |
| **3 Follow-up** | Handle locally if context available | Use conversation state for contextual responses |

**CRITICAL DELEGATION PATTERN**: When you need to call `delegate_to_supervisor`, ALWAYS provide immediate acknowledgment first:

✅ **Correct Pattern**:
"Great choice, let me pull that up!" [calls delegate_to_supervisor tool]

✅ **Contextual Acknowledgment Examples by Request Type**:

**Product Search/Discovery**:
- User: "Find me something under $500"
- ChatAgent: "Nice! Let me look up some great options for you." [calls delegate_to_supervisor]
- User: "What's available in my size?"
- ChatAgent: "Perfect, let me check what we have for you!" [calls delegate_to_supervisor]

**Product Display/Details**:
- User: "Show me the latest product"
- ChatAgent: "Great choice, let me pull that up!" [calls delegate_to_supervisor]
- User: "Can you display the specs?"
- ChatAgent: "Absolutely, let me grab those details for you." [calls delegate_to_supervisor]

**Deep Product Information**:
- User: "Tell me more about the carbon fiber construction"
- ChatAgent: "Good question! Let me get you the detailed technical info." [calls delegate_to_supervisor]
- User: "How does the suspension work?"
- ChatAgent: "Great question, let me pull up the engineering details." [calls delegate_to_supervisor]

**Knowledge Base Queries**:
- User: "What's your return policy?"
- ChatAgent: "Let me look that up for you." [calls delegate_to_supervisor]
- User: "Do you offer financing?"
- ChatAgent: "Good question, let me check on our current options." [calls delegate_to_supervisor]

**Inventory/Availability**:
- User: "Do you have this in stock?"
- ChatAgent: "Let me check availability for you." [calls delegate_to_supervisor]
- User: "What colors are available?"
- ChatAgent: "One sec, let me see what we have in stock." [calls delegate_to_supervisor]

**Comparison Requests** (when requiring real-time data):
- User: "Compare these two models"
- ChatAgent: "Perfect! Let me pull up a detailed comparison for you." [calls delegate_to_supervisor]

**Acknowledgment Phrase Guidelines**:
- **Product Search**: "Let me look up...", "Let me find...", "Let me check what we have..."
- **Product Display**: "Let me pull that up!", "Let me grab that for you!", "Great choice, let me show you..."
- **Technical Details**: "Let me get you the detailed info...", "Let me pull up the technical specs..."
- **Quick Lookups**: "One sec...", "Let me check...", "Give me a moment..."
- **Availability**: "Let me check availability...", "Let me see what's in stock..."

**CRITICAL**: The user should never know there are two agents working together. You are providing a seamless, single-agent experience. Never reference "delegating" or "checking with someone else" - always speak as if YOU are personally handling their request.

**NEVER** call delegate_to_supervisor without first acknowledging the user's request with enthusiasm and contextually appropriate language.

## Dynamic Conversation Context
You will receive dynamic conversation state updates as separate system messages during the conversation. These provide:
- Current conversation phase and context summary
- Products that have been discussed so far
- User preferences and capabilities you can handle locally
- Suggested response strategies based on conversation flow

Use this contextual information to provide more personalized and appropriate responses while maintaining the seamless single-agent experience.
"""
        
        return chat_agent_instructions 

    # ================================
    # Core delegation decision methods (SIMPLIFIED)
    # ================================

    async def delegate_to_supervisor_streaming(self, user_message: str, reason: str = "tool_required"):
        """
        Delegate to SupervisorAssistant with streaming response for immediate speech.
        
        This method streams the SupervisorAssistant's response back as chunks so 
        ChatAgent can start speaking immediately rather than waiting for complete response.
        
        Args:
            user_message: The user's message that requires delegation
            reason: Why delegation is needed
            
        Yields:
            Response chunks from SupervisorAssistant for immediate speaking
        """
        try:
            logger.info(f"ChatAgent delegating to SupervisorAssistant (streaming): {reason}")
            
            # 1. Get current chat context
            chat_ctx = self.chat_ctx.copy()
                
            # 2. If a reason is provided, add it to the chat context
            if reason:
                chat_ctx.items.append(llm.ChatMessage(role="system", content=[reason]))
            
            # 3. Call SupervisorAssistant with full context
            context = {
                "conversation_state": self._conversation_state,
                "chat_context": chat_ctx.items if chat_ctx.items else [],
                "account": self._account,
                "user_id": self._user_id
            }
            
            # 4. Stream response from SupervisorAssistant
            async for chunk in self._supervisor.handle_delegation_streaming(
                user_message, reason, context
            ):
                # Process each chunk and yield immediately for speaking
                processed_chunk = await self._process_supervisor_chunk(chunk)
                if processed_chunk:
                    yield processed_chunk
            
        except Exception as e:
            logger.error(f"Error in delegate_to_supervisor_streaming: {e}")
            yield "I'm having trouble with that request right now. Could you try asking differently?"

    @function_tool()
    async def delegate_to_supervisor(self, user_message: str, reason: str = "tool_required") -> str:
        """
        Delegate to SupervisorAssistant when tools or complex actions are needed.
        
        This is a simple wrapper that will trigger streaming delegation via ChatAgent's
        own LLM processing logic.
        
        Args:
            user_message: The user's message that requires delegation
            reason: Why delegation is needed (optional, but strongly recommended)
            
        Returns:
            Streaming indicator that triggers streaming delegation
        """
        # Store delegation request for streaming processing
        self._pending_delegation = {
            "user_message": user_message,
            "reason": reason,
            "timestamp": time.time()
        }
        
        # Return a special indicator that tells ChatAgent to use streaming
        return "__STREAMING_DELEGATION_REQUESTED__"

    # Note: Delegation is now handled automatically by the @function_tool() decorator
    # The SLM will call delegate_to_supervisor when it determines delegation is needed

    async def _update_conversation_state(self, chat_ctx: llm.ChatContext) -> None:
        """Update conversation state using SupervisorAssistant analysis"""
        try:
            # Delegate state analysis to SupervisorAssistant
            updated_state = await self._supervisor.analyze_conversation_state(
                chat_ctx.copy(), self._conversation_state
            )
            
            # Update our state
            self._conversation_state.update(updated_state)
            
            # Rebuild instructions with new state context
            await self._rebuild_instructions_with_state()
            
        except Exception as e:
            logger.error(f"Error updating conversation state: {e}")

    async def _rebuild_instructions_with_state(self) -> None:
        """
        Update conversation state as separate message for optimal token caching.
        
        This preserves the base system prompt (cacheable) and only adds/updates
        the conversation state as a separate system message.
        """
        try:
            # Build conversation state message
            state_message = f"""Conversation State:
{json.dumps(self._conversation_state, indent=2)}

**Context Summary**: {self._conversation_state.get('context_summary', '')}
**Phase**: {self._conversation_state.get('conversation_phase', 'greeting')}
**Active Products**: {', '.join(self._conversation_state.get('active_products', []))}
**Can Handle Locally**: {', '.join(self._conversation_state.get('can_handle_locally', []))}

Use this context to provide more personalized and contextually appropriate responses."""

            # Get current chat context
            chat_ctx = self.chat_ctx.copy()
            
            # Remove any existing conversation state messages (for updates)
            new_items = []
            for m in chat_ctx.items:
                if not hasattr(m, "role"):
                    new_items.append(m)
                else:
                    content = m.content if isinstance(m.content, str) else " ".join(m.content)
                    if m.role != "system" or not content.startswith("Conversation State:"):
                        new_items.append(m)
            
            # Clear and rebuild with filtered items
            chat_ctx.items.clear()
            chat_ctx.items.extend(new_items)
            
            # Add current conversation state as separate system message
            chat_ctx.add_message(
                role="system",
                content=state_message
            )
            
            # Update the chat context (preserves base system prompt cache)
            await self.update_chat_ctx(chat_ctx=chat_ctx)
            
        except Exception as e:
            logger.error(f"Error rebuilding instructions with state: {e}")

    # ================================
    # Delegation execution methods
    # ================================

    async def _process_supervisor_chunk(self, chunk: str) -> str:
        """
        Process individual chunks from SupervisorAssistant for immediate speaking.
        
        Args:
            chunk: Individual response chunk from SupervisorAssistant
            
        Returns:
            Processed chunk ready for immediate speaking
        """
        try:
            # For now, return chunks as-is, but could add processing here
            # e.g., adding conversational context, filtering, etc.
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing supervisor chunk: {e}")
            return chunk  # Return original chunk if processing fails

    async def _process_supervisor_result(self, supervisor_result: SupervisorResponse) -> str:
        """
        ChatAgent processes supervisor response and crafts final response.
        This gives ChatAgent full control over the final user experience.
        """
        try:
            suggested_message = supervisor_result.suggested_message
            context_updates = supervisor_result.context_updates
            follow_up_options = supervisor_result.follow_up_suggestions
            confidence = supervisor_result.confidence
            
            # Update conversation state with supervisor's context
            self._conversation_state.update(context_updates)
            
            # ChatAgent decides how to use supervisor's response based on confidence
            if confidence > 0.9 and self._message_fits_conversation_flow(suggested_message):
                # High confidence: use supervisor's message as-is
                return suggested_message
                
            elif confidence > 0.7:
                # Medium confidence: enhance supervisor's message with ChatAgent context
                enhanced_message = self._add_conversational_context(suggested_message)
                if follow_up_options:
                    enhanced_message += f" {self._add_natural_follow_up(follow_up_options[0])}"
                return enhanced_message
                
            else:
                # Low confidence: ChatAgent crafts new message using supervisor's work
                return self._craft_contextual_response(context_updates, follow_up_options)
                
        except Exception as e:
            logger.error(f"Error processing supervisor result: {e}")
            return "I found what you're looking for! Let me know if you need anything else."

    def _message_fits_conversation_flow(self, message: str) -> bool:
        """Check if supervisor's message fits current conversation flow"""
        # Simple heuristic - could be enhanced
        return len(message) > 10 and len(message) < 200

    def _add_conversational_context(self, message: str) -> str:
        """Add conversational flow context to supervisor's message"""
        try:
            # Add context based on conversation state
            if self._conversation_state.get("conversation_phase") == "consideration":
                message = f"Great question! {message}"
            elif len(self._conversation_state.get("active_products", [])) > 1:
                message += " Much better than the other options we looked at."
                
            return message
        except Exception as e:
            logger.error(f"Error adding conversational context: {e}")
            return message

    def _add_natural_follow_up(self, follow_up_suggestion: str) -> str:
        """Convert follow-up suggestion to natural Spence voice"""
        follow_up_map = {
            "Ask about sizing": "What size are you thinking?",
            "Ask about colors": "Any particular color catching your eye?", 
            "Show product details": "Want me to pull up all the details?",
            "Compare with other models": "Should we compare it with some other options?",
            "Add to cart": "Ready to add it to your cart?"
        }
        
        return follow_up_map.get(follow_up_suggestion, "What do you think?")

    def _craft_contextual_response(self, context_updates: dict, follow_up_options: list) -> str:
        """ChatAgent crafts new message using supervisor's work when confidence is low"""
        # Fallback contextual response
        if context_updates.get("conversation_phase") == "focus":
            return "I found some great options for you! What interests you most?"
        elif context_updates.get("product_shown"):
            return "Here's what you're looking for! Pretty awesome, right?"
        else:
            return "I've got what you need. What would you like to know?" 

    # ================================
    # Main processing logic (llm_node)
    # ================================

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        """
        Main processing logic for ChatAgent with streaming delegation support.
        
        The SLM (gpt-4.1-nano) has access to delegate_to_supervisor tool and will call it
        when it determines delegation is needed. When delegation is requested, this switches
        to streaming mode for immediate response.
        """
        try:
            # Always update conversation state first (every turn)
            await self._update_conversation_state(chat_ctx)
            
            # Log the processing
            last_user_message = None
            for m in chat_ctx.items:
                if hasattr(m, "role") and m.role == "user":
                    last_user_message = m
            
            if last_user_message and hasattr(last_user_message, 'text_content'):
                logger.info(f"ChatAgent processing: {last_user_message.text_content[:100]}...")
            
            # Use standard LiveKit Agent processing with ChatAgent's tools
            # The SLM will automatically call delegate_to_supervisor when needed
            streaming_triggered = False
            
            async for chunk in Agent.default.llm_node(
                self, 
                chat_ctx=chat_ctx, 
                tools=tools,  # Tools include delegate_to_supervisor 
                model_settings=model_settings
            ):
                # Check if this chunk contains the streaming delegation indicator
                if hasattr(chunk, 'content') and chunk.content and "__STREAMING_DELEGATION_REQUESTED__" in str(chunk.content):
                    logger.info("ChatAgent: Streaming delegation requested, switching to streaming mode")
                    streaming_triggered = True
                    
                    # Start streaming delegation if we have a pending request
                    if self._pending_delegation:
                        async for stream_chunk in self.delegate_to_supervisor_streaming(
                            self._pending_delegation["user_message"],
                            self._pending_delegation["reason"]
                        ):
                            # Create and yield chat chunks from streamed response
                            stream_chat_chunk = llm.ChatChunk(
                                choices=[
                                    llm.ChatChunk.Choice(
                                        delta=llm.ChatMessage(role="assistant", content=[stream_chunk]),
                                        index=0
                                    )
                                ]
                            )
                            yield stream_chat_chunk
                        
                        # Clear pending delegation
                        self._pending_delegation = None
                        break  # Exit the normal chunk processing
                else:
                    # Normal chunk processing if not streaming
                    if not streaming_triggered:
                        yield chunk
                    
        except Exception as e:
            logger.error(f"Error in ChatAgent llm_node: {e}")
            # Fallback to basic behavior
            async for chunk in Agent.default.llm_node(self, chat_ctx=chat_ctx, tools=[], model_settings=model_settings):
                yield chunk

    # ================================
    # Lifecycle methods
    # ================================

    async def on_user_turn_completed(
        self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Handle user turn completion - update state and save"""
        try:
            # Update conversation state after user turn
            await self._update_conversation_state(chat_ctx)
            
            # Update session userdata
            user_state: UserState = self.session.userdata if hasattr(self, 'session') and self.session else None or UserManager.get_user_state(user_id=self._user_id)
            if not user_state:
                user_state = UserState(user_id=self._user_id)
            if hasattr(self, 'session') and self.session:
                self.session.userdata = user_state
                self.session.userdata.last_interaction_time = time.time()
            self.last_stt_message = new_message
                
            UserManager.save_user_state(user_state)
            
            # Save conversation asynchronously
            asyncio.create_task(self._save_conversation_to_storage())
            
        except Exception as e:
            logger.error(f"Error in on_user_turn_completed: {e}")

    async def _save_conversation_to_storage(self, cache: bool = True) -> Optional[str]:
        """Save conversation to storage - delegated to SupervisorAssistant"""
        try:
            if hasattr(self._supervisor, 'save_conversation_to_storage'):
                return await self._supervisor.save_conversation_to_storage(cache)
            else:
                logger.warning("SupervisorAssistant does not have save_conversation_to_storage method")
                return None
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return None

    # ================================
    # Agent lifecycle delegation
    # ================================

    async def on_exit(self) -> None:
        """Handle agent exit"""
        try:
            logger.info("ChatAgent exiting")
            await super().on_exit()
            if hasattr(self._supervisor, 'on_exit'):
                await self._supervisor.on_exit()
        except Exception as e:
            logger.error(f"Error in ChatAgent on_exit: {e}")

    def set_session(self, session) -> None:
        """Set the LiveKit session for proper RPC access"""
        self._session = session
        # Pass session context to SupervisorAssistant
        if hasattr(self._supervisor, 'set_session'):
            self._supervisor.set_session(session)

    def set_user_id(self, user_id: str) -> None:
        """Set user ID for both ChatAgent and SupervisorAssistant"""
        if self._user_id == user_id:
            return
        self._user_id = user_id
        self._session_id = f"{int(time.time())}_{user_id}" if user_id else None
        if hasattr(self._supervisor, 'set_user_id'):
            self._supervisor.set_user_id(user_id)
    
    def get_user_id(self) -> str:
        return self._user_id

    def set_account(self, account: str) -> None:
        """Set account for both ChatAgent and SupervisorAssistant"""
        self._account = account
        if hasattr(self._supervisor, 'set_account'):
            self._supervisor.set_account(account)
    
    def get_account(self) -> str:
        return self._account

    # ================================
    # Voice/session event delegation
    # ================================

    async def on_user_state_changed(self, ev: UserStateChangedEvent):
        """Delegate user state changes to SupervisorAssistant"""
        try:
            if hasattr(self._supervisor, 'on_user_state_changed'):
                await self._supervisor.on_user_state_changed(ev)
        except Exception as e:
            logger.error(f"Error in on_user_state_changed: {e}")

    async def on_agent_state_changed(self, ev: AgentStateChangedEvent):
        """Delegate agent state changes to SupervisorAssistant"""
        try:
            if hasattr(self._supervisor, 'on_agent_state_changed'):
                await self._supervisor.on_agent_state_changed(ev)
        except Exception as e:
            logger.error(f"Error in on_agent_state_changed: {e}")

    async def on_participant_connected(self, participant: RemoteParticipant):
        """Delegate participant connected to SupervisorAssistant"""
        try:
            logger.debug(f"ChatAgent: Participant connected: {participant.identity}")
            if hasattr(self._supervisor, 'on_participant_connected'):
                await self._supervisor.on_participant_connected(participant)
        except Exception as e:
            logger.error(f"Error in on_participant_connected: {e}")

    async def on_participant_disconnected(self, participant: RemoteParticipant):
        """Delegate participant disconnected to SupervisorAssistant"""
        try:
            logger.debug(f"ChatAgent: Participant disconnected: {participant.identity}")
            if hasattr(self._supervisor, 'on_participant_disconnected'):
                await self._supervisor.on_participant_disconnected(participant)
        except Exception as e:
            logger.error(f"Error in on_participant_disconnected: {e}")

    async def on_end_of_turn(self,
                             chat_ctx: llm.ChatContext,
                             new_message: llm.ChatMessage,
                             generating_reply: bool,
                             ) -> None:
        """Handle end of turn"""
        try:
            # Update state after turn
            await self._update_conversation_state(chat_ctx)
        except Exception as e:
            logger.error(f"Error in on_end_of_turn: {e}")

    async def on_room_disconnected(self, ev):
        """Delegate room disconnected to SupervisorAssistant"""
        try:
            logger.debug(f"ChatAgent: Room disconnected: {ev}")
            if hasattr(self._supervisor, 'on_room_disconnected'):
                await self._supervisor.on_room_disconnected(ev)
        except Exception as e:
            logger.error(f"Error in on_room_disconnected: {e}") 