import asyncio
import json
import logging
import os
import re
import random
import time
import hashlib
import psutil

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
from spence.conversation.analyzer import ConversationAnalyzer

from spence.product import Product
from spence.session_state_manager import SessionStateManager
from spence.rag import PineconeRAG
from spence.model import UserState, BasicChatMessage
from spence.llm_service import LlmService
from redis_client import get_user_state, save_user_state, update_user_room_reconnect_time, save_user_latest_conversation
# from spence.storage import get_storage_provider
from spence.account_manager import get_account_manager
from spence.search_service import SearchService
from livekit.rtc import RemoteParticipant
from spence.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class Assistant(Agent):
    
    _current_context: Optional[tuple[str, str]] = None
    
    def __init__(self, ctx: JobContext, primary_model:str, user_id:str=None, chat_ctx: Optional[llm.ChatContext | None] = None, account:str=None):
        # Track startup timing for optimization
        startup_start = time.time()
        
        # Phase 1: Prompt Manager initialization
        prompt_init_start = time.time()
        prompts_dir = os.getenv("PROMPTS_DIR", None)
        self._prompt_manager = PromptManager(account=account, prompts_dir=prompts_dir)
        prompt_init_time = time.time() - prompt_init_start
        
        # Phase 2: Product loading setup (KISS approach)
        product_setup_start = time.time()
        self._products_loaded = False
        self._product_load_task = None
        self._product_load_timing = None  # Will store detailed timing after loading
        
        if account:
            # Start product loading in background - doesn't block Assistant startup
            self._product_load_task = asyncio.create_task(self._load_products_for_account(account))
        
        product_setup_time = time.time() - product_setup_start
        
        # Phase 3: Agent initialization
        agent_init_start = time.time()
        super().__init__(
            instructions=self._prompt_manager.build_system_instruction_prompt(account=account), 
            chat_ctx=chat_ctx
        )
        agent_init_time = time.time() - agent_init_start
        
        # Phase 4: Core attribute setup
        attr_setup_start = time.time()
        self._ctx = ctx
        self._primary_model = primary_model
        self._user_id = user_id
        self._session_id = f"{int(time.time())}_{user_id}" if user_id else None
        self._current_context = None
        self._usage_collector = metrics.UsageCollector()
        self._account = account
        self.last_stt_message = None
        attr_setup_time = time.time() - attr_setup_start
        
        # Phase 5: LLM client setup
        llm_setup_start = time.time()
        # self._fast_llm = groq.LLM(model="llama-3.1-8b-instant")
        self._fast_llm = openai.LLM(model="gpt-4.1-nano")
        self._fast_llm_prompt = llm.ChatMessage(
            role="system",
            content=[
                "Based on the context of the conversation, first:\n"
                "1. Determine if a filler response should be used to give the assistant time to think and/or perform the appropriate tool call.\n"
                "2. If a filler response is needed, generate a short instant response to the user's message with 2 to 5 words.\n"
                "3. If no response is needed, reply with <NO_RESPONSE>.\n"
                "4. If a response is needed, generate a short instant response to the user's message with 2 to 5 words.\n"
                "Example: "
                "- **BAD**: "
                "  - User: 'what's the weather like?' "
                "  - Assistant: 'let me check that for you'. "
                "  - Your Reply: 'one sec...'\n"
                "- **GOOD**: "
                "  - User: 'what's the weather like?' "
                "  - Assistant: 'let me check that for you'. "
                "  - Your Reply: '<NO_RESPONSE>'\n"
                "Example: "
                "- **BAD**: "
                "  - User: 'what's the weather like?' "
                "  - Your Reply: '<NO_RESPONSE>'.\n"
                "- **GOOD**: "
                "  - User: 'what's the weather like?' "
                "  - Your Reply: 'one sec...'\n"
                
            ],
        )

        # Use AsyncCerebras for non-blocking API calls
        self.async_cerebras_client = AsyncCerebras(
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )
        # Keep the sync client for any remaining sync usage
        self.cerebras_client = Cerebras(
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )
        llm_setup_time = time.time() - llm_setup_start
        
        # Phase 6: Background task setup
        bg_task_start = time.time()
        # prewarm the LLM
        asyncio.create_task(self._prewarm_llm())
        
        self.inactivity_task: asyncio.Task | None = None
        self._closing_task: asyncio.Task[None] | None = None
        self._is_closing_session = False
        
        self._planned_reconnect: bool = False
        self._initital_close_on_disconnect: bool | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        bg_task_time = time.time() - bg_task_start
        
        # Store startup timing for monitoring
        total_startup_time = time.time() - startup_start
        self._startup_timing = {
            "account": account,
            "timing": {
                "prompt_init": prompt_init_time,
                "product_setup": product_setup_time, 
                "agent_init": agent_init_time,
                "attr_setup": attr_setup_time,
                "llm_setup": llm_setup_time,
                "bg_task_setup": bg_task_time,
                "total_startup": total_startup_time
            },
            "performance": {
                "startup_target": 1.0,  # 1 second target
                "meets_target": total_startup_time < 1.0,
                "startup_efficiency": 1.0 / total_startup_time if total_startup_time > 0 else 0
            },
            "timestamp": time.time()
        }
        
        # Log startup performance
        logger.info(f"🚀 Assistant startup complete for {account}: {total_startup_time:.3f}s "
                   f"(prompt: {prompt_init_time:.3f}s, agent: {agent_init_time:.3f}s, "
                   f"llm: {llm_setup_time:.3f}s) | "
                   f"Target: {'✅' if total_startup_time < 1.0 else '⚠️'} <1s")
        
        if total_startup_time > 1.0:
            logger.warning(f"⚠️ Assistant startup exceeded 1s target: {total_startup_time:.3f}s for {account}")

    def get_performance_metrics(self) -> dict:
        """
        Get comprehensive performance metrics for this Assistant instance.
        
        Returns:
            Dictionary with startup timing, product loading performance, and memory usage
        """
        metrics = {
            "account": self._account,
            "startup": self._startup_timing,
            "product_loading": self._product_load_timing,
            "status": {
                "products_loaded": self._products_loaded,
                "startup_completed": hasattr(self, '_startup_timing'),
                "product_task_status": "completed" if self._products_loaded else (
                    "running" if self._product_load_task and not self._product_load_task.done() else "pending"
                )
            },
            "targets": {
                "startup_time": "< 1.0s",
                "product_loading": "< 2.0s", 
                "memory_usage": "< 10MB",
                "meets_startup_target": (
                    self._startup_timing.get("performance", {}).get("meets_target", False) 
                    if hasattr(self, '_startup_timing') else False
                ),
                "meets_product_target": (
                    self._product_load_timing.get("timing", {}).get("total_time", 0) < 2.0
                    if self._product_load_timing else False
                ),
                "meets_memory_target": (
                    self._product_load_timing.get("memory", {}).get("delta_mb", 0) < 10.0
                    if self._product_load_timing else False
                )
            }
        }
        
        return metrics

    async def get_performance_summary(self) -> str:
        """
        Get a human-readable performance summary for monitoring and debugging.
        
        Returns:
            Formatted string with performance metrics and target compliance
        """
        # Ensure products are loaded to get complete metrics
        await self.ensure_products_loaded()
        
        metrics = self.get_performance_metrics()
        
        # Extract key metrics
        startup = metrics.get("startup", {}).get("timing", {})
        product = metrics.get("product_loading", {}).get("timing", {}) if metrics.get("product_loading") else {}
        memory = metrics.get("product_loading", {}).get("memory", {}) if metrics.get("product_loading") else {}
        targets = metrics.get("targets", {})
        
        startup_time = startup.get("total_startup", 0)
        product_time = product.get("total_time", 0)
        memory_mb = memory.get("delta_mb", 0)
        product_count = metrics.get("product_loading", {}).get("product_count", 0) if metrics.get("product_loading") else 0
        
        # Format summary
        summary = f"""
🚀 Assistant Performance Summary - {self._account}

Startup Performance:
  ⏱️  Total Startup: {startup_time:.3f}s {'✅' if targets.get('meets_startup_target') else '⚠️'} (target: <1s)
  📊 Phase Breakdown:
     - Prompt Init: {startup.get('prompt_init', 0):.3f}s
     - Agent Init: {startup.get('agent_init', 0):.3f}s  
     - LLM Setup: {startup.get('llm_setup', 0):.3f}s

Product Loading Performance:
  ⏱️  Loading Time: {product_time:.3f}s {'✅' if targets.get('meets_product_target') else '⚠️'} (target: <2s)
  📦 Products: {product_count} loaded
  💾 Memory Usage: {memory_mb:.1f}MB {'✅' if targets.get('meets_memory_target') else '⚠️'} (target: <10MB)
  📈 Efficiency: {memory.get('load_efficiency', 0):.1f} products/MB

Overall Status: {'🎯 All targets met!' if all([targets.get('meets_startup_target'), targets.get('meets_product_target'), targets.get('meets_memory_target')]) else '⚠️ Some targets exceeded'}
"""
        
        return summary.strip()

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

    async def _load_products_for_account(self, account: str) -> None:
        """
        Load product catalog for this Assistant's account (KISS approach).
        
        Voice agents serve one account at a time, so we load only what we need:
        - Single account catalog loading
        - Direct memory access after loading  
        - No Redis complexity - perfect for stateful applications
        
        Args:
            account: Account domain for this voice agent
        """
        # Get process for memory monitoring
        process = psutil.Process(os.getpid())
        
        try:
            # Phase 1: Initialization timing
            total_start = time.time()
            init_start = time.time()
            
            from .product import Product
            from .product_manager import load_product_catalog_for_assistant
            
            init_time = time.time() - init_start
            
            # Phase 2: Memory baseline measurement
            memory_start = process.memory_info().rss / 1024 / 1024  # MB
            
            # Phase 3: Product loading timing
            load_start = time.time()
            manager = await load_product_catalog_for_assistant(account)
            products = await manager.get_product_objects()
            load_time = time.time() - load_start
            
            # Phase 4: Memory after loading measurement  
            memory_end = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_end - memory_start
            
            # Phase 5: Finalization
            finalize_start = time.time()
            self._products_loaded = True
            
            # Store timing metadata for monitoring
            self._product_load_timing = {
                "account": account,
                "product_count": len(products),
                "timing": {
                    "init_time": init_time,
                    "load_time": load_time,
                    "total_time": time.time() - total_start,
                    "finalize_time": time.time() - finalize_start
                },
                "memory": {
                    "baseline_mb": memory_start,
                    "final_mb": memory_end,
                    "delta_mb": memory_delta,
                    "products_mb": memory_delta  # Approximate product catalog memory
                },
                "performance": {
                    "products_per_second": len(products) / load_time if load_time > 0 else 0,
                    "mb_per_second": memory_delta / load_time if load_time > 0 else 0,
                    "load_efficiency": len(products) / memory_delta if memory_delta > 0 else 0
                },
                "timestamp": time.time()
            }
            
            total_time = time.time() - total_start
            
            # Enhanced logging with performance metrics
            logger.info(f"🚀 Assistant product loading complete for {account}: "
                       f"{len(products)} products loaded in {total_time:.3f}s "
                       f"(init: {init_time:.3f}s, load: {load_time:.3f}s) "
                       f"| Memory: {memory_delta:.1f}MB | "
                       f"Efficiency: {len(products) / memory_delta if memory_delta > 0 else 0:.1f} products/MB")
            
            # Performance validation and warnings
            if total_time > 2.0:
                logger.warning(f"⚠️ Product loading exceeded 2s target: {total_time:.3f}s for {account}")
            
            if memory_delta > 10.0:
                logger.warning(f"⚠️ Product catalog memory exceeded 10MB target: {memory_delta:.1f}MB for {account}")
            
            if len(products) == 0:
                logger.warning(f"⚠️ No products loaded for account {account}")
            
        except Exception as e:
            logger.error(f"Error loading products for Assistant account {account}: {e}")
            self._products_loaded = False
            
            # Store error timing for debugging
            self._product_load_timing = {
                "account": account,
                "error": str(e),
                "timing": {
                    "total_time": time.time() - total_start,
                    "failed_at": "product_loading"
                },
                "timestamp": time.time()
            }

    async def ensure_products_loaded(self) -> bool:
        """
        Ensure products are loaded before using product-related functions.
        
        This method can be called by product search functions to ensure
        the product catalog is ready.
        
        Returns:
            True if products are loaded or loading completed successfully
        """
        if self._products_loaded:
            return True
            
        if self._product_load_task and not self._product_load_task.done():
            try:
                # Wait for product loading to complete
                await asyncio.wait_for(self._product_load_task, timeout=30.0)
                return self._products_loaded
            except asyncio.TimeoutError:
                logger.warning(f"Product loading timeout for account {self._account}")
                return False
            except Exception as e:
                logger.error(f"Error waiting for product loading: {e}")
                return False
        
        return self._products_loaded

    def set_user_id(self, user_id:str) -> None:
        if self._user_id == user_id:
            return
        self._user_id = user_id
        self._session_id = f"{int(time.time())}_{user_id}" if user_id else None
    
    def get_user_id(self) -> str:
        return self._user_id

    def set_account(self, account:str) -> None:
        self._account = account
    
    def get_account(self) -> str:
        return self._account
    
    def set_current_context(self, context:tuple[str, str]) -> None:
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
        
    async def _process_intent_detection(self, last_user_message, chat_ctx: llm.ChatContext) -> None:
        """Process intent detection asynchronously in the background"""
        try:
            if last_user_message is None or not last_user_message.text_content:
                return
                
            # get the first `system` message from the chat_ctx
            system_prompt = self._prompt_manager.get_system_prompt()
            last_system_message = None
            for m in chat_ctx.items:
                if hasattr(m, "role") and m.role == "system":
                    last_system_message = m.content
            
            available_intents = [
                {
                    "name": "display_product",
                    "description": "Use this to display or show exactly one products. The tool call itself will handle actually doing this. Never try to display or show products on your own, ALWAYS use this tool call. It is not possible to show multiple products at once, so you must narrow it down to a single product.",
                    "parameters": {
                        "product_id": "string",
                        "variant": "string",
                        "resumption_message": "string"
                    }
                },
                {
                    "name": "display_product_list",
                    "description": "Use this to display a list of products. The tool call itself will handle actually doing this. Never try to display or show products on your own, ALWAYS use this tool call. It is not possible to show multiple products at once, so you must narrow it down to a single product.",
                    "parameters": {
                        "product_ids": "string[]",
                        "resumption_message": "string"
                    }
                },
                {
                    "name": "display_product_image",
                    "description": "Use this to display an image of a product. The tool call itself will handle actually doing this. Never try to display or show products on your own, ALWAYS use this tool call. The query is a natural language query that will be used to search for the product image.",
                    "parameters": {
                        "product_id": "string",
                        "query": "string"
                    }
                },
                {
                    "name": "add_to_cart",
                    "description": "Use this to add a product to the user's cart. The tool call itself will handle actually doing this. Never try to add to the cart on your own, ALWAYS use this tool call. The product_id is the ID of the product to add to the cart.",
                    "parameters": {
                        "product_id": "string",
                        "variant": "string",
                        "quantity": "integer"
                    }
                }
            ]    
                
            # build the system message for the intent detection
            intent_system_message = f"""
Your goal is to translate the user text into as many **Intents** as possible. The Intents are defined below and are restricted to User Interface Intents. So only identify Intents that are related to the User Interface.

* The user may not be obvious with their **Intents**
  * Thus, you must imply the user's intent from their input
* The user may give multiple commands in one input, and you should try to identify all of them
* It is given that the user is expressing one of the following intents
* It is always better to identify an intent than to not identify one
* It may be helpful not to take the user's input literally, but rather to interpret it in the context of the intents

"""
            if system_prompt and False:
                intent_system_message += f"""
The user gives you the following additional instructions and information about their situation:
```
{system_prompt}
```
"""
            if last_system_message:
                intent_system_message += f"""
The user gives you the following context and data about their situation:
```
{last_system_message}
```
"""
            if len(available_intents) > 0:
                intent_system_message += f"""
# Intent Descriptions

Here is the schema for the parameters, as well as descriptions for some parameters that you should use to guide your decisions.

"""
                for intent in available_intents:
                    intent_system_message += f"""
## {intent["name"]}: {intent["description"]}

Parameter Schema and Description:
{intent["parameters"]}
"""
            intent_system_message += f"""

# Output Format

* You will return an array of objects with two properties:
    * `name` of the intent
    * `parameters` of the intent
* Follow the given schema when returning the intents
* You may and should return multiple intents when the user's command includes multiple actions
* You should always return an intent, if not more
* Only if you are completely unsure, you should return a length-1 array with one intent with name \\`other\\` and blank parameters, but use this sparingly.
* NEVER return `other` with other intents
"""
                        
            start_time = time.time()
            
            # Use the async Cerebras client - no need for run_in_executor!
            completion_create_response = await self.async_cerebras_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": intent_system_message
                    },
                    {
                        "role": "user",
                        "content": f"Translate the following user input into intents. Respond with JSON:\n\n```{last_user_message.text_content}```"
                    }
                ],
                model="llama-4-scout-17b-16e-instruct",
                stream=False,
                max_completion_tokens=4096,
                temperature=0.2,
                top_p=1,
                response_format={ "type": "json_object" }
            )
            
            end_time = time.time()
            print(f"Cerebras response time: {end_time - start_time} seconds")

            # fish out the content of the response as JSON
            response_content = completion_create_response.choices[0].message.content
            response_json = json.loads(response_content)
            
            # if the response is not empty, then nicely print out the intent(s)
            if response_json:
                print(f"Intents: {response_json}")
                # Here you could add logic to handle the detected intents
                # For example, trigger UI actions or store them for later use
            else:
                print("No intents found")
                
        except Exception as e:
            logger.error(f"Error in _process_intent_detection: {e}")

    @observe(name="llm_node", as_type="generation")
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        """Process the text with the LLM model"""

        self._prompt_manager.update_current_generation()
        
        # get the last user message from the chat_ctx
        last_user_message = None
        for m in chat_ctx.items:
            if hasattr(m, "role") and m.role == "user":
                last_user_message = m
        if last_user_message and (not self.last_stt_message or last_user_message.id != self.last_stt_message.id):
            try:
                json.loads(last_user_message.text_content)
                is_json_payload = True
                
                # if the agent is speaking, then do nothing
                if self.session.current_speech:
                    return
            except:
                pass
        
        await self._update_user_state_prompt()

        # # Run intent detection asynchronously in the background
        # if last_user_message is not None and last_user_message.text_content:
        #     # Create a background task for intent detection
        #     asyncio.create_task(self._process_intent_detection(last_user_message, chat_ctx))

        return Agent.default.llm_node(self, chat_ctx=chat_ctx, tools=tools, model_settings=model_settings)

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        """Process text-to-speech with timeout handling for responsiveness"""
        pronunciations = {
            "API": "A P I",
            "REST": "rest",
            "SQL": "sequel",
            "kubectl": "kube control",
            "AWS": "A W S",
            "UI": "U I",
            "URL": "U R L",
            "npm": "N P M",
            "LiveKit": "Live Kit",
            "async": "a sink",
            "nginx": "engine x",
            "Dara": "Dare-uh",
            "Dara's": "Dare-uh's",
            "Daras": "Dare-uhs",
            "DaraKaye": "Dare-uh kay",
            "DaraKayes": "Dare-uh kayes",
            "DaraKayes's": "Dare-uh kayes's",
            "derailleur": "dih-RAY-ler",
        }
        
        async def adjust_pronunciation(input_text: AsyncIterable[str]) -> AsyncIterable[str]:
            async for chunk in input_text:
                modified_chunk = chunk
                
                # Apply pronunciation rules
                for term, pronunciation in pronunciations.items():
                    # Use word boundaries to avoid partial replacements
                    modified_chunk = re.sub(
                        rf'\b{term}\b',
                        pronunciation,
                        modified_chunk,
                        flags=re.IGNORECASE
                    )
                
                yield modified_chunk
        
        # Process with modified text through base TTS implementation
        async for frame in Agent.default.tts_node(
            self,
            adjust_pronunciation(text),
            model_settings
        ):
            yield frame

        # return Agent.default.tts_node(
        #     self, 
        #     tokenize.utils.replace_words(
        #         text=text, replacements={
        #             # "livekit": r"<<l|aɪ|v|k|ɪ|t|>>",
        #             # "derailleur": r"<<d|ɪ|ˈr|eɪ|l|ər|>>",
        #             # "alloy": r"<<æ|l|ɔɪ|>>",
        #         }
        #     ),
        #     model_settings
        # )

    async def _update_user_state_prompt(self) -> None:
        try:
            user_id = self._user_id
            if user_id:
                user_state_message = await SessionStateManager.build_user_state_message(user_id=user_id, include_current_page=True, user_state=self.session.userdata, include_product_history=True)
                if user_state_message:
                    chat_ctx = self.chat_ctx.copy()
                    new_items = []
                    for m in chat_ctx.items:
                        if not hasattr(m, "role"):
                            new_items.append(m)
                        else:
                            content = m.content if isinstance(m.content, str) else " ".join(m.content)
                            if m.role != "system" or not content.startswith("User State:"):
                                new_items.append(m)
                    chat_ctx.items.clear()
                    chat_ctx.items.extend(new_items)

                    chat_ctx.add_message(
                        role="system",
                        content=f"User State:\n{user_state_message}"
                    )
                    
                    await self.update_chat_ctx(chat_ctx=chat_ctx)
        except Exception as e:
            logger.error(f"Error updating user state prompt: {e}")


    async def on_user_turn_completed(
        self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        user_state: UserState = self.session.userdata or get_user_state(user_id=self._user_id)
        if not user_state:
            user_state = UserState(user_id=self._user_id)
        self.session.userdata = user_state
        self.session.userdata.last_interaction_time = time.time()
        self.last_stt_message = new_message
            
        save_user_state(user_state=user_state)
        
        # asyncio.create_task(self.save_conversation_to_storage())

    async def on_agent_state_changed(self, ev: AgentStateChangedEvent):
        pass
    

    async def user_presence_task(self, initial_delay: float = 0.0):
        if self._is_closing_session:
            return
    
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        
        # try to ping the user 3 times, if we get no answer, close the session
        try:
            for _ in range(3):
                await self.session.generate_reply(
                    instructions=(
                        "The user has been inactive. Politely check if the user is still present, "
                        "gently guiding the conversation back toward your intended goal. For example, "
                        "you can simply say 'Just checking in' or 'Just making sure you're still here'."
                    )
                )
                await asyncio.sleep(15)
        except Exception as e:
            logger.error(f"Error in user_presence_task: {e}")
        
        self._closing_task = asyncio.create_task(self.session.aclose())

    async def on_user_state_changed(self, ev: UserStateChangedEvent):
        # pass
        if ev.new_state == "away":
            self.inactivity_task = asyncio.create_task(self.user_presence_task())
            return

        # ev.new_state: listening, speaking, ..
        self._cancel_inactivity_task()
    
    async def on_participant_connected(self, participant: RemoteParticipant):
        logger.debug(f"Participant connected: {participant.identity}")
        self._cancel_inactivity_task()
                
        # say something to the user if not already speaking
        speech = self.session.current_speech
        if not speech and not self._planned_reconnect:
            await self.session.generate_reply(
                instructions="The user has just connected, say something appropriate based on the context"
            )
        else:
            logger.debug("User already speaking, skipping greeting")
        self._cancel_planned_reconnect()
        # await self.session.generate_reply(
        #     instructions="The user has just connected, say something appropriate based on the context"
        # )


    async def on_participant_disconnected(self, participant: RemoteParticipant):
        logger.debug(f"Participant disconnected: {participant.identity}")
        # try:
        #     user_id = self._user_id
        #     if not user_id:
        #         logger.warning("No user_id available for participant disconnect event")
        #         return
        
        #     # try:
        #     #     # messages = self.chat_ctx.copy().items
        #     #     # if not messages:
        #     #     #     logger.info("No messages to analyze for disconnected participant")
        #     #     #     return
                
        #     #     # await self.save_conversation_to_storage()

        #     #     # TODO: Should this be done "offline"?                
        #     #     # try:
        #     #     #     conversation_analyzer = ConversationAnalyzer(user_id=user_id)
        #     #     #     userdata = getattr(self.session, "userdata", None)
        #     #     #     await conversation_analyzer.analyze_chat(
        #     #     #         user_state=userdata, 
        #     #     #         transcript=messages, 
        #     #     #         source="chat", 
        #     #     #         source_id=userdata.voice_session_id if userdata and hasattr(userdata, "voice_session_id") else None
        #     #     #     )
        #     #     #     logger.debug(f"Generated resumption message for {user_id}")
        #     #     # except Exception as analysis_error:
        #     #     #     logger.warning(f"Error in conversation analysis: {analysis_error}")
        #     # except Exception as inner_error:
        #     #     logger.error(f"Error processing disconnect event data: {inner_error}")
            
        # except Exception as e:
        #     logger.error(f"Error handling participant disconnect: {e}")

        return

    def _cancel_inactivity_task(self):
        if self.inactivity_task is not None:
            try:
                self.inactivity_task.cancel()
            except Exception as e:
                logger.error(f"Error in _cancel_inactivity_task: {e}")
            self.inactivity_task = None
    
    def _setup_planned_reconnect(self):
        self._planned_reconnect = True
        # hack to turn off auto disconnect
        if self.session._room_io._input_options.close_on_disconnect:
            self._initital_close_on_disconnect = True
            self.session._room_io._input_options.close_on_disconnect = False
        else:
            self._initital_close_on_disconnect = None
        self._reconnect_task = asyncio.create_task(self._reconnect_timer())
    
    # setup a timer to turn off the planned reconnect
    async def _reconnect_timer(self):
        await asyncio.sleep(10)
        self._cancel_planned_reconnect()

    def _cancel_planned_reconnect(self):
        self._planned_reconnect = False
        if self._initital_close_on_disconnect is not None:
            self.session._room_io._input_options.close_on_disconnect = self._initital_close_on_disconnect
            self._initital_close_on_disconnect = None
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            self._reconnect_task = None

    async def on_end_of_turn(self,
                             chat_ctx: llm.ChatContext,
                             new_message: llm.ChatMessage,
                             generating_reply: bool,
                             ) -> None:
        pass

    async def on_room_disconnected(self, ev):
        logger.debug(f"Room disconnected: {ev}")
        # try:
        #     await self.save_conversation_to_storage()
        # except Exception as e:
        #     logger.error(f"Error in room disconnect handler: {e}")

    wait_phrases = [
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
    
    async def on_participant_attributes_changed(self, attributes: dict[str, str], participant: Participant):
        # logger.debug(f"Participant attributes changed: {attributes}")
        pass

    async def on_participant_metadata_changed(self, ev):
        # logger.debug(f"Participant metadata changed: {ev}")
        pass
    
    async def generate_and_say_wait_phrase_if_needed(self, use_fast_llm: bool = False, say_filler: bool = True, force: bool=False) -> Optional[str]:
        try:
            # if self.session.current_speech:
            #     await self.session.current_speech.wait_for_playout()
            
            if not use_fast_llm:
                wait_phrase = random.choice(self.wait_phrases)
                if say_filler:
                    await self.session.say(wait_phrase, add_to_chat_ctx=False)
                return wait_phrase
            
            chat_ctx = self.chat_ctx.copy(
                exclude_function_call=True, exclude_instructions=True, tools=[]
            ).truncate(max_items=5)
            
            if chat_ctx.items and len(chat_ctx.items) > 0:
                most_recent_message = chat_ctx.items[-1]
                if force or hasattr(most_recent_message, "role") and most_recent_message.role == "user":
                    chat_ctx.items.insert(0, self._fast_llm_prompt)
                    chat_ctx.items.append(llm.ChatMessage(role="user", content=["Based on the above conversation, generate a short instant response to the user's message with 2 to 10 words indicating that you are looking up the information. For example, 'one sec...' or 'let me check that for you'. Be sure your response is short and concise yet contextually relevant and natural."]))
                    fast_response = await LlmService.chat_wrapper(
                        llm_service=self._fast_llm,
                        chat_ctx=chat_ctx,
                    )
                    logger.debug(f"Fast response: {fast_response}")
                    if say_filler:
                        await self.session.say(fast_response, add_to_chat_ctx=True)
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
        _context: RunContext,
        query: str,
        topic_name: str = "",
        top_k: int = 10,
        top_n: int = 5,
        min_score: float = 0.15,
        _min_n: int = 0
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
            # Use enhanced knowledge search with status updates
            results = await self.perform_search_with_status(
                search_fn=lambda q, **params: SearchService.search_knowledge_rag_with_context(
                    query=q,
                    user_state=params.get("user_state"),
                    chat_ctx=params.get("chat_ctx"),
                    account=params.get("account"),
                    knowledge_base=params.get("knowledge_base", ""),
                    top_k=params.get("top_k", 20),
                    top_n=params.get("top_n", 5),
                    min_score=params.get("min_score", 0.15)
                ),
                query=search_query,
                search_params={
                    "user_state": self.session.userdata,
                    "chat_ctx": self.chat_ctx,
                    "account": self._account,
                    "knowledge_base": self._prompt_manager.knowledge_search_guidance or "",
                    "top_k": top_k,
                    "top_n": top_n,
                    "min_score": min_score
                }
            )

            # Process results with enhanced metadata
            articles = []
            if results and not isinstance(results, dict) or not results.get("error"):
                for res in results:
                    article_data = {
                        "title": res.get("metadata", {}).get("title", "Untitled"),
                        "content": res.get("text", ""),
                        "score": res.get("score", 0),
                        "type": res.get("type", "knowledge")
                    }
                    
                    # Add any additional metadata
                    if "metadata" in res and isinstance(res["metadata"], dict):
                        for key, value in res["metadata"].items():
                            if key not in article_data:
                                article_data[key] = value
                    
                    articles.append(article_data)
            
            # Sort by relevance score
            articles.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            logger.info(f"Enhanced knowledge search returned {len(articles)} articles")
            
            return {"articles": articles}
            
        except Exception as e:
            logger.error(f"Error in knowledge_search function: {e}")
            return {"error": str(e)}

    @observe(name="product_search")
    @function_tool()
    async def product_search(
        self,
        _context: RunContext[UserState],
        query: str,
    ):
        """Use this to search for products that the user may be interested in. Include as much depth and detail in the query as possible to get the best results. Generally, the only way for you to know about individual products is through this tool call, so feel free to liberally call this tool call.
        
        Args:
            query: The query to use for the search. Be as descriptive and detailed as possible based on the context of the entire conversation. For example, the query "mountain bike" is not very descriptive, but "mid-range non-electric mountain bike with electronic shifting for downhill flowy trails for advanced rider" is much more descriptive. The more descriptive the query, the better the results will be. You can also use this to search for specific products by name or ID, but be sure to include as much context as possible to help narrow down the results.
        """
        try:
            # Input validation
            if not query:
                raise ToolError("`query` is required")
        
            # Ensure products are loaded before searching (KISS approach)
            products_ready = await self.ensure_products_loaded()
            if not products_ready:
                logger.warning(f"Products not loaded for account {self._account}, falling back to legacy loading")
        
            # Get products using ProductManager
            from .product_manager import get_products_for_account
            products = await get_products_for_account(self._account)
            products_results = []
            
            account_manager = await get_account_manager(self._account)
            index_name = account_manager.get_rag_details()
            
            # Choose search approach based on catalog size and use enhanced search
            if not index_name:
                # Small catalog or no RAG: Use LLM-based search with enhanced query
                logger.info(f"Using LLM-based search for {self._account} (no RAG index)")
                
                # Enhance the query even for LLM search
                enhanced_query, extracted_filters = await SearchService.enhance_product_query_with_filters(
                    query=query,
                    user_state=self.session.userdata,
                    chat_ctx=self.chat_ctx,
                    account=self._account,
                    product_knowledge=self._prompt_manager.product_search_knowledge or ""
                )
                
                logger.debug(f"Enhanced query for LLM search: {enhanced_query}")
                logger.debug(f"Extracted filters: {extracted_filters}")
                
                # Use enhanced query for LLM search with status updates
                products_results = await self.perform_search_with_status(
                    search_fn=lambda q, **params: SearchService.search_products_llm(
                        q,
                        products=params.get("products", []),
                        user_state=self.session.userdata
                    ),
                    query=enhanced_query,  # Use enhanced query
                    search_params={
                        "products": products,
                        "account": self._account
                    }
                )
            else:
                # Large catalog with RAG: Use enhanced RAG search with filters
                logger.info(f"Using enhanced RAG search for {self._account}")
                
                # First, enhance the query and extract filters
                enhanced_query, extracted_filters = await SearchService.enhance_product_query_with_filters(
                    query=query,
                    user_state=self.session.userdata,
                    chat_ctx=self.chat_ctx,
                    account=self._account,
                    product_knowledge=self._prompt_manager.product_search_knowledge or ""
                )
                
                logger.info(f"Query enhanced: '{query}' -> '{enhanced_query}'")
                logger.info(f"Filters extracted: {extracted_filters}")
                
                # Use the new enhanced RAG search with filters
                rag_results = await self.perform_search_with_status(
                    search_fn=lambda q, **params: SearchService.search_products_rag_with_filters(
                        query=q,
                        filters=params.get("filters", {}),
                        account=params.get("account"),
                        top_k=params.get("top_k", 35),
                        top_n=params.get("top_n", 10),
                        min_score=params.get("min_score", 0.15),
                        min_n=params.get("min_n", 3)
                    ),
                    query=enhanced_query,
                    search_params={
                        "filters": extracted_filters,
                        "account": self._account,
                        "top_k": 35,
                        "top_n": 10,
                        "min_score": 0.15,
                        "min_n": 3
                    }
                )
                
                # Process RAG results into product objects
                if rag_results and not isinstance(rag_results, dict) or not rag_results.get("error"):
                    for result in rag_results:
                        if result.get('metadata'):
                            # Add filter match information for better ranking
                            product = Product.from_metadata(result['metadata'])
                            # Store relevance and filter match info
                            product._relevance_score = result.get('score', 0)
                            product._matched_filters = result.get('matched_filters', [])
                            products_results.append(product)
                            
                logger.info(f"Found {len(products_results)} products via enhanced RAG search")
                
                # Log filter effectiveness
                if products_results and extracted_filters:
                    filter_matches = sum(len(p._matched_filters) for p in products_results if hasattr(p, '_matched_filters'))
                    logger.info(f"Filter effectiveness: {filter_matches} total filter matches across results")
            
            # Format results as markdown
            markdown_results = "# Products (in descending order):\n\nRemember that you are an AI sales agent speaking to a human, so be sure to use natural language to respond and not just a list of products. For example, never say the product URL unless the user specifically asks for it.\n\n"
            for i, result in enumerate(products_results):
                markdown_results += f"{Product.to_markdown(result, depth=1)}\n\n"
                
            markdown_results += f"\n\n**CRITICAL**: Be sure to call the `display_product` tool call to display the product that you talk about. You can use the product ID from the results above to do this.\n\n"

            return markdown_results
            
        except Exception as e:
            logger.error(f"Error in product_search: {e}")
            raise ToolError("Unable to search for products")
    
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
            
            from .product_manager import find_product_by_id, get_products_for_account
            
            product = await find_product_by_id(self._account, product_id)
            products = []
            
            if not product:
                products = await get_products_for_account(self._account)
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
            
            user_state = self.session.userdata
            history = SessionStateManager.get_user_recent_history(user_id=user_state.user_id)

            current_product = None
            if history and len(history) > 0:
                most_recent_history = history[0]
                if most_recent_history.url:
                    product_url = most_recent_history.url
                    from .product_manager import find_product_by_url
                    current_product = await find_product_by_url(user_state.account, product_url)
            
            await self.session.current_speech.wait_for_playout()
            
            participant_identity = next(iter(get_job_context().room.remote_participants))
            local_participant = get_job_context().room.local_participant
            
            # Because this potentially causes the UI to do a hard nav to a new page, update the should_reconnect_until timeout for this account/user_id
            update_user_room_reconnect_time(
                account=self._account,
                user_id=user_state.user_id,
                room_id=get_job_context().room.name,
                auto_reconnect_until=time.time() + 60 * 5,  # 30 seconds
                resumption_message=resumption_message
            )
            
            try:
                self._setup_planned_reconnect()
                rpc_result = await local_participant.perform_rpc(
                    destination_identity=participant_identity,
                    method="display_products",
                    payload=json.dumps({"products": [product_results], "resumptionMessage": resumption_message}),
                    response_timeout=10.0,
                )
            except Exception as e:
                if isinstance(e, RpcError) or e.message == "Response timeout":
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
            if isinstance(e, RpcError) or e.message == "Response timeout":
                logger.error(f"RpcError in display_product: {e}")
                # Ignore for now
            elif isinstance(e, ConnectionError):
                logger.error(f"ConnectionError in display_product: {e}")
                # Ignore for now
            else:
                logger.error(f"Error displaying product: {e}")
                raise ToolError("Unable to show product: {e}")

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
            SessionStateManager.store_conversation_exit_reason(user_state=self.session.userdata, reason=reason)
            handle = self.session.current_speech
            await handle.wait_for_playout()
            await self.session.generate_reply(instructions="say goodbye to the user")
            participant_identity = next(iter(get_job_context().room.remote_participants))
            local_participant = get_job_context().room.local_participant
            end_result = await local_participant.perform_rpc(
                destination_identity=participant_identity, 
                method="end_conversation",
                payload=json.dumps({
                    "reason": reason,
                }),
                response_timeout=30.0,
            )
            logger.debug(f"End conversation result: {end_result}")
            # await self.session.aclose()
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")
            # raise ToolError("Unable to end conversation")

        self._closing_task = asyncio.create_task(self.session.aclose())

    async def save_conversation_to_storage(self, cache: bool = True) -> Optional[str]:
        pass
        # """Save the entire conversation to cloud storage"""
        # try:
        #     user_id = self._user_id
        #     if not user_id:
        #         logger.warning("Cannot save conversation: user_id is not set")
        #         return None
                
        #     if not hasattr(self, "session") or self.session is None:
        #         logger.warning("Cannot save conversation: session is no longer available")
        #         return None
                
        #     if not hasattr(self, "chat_ctx") or self.chat_ctx is None:
        #         logger.warning("Cannot save conversation: chat context is no longer available")
        #         return None
            
        #     domain = self._account if self._account else "specialized.com"  # Default domain as fallback
            
        #     messages = []
        #     try:
        #         messages = self.chat_ctx.copy(
        #             exclude_function_call=False,
        #             exclude_instructions=True,
        #             tools=[],
        #         ).items
        #         if not messages or len(messages) == 0:
        #             logger.info("No messages to save in conversation")
        #             return None
        #     except Exception as e:
        #         logger.warning(f"Error copying chat context: {e}, proceeding with empty messages")
            
        #     formatted_messages = []
        #     for message in messages:
        #         try:
        #             # Handle text content properly with fallbacks
        #             if hasattr(message, "type") and message.type == "message":
        #                 content = ""
        #                 if hasattr(message, "text_content") and message.text_content is not None:
        #                     content = message.text_content
        #                 elif hasattr(message, "content"):
        #                     content = str(message.content) if message.content is not None else ""
                        
        #                 formatted_messages.append({
        #                     "type": "message",
        #                     "role": getattr(message, "role", "unknown"),
        #                     "content": content,
        #                     "interrupted": getattr(message, "interrupted", False),
        #                     "timestamp": getattr(message, "timestamp", time.time())
        #                 })
        #             elif hasattr(message, "type") and message.type == "function_call":
        #                 formatted_messages.append({
        #                     "type": "function_call",
        #                     "name": getattr(message, "name", ""),
        #                     "arguments": getattr(message, "arguments", "{}"),
        #                     "call_id": getattr(message, "call_id", ""),
        #                     "timestamp": getattr(message, "timestamp", time.time())
        #                 })
        #             elif hasattr(message, "type") and message.type == "function_call_output":
        #                 formatted_messages.append({
        #                     "type": "function_call_output",
        #                     "name": getattr(message, "name", ""),
        #                     "output": getattr(message, "output", ""),
        #                     "call_id": getattr(message, "call_id", ""),
        #                     "is_error": getattr(message, "is_error", False),
        #                     "timestamp": getattr(message, "timestamp", time.time())
        #                 })
        #         except Exception as msg_err:
        #             logger.warning(f"Error processing message: {msg_err}, skipping")
        #             continue
            
        #     voice_session_id = None
        #     interaction_start_time = time.time()
            
        #     try:
        #         if hasattr(self.session, "userdata") and self.session.userdata:
        #             if hasattr(self.session.userdata, "voice_session_id"):
        #                 voice_session_id = self.session.userdata.voice_session_id
        #             if hasattr(self.session.userdata, "interaction_start_time"):
        #                 interaction_start_time = self.session.userdata.interaction_start_time
        #     except Exception as e:
        #         logger.warning(f"Error accessing session userdata: {e}")
            
        #     conversation_data = {
        #         "user_id": user_id,
        #         "conversation_id": self._session_id,
        #         "model": self._primary_model if hasattr(self, "_primary_model") else "unknown",
        #         "liddy_version": "0.1.0",
        #         "domain": domain,
        #         "start_time": interaction_start_time,
        #         "end_time": time.time(),
        #         "messages": formatted_messages,
        #         "metadata": {
        #             "voice_session_id": voice_session_id
        #         }
        #     }
            
        #     if cache:
        #         try:
        #             save_user_latest_conversation(
        #                 account=self._account,
        #                 user_id=user_id,
        #                 conversation_data=conversation_data
        #             )
        #         except Exception as save_err:
        #             logger.warning(f"Error saving latest conversation: {save_err}")
            
        #     try:
        #         storage = get_storage_provider()
        #         path = await storage.save_json(self._session_id, conversation_data, domain=domain)
        #         logger.info(f"Saved conversation to {path}")
        #         return path
        #     except ModuleNotFoundError as module_err:
        #         logger.error(f"Missing required module for storage: {module_err}")
        #         return None
        #     except Exception as storage_err:
        #         logger.error(f"Error saving to storage: {storage_err}")
        #         return None
            
        # except Exception as e:
        #     logger.error(f"Error saving conversation: {e}")
        #     return None