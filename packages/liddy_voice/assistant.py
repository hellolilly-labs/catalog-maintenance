import asyncio
import json
import logging
import os
import re
import random
import time
import psutil

from langfuse import observe, get_client

from cerebras.cloud.sdk import Cerebras, AsyncCerebras

from livekit.rtc import Participant

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
    metrics,
)
# from livekit.agents.voice.transcription.filters import filter_markdown
from livekit.protocol.models import RpcError
from livekit.agents.voice import UserStateChangedEvent, AgentStateChangedEvent
from livekit.plugins import deepgram, groq, openai, silero
from typing import Optional, AsyncIterable, List, Coroutine, Any
from liddy_voice.conversation.analyzer import ConversationAnalyzer

from liddy.models.product import Product
from liddy_voice.session_state_manager import SessionStateManager
from liddy_voice.rag_unified import PineconeRAG
from liddy.model import UserState, BasicChatMessage
from liddy_voice.llm_service import LlmService
from liddy_voice.user_manager import UserManager

from liddy.account_manager import get_account_manager, AccountManager
from liddy_voice.voice_search_wrapper import VoiceSearchService as SearchService
from livekit.rtc import RemoteParticipant
from liddy_voice.account_prompt_manager import get_account_prompt_manager, AccountPromptManager
from liddy_voice.product_manager import ProductManager

logger = logging.getLogger(__name__)

class Assistant(Agent):
    
    _current_context: Optional[tuple[str, str]] = None
    
    def __init__(self, ctx: JobContext, primary_model:str, user_id:str=None, chat_ctx: Optional[llm.ChatContext | None] = None, account:str=None):
        # Track startup timing for optimization
        startup_start = time.time()
        
        # Phase 1: Prompt Manager initialization
        prompt_init_start = time.time()
        
        # Try to get cached PromptManager if available
        self._account_prompt_manager: AccountPromptManager = get_account_prompt_manager(account=account)
        
        self._product_manager: ProductManager = None
        
        # # Only preload if not already cached (new instance)
        # if account and not hasattr(self._account_prompt_manager, '_already_preloaded'):
        #     asyncio.create_task(self._account_prompt_manager.load_account_config_async())
        #     self._account_prompt_manager._already_preloaded = True
        
        prompt_init_time = time.time() - prompt_init_start
        
        # Phase 2: Product loading setup (KISS approach)
        product_setup_start = time.time()
        self._instructions_loaded = False
        self._products_loaded = False
        self._product_load_task = None
        self._product_load_timing = None  # Will store detailed timing after loading
        self._search_prewarm_task = None  # For prewarming search instance
        self._word_boost_terms = []  # STT word boost terms
        self._pronunciation_guide = {}  # TTS pronunciation guide
        self._word_boost_task = None  # For loading word boost terms
        
        # Track all background tasks for proper cleanup
        self._background_tasks = []
        
        # Defer prewarming until after greeting is sent
        self._prewarming_started = False
        self._search_prewarmed = False  # Track if search has been prewarmed
        
        if account:
            # Only load the most critical items during init
            # Everything else will be loaded after greeting via start_deferred_prewarming()
            pass
        
        product_setup_time = time.time() - product_setup_start
        
        # Phase 3: Agent initialization
        agent_init_start = time.time()
        super().__init__(
            instructions=self._account_prompt_manager.build_system_instruction_prompt(), 
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
        
        # Phase 5: Placeholder for deferred LLM clients
        llm_setup_start = time.time()
        # Fast LLM and Cerebras clients will be initialized in start_deferred_prewarming()
        self._fast_llm = None
        self._fast_llm_prompt = None
        self.async_cerebras_client = None
        self.cerebras_client = None
        llm_setup_time = time.time() - llm_setup_start
        
        # Phase 6: Background task setup
        bg_task_start = time.time()
        
        # Comprehensive prewarming is now deferred until after greeting
        # Call start_deferred_prewarming() after session.say(greeting)
        
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

    async def start_deferred_prewarming(self):
        """
        Start all prewarming tasks after the greeting has been sent.
        This allows the user to hear something immediately while the system warms up.
        """
        if self._prewarming_started or not self._account:
            return
            
        self._prewarming_started = True
        logger.info(f"🚀 Starting deferred prewarming for {self._account}...")
        
        # Initialize fast LLM and Cerebras clients
        llm_init_start = time.time()
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
        
        # Initialize Cerebras clients
        self.async_cerebras_client = AsyncCerebras(
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )
        self.cerebras_client = Cerebras(
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )
        logger.info(f"✅ LLM clients initialized in {time.time() - llm_init_start:.3f}s")
        
        # Start product loading in background
        self._product_load_task = asyncio.create_task(self._load_products_for_account(self._account))
        self._background_tasks.append(self._product_load_task)
        
        # Prewarm the search instance
        self._search_prewarm_task = asyncio.create_task(self._prewarm_search_instance(self._account))
        self._background_tasks.append(self._search_prewarm_task)
        
        # Load STT word boost terms
        self._word_boost_task = asyncio.create_task(self._load_word_boost_terms(self._account))
        self._background_tasks.append(self._word_boost_task)
        
        # Start comprehensive prewarming
        prewarm_task = asyncio.create_task(self._prewarm_all_systems())
        self._background_tasks.append(prewarm_task)
        
        logger.info(f"✅ Deferred prewarming tasks started for {self._account}")

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
            "prewarming": {
                "overall": getattr(self, '_overall_prewarm_timing', None),
                "search": getattr(self, '_search_prewarm_timing', None),
                "account_manager": getattr(self, '_account_manager_prewarm_timing', None),
                "llm": {"status": "completed" if hasattr(self, 'instructions') else "skipped"}
            },
            "status": {
                "products_loaded": self._products_loaded,
                "startup_completed": hasattr(self, '_startup_timing'),
                "product_task_status": "completed" if self._products_loaded else (
                    "running" if self._product_load_task and not self._product_load_task.done() else "pending"
                ),
                "all_systems_prewarmed": hasattr(self, '_overall_prewarm_timing')
            },
            "targets": {
                "startup_time": "< 1.0s",
                "product_loading": "< 2.0s", 
                "memory_usage": "< 10MB",
                "search_prewarm": "< 3.0s",
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
                ),
                "meets_search_prewarm_target": (
                    getattr(self, '_search_prewarm_timing', {}).get("prewarm_time", 0) < 3.0
                    if hasattr(self, '_search_prewarm_timing') and not getattr(self, '_search_prewarm_timing', {}).get("error") else False
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
        """
        Prewarm the LLM with the system prompt to cache tokens.
        
        This ensures the initial response can leverage cached tokens for faster TTFT.
        """
        try:
            start_time = time.time()
            logger.info(f"🔥 Prewarming main LLM ({self._primary_model}) with system prompt...")
            
            if not self._instructions_loaded:
                self._instructions_loaded = True
                await self.update_instructions(self._account_prompt_manager.build_system_instruction_prompt())
            
            chat_ctx_copy = self.chat_ctx.copy()
            chat_ctx_copy.items.append(llm.ChatMessage(
                role="user",
                content=["ping"]
            ))
            response = await LlmService.chat_wrapper(
                llm_service=self._llm,
                chat_ctx=chat_ctx_copy,
            )
            logger.info(f"✅ LLM prewarmed in {time.time() - start_time:.3f}s")
            logger.info(f"✅ LLM response: {response}")
                
            elapsed = time.time() - start_time
            logger.info(f"✅ Main LLM prewarmed in {elapsed:.3f}s - cached {len(self.instructions)} chars of system prompt") 
                
        except Exception as e:
            logger.error(f"Error prewarming main LLM: {e}")
            
    async def _prewarm_search_instance(self, account: str) -> None:
        """
        Prewarm the search instance for faster first search.
        
        This initializes the PineconeRAG instance and its indexes in the background
        with full prewarming to eliminate cold start penalties:
        - Connection pooling initialization
        - Serverless instance wake-up queries
        - Reranking service prewarming
        """
        # Check if already prewarmed
        if hasattr(self, '_search_prewarmed') and self._search_prewarmed:
            logger.debug(f"Search instance already prewarmed for {account}, skipping")
            return
            
        # Check if recently prewarmed (within last 5 minutes)
        if hasattr(self, '_search_prewarm_timing') and self._search_prewarm_timing:
            last_prewarm = self._search_prewarm_timing.get('timestamp', 0)
            if time.time() - last_prewarm < 300:  # 5 minutes
                logger.debug(f"Search instance was prewarmed {time.time() - last_prewarm:.1f}s ago, skipping")
                self._search_prewarmed = True
                return
            
        try:
            start_time = time.time()
            logger.info(f"🔥 Prewarming search instance for {account}...")
            
            # Get account manager to check if RAG is configured
            account_manager: AccountManager = await get_account_manager(account)
            # index_name, embedding_model = account_manager.get_rag_details()
            
            # if not index_name:
            #     logger.info(f"No RAG index configured for {account}, skipping search prewarm")
            #     return
            
            # Import and initialize the search service
            from liddy.search.service import SearchService
            
            # Get search config for the account
            search_config = account_manager.get_search_config()
            
            # Phase 1: Initialize the search instance with FULL prewarming
            init_start = time.time()
            
            # Check if we're using separate or unified indexes
            if search_config.unified_index:
                # For unified index, use the SearchService approach with full prewarming
                self._search_instance = await SearchService._get_search_instance(
                    account_manager,
                    prewarm_level="full"  # Full prewarming including reranker
                )
                init_time = time.time() - init_start
                logger.info(f"  ✓ Search instance initialized (unified) with full prewarming in {init_time:.3f}s")
            else:
                # For separate indexes, use PineconeRAG directly with full prewarming
                from liddy.search.pinecone import get_search_pinecone
                
                self._search_instance = await get_search_pinecone(
                    brand_domain=account,
                    dense_index_name=search_config.dense_index,
                    sparse_index_name=search_config.sparse_index,
                    prewarm_level="full"  # Full prewarming including reranker
                )
                init_time = time.time() - init_start
                logger.info(f"  ✓ Search instance initialized with full prewarming in {init_time:.3f}s")
            
            # Phase 2: Run a minimal test search to verify everything is working
            test_start = time.time()
            try:
                test_results = await self._search_instance.search(
                    query="test",
                    top_k=1,
                    search_mode="hybrid",
                    rerank=False,  # Skip reranking for test
                    smart_rerank=False
                )
                test_time = time.time() - test_start
                logger.info(f"  ✓ Test search completed in {test_time:.3f}s")
            except Exception as test_error:
                logger.warning(f"Test search failed (non-critical): {test_error}")
                test_time = 0
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Search instance fully prewarmed for {account} in {elapsed_time:.3f}s")
            
            # Mark as prewarmed to avoid duplicate prewarming
            self._search_prewarmed = True
            
            # Store timing info for monitoring
            self._search_prewarm_timing = {
                "account": account,
                "prewarm_time": elapsed_time,
                "init_time": init_time,
                "test_time": test_time,
                "index_type": "unified" if search_config.unified_index else "separate",
                "timestamp": time.time()
            }
            
        except asyncio.CancelledError:
            logger.debug(f"Search prewarming cancelled for {account}")
            raise  # Re-raise to properly handle cancellation
        except Exception as e:
            logger.error(f"Error prewarming search instance for {account}: {e}")
            # Don't let this fail the entire Assistant initialization
            self._search_prewarm_timing = {
                "account": account,
                "error": str(e),
                "timestamp": time.time()
            }

    async def _prewarm_account_manager(self, account: str) -> None:
        """
        Prewarm the account manager to cache account configuration.
        
        This avoids the overhead on first product search.
        """
        try:
            start_time = time.time()
            logger.info(f"🔥 Prewarming account manager for {account}...")
            
            # Get and cache the account manager
            account_manager: AccountManager = await get_account_manager(account)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Account manager prewarmed in {elapsed:.3f}s")
            
            self._account_manager_prewarm_timing = {
                "account": account,
                "prewarm_time": elapsed,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error prewarming account manager: {e}")
            self._account_manager_prewarm_timing = {
                "account": account,
                "error": str(e),
                "timestamp": time.time()
            }

    async def _prewarm_all_systems(self) -> None:
        """
        Comprehensive prewarming strategy for optimal performance.
        
        Runs all prewarming tasks in parallel to minimize startup time while
        maximizing first-request performance.
        """
        try:
            overall_start = time.time()
            logger.info(f"🚀 Starting additional system prewarming for {self._account} (LLM & account manager)...")
            
            # Create all prewarming tasks
            prewarm_tasks = [
                ("llm", self._prewarm_llm()),
                # Search is already prewarming via self._search_prewarm_task
                ("account_manager", self._prewarm_account_manager(self._account)),
                # Products are already loading via self._product_load_task
            ]
            
            # Run all prewarm tasks in parallel
            results = await asyncio.gather(
                *[task for _, task in prewarm_tasks],
                return_exceptions=True
            )
            
            # Log results
            for (name, _), result in zip(prewarm_tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Prewarming {name} failed: {result}")
                else:
                    logger.debug(f"✅ Prewarming {name} completed")
            
            total_time = time.time() - overall_start
            logger.info(f"🎯 All systems prewarmed in {total_time:.3f}s for {self._account}")
            
            # Store overall prewarm timing
            self._overall_prewarm_timing = {
                "account": self._account,
                "total_time": total_time,
                "components": {
                    "llm": hasattr(self, '_llm_prewarm_timing'),
                    "search": hasattr(self, '_search_prewarm_timing'), 
                    "account_manager": hasattr(self, '_account_manager_prewarm_timing'),
                    "products": self._products_loaded
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive prewarming: {e}")

    async def get_product_manager(self) -> ProductManager:
        if not self._product_manager:
            # self._product_manager = await self._load_products_for_account()
            from liddy_voice.product_manager import get_product_manager
            self._product_manager = await get_product_manager(account=self._account)
        return self._product_manager

    async def _load_products_for_account(self) -> None:
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
            
            from liddy.models.product import Product
            from .product_manager import load_product_catalog_for_assistant
            
            init_time = time.time() - init_start
            
            # Phase 2: Memory baseline measurement
            memory_start = process.memory_info().rss / 1024 / 1024  # MB
            
            # Phase 3: Product loading timing
            load_start = time.time()
            self._product_manager = await load_product_catalog_for_assistant(account=self._account)
            products = await self._product_manager.get_product_objects()
            load_time = time.time() - load_start
            
            # Phase 4: Memory after loading measurement  
            memory_end = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_end - memory_start
            
            # Phase 5: Finalization
            finalize_start = time.time()
            self._products_loaded = True
            
            # Store timing metadata for monitoring
            self._product_load_timing = {
                "account": self._account,
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
            logger.info(f"🚀 Assistant product loading complete for {self._account}: "
                       f"{len(products)} products loaded in {total_time:.3f}s "
                       f"(init: {init_time:.3f}s, load: {load_time:.3f}s) "
                       f"| Memory: {memory_delta:.1f}MB | "
                       f"Efficiency: {len(products) / memory_delta if memory_delta > 0 else 0:.1f} products/MB")
            
            # Performance validation and warnings
            if total_time > 2.0:
                logger.warning(f"⚠️ Product loading exceeded 2s target: {total_time:.3f}s for {self._account}")
            
            if memory_delta > 10.0:
                logger.warning(f"⚠️ Product catalog memory exceeded 10MB target: {memory_delta:.1f}MB for {self._account}")
            
            if len(products) == 0:
                logger.warning(f"⚠️ No products loaded for account {self._account}")
            
        except asyncio.CancelledError:
            logger.debug(f"Product loading cancelled for {self._account}")
            self._products_loaded = False
            raise  # Re-raise to properly handle cancellation
        except Exception as e:
            logger.error(f"Error loading products for Assistant account {self._account}: {e}")
            self._products_loaded = False
            
            # Store error timing for debugging
            self._product_load_timing = {
                "account": self._account,
                "error": str(e),
                "timing": {
                    "total_time": time.time() - total_start,
                    "failed_at": "product_loading"
                },
                "timestamp": time.time()
            }

    async def _load_word_boost_terms(self, account: str) -> None:
        """
        Load STT word boost terms from optimized vocabulary file.
        
        This loads pre-extracted word_boost terms to improve speech recognition
        accuracy for brand-specific terminology.
        
        Args:
            account: Account domain for this voice agent
        """
        try:
            start_time = time.time()
            logger.info(f"🎤 Loading STT word boost terms for {account}...")
            
            # Try to load from storage
            from liddy.storage import get_account_storage_provider
            storage_manager = get_account_storage_provider()
            
            # Try optimized file first
            try:
                word_boost_content = await storage_manager.read_file(
                    account=account,
                    file_path="stt_word_boost.json"
                )
                word_boost_data = json.loads(word_boost_content)
                self._word_boost_terms = word_boost_data.get('word_boost', [])
                
                # Also try to load pronunciation guide
                try:
                    vocab_content = await storage_manager.read_file(
                        account=account,
                        file_path="stt_vocabulary.json"
                    )
                    vocab_data = json.loads(vocab_content)
                    self._pronunciation_guide = vocab_data.get('pronunciation_guide', {})
                    logger.info(f"✅ Loaded {len(self._word_boost_terms)} word boost terms and {len(self._pronunciation_guide)} pronunciations in {time.time() - start_time:.3f}s")
                except:
                    logger.info(f"✅ Loaded {len(self._word_boost_terms)} optimized word boost terms in {time.time() - start_time:.3f}s")
                
                return
            except Exception as e:
                logger.debug(f"Optimized word boost file not found, trying research file: {e}")
            
            # Fallback to product catalog research
            try:
                research_content = await storage_manager.read_file(
                    account=account,
                    file_path="research/product_catalog/research.md"
                )
            except Exception as e:
                logger.debug(f"Could not load product catalog research: {e}")
                return
            
            # Extract word boost terms from Part E
            word_boost_section = self._extract_word_boost_section(research_content)
            if not word_boost_section:
                logger.info("No word boost section found in product catalog research")
                return
            
            # Parse the JSON from the word boost section
            import re
            json_match = re.search(r'```json\s*({[^}]+})\s*```', word_boost_section, re.DOTALL)
            if json_match:
                try:
                    word_boost_data = json.loads(json_match.group(1))
                    self._word_boost_terms = word_boost_data.get('word_boost', [])
                    logger.info(f"✅ Loaded {len(self._word_boost_terms)} word boost terms from research in {time.time() - start_time:.3f}s")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse word boost JSON: {e}")
            else:
                # Fallback: extract basic terms from the section
                self._word_boost_terms = self._extract_basic_word_boost_terms(word_boost_section)
                logger.info(f"✅ Extracted {len(self._word_boost_terms)} basic word boost terms in {time.time() - start_time:.3f}s")
            
        except asyncio.CancelledError:
            logger.debug(f"Word boost loading cancelled for {account}")
            self._word_boost_terms = []
            raise  # Re-raise to properly handle cancellation  
        except Exception as e:
            logger.error(f"Error loading word boost terms for {account}: {e}")
            self._word_boost_terms = []
    
    def _extract_word_boost_section(self, content: str) -> str:
        """
        Extract Part E (STT Word Boost Vocabulary) from the research content.
        """
        # Look for Part E section
        part_e_start = content.find("## Part E: STT Word Boost Vocabulary")
        if part_e_start == -1:
            return ""
        
        # Find the end of Part E (next major section or end of content)
        part_e_end = content.find("\n## ", part_e_start + 1)
        if part_e_end == -1:
            part_e_end = len(content)
        
        return content[part_e_start:part_e_end]
    
    def _extract_basic_word_boost_terms(self, section: str) -> list:
        """
        Extract basic word boost terms from the section text.
        """
        terms = []
        
        # Extract brand names and product lines from bullet points
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                # Extract quoted terms
                import re
                quoted_terms = re.findall(r'"([^"]+)"', line)
                terms.extend(quoted_terms)
                
                # Also extract terms in brackets
                bracketed_terms = re.findall(r'\[([^\]]+)\]', line)
                for term in bracketed_terms:
                    # Skip meta descriptions
                    if not any(skip in term.lower() for skip in ['list', 'top', 'weight', 'specific']):
                        terms.append(term)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen and len(term) > 2:  # Skip very short terms
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:500]  # Limit to 500 terms
    
    def get_word_boost_terms(self) -> list:
        """
        Get the loaded word boost terms for STT configuration.
        
        Returns:
            List of word boost terms
        """
        return self._word_boost_terms.copy() if self._word_boost_terms else []

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

    async def cleanup(self) -> None:
        """
        Clean up background tasks when shutting down.
        
        This should be called when the Assistant is being destroyed to prevent
        async tasks from continuing to run after the session closes.
        """
        logger.debug(f"Cleaning up {len(self._background_tasks)} background tasks for Assistant")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if task and not task.done():
                task.cancel()
        
        # Wait for all tasks to complete cancellation
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Clear the task list
        self._background_tasks.clear()
        
        # Clean up search instance if it exists
        if hasattr(self, '_search_instance') and self._search_instance:
            try:
                await self._search_instance.cleanup()
                logger.debug("Search instance cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up search instance: {e}")
        
        logger.debug("Assistant cleanup completed")

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
            system_prompt = self._account_prompt_manager.get_system_prompt()
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
            if not self.async_cerebras_client:
                logger.warning("Cerebras client not initialized yet, skipping intent extraction")
                return []
                
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

        # self._prompt_manager.update_current_generation()
        if not self._instructions_loaded:
            self._instructions_loaded = True
            await self.update_instructions(self._account_prompt_manager.build_system_instruction_prompt())
            logger.info(f"🔍 Updated instructions for {self._account}")
        
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
        
        # # Check to see if we are going to call the product search tool
        # if last_user_message and last_user_message.text_content:
        #     if "product_search" in last_user_message.text_content.lower():
        #         logger.info(f"🔍 User is searching for products, calling product_search")
        #         await self.session.say(text="Searching for products", allow_interruptions=False)
        #         return


        # # Run intent detection asynchronously in the background
        # if last_user_message is not None and last_user_message.text_content:
        #     # Create a background task for intent detection
        #     asyncio.create_task(self._process_intent_detection(last_user_message, chat_ctx))

        return Agent.default.llm_node(self, chat_ctx=chat_ctx, tools=tools, model_settings=model_settings)

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        """Process text-to-speech with timeout handling for responsiveness"""
        # filtered_text = filter_markdown(text)
        filtered_text = text

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
        return super().tts_node(adjust_pronunciation(filtered_text), model_settings)

        # async for frame in Agent.default.tts_node(
        #     self,
        #     adjust_pronunciation(filtered_text),
        #     model_settings
        # ):
        #     yield frame

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
            import traceback
            traceback.print_exc()
            logger.error(f"Error updating user state prompt: {e}")


    async def on_user_turn_completed(
        self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        user_state: UserState = self.session.userdata or UserManager.get_user_state(user_id=self._user_id)
        if not user_state:
            user_state = UserState(user_id=self._user_id)
        self.session.userdata = user_state
        self.session.userdata.last_interaction_time = time.time()
        self.last_stt_message = new_message
            
        UserManager.save_user_state(user_state)
        
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
        try:
            # Clean up background tasks to prevent async warnings
            await self.cleanup()
        except Exception as e:
            logger.error(f"Error during Assistant cleanup: {e}")
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
                    if not self._fast_llm or not self._fast_llm_prompt:
                        logger.debug("Fast LLM not initialized yet, using default wait phrase")
                        wait_phrase = random.choice(self.wait_phrases)
                        if say_filler:
                            await self.session.say(wait_phrase, add_to_chat_ctx=False)
                        return wait_phrase
                    
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
    ) -> List[Product]:
        """Unified search with status updates and proper timeout handling"""
        status_task = None
        search_start = time.time()
        
        try:
            # Create a delayed status update task that only runs after status_delay
            async def delayed_status_update():
                await asyncio.sleep(status_delay)
                await self.generate_and_say_wait_phrase_if_needed(use_fast_llm=False, say_filler=True)
            
            status_task = asyncio.create_task(delayed_status_update())
            
            # Log search start
            logger.debug(f"🔎 Starting search for query: '{query[:50]}...' with timeout={timeout}s")
            
            # Perform search with timeout
            search_results = await asyncio.wait_for(
                search_fn(query, **(search_params or {})),
                timeout=timeout
            )
            
            # Log search completion
            search_time = time.time() - search_start
            result_count = len(search_results.get("results", [])) if isinstance(search_results, dict) else len(search_results)
            logger.debug(f"✅ Search completed in {search_time:.3f}s with {result_count} results")
            
            return search_results
            
        except asyncio.TimeoutError:
            timeout_time = time.time() - search_start
            logger.warning(f"⏱️ Search timed out after {timeout_time:.3f}s for query: '{query}'")
            return {"error": "Search timed out", "results": []}
        except Exception as e:
            error_time = time.time() - search_start
            logger.error(f"❌ Error in perform_search_with_status after {error_time:.3f}s: {e}")
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
        
        ALWAYS say something to the user in parallel to calling this function to let them know you are finding the relevant information (using whatever language is contextually appropriate).

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
            results = await self.perform_search_with_status(
                search_fn=lambda q, **params: SearchService.perform_search(
                    query=q,
                    search_function=SearchService.search_knowledge,
                    search_params=params,
                    user_state=self.session.userdata,
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
                        # "knowledge_base": self._account_prompt_manager.knowledge_search_guidance or ""
                    }
                }
            )

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

    @observe(name="product_search")
    @function_tool()
    async def product_search(
        self,
        context: RunContext[UserState],
        query: str,
        top_k: str | int = "auto",
        # message_to_user: str,
    ):

        """Use this to search for products that the user may be interested in. Include as much depth and detail in the query as possible to get the best results. Generally, the only way for you to know about individual products is through this tool call, so feel free to liberally call this tool call.
        
        ALWAYS speak to the user in the same turn as the tool call.
        
        Args:
            query: The query to use for the search. Be as descriptive and detailed as possible based on the context of the entire conversation. For example, the query "mountain bike" is not very descriptive, but "mid-range non-electric mountain bike with electronic shifting for downhill flowy trails for advanced rider" is much more descriptive. The more descriptive the query, the better the results will be. You can also use this to search for specific products by name or ID, but be sure to include as much context as possible to help narrow down the results.
            top_k: The number of products to return. Defaults to "auto" which means the number of products to return is determined by quality of the results. If you want to return a specific number of products, you can set this to an integer. For example, if the product search is not returning back satisfactory results, you can set this to 10 to get more results.
        """
        # Track overall timing
        overall_start = time.time()
        timing_breakdown = {}
        
        logger.info(f"🔍 Product search called with query: {query}, top_k: {top_k}")
        
        try:
            # async def _say_message_to_user():
            #     handle = context.session.current_speech
            #     if handle and not handle.done():
            #         await handle.wait_for_playout()
            #     chat_ctx = self.chat_ctx.copy(exclude_instructions=True, exclude_function_call=True, tools=[])
            #     last_user_message = next((msg for msg in reversed(chat_ctx.items) if msg.role == "user"), None)
            #     await context.session.generate_reply(instructions="Affirm the user's query and let them know you are lookingfor products", 
            #                                          user_input=last_user_message.content[0] if last_user_message else None,
            #                                          allow_interruptions=True)
            # say_message_to_user_task = asyncio.create_task(_say_message_to_user())
            # # if not context.speech_handle or context.speech_handle.done():
            # #     logger.info(f"🔍 Agent is not currently saying anything, saying message_to_user")
            # #     await self.session.say(text=message_to_user, allow_interruptions=False)
            # # else:
            # #     logger.info(f"🔍 Agent is currently saying something, skipping message_to_user")
            # #     # # if the agent is currently saying something, then we need to wait for it to finish
            # #     # await context.current_speech.wait_for_playout()
            
            # Input validation
            validation_start = time.time()
            if not query:
                raise ToolError("`query` is required")
            timing_breakdown['validation'] = time.time() - validation_start
        
            # Ensure products are loaded before searching (KISS approach)
            products_load_start = time.time()
            products_ready = await self.ensure_products_loaded()
            if not products_ready:
                logger.warning(f"Products not loaded for account {self._account}, falling back to legacy loading")
            timing_breakdown['ensure_products'] = time.time() - products_load_start
        
            # Get products using ProductManager (should be cached after initial load)
            get_products_start = time.time()
            products = await (await self.get_product_manager()).get_products()
            products_results: List[Product] = []
            timing_breakdown['get_products'] = time.time() - get_products_start
            
            # Log if getting products is slow (shouldn't be if cached properly)
            if timing_breakdown['get_products'] > 0.1:
                logger.warning(f"⚠️ Getting products took {timing_breakdown['get_products']:.3f}s - cache may not be working")
            
            # Get account manager and RAG details
            account_mgr_start = time.time()
            account_manager: AccountManager = await get_account_manager(self._account)
            index_name, embedding_model = account_manager.get_rag_details()
            timing_breakdown['account_manager'] = time.time() - account_mgr_start
            
            # Choose search approach based on catalog size
            search_start = time.time()
            # if len(products) < 100:
            if not index_name:
                # For small catalogs: Use LLM-based search with standardized status updates
                logger.info(f"🔍 Using LLM-based search for {self._account} (no RAG index)")
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
                            # "product_knowledge": self._account_prompt_manager.product_search_knowledge
                        }
                    }
                )
                timing_breakdown['search_type'] = 'llm'
            else:
                # For large catalogs: Use RAG search with standardized status updates
                logger.info(f"🔍 Using RAG search for {self._account} with index: {index_name}")
                rag_results = await self.perform_search_with_status(
                    search_fn=lambda q, **params: SearchService.perform_search(
                        query=q,
                        search_function=SearchService.search_products_rag,
                        search_params=params,
                        user_state=self.session.userdata,
                        # query_enhancer=SearchService.enhance_query
                    ),
                    query=query,
                    search_params={
                        "account": self._account,
                        "enhancer_params": {
                            "chat_ctx": self.chat_ctx,
                            # "product_knowledge": self._account_prompt_manager.product_search_knowledge
                        }
                    }
                )
                timing_breakdown['search_type'] = 'rag'
                
                # Process RAG results into product objects
                process_start = time.time()
                if rag_results and not isinstance(rag_results, dict) or not rag_results.get("error"):
                    for result in rag_results.get("results"):
                        if result.get('id'):
                            product = await (await self.get_product_manager()).find_product_by_id(product_id=result.get('id'))
                            if product:
                                products_results.append(product)
                                continue
                        if result.get('metadata'):
                            products_results.append(Product.from_metadata(result['metadata']))
                timing_breakdown['process_results'] = time.time() - process_start
            
            timing_breakdown['search_execution'] = time.time() - search_start
            
            # Format results as markdown
            format_start = time.time()
            markdown_results = ""
            if 'rag_results' in locals() and rag_results and rag_results.get("agent_instructions"):
                markdown_results += f"**INSTRUCTIONS**: {rag_results.get('agent_instructions')}\n\n"
            markdown_results += f"**CRITICAL**: Be sure to call the `display_product` tool call to display the product that you talk about. You can use the product ID from the results above to do this.\n\n"

            markdown_results = "# Products (in descending order):\n\nRemember that you are an AI sales agent speaking to a human, so be sure to use natural language to respond and not just a list of products. For example, never say the product URL unless the user specifically asks for it.\n\n"
            
            # Use a more concise format for search results to reduce LLM processing time
            for i, result in enumerate(products_results):
                # Only include essential info for decision making
                markdown_results += f"## {i+1}. {result.name}\n"
                markdown_results += f"- **ID**: {result.id}\n"
                if result.originalPrice:
                    markdown_results += f"- **Price**: ${result.originalPrice}\n"
                elif result.salePrice:
                    markdown_results += f"- **Sale Price**: ${result.salePrice}\n"
                
                # Add brief description if available
                if hasattr(result, 'description') and result.description:
                    # Truncate long descriptions
                    desc = result.description[:200] + "..." if len(result.description) > 200 else result.description
                    markdown_results += f"- **Description**: {desc}\n"
                
                # Add key highlights if available
                if hasattr(result, 'keySellingPoints') and result.keySellingPoints:
                    markdown_results += f"- **Key Features**: {', '.join(result.keySellingPoints[:3])}\n"
                
                markdown_results += "\n"
            
            timing_breakdown['format_results'] = time.time() - format_start
            
            # Log final timing
            total_time = time.time() - overall_start
            timing_breakdown['total'] = total_time
            
            # Create detailed timing log
            timing_details = [f"{k}: {v:.3f}s" for k, v in timing_breakdown.items() if k != 'search_type']
            logger.info(f"📊 product_search completed in {total_time:.3f}s | "
                       f"query: '{query[:30]}...' | "
                       f"results: {len(products_results)} | "
                       f"breakdown: {', '.join(timing_details)}")
            
            # Warn if total time exceeds threshold
            if total_time > 3.0:
                logger.warning(f"⚠️ product_search took {total_time:.3f}s (>3s threshold) for query: '{query}'")
                # Log the slowest components
                sorted_timings = sorted([(k, v) for k, v in timing_breakdown.items() if k not in ['total', 'search_type']], 
                                      key=lambda x: x[1], reverse=True)
                logger.warning(f"   Slowest components: {sorted_timings[:3]}")
            
            # say_message_to_user_task.cancel()
            
            return markdown_results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Log error with timing info
            error_time = time.time() - overall_start
            logger.error(f"Error in product_search after {error_time:.3f}s: {e}")
            logger.error(f"   Timing at error: {timing_breakdown}")
            raise ToolError("Unable to search for products")
    
    @observe(name="display_product")
    @function_tool()
    async def display_product(
        self,
        product_id: str,
        variant: Optional[dict[str, str]]=None,
        resumption_message: str="",
    ):
        """Use this to display or show exactly one product. The tool call itself will handle actually doing this. Never try to display or show products on your own, ALWAYS use this tool call. It is not possible to show multiple products at once, so you must narrow it down to a single product. For example, if the user wants to see multiple products at a time, you should show them the first product only. Then move on to the next product when the user is ready. Never make multiple calls to this function at once.
        
        You typically would first use `product_search` to find the product ID, and then use this function to display the product.
        
        NOTE: If you know the user is already looking at that product (because they are viewing that page or something like that) then you can silently call this function without necessarily saying anything to the user. This is useful for showing the product details without interrupting the conversation. Otherwise, it is generally a good idea to say something to the user to let them know you are displaying the product.
        
        Args:
            product_id: The ID of the product to display. Required.
            variant: The ID or name of the product variant to display. This is optional and can be used to show a specific variant of the product. If not provided, the default variant will be shown. Include the name of the variant, such as "color" or "size", and the value of the variant, such as "red" or "large". For example, "color=red" or "size=large". If you don't know the variant ID, you can use the `product_search` function to find it.
            resumption_message: The message to say back to the user after displaying the product. This is required and should be a natural language message that is appropriate for the context of the conversation.
        """
        start_time = time.time()
        timing_breakdown = {}
        
        if not isinstance(product_id, str):
            raise ToolError("`product_id` is required")
        try:
            logger.info(f"🔍 display_product started: {product_id}")
            
            # Time product lookup
            product_lookup_start = time.time()
            product = await (await self.get_product_manager()).find_product_by_id(product_id=product_id)
            timing_breakdown['product_lookup'] = time.time() - product_lookup_start
            products = []
            
            if not product:
                products = await (await self.get_product_manager()).get_products()
                products_names_urls_ids = [(p.name, p.productUrl, p.id) for p in products]
                raise ToolError(f"The product ID's provided are not in the product catalog. Please provide valid product ID's. Use the `product_search` function to find a valid product ID. If at all possible, don't ask the user for the product ID, instead use the `product_search` function to find it silently and use that.\n\nValid product ID's: {json.dumps(products_names_urls_ids)}")
            
            print(f"Product URL: {product.productUrl}")
            
            # Time building product results
            build_results_start = time.time()
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
                    account_manager: AccountManager = await get_account_manager(account=self._account)
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
            
            timing_breakdown['build_results'] = time.time() - build_results_start
            
            # Time user state and history lookup
            user_state_start = time.time()
            user_state = self.session.userdata
            history = SessionStateManager.get_user_recent_history(user_id=user_state.user_id)
            timing_breakdown['user_state_lookup'] = time.time() - user_state_start

            current_product = None
            if history and len(history) > 0:
                history_product_start = time.time()
                most_recent_history = history[0]
                if most_recent_history.url:
                    product_url = most_recent_history.url
                    current_product = await (await self.get_product_manager()).find_product_by_url(product_url=product_url)
                timing_breakdown['history_product_lookup'] = time.time() - history_product_start
            
            # Time waiting for speech to finish
            wait_speech_start = time.time()
            await self.session.current_speech.wait_for_playout()
            timing_breakdown['wait_for_speech'] = time.time() - wait_speech_start
            
            participant_identity = next(iter(get_job_context().room.remote_participants))
            local_participant = get_job_context().room.local_participant
            
            # Time room update
            room_update_start = time.time()
            # Because this potentially causes the UI to do a hard nav to a new page, update the should_reconnect_until timeout for this account/user_id
            UserManager.update_user_room_reconnect_time(
                user_id=user_state.user_id,
                room_id=get_job_context().room.name,
                account=self._account,
                auto_reconnect_until=time.time() + 60 * 5,  # 5 minutes
                resumption_message=resumption_message
            )
            timing_breakdown['room_update'] = time.time() - room_update_start
            
            # Time RPC call
            rpc_start = time.time()
            try:
                self._setup_planned_reconnect()
                rpc_result = await local_participant.perform_rpc(
                    destination_identity=participant_identity,
                    method="display_products",
                    payload=json.dumps({"products": [product_results], "resumptionMessage": resumption_message}),
                    response_timeout=10.0,
                )
                timing_breakdown['rpc_call'] = time.time() - rpc_start
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
            
            # Log timing summary
            total_time = time.time() - start_time
            timing_breakdown['total'] = total_time
            
            logger.info(f"📊 display_product completed in {total_time:.3f}s | product_id: {product_id} | breakdown: " +
                       f"product_lookup: {timing_breakdown.get('product_lookup', 0):.3f}s, " +
                       f"build_results: {timing_breakdown.get('build_results', 0):.3f}s, " +
                       f"user_state: {timing_breakdown.get('user_state_lookup', 0):.3f}s, " +
                       f"history_lookup: {timing_breakdown.get('history_product_lookup', 0):.3f}s, " +
                       f"wait_speech: {timing_breakdown.get('wait_for_speech', 0):.3f}s, " +
                       f"room_update: {timing_breakdown.get('room_update', 0):.3f}s, " +
                       f"rpc: {timing_breakdown.get('rpc_call', 0):.3f}s")
            
            return result
        except ToolError as e:
            logger.error(f"ToolError in display_product: {e}")
            raise e
        except ConnectionError as e:
            logger.error(f"ConnectionError in display_product: {e}")
            # Ignore for now
        except Exception as e:
            import traceback
            traceback.print_exc()
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
            if handle and not handle.done():
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