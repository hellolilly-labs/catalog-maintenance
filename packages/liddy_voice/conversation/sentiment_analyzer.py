import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

from livekit.agents import llm, utils, Agent
from livekit.agents.llm import ChatMessage

from liddy_voice.spence.sentiment import SentimentService, SentimentAnalysis, SentimentStorage
from liddy_voice.spence.session_state_manager import SessionStateManager
from liddy_voice.spence.model import UserState
from redis_client import get_user_state, save_user_state

logger = logging.getLogger("sentiment-analyzer")


class SentimentAnalyzer():
    """
    Analyzes conversation sentiment in real-time using LiveKit's event system
    to provide emotional intelligence guidance to the assistant.
    """
    
    def __init__(
        self,
        agent: Agent | None,
        llm_model: any,
        user_id: str,
    ):
        """
        Initialize the sentiment analyzer
        
        Args:
            model: VoicePipelineAgent instance to hook events to
            llm_model: The LLM model to use for analysis
            user_id: The unique ID for the current user
        """
        super().__init__()
        
        self._agent = agent
        self.llm_model = llm_model
        self.user_id = user_id
        
        # Analysis state tracking
        self.sentiment_service = SentimentService()
        self.sentiment_storage = SentimentStorage()
        self.current_sentiment: Optional[SentimentAnalysis] = self.sentiment_storage.load(user_id=user_id)
        self.analysis_lock = asyncio.Lock()
        self.last_analysis_time = 0
        self.analysis_task = None
        self.min_analysis_interval = 60  # seconds between analyses (sentiment changes more slowly)
        self.min_messages_for_analysis = 3
        self.message_count_since_analysis = 0
        
        # Event tracking
        self._conversation = []
        self._log_q = asyncio.Queue()
        
        # if we have a current_sentiment, update the communication directive
        if self.current_sentiment:
            asyncio.create_task(self._update_comm_directive(self.current_sentiment))
    
    async def aclose(self) -> None:
        """Exits"""
        self._log_q.put_nowait(None)
        if self._main_task:
            await self._main_task
    
    async def _main_atask(self) -> None:
        """Main task that processes the queue of events"""
        while True:
            event = await self._log_q.get()
            
            if event is None:
                break
            
            # Process the event
            event_name = event.get("event")
            message = event.get("message")
            
            if event_name == "user_speech_committed":
                if message:
                    self._conversation.append(message)
                self.message_count_since_analysis += 1
            
            elif event_name == "agent_speech_committed":
                if message:
                    self._conversation.append(message)
                self.message_count_since_analysis += 1
                
                # Full sentiment analysis after agent responds is a good time
                if (self.message_count_since_analysis >= 3 or 
                    time.time() - self.last_analysis_time > self.min_analysis_interval):
                    asyncio.create_task(self._run_analysis())
    
    def start(self) -> None:
        """Start the analyzer and attach to events"""
        self._main_task = asyncio.create_task(self._main_atask())
        
        if not self._agent:
            logger.warning("No agent model provided, cannot attach to events")
            return
        
        @self._agent.on("user_speech_committed")
        def _user_speech_committed(user_msg: ChatMessage):
            self._log_q.put_nowait({"event": "user_speech_committed", "message": user_msg})
        
        @self._agent.on("agent_speech_committed")
        def _agent_speech_committed(agent_msg: ChatMessage):
            self._log_q.put_nowait({"event": "agent_speech_committed", "message": agent_msg})
    
    async def _run_analysis(self):
        """Run sentiment analysis on the conversation"""
        async with self.analysis_lock:
            # Reset counter
            self.message_count_since_analysis = 0
            self.last_analysis_time = time.time()
            
            # Only analyze if we have enough messages
            relevant_messages = [m for m in self._conversation if m.role in ["user", "assistant"]]
            if len(relevant_messages) < self.min_messages_for_analysis:
                return
                
            # Format messages for analysis
            messages = []
            for m in relevant_messages:
                # Get role, defaulting to "unknown" if missing
                role = getattr(m, 'role', 'unknown')
                
                # Get content with different fallbacks
                content = ""
                if hasattr(m, 'content') and m.content:
                    content = m.content
                elif hasattr(m, 'text') and m.text:
                    content = m.text
                    
                # Process content - split if contains the marker
                if content and isinstance(content, str) and "# Context" in content:
                    content = content.split("# Context")[0]
                    
                messages.append(f"{role}: {content}")
            
            try:
                # Use the sentiment service to analyze
                sentiment_analysis = await self.sentiment_service.analyze_sentiment(
                    user_id=self.user_id,
                    source="voice",
                    source_id=self.user_id,
                    source_date_time=datetime.now(),
                    texts=messages
                )
                
                if sentiment_analysis is not None and sentiment_analysis.communication_directive is not None:
                    # Store the sentiment analysis - fixed here with correct argument count
                    self.current_sentiment = sentiment_analysis
                    self.sentiment_storage.save(sentiment_analysis)
                    
                    # Update the communication directive in the agent's chat context
                    await self._update_comm_directive(sentiment_analysis)
                    
                    logger.debug(f"Updated sentiment analysis for {self.user_id}")
            except Exception as e:
                logger.debug(f"Error in sentiment analysis: {e}")
                # import traceback
                # logger.debug(traceback.format_exc())
    
    async def _update_comm_directive(self, sentiment_analysis: SentimentAnalysis) -> None:
        """Update the communication directive in the agent's chat context"""
        if not self._agent or not sentiment_analysis.communication_directive:
            return
            
        # Format the communication directive
        comm_dir_str = (
            f"Communication Directive: {sentiment_analysis.communication_directive.directive}\n\n"
            f"Formality: {sentiment_analysis.communication_directive.formality.score}\n"
            f"{sentiment_analysis.communication_directive.formality.reasoning}"
        )

        user_state = get_user_state(self.user_id)
        if not user_state:
            user_state = UserState(user_id=self.user_id)
        
        user_state.sentiment_analysis = sentiment_analysis
        user_state.communication_directive = sentiment_analysis.communication_directive if sentiment_analysis else None
        save_user_state(user_state=user_state)
        
        # # Find existing directive or create new one
        # if self._model.chat_ctx:
        #     chat_ctx_comm_dir_message = next(
        #         (m for m in self._model.chat_ctx.messages 
        #          if m.role == "system" and "Communication Directive" in m.content), 
        #         None
        #     )
            
        #     if chat_ctx_comm_dir_message is None:
        #         chat_ctx_comm_dir_message = ChatMessage(role="system", content="")
        #         self._model.chat_ctx.messages.append(chat_ctx_comm_dir_message)
                
        #     chat_ctx_comm_dir_message.content = comm_dir_str
    
    async def get_current_directive(self) -> Optional[str]:
        """Get the current communication directive"""
        if not self.current_sentiment or not self.current_sentiment.communication_directive:
            return None
            
        return self.current_sentiment.communication_directive.directive