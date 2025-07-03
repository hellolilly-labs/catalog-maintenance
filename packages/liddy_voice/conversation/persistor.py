# import asyncio
# import json
# import logging
# from dataclasses import dataclass
# from datetime import datetime
# from typing import List, Optional, Union

# from livekit.agents import utils
# from livekit.agents.llm import ChatMessage

# from liddy_voice.session_state_manager import SessionStateManager

# import redis_client as redis_client

# logger = logging.getLogger("conversation-persistor")

# @dataclass
# class EventLog:
#     eventname: str
#     """Name of recorded event"""
#     time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
#     """Time the event is recorded"""


# @dataclass
# class TranscriptionLog:
#     role: str
#     """Role of the speaker"""
#     transcription: str
#     """Transcription of speech"""
#     time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
#     """Time the event is recorded"""

    
# class ConversationPersistor(utils.EventEmitter[EventTypes]):
#     """
#     Records and persists conversation events and transcriptions from a VoicePipelineAgent.
#     Stores conversations in Redis for persistence across sessions.
#     """
    
#     def __init__(
#         self,
#         *,
#         model: Optional[VoicePipelineAgent] = None,
#         log: Optional[str] = None,  # Kept for backwards compatibility
#         transcriptions_only: bool = False,
#         user_id: str,
#     ):
#         """
#         Initializes a ConversationPersistor instance.

#         Args:
#             model: An instance of a VoicePipelineAgent
#             log: Name for the conversation log (used as Redis key suffix)
#             transcriptions_only: If True, only transcriptions will be stored
#             user_id: Unique ID for the current user
#         """
#         super().__init__()

#         self._model = model
#         self._log_name = log
#         self._transcriptions_only = transcriptions_only
#         self._user_id = user_id
        
#         # Redis keys
#         self._conversation_key = f"conversation:{self._user_id}"
#         self._event_key = f"events:{self._user_id}"

#         # Conversation storage (in-memory)
#         self._user_transcriptions: List[TranscriptionLog] = []
#         self._agent_transcriptions: List[TranscriptionLog] = []
#         self._events: List[EventLog] = []
#         self._conversation: List[ChatMessage] = []

#         # Event queue for async processing
#         self._log_q = asyncio.Queue()
#         self._main_task = None

#     async def _main_atask(self) -> None:
#         """Main async task that processes the log queue"""
#         while True:
#             log = await self._log_q.get()

#             if log is None:
#                 break

#             try:
#                 if isinstance(log, EventLog) and not self._transcriptions_only:
#                     self._events.append(log)
#                     # Store event in Redis
#                     await self._store_event_in_redis(log)

#                 elif isinstance(log, TranscriptionLog):
#                     if log.role == "user":
#                         # Avoid duplicate user transcriptions
#                         if (self._user_transcriptions and 
#                             self._user_transcriptions[-1].transcription == log.transcription):
#                             continue
                        
#                         self._user_transcriptions.append(log)
                        
#                     else:  # assistant
#                         # Avoid duplicate agent transcriptions
#                         if (self._agent_transcriptions and 
#                             self._agent_transcriptions[-1].transcription == log.transcription):
#                             continue
                        
#                         self._agent_transcriptions.append(log)
                    
#                     # Create ChatMessage and add to conversation history
#                     if log.transcription:
#                         chat_message = ChatMessage.create(
#                             text=log.transcription,
#                             role=log.role,
#                         )
#                         self._conversation.append(chat_message)
                        
#                         # Store in Redis
#                         await self._store_message_in_redis(chat_message)
                        
#             except Exception as e:
#                 logger.error(f"Error processing log entry: {e}")

#     async def _store_message_in_redis(self, message: ChatMessage) -> None:
#         """Store a message in Redis"""
#         try:
#             SessionStateManager.add_to_conversation_history(
#                 self._user_id, 
#                 message.role, 
#                 message.content
#             )
#         except Exception as e:
#             logger.error(f"Error storing message in Redis: {e}")
            
#     async def _store_event_in_redis(self, event: EventLog) -> None:
#         """Store an event in Redis"""
#         try:
#             # Convert EventLog to serializable dict
#             event_data = {
#                 "eventname": event.eventname,
#                 "time": event.time
#             }
            
#             # Get existing events or create new list
#             events = await redis_client.async_get_list(self._event_key) or []
#             events.append(event_data)
            
#             # Limit events length
#             if len(events) > 200:
#                 events = events[-200:]
                
#             # Store updated events
#             await redis_client.async_set_list(self._event_key, events)
            
#         except Exception as e:
#             logger.error(f"Error storing event in Redis: {e}")

#     async def aclose(self) -> None:
#         """Clean shutdown of the persistor"""
#         self._log_q.put_nowait(None)
#         if self._main_task:
#             await self._main_task

#     def start(self) -> None:
#         """Start the persistor and attach to agent events"""
#         # Load existing conversation if available
#         self._load_conversation_from_redis()
        
#         # Start async processing task
#         self._main_task = asyncio.create_task(self._main_atask())
        
#         if not self._model:
#             logger.warning("No agent model provided, cannot attach to events")
#             return

#         # Attach event handlers
#         @self._model.on("user_started_speaking")
#         def _user_started_speaking():
#             self._log_q.put_nowait(EventLog(eventname="user_started_speaking"))

#         @self._model.on("user_stopped_speaking")
#         def _user_stopped_speaking():
#             self._log_q.put_nowait(EventLog(eventname="user_stopped_speaking"))

#         @self._model.on("agent_started_speaking")
#         def _agent_started_speaking():
#             self._log_q.put_nowait(EventLog(eventname="agent_started_speaking"))

#         @self._model.on("agent_stopped_speaking")
#         def _agent_stopped_speaking():
#             self._log_q.put_nowait(EventLog(eventname="agent_stopped_speaking"))

#         @self._model.on("user_speech_committed")
#         def _user_speech_committed(user_msg: ChatMessage):
#             self._log_q.put_nowait(TranscriptionLog(
#                 role="user", transcription=user_msg.content
#             ))
#             self._log_q.put_nowait(EventLog(eventname="user_speech_committed"))

#         @self._model.on("agent_speech_committed")
#         def _agent_speech_committed(agent_msg: ChatMessage):
#             self._log_q.put_nowait(TranscriptionLog(
#                 role="assistant", transcription=agent_msg.content
#             ))
#             self._log_q.put_nowait(EventLog(eventname="agent_speech_committed"))

#         @self._model.on("agent_speech_interrupted")
#         def _agent_speech_interrupted():
#             self._log_q.put_nowait(EventLog(eventname="agent_speech_interrupted"))

#         @self._model.on("function_calls_collected")
#         def _function_calls_collected():
#             self._log_q.put_nowait(EventLog(eventname="function_calls_collected"))

#         @self._model.on("function_calls_finished")
#         def _function_calls_finished():
#             self._log_q.put_nowait(EventLog(eventname="function_calls_finished"))
    
#     def get_conversation(self) -> List[ChatMessage]:
#         """Get the current conversation as a list of ChatMessage objects"""
#         return self._conversation

#     def _load_conversation_from_redis(self) -> None:
#         """Load the existing conversation from Redis"""
#         try:
#             # Load conversation messages
#             conversation_data = redis_client.get_list(self._conversation_key)
#             if conversation_data:
#                 # Convert each message dict back to ChatMessage
#                 for msg_data in conversation_data:
#                     if "role" in msg_data and "content" in msg_data:
#                         chat_message = ChatMessage.create(
#                             text=msg_data["content"],
#                             role=msg_data["role"],
#                         )
#                         self._conversation.append(chat_message)
                        
#                         # Add to appropriate transcription list
#                         if msg_data["role"] == "user":
#                             self._user_transcriptions.append(TranscriptionLog(
#                                 role="user", 
#                                 transcription=msg_data["content"],
#                                 time=msg_data.get("timestamp", "")
#                             ))
#                         elif msg_data["role"] == "assistant":
#                             self._agent_transcriptions.append(TranscriptionLog(
#                                 role="assistant", 
#                                 transcription=msg_data["content"],
#                                 time=msg_data.get("timestamp", "")
#                             ))
                
#             # Load events
#             events_data = redis_client.get_list(self._event_key)
#             if events_data:
#                 for event_data in events_data:
#                     if "eventname" in event_data:
#                         self._events.append(EventLog(
#                             eventname=event_data["eventname"],
#                             time=event_data.get("time", "")
#                         ))
                        
#             logger.info(f"Loaded {len(self._conversation)} messages from Redis for user {self._user_id}")
                        
#         except Exception as e:
#             logger.error(f"Error loading conversation from Redis: {e}")