import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple

from openai import AsyncOpenAI
import google.generativeai as genai
from livekit.agents import llm
from livekit.agents.llm import ChatMessage
# from livekit.agents.pipeline import VoicePipelineAgent
# from livekit.agents.pipeline.pipeline_agent import EventTypes

if __name__ == "__main__":
    # add parent directory to path
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    # add parent's parent directory to path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


from liddy_voice.session_state_manager import SessionStateManager
from liddy.model import BasicChatMessage, UserState, ConversationExitState
from liddy_voice.llm_service import LlmService
from liddy_voice.sentiment import SentimentService
from liddy_voice.user_manager import UserManager

logger = logging.getLogger("conversation-analyzer")

# class ConversationAnalyzer(utils.EventEmitter[EventTypes]):
class ConversationAnalyzer:
    """
    Analyzes conversation in real-time by hooking into LiveKit's event system.
    Identifies key topics, preferences and generates appropriate RAG filters.
    """
    
    def __init__(
        self,
        *,
        # model: VoicePipelineAgent | None,
        # llm_model: any,
        user_id: str,
    ):
        """
        Initialize the conversation analyzer
        
        Args:
            model: VoicePipelineAgent instance to hook events to
            llm_model: The LLM model to use for analysis
            user_id: The unique ID for the current user
        """
        super().__init__()
        
        # self._model = model
        # self.llm_model = llm_model
        self.user_id = user_id
        
        # Create a Gemini model client
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        GOOGLE_MODEL = os.getenv('GOOGLE_MODEL', 'gemini-2.5-pro-preview-03-25')
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel(GOOGLE_MODEL)
        
        # Analysis state tracking
        self.conversation_filters: Dict = {}
        self.analysis_lock = asyncio.Lock()
        self.last_analysis_time = 0
        self.analysis_task = None
        self.min_analysis_interval = 30  # seconds between analyses
        self.min_messages_for_analysis = 3
        self.message_count_since_analysis = 0
        self.category_mentions: Set[str] = set()
        self.price_mentions: Dict = {}
        
        # Event tracking
        self.events_since_analysis = []
        self._conversation = []
        self._log_q = asyncio.Queue()
        self._main_task = None
        
    
    async def append_messages(self, messages: List[any], realtime: bool=False) -> None:
        """Append a message to the conversation"""
        contains_user_message = False
        
        for message in messages:
            if message and (message.get('content') or message.get('message')):
                # self._extract_keywords_from_message(message.content)
                # self.message_count_since_analysis += 1
                if message.get('role') == "user":
                    contains_user_message = True
                
                # Store user message in conversation
                self._conversation.append(message)
            
        # run full analysis if this is a user message
        if contains_user_message:
            if realtime:
                await self._generate_search_context()
            else:
                # asyncio.create_task(self._run_analysis())
                asyncio.create_task(self._generate_search_context())
    
    async def aclose(self) -> None:
        """Exits"""
        self._log_q.put_nowait(None)
        if self._main_task:
            await self._main_task
    
    # async def _main_atask(self) -> None:
    #     """Main task that processes the queue of events"""
    #     while True:
    #         event = await self._log_q.get()
            
    #         if event is None:
    #             break
            
    #         # Process the event
    #         event_name = event.get("event")
    #         message = event.get("message")
            
    #         if event_name == "user_speech_committed":
    #             # Always do fast keyword extraction on user messages
    #             if message and message.content:
    #                 self._extract_keywords_from_message(message.content)
    #                 self.message_count_since_analysis += 1
                    
    #                 # Store user message in conversation
    #                 self._conversation.append(message)
                    
    #                 # Check if we should run full analysis
    #                 if (self.message_count_since_analysis >= 3 or 
    #                     time.time() - self.last_analysis_time > self.min_analysis_interval):
    #                     asyncio.create_task(self._run_analysis())
    #                 else:
    #                     # log the event for potential analysis
    #                     logger.debug(f"NOT Analyzing User message: {message.content}")
            
    #         elif event_name == "agent_speech_committed":
    #             # Store agent message in conversation
    #             if message:
    #                 self._conversation.append(message)
    #             self.message_count_since_analysis += 1
                
    #             # Full analysis after agent responds is a good time
    #             if (self.message_count_since_analysis >= 2 or 
    #                 time.time() - self.last_analysis_time > self.min_analysis_interval * 1.5):
    #                 asyncio.create_task(self._run_analysis())
    #             else:
    #                 logger.debug(f"NOT Analyzing Agent message: {message.content}")
    
    # def start(self) -> None:
    #     """Start the analyzer and attach to events"""
    #     self._main_task = asyncio.create_task(self._main_atask())
        
    #     if not self._model:
    #         logger.warning("No agent model provided, cannot attach to events")
    #         return
        
    #     @self._model.on("user_speech_committed")
    #     def _user_speech_committed(user_msg: ChatMessage):
    #         """Quick keyword extraction and potential analysis on user message"""
    #         self._log_q.put_nowait({"event": "user_speech_committed", "message": user_msg})
        
    #     @self._model.on("agent_speech_committed")
    #     def _agent_speech_committed(agent_msg: ChatMessage):
    #         """Analyze after the agent completes their turn"""
    #         self._log_q.put_nowait({"event": "agent_speech_committed", "message": agent_msg})
    
    # def _extract_keywords_from_message(self, message: str):
    #     """Extract keywords from a message without using LLM"""
    #     # Enhanced category keywords for better detection
    #     category_keywords = {
    #         "mountain": ["mountain", "trail", "downhill", "enduro", "xc", "rough terrain", "off-road"],
    #         "road": ["road", "racing", "race bike", "tarmac", "aethos", "pavement"],
    #         "gravel": ["gravel", "all-road", "diverge", "dirt road", "mixed terrain"],
    #         "helmet": ["helmet", "protection", "head"],
    #         "shoe": ["shoe", "footwear", "cleat", "cycling shoe"],
    #         "electric": ["electric", "e-bike", "turbo", "battery", "motor", "pedal assist"]
    #     }
        
    #     # Enhanced detection logic
    #     message_lower = message.lower()
    #     for category, keywords in category_keywords.items():
    #         if any(kw in message_lower for kw in keywords):
    #             # Store both in category_mentions and directly in filters
    #             self.category_mentions.add(category)
    #             if "all_categories" not in self.conversation_filters:
    #                 self.conversation_filters["all_categories"] = []
    #             if category not in self.conversation_filters["all_categories"]:
    #                 self.conversation_filters["all_categories"].append(category)
        
    #     # Extract price information
    #     price_patterns = [
    #         (r'under\s+\$?(\d+)', lambda x: float(x)),
    #         (r'less\s+than\s+\$?(\d+)', lambda x: float(x)),
    #         (r'(\d+)\s+dollars', lambda x: float(x)),
    #         (r'\$(\d+)', lambda x: float(x))
    #     ]
        
    #     for pattern, converter in price_patterns:
    #         if match := re.search(pattern, message_lower):
    #             price = converter(match.group(1))
    #             self.price_mentions[price] = time.time()
                
    #             # Set price range based on most recent mention
    #             if price <= 500:
    #                 self.conversation_filters["price_range"] = "budget"
    #             elif price <= 2000:
    #                 self.conversation_filters["price_range"] = "mid-range"
    #             elif price <= 5000:
    #                 self.conversation_filters["price_range"] = "premium"
    #             else:
    #                 self.conversation_filters["price_range"] = "high-end"
        
    #     # Add size extraction
    #     size_patterns = [
    #         # Common bike sizes
    #         (r'\b(small|medium|large|x-?large|s|m|l|xl)\b', "size"),
    #         # Numeric sizes for wheels, etc.
    #         (r'\b(27\.5|29|26|700c)\b', "size"),
    #         # Numeric with units
    #         (r'\b(\d+)(mm|cm|in)\b', "size"),
    #         # Specific size mentions
    #         (r'size\s+(\w+)', "size")
    #     ]
        
    #     for pattern, key in size_patterns:
    #         if matches := re.findall(pattern, message_lower):
    #             # If we find any size mentions, add them to filters
    #             if "size" not in self.conversation_filters:
    #                 self.conversation_filters["size"] = []
                
    #             # Add all found sizes (could be multiple in one message)
    #             for match in matches:
    #                 size_value = match[0] if isinstance(match, tuple) else match
    #                 # Normalize size values
    #                 if size_value.lower() in ('s', 'sm'):
    #                     size_value = 'small'
    #                 elif size_value.lower() in ('m', 'med'):
    #                     size_value = 'medium'
    #                 elif size_value.lower() in ('l', 'lg'):
    #                     size_value = 'large'
    #                 elif size_value.lower() in ('xl', 'x-large', 'xlg'):
    #                     size_value = 'xlarge'
                        
    #                 if size_value not in self.conversation_filters["size"]:
    #                     self.conversation_filters["size"].append(size_value)
    
    async def _run_analysis(self):
        """Run a full LLM-based analysis of the conversation"""
        async with self.analysis_lock:
            # Reset counters
            self.message_count_since_analysis = 0
            self.last_analysis_time = time.time()
            
            # Only analyze if we have enough messages
            relevant_messages = [m for m in self._conversation if m.get('role') in ["user", "assistant"]]
            if len(relevant_messages) < self.min_messages_for_analysis:
                return
            
            system_prompt_conversation_resumption = """You are an AI that analyzes conversations to extract context and product preferences for the bike brand Specialized brand bikes and accessories. Extract only clearly expressed preferences and facts, not assumptions. Format your response as JSON with these fields only:\n- product_categories: List of product categories mentioned (bikes, helmets, etc.)\n- bike_type: List of bike types mentioned (mountain, road, gravel, etc.)\n- price_range: Budget category if price is mentioned (budget, mid-range, premium, high-end)\n- features: List of specific features the customer wants\n- colors: List of colors explicitly mentioned\n- size: List of sizes mentioned (small, medium, large, xlarge, numeric sizes like 29, 27.5, etc.)\n- context: A freeform field to describe the full context of the conversation. For example, if the user has been looking at a particular bike and then moved on to look at pedals for that bike then moved on to look at shoes, the context field would contain all that detail so that an AI sales agent can fully understand the consumer's journey.\n\nOnly include fields where you have clear evidence. """
            
            # extract conversation into a string
            # Format conversation for analysis
            conversation_text = "\n".join([
                f"{m.get('role')}: {m.get('content') or m.get('message')}" 
                for m in relevant_messages # for m in relevant_messages[-25:] 
            ])
            # transcript_str = ""
            # for msg in relevant_messages:
            #     if 'role' in msg and 'message' in msg:
            #         # Add message to conversation
            #         transcript_str += f"{msg['role']}: {msg['message']}\n"

            formatted_messages = [
                {"role": "user", "parts": [{"text": system_prompt_conversation_resumption}]},
                {"role": "user", "parts": [{"text": f"Analyze this conversation and extract context and product preferences:\n\n{conversation_text}"}]}
            ]

            try:
                # Run analysis
                response = self.gemini_model.generate_content(formatted_messages)
                analysis_content = response.text.strip()
                            
                # Extract JSON from potential text wrapper
                json_match = re.search(r'\{.*\}', analysis_content, re.DOTALL)
                if json_match:
                    try:
                        analysis_json = json.loads(json_match.group(0))
                        
                        # # Merge fast keyword extraction with LLM analysis
                        # analysis_json = self._merge_with_keywords(analysis_json)
                        
                        # Update conversation filters
                        # self.conversation_filters = analysis_json
                        
                        # Store in user state for persistence
                        SessionStateManager.merge_analysis_results(self.user_id, analysis_json)
                        
                        logger.debug(f"Updated conversation filters: {analysis_json}")
                        # return analysis_json
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing analysis JSON: {e}")
                        logger.error(f"Raw content: {analysis_content}")
            except Exception as e:
                logger.error(f"Error in conversation analysis: {e}")

            # # After updating conversation filters with analysis results
            # if len(self._conversation) >= 2:
            #     # Generate search context every few messages
            #     await self._generate_search_context()
    
    def _merge_with_keywords(self, analysis_json: Dict) -> Dict:
        """Merge LLM analysis with keyword extraction results"""
        # Add any categories from keyword extraction not in LLM analysis
        if "product_categories" not in analysis_json:
            analysis_json["product_categories"] = list(self.category_mentions)
        elif self.category_mentions:
            existing_categories = set(analysis_json["product_categories"])
            analysis_json["product_categories"] = list(existing_categories.union(self.category_mentions))
            
        # Ensure price range from keyword extraction is included
        if "price_range" not in analysis_json and "price_range" in self.conversation_filters:
            analysis_json["price_range"] = self.conversation_filters["price_range"]
            
        return analysis_json
                
    async def get_filters(self) -> Dict:
        """Get current conversation filters for RAG queries"""
        async with self.analysis_lock:
            return self.conversation_filters.copy()
            
    def get_conversation(self) -> List[any]:
        """Get the current conversation messages"""
        return self._conversation

    async def reset_filters(self, filter_names=None):
        """Reset specified filters or all filters if none specified"""
        async with self.analysis_lock:
            if filter_names is None:
                # Reset all filters except very recent ones
                self.conversation_filters = {}
                self.category_mentions = set()
                self.price_mentions = {}
            else:
                # Reset only specified filters
                for name in filter_names:
                    if name in self.conversation_filters:
                        del self.conversation_filters[name]
                
                # If resetting categories, also reset the set
                if "product_categories" in filter_names:
                    self.category_mentions = set()
                
                # If resetting price, also reset the dict
                if "price_range" in filter_names:
                    self.price_mentions = {}

    async def _generate_search_context(self):
        """Generate a search context from conversation history"""
        # Only run this periodically to avoid excessive LLM calls
        if len(self._conversation) < 2:  # Need enough context
            return
        
        # # Take recent messages but exclude the very latest user message
        # # (which will be used directly in the RAG query)
        # recent_messages = self._conversation[-8:-1] if len(self._conversation) > 8 else self._conversation[:-1]
        # we want as much context as possible
        # we are going in reverse order to capture the most relevant context (latest first)
        recent_messages = self._conversation
        
        # reverse the order to get the most recent messages first
        recent_messages = recent_messages[::-1]
        
        if not recent_messages:
            return
        
        # Format conversation for the LLM
        conversation_text = ""
        for msg in recent_messages:
            if 'role' in msg and ('message' in msg or 'content' in msg):
                # Add message to conversation
                conversation_text = f"{msg.get('role')}: {msg.get('content') or msg.get('message')}\n{conversation_text}"

                if (len(conversation_text) / 4) > 8000:
                    break
        # conversation_text = "\n".join([
        #     f"{m.get('role')}: {m.get('content') or m.get('message')}" for m in recent_messages
        # ])
        
        system_prompt = ("You are an AI that creates search contexts based on conversation history. "
                "Review the bike shop conversation and extract key topics, preferences, and requirements. "
                "Output a concise paragraph that captures what information would help continue this conversation. "
                "Focus on bike types, features, price ranges, and other preferences mentioned. "
                "This will be combined with the customer's new question to retrieve relevant information.")

        user_state = UserManager.get_user_state(self.user_id)
        current_search_context = user_state.get("search_context", "") if user_state else ""
        
        user_message = f"Create a search context from this conversation:\n\n\"\"\"\n{conversation_text}\n\"\"\""
        
        if current_search_context:
            user_message += f"\n\nPrevious search context:\n\n\"\"\"\n{current_search_context}\n\"\"\""

        formatted_messages = [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": user_message}]}
        ]

        try:
            # Run analysis
            response = self.gemini_model.generate_content(formatted_messages)
            search_context = response.text.strip()
        
            # self.conversation_filters["search_context"] = search_context
            # Store in user state for persistence
            SessionStateManager.update_search_context(self.user_id, search_context)
            
            logger.debug(f"Generated search context: {search_context}")
        except Exception as e:
            logger.error(f"Error generating search context: {e}")

    async def generate_resumption_message(self, resumption_data: Dict) -> str:
        """Generate a contextual message for resuming the conversation"""
        try:
            analysis_ctx = llm.ChatContext().append(
                role="system",
                text=(
                    "You are an AI that analyzes conversations to extract product preferences "
                    "for the bike brand Specialized brand bikes and accessories. "
                    "Extract only clearly expressed preferences and facts, not assumptions. "
                    "Format your response as JSON with these fields only:\n"
                    "- product_categories: List of product categories mentioned (bikes, helmets, etc.)\n"
                    "- bike_type: List of bike types mentioned (mountain, road, gravel, etc.)\n" 
                    "- price_range: Budget category if price is mentioned (budget, mid-range, premium, high-end)\n"
                    "- features: List of specific features the customer wants\n"
                    "- colors: List of colors explicitly mentioned\n"
                    "- size: List of sizes mentioned (small, medium, large, xlarge, numeric sizes like 29, 27.5, etc.)\n"
                    "Only include fields where you have clear evidence."
                )
            ).append(
                role="user",
                text=f"Analyze this conversation and come up with a prompt for the user when they come back to this conversation. Remember that the user will have been gone for some time (up to a day), so you should start with some sort of greeting and then remind them of the context. Respond with the prompt only:\n\n{json.dumps(resumption_data)}"
            )
            
            analysis_stream = self.llm_model.chat(chat_ctx=analysis_ctx)
            analysis_content = ""
            async for chunk in analysis_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    analysis_content += chunk.choices[0].delta.content
            
            logger.debug(f"Resumption message: {analysis_content}")
            return analysis_content
    
        except Exception as e:
            logger.error(f"Error generating resumption message: {e}")
            # return cls._fallback_resumption_message(resumption_data)

        last_agent_msg = resumption_data.get('last_agent_message', '')
        topics = resumption_data.get('topics', [])
        
        # If we have product categories, mention them
        if topics:
            topic_str = ", ".join(topics[:2])  # Mention up to 2 topics
            return f"Welcome back! Last time we were discussing {topic_str}. Would you like to continue that conversation?"
        
        # If we have the last agent message, base it on that
        if last_agent_msg:
            # Keep it short - just first sentence
            first_sentence = last_agent_msg.split('.')[0]
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:100] + "..."
            
            return f"Welcome back! Last time I mentioned: '{first_sentence}'. Would you like to continue where we left off?"
        
        # Generic fallback
        return "Welcome back! Would you like to continue our previous conversation?"

    async def analyze_chat(self, user_state: UserState, transcript: List[any], source: str = "chat", source_id: str = None, source_date_time: float = None) -> Dict:
        """Analyze the chat messages"""
        
        source_date_time = source_date_time or time.time()
        source_id = source_id or user_state.voice_session_id or str(source_date_time)
        source = source or "chat"
        
        try:
            GENERAL_ANALYSIS_MODEL = os.getenv('GENERAL_ANALYSIS_MODEL', 'gemini-2.5-pro-preview-03-25')
            if not GENERAL_ANALYSIS_MODEL:
                return {"success": False, "error": "Google model not configured"}

            if not user_state or not user_state.user_id:
                return {"success": False, "error": "User ID is required"}

            if not transcript or not isinstance(transcript, list):
                return {"success": False, "error": "Invalid transcript format"}

            # transcript_summary = analysis.get('transcript_summary')
            transcript_summary = ''

            system_prompt_conversation_resumption = """Given the entire conversation, capture the essence and context of the conversation to generate the following:
                1. A custom message for when the user reconnects. The idea is to pick up the conversation with the user, but be mindful that this will replace the first message the agent says to the user so it should include some sort of "welcome."
                2. A summary of the conversation transcript. This should be a brief summary of the conversation that the agent can use to quickly catch up on the conversation. It should be concise and to the point but not lose any nuance or detail. In particular, capture details about products, preferences, and any other important information that was discussed. This will help the agent quickly understand the context of the conversation. Take in to consideration the current product details as well as any browsing history (if included).

                Respond in the following JSON format:
                {
                    "resumption_message": "Welcome back! I'm here to help you with your questions. How can I assist you today?",
                    "transcript_summary": "Here is a summary of the conversation transcript..."
                }
                """

            # it's only worth processing the transcript if there's more than 3 messages
            if len(transcript) < 3:
                return {"success": True}
            
            formatted_messages = [
                # {"role": "user", "parts": [{"text": system_prompt_conversation_resumption}]},
            ]

            # check to see if we have a summary of the last conversation and pull that in as context
            previous_transcript_summary = None
            conversation_exit_state = user_state.conversation_exit_state
            if conversation_exit_state:
                previous_transcript_summary = conversation_exit_state.transcript_summary
            
                if previous_transcript_summary:
                    formatted_messages.append({"role": "user", "parts": [{"text": f"Previous Conversation Summary:\n\n{previous_transcript_summary}"}]})
            
            # see if we have information on the current URL and any product details
            history = UserManager.get_user_recent_history(user_id=user_state.user_id)
            most_recent_history = history[0] if history else None
            if most_recent_history:
                browsing_history_message = None
                    
                # if there are more than 1 history, then pull the title and url from each one into the browsing history message
                include_browsing_history = True
                if include_browsing_history and len(history) > 1:
                    browsing_history_message = "Browsing History (in descending order):\n\n"
                    for url in history:
                        if url.title and url.url:
                            # grab the details if they exist
                            details = None
                            if url.product_details:
                                if 'product' in url.product_details and 'details' in url.product_details['product']:
                                    details = url.product_details['product']['details']
                            browsing_history_message += f"- [{url.title}]({url.url}): {details if details else ''}\n\n"
                    
                if most_recent_history:
                    current_url = None
                    title = None
                    details = None
                    current_url = most_recent_history.url
                    title = most_recent_history.title
                    product_details = most_recent_history.product_details
                    if 'product' in product_details:
                        product = product_details['product']
                        if 'details' in product:
                            details = product['details']
                    # if 'recommended_products' in product_details:
                    #     product_recommendations = product_details['recommended_products']
                    
                    last_product_text = ""
                    if current_url:
                        if title:
                            last_product_text = f"Last URL: [{title}]({current_url})"
                        else:
                            last_product_text = f"Last URL: {current_url}"
                    if details:
                        if last_product_text:
                            last_product_text += f"\n\nDetails:\n{details}"
                        else:
                            last_product_text = f"Last Product Details:\n{details}"
                    
                    if browsing_history_message:
                        last_product_text += f"\n\n{browsing_history_message}"

                    if last_product_text:
                        formatted_messages.append({"role": "user", "parts": [{"text": last_product_text}]})
            
            # # see if we have a history object for this chat (from a timestamp perspective) that has a reason
            # history = user_state.get('conversation_history')
            # if history:
            #     last_history = history[-1]
            #     last_timestamp = last_history.get('timestamp')
            #     last_timestamp = datetime.fromisoformat(last_timestamp) if last_timestamp else None
            #     if last_timestamp and 'end_reason' in last_history:
            #         # make sure this timestamp is not too old (older than 1 minute)
            #         if datetime.now() - last_timestamp < timedelta(minutes=1):
            #             reason = last_history.get('end_reason')
            #             formatted_messages.append({"role": "user", "parts": [{"text": f"Reason for ending conversation: {reason}"}]})
            
            transcript_str = ""
            conversation_transcript: List[BasicChatMessage] = []
            for msg in transcript:
                if not hasattr(msg, "role"):
                    continue
                if isinstance(msg, BasicChatMessage):
                    transcript_str += f"{msg.role}: {msg.content}\n"
                    # if msg.role != "system":
                    conversation_transcript.append(msg)
                elif isinstance(msg, ChatMessage):
                    transcript_str += f"{msg.role}: {msg.content}\n"
                    # if msg.role != "system":
                    conversation_transcript.append(BasicChatMessage(msg.role, msg.content, msg.created_at if msg.created_at else None))
                elif 'role' in msg and 'message' in msg:
                    transcript_str += f"{msg['role']}: {msg['message']}\n"
                    # if msg.role != "system":
                    conversation_transcript.append((BasicChatMessage(msg['role'], msg['message'])))
                elif 'role' in msg and 'content' in msg:
                    transcript_str += f"{msg['role']}: {msg['content']}\n"
                    # if msg.role != "system":
                    conversation_transcript.append((BasicChatMessage(msg['role'], msg['content'])))
                elif 'role' in msg and 'text' in msg:
                    transcript_str += f"{msg['role']}: {msg['text']}\n"
                    # if msg.role != "system":
                    conversation_transcript.append((BasicChatMessage(msg['role'], msg['text'])))

            formatted_messages.append({"role": "user", "parts": [{"text": f"Please analyze the conversation transcript. Respond with the `resumption_message` and `transcript_summary` in JSON format. The resumption is intended to pick up the conversation with the user, but be mindful that this will replace the first message the agent says to the user so it should include some sort of 'welcome.'\n\nConversation Transcript:\n\"\"\"\n{transcript}\n\"\"\""}]})

            response_obj = None
            try:
                llm_service = LlmService.fetch_model_service_from_model(GENERAL_ANALYSIS_MODEL, account=user_state.account, user=user_state.user_id, model_use="conversation_analysis")
                initial_chat_message = llm.ChatMessage(role="user", content=[system_prompt_conversation_resumption])
                chat_ctx = llm.ChatContext([
                    initial_chat_message
                ])
                for message in formatted_messages:
                    chat_ctx.add_message(role=message['role'], content=message['parts'][0]['text'])
                
                analysis_content = await LlmService.chat_wrapper(
                    llm_service=llm_service,
                    chat_ctx=chat_ctx
                )
                response_obj = LlmService.parse_json_response(analysis_content)
            except Exception as e:
                logger.error(f"Error in conversation analysis: {e}")
                generative_model = genai.GenerativeModel(
                    model_name=os.getenv('GOOGLE_MODEL', 'gemini-2.0-flash-001'),
                    system_instruction=system_prompt_conversation_resumption
                )

                response = generative_model.generate_content(formatted_messages)
                response_text = response.text.strip()
                # convert the response text to a JSON object
                # strip down response_text to the first and last curly braces
                response_text = response_text[response_text.find('{'):response_text.rfind('}') + 1]
                response_obj = json.loads(response_text)

            resumption_message = None
            resumption_transcript_summary = None
            if response_obj:
                try:
                    # get the resumption message and transcript summary
                    resumption_message = response_obj.get('resumption_message')
                    resumption_transcript_summary = response_obj.get('transcript_summary')
                except Exception as e:
                    logger.error(f"Error processing transcript: {str(e)}")
                    logger.exception(e)
            
            conversation_exit_state = ConversationExitState(
                exit_reason="",
                last_interaction_time=time.time(),
                transcript_summary=resumption_transcript_summary or transcript_summary,
                resumption_message=resumption_message or ""
            )
            
            # update the resumption message in the user state
            user_state.conversation_exit_state = conversation_exit_state
            
            sentiment_service = SentimentService()
            await sentiment_service.analyze_sentiment(
                user_state=user_state,
                source=source,
                source_date_time=time.time(),
                source_id=source_id,
                texts=conversation_transcript
            )
            
            # save the user state
            UserManager.save_user_state(user_state=user_state)
            
            # logger.info(f"Chat completion response: {response_text}")

            return {"success": True}
        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}")
            logger.exception(e)  # Log the full traceback
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Example usage
    # user_id="4d1b1f89-563e-483f-9d9d-a17c6524c633"
    user_id="cf8ea811-9e56-4357-b1ff-e4e917171034"
    analyzer = ConversationAnalyzer(user_id=user_id)
    # Add your test cases here
    # analyzer.analyze_chat(user_state, transcript)
    # pass