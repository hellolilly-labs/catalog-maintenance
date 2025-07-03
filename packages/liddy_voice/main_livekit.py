import asyncio
import json
import logging
import os
from datetime import datetime
import time

import aiofiles
from dotenv import load_dotenv
from typing import List

from livekit.agents import (
    stt,
    tts,
    llm,
    MetricsCollectedEvent,
    metrics,
    WorkerOptions,
    AgentSession,
    JobContext,
    JobProcess,
    AutoSubscribe,
    RoomInputOptions,
    ChatContext,
    BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip
)
from livekit.plugins import (
    assemblyai,
    deepgram, 
    google,
    silero, 
    elevenlabs, 
    noise_cancellation
)
from google.cloud import texttospeech
from livekit.plugins.turn_detector.english import EnglishModel
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
from liddy_voice.spence.llm_service import LlmService
from liddy_voice.spence.session_state_manager import SessionStateManager, ConversationResumptionState
from liddy_voice.spence.assistant import Assistant
from liddy_voice.spence.model import BasicChatMessage, UserState
from redis_client import get_user_state, get_user_latest_conversation
from liddy_voice.spence.account_manager import ElevenLabsTtsProviderSettings, get_account_manager

# Configure logging first, before any functions try to use it
def setup_logging():
    """Configure logging to prevent duplicate log messages"""
    # Clear all existing handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure the formatter we want to use
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add a single handler to the root logger
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    # Disable propagation for the LiveKit loggers to avoid duplicate logs
    for logger_name in ['livekit', 'livekit.agents', 'livekit.plugins']:
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        # Clear any existing handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        # Add our standard handler
        logger.addHandler(handler)
    
    # Get our application logger
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured")
    
    return logger

# Initialize logging
logger = setup_logging()

logger.info(f"===== Voice Agent Process: {__name__} =====")


# Load environment variables with stricter fallback mechanism
def load_environment_variables():
    """Load environment variables with explicit environment handling."""
    # Determine deployment environment from ENV_TYPE
    env_type = os.getenv('ENV_TYPE', 'development').lower()
    logger.info(f"Initializing with environment type: {env_type}")
    
    # Load the environment-specific file
    loaded = False
    env_file = f'.env.{env_type}'
    
    # First try from /config directory
    if os.path.exists(f'config/{env_file}'):
        loaded = load_dotenv(dotenv_path=f'config/{env_file}', override=True)
        logger.info(f"Loaded environment from config/{env_file}")
    # Then try from root directory
    elif os.path.exists(env_file):
        loaded = load_dotenv(dotenv_path=env_file, override=True)
        logger.info(f"Loaded environment from {env_file}")
    # For development environment only, try the default .env file as fallback
    elif env_type == 'development' and os.path.exists('.env'):
        loaded = load_dotenv(dotenv_path='.env', override=True)
        logger.info("Development environment: loaded from .env file")
    
    if not loaded:
        logger.warning(f"No environment file found for {env_type}. Using system environment variables only.")
    
    # Validate critical environment variables
    required_vars = [
        'ELEVENLABS_API_KEY',
        'OPENAI_API_KEY',
        'GOOGLE_API_KEY',
        'GROQ_API_KEY',
        'LIVEKIT_API_KEY',
        'LIVEKIT_API_SECRET',
        'LIVEKIT_URL',
        'MODEL_NAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        # In production, you might want to raise an exception here
        if env_type != 'development':
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Log loaded configuration (without sensitive values)
    config_summary = {
        'ENV_TYPE': env_type,
        'LLM_PROVIDER': os.getenv('LLM_PROVIDER', 'google'),
        'MODEL_NAME': os.getenv('MODEL_NAME', 'not set'),
        'GENERAL_ANALYSIS_MODEL': os.getenv('GENERAL_ANALYSIS_MODEL', 'not set'),
        'OPENAI_MODEL': os.getenv('OPENAI_MODEL', 'not set'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY', 'not set'),
        'REDIS_HOST': os.getenv('REDIS_HOST', 'not set'),
        'REDIS_PORT': os.getenv('REDIS_PORT', 'not set'),
        'REDIS_PREFIX': os.getenv('REDIS_PREFIX', ''),
        'CONVERSATION_TTL': os.getenv('CONVERSATION_TTL', 'not set'),
        'LIVEKIT_URL': os.getenv('LIVEKIT_URL', 'not set'),
        'LIVEKIT_API_KEY': os.getenv('LIVEKIT_API_KEY', 'not set'),
        'LIVEKIT_API_SECRET': os.getenv('LIVEKIT_API_SECRET', 'not set'),
        'VOICE_MODEL': os.getenv('VOICE_MODEL', 'not set'),
        'VOICE_STABILITY': os.getenv('VOICE_STABILITY', 'not set'),
        'VOICE_SIMILARITY_BOOST': os.getenv('VOICE_SIMILARITY_BOOST', 'not set'),
        'VOICE_STYLE': os.getenv('VOICE_STYLE', 'not set'),
        'USE_NOISE_CANCELLATION': os.getenv('USE_NOISE_CANCELLATION', 'not set'),
    }
    logger.info(f"Active configuration: {json.dumps(config_summary)}")

# Call the function to load environment variables
load_environment_variables()


logger.info(f"===== Voice Agent Starting =====\nLIVEKIT_URL: {os.environ.get('LIVEKIT_URL', 'not set')}")


# ==============================
# PREWARM FUNCTION
# ==============================

def prewarm(proc: JobProcess):
    # """Load VAD model before agent starts"""
    # proc.userdata["vad"] = silero.VAD.load()
    pass

# ==============================
# REQUEST FUNCTION
# ==============================

async def request_fnc(ctx: JobProcess):
    await ctx.accept()

# ==============================
# MAIN ENTRYPOINT
# ==============================

async def setup_agent(user_state: UserState):
    """Configure and return the voice agent"""
    # Configure STT
    # credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_file = None
    try:
        primary_stt = deepgram.STT()
        # Get path to credentials file - check both absolute and relative paths
        if credentials_file and not os.path.isabs(credentials_file):
            # Convert relative path to absolute
            credentials_file = os.path.join(os.getcwd(), credentials_file)
        
        if credentials_file and os.path.exists(credentials_file):
            logger.info(f"Using Google STT with credentials file: {credentials_file}")
            secondary_stt = google.STT(credentials_file=credentials_file)
        else:
            logger.warning("Google credentials file not found, falling back to Deepgram only")
            primary_stt = assemblyai.STT(
                end_of_turn_confidence_threshold=0.8,
                min_end_of_turn_silence_when_confident=65,
                max_turn_silence=2250,
            )
            secondary_stt = deepgram.STT()
            # secondary_stt = google.STT()
            # primary_stt = deepgram.STT()
        
        if not secondary_stt:
            stt_model = primary_stt
        else:
            stt_model = stt.FallbackAdapter([primary_stt, secondary_stt])
    except Exception as e:
        logger.warning(f"Error initializing Google STT: {e}. Using Deepgram only.")
        primary_stt = assemblyai.STT()
        secondary_stt = deepgram.STT()
        stt_model = stt.FallbackAdapter([primary_stt, secondary_stt])
        # stt_model = deepgram.STT()
            
    try:
        # Get TTS settings from account manager
        account_manager = await get_account_manager(user_state.account)
        tts_settings = account_manager.get_tts_settings()
        tts_models = []

        # Create a TTS model for each provider
        for provider in tts_settings.providers:
            if isinstance(provider, ElevenLabsTtsProviderSettings):
            # if provider.voice_provider == "elevenlabs":
                elevenlabs_api_key = os.getenv("ELEVEN_API_KEY")
                
                logger.info(f"Setting up ElevenLabs TTS with voice: {provider.voice_name} ({provider.voice_id})")
                tts_model = elevenlabs.TTS(
                    voice_id=provider.voice_id,
                    model=provider.voice_model,
                    api_key=elevenlabs_api_key,
                    auto_mode=True,
                    voice_settings=elevenlabs.VoiceSettings(
                        stability=provider.voice_stability, 
                        similarity_boost=provider.voice_similarity_boost, 
                        style=provider.voice_style, 
                        use_speaker_boost=provider.voice_use_speaker_boost,
                        speed=provider.voice_speed if provider.voice_speed else 1.0
                    )
                )
                tts_models.append(tts_model)
            
            elif provider.voice_provider == "google":
                logger.info(f"Setting up Google TTS with voice: {provider.voice_name} ({provider.voice_id})")
                tts_model = google.TTS(
                    credentials_file=credentials_file,
                    voice_name=provider.voice_id,
                    language="en-US"
                    # voice=texttospeech.VoiceSelectionParams(
                    #     language_code="en-US",
                    # )
                )
                tts_models.append(tts_model)
            
            # You can add more provider types here in the future
            else:
                logger.warning(f"Unknown voice provider: {provider.voice_provider}")

        # Set up the TTS model using the FallbackAdapter if more than one model
        if not tts_models:
            logger.warning("No TTS models were created. Falling back to default Google TTS.")
            tts_model = google.TTS(
                credentials_file=credentials_file,
                voice=texttospeech.VoiceSelectionParams(
                    name="en-US-Chirp3-HD-Orus",
                    language_code="en-US",
                )
            )
        elif len(tts_models) == 1:
            logger.info("Using single TTS model without fallback")
            tts_model = tts_models[0]
        else:
            logger.info(f"Using FallbackAdapter with {len(tts_models)} TTS models")
            tts_model = tts.FallbackAdapter(tts_models)
    except Exception as e:
        logger.error(f"Error setting up TTS: {e}")
        # Fallback to a default TTS model if anything fails
        tts_model = google.TTS(
            credentials_file=credentials_file,
            voice=texttospeech.VoiceSelectionParams(
                name="en-US-Chirp3-HD-Orus",
                language_code="en-US",
            )
        )
    
    # Configure LLM
    primary_model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash-001")
    primary_llm_model = LlmService.fetch_model_service_from_model(primary_model_name, account=user_state.account, user=user_state.user_id, model_use="voice_assistant")
    fallback_model_name = os.getenv("FALLBACK_MODEL_NAME", "gpt-4.1-mini")
    fallback_model = LlmService.fetch_model_service_from_model(fallback_model_name, account=user_state.account, user=user_state.user_id, model_use="voice_assistant")
    
    
    llm_model = llm.FallbackAdapter([
        primary_llm_model,
        fallback_model
    ])
        
    # Create the agent session
    session = AgentSession[UserState](
        userdata=user_state,
        user_away_timeout=30.0,
        stt=stt_model,
        llm=llm_model,
        tts=tts_model,
        # vad=ctx.proc.userdata["vad"],
        vad=silero.VAD.load(),
        turn_detection="stt"
        # turn_detection=EnglishModel() # MultilingualModel()
        # turn_detection=MultilingualModel()
    )
    
    return session


async def setup_user_state(ctx, agent: Assistant):
    """Set up user state and personalization"""
    # user_id = ctx.room.name
    user_id = agent.get_user_id()
    
    # Update user state to record this voice interaction
    user_state = agent.session.userdata or get_user_state(user_id)
    if user_state is None:
        user_state = UserState(user_id=user_id)
    agent.session.userdata = user_state

    user_state.interaction_start_time = time.time()    
    user_state.last_interaction_time = time.time()
    user_state.voice_session_id = ctx.room.name
    user_state.conversation_exit_state = None
    # # now clear the user_state transcript
    # SessionStateManager.clear_conversation_resumption(user_id)

    # Handle SIP calls specially
    if participant.kind == 3:  # "SIP"
        current_date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        phone_number = participant.attributes.get("sip.phoneNumber")
        # Extract just the digits from the phone number
        phone_digits = ''.join(filter(str.isdigit, phone_number))
        # Ensure logs directory exists
        os.makedirs(f"logs/{phone_digits}", exist_ok=True)
        
        log_file_name = f"logs/{phone_digits}/{current_date_time_string}.txt"
        user_id = phone_digits
        
        call_id = participant.attributes.get("sip.callID")
        # Write the call id to the log file
        async with aiofiles.open(log_file_name, "w") as file:
            await file.write(f"Call ID: {call_id}\nPhone Number: {phone_number}\nDate and Time: {current_date_time_string}\nType: Interview\n\n")
    else:
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        log_file_name = f"logs/{user_id}.txt"
        
    await agent._update_user_state_prompt()
        
    return user_id, log_file_name

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the Spence voice agent with optimized user response time"""
    start_total = time.monotonic()  # total timer
    global participant, agent
    
    logger.info(f"Connecting to Room: {ctx.room.name}")
    start_connect = time.monotonic()
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.debug(f"Connected to room in {time.monotonic() - start_connect:.2f} seconds")
    
    # PHASE 1: Fast path - minimal agent setup for immediate response
    phase1_start = time.monotonic()

    use_noise_cancellation = os.getenv("USE_NOISE_CANCELLATION", "false").lower() == "true"
    if use_noise_cancellation:
        room_input_options = RoomInputOptions(
            close_on_disconnect=False,
            noise_cancellation=noise_cancellation.BVC()
        )
    else:
        room_input_options = RoomInputOptions(
            close_on_disconnect=False,
        )
    
    primary_model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash-001")
    
    logger.debug("Waiting for first participant...")
    start_wait = time.monotonic()
    participant = await ctx.wait_for_participant()
    logger.debug(f"First participant connected in {time.monotonic() - start_wait:.2f} seconds")
    user_state = get_user_state(participant.identity)
    metadata = json.loads(participant.metadata) if participant.metadata else {}
    account = None
    if metadata:
        if "account" in metadata:
            account = metadata["account"]

    account_manager = await get_account_manager(account)

    user_state.account = account_manager.get_account()
    
    session = await setup_agent(user_state)
    
    usage_collector = metrics.UsageCollector()
    # current_speech_metrics: List[metrics.AgentMetrics] = []
    eou_metric: metrics.EOUMetrics = None
    llm_metric: metrics.LLMMetrics = None
    tts_metric: metrics.TTSMetrics = None
    def handle_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        nonlocal eou_metric, llm_metric, tts_metric
        if isinstance(ev.metrics, metrics.EOUMetrics):
            eou_metric = ev.metrics
        elif isinstance(ev.metrics, metrics.LLMMetrics):
            llm_metric = ev.metrics
        elif isinstance(ev.metrics, metrics.TTSMetrics):
            tts_metric = ev.metrics
            if eou_metric and llm_metric and tts_metric:
                total_latency = eou_metric.end_of_utterance_delay + llm_metric.ttft + tts_metric.ttfb
                logger.info(f"EOU metrics: {eou_metric}")
                logger.info(f"LLM metrics: {llm_metric}")
                logger.info(f"TTS metrics: {tts_metric}")
                logger.info(f"Total latency: {total_latency:.2f} seconds")
                eou_metric = None
                llm_metric = None
                tts_metric = None
        usage_collector.collect(ev.metrics)
        # asyncio.create_task(agent.on_metrics_collected(ev))
    session.on("metrics_collected", handle_metrics_collected)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # At shutdown, generate and log the summary from the usage collector
    ctx.add_shutdown_callback(log_usage)

    agent = Assistant(ctx=ctx, primary_model=primary_model_name, account=account_manager.get_account())
    print("session room name:", ctx.room.name)
    phase1_session_start = time.monotonic()
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=room_input_options
    )
    logger.debug(f"Session start completed in {time.monotonic() - phase1_session_start:.2f} seconds")
    logger.debug(f"Phase 1 completed in {time.monotonic() - phase1_start:.2f} seconds")
    
    use_background_audio = os.getenv("USE_BACKGROUND_AUDIO", "false").lower() == "true"
    if use_background_audio:
        background_audio = BackgroundAudioPlayer(
            # play office ambience sound looping in the background
            ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.05),
            # play keyboard typing sound when the agent is thinking
            thinking_sound=[
                    AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.125),
                    AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.1),
                ],
            )
        await background_audio.start(room=ctx.room, agent_session=session)

    def handle_agent_state_changed(ev):
        asyncio.create_task(agent.on_agent_state_changed(ev))
    session.on("agent_state_changed", handle_agent_state_changed)
    
    def handle_user_state_changed(ev):
        asyncio.create_task(agent.on_user_state_changed(ev))
    session.on("user_state_changed", handle_user_state_changed)
    
    def handle_participant_disconnected(p):
        asyncio.create_task(agent.on_participant_disconnected(p))
    ctx.room.on("participant_disconnected", handle_participant_disconnected)
    
    def handle_participant_connected(p):
        asyncio.create_task(agent.on_participant_connected(p))
    ctx.room.on("participant_connected", handle_participant_connected)
    
    def handle_room_disconnected(ev):
        asyncio.create_task(agent.on_room_disconnected(ev))
    ctx.room.on("disconnected", handle_room_disconnected)
    
    # participant_metadata_changed
    def handle_participant_metadata_changed(p):
        asyncio.create_task(agent.on_participant_metadata_changed(p))
    ctx.room.on("participant_metadata_changed", handle_participant_metadata_changed)
    
    # participant_attributes_changed
    def handle_participant_attributes_changed(attributes, p):
        asyncio.create_task(agent.on_participant_attributes_changed(attributes=attributes, participant=p))
    ctx.room.on("participant_attributes_changed", handle_participant_attributes_changed)
    
    logger.debug(f"Starting voice assistant for participant {participant.identity}")
    user_id = participant.identity
    agent.set_user_id(user_id)
    
    # Check if there's a recent conversation to resume
    resumption_data: ConversationResumptionState = SessionStateManager.get_conversation_resumption(user_state=user_state)
    is_resumable = resumption_data is not None and resumption_data.is_resumable
    resumption_message = None
    has_previous_conversation = resumption_data is not None
    
    time_since_last_conversation = None
    
    # TODO: See ROADMAP/CONVERSATION_RESUMPTION_STRATEGY.md for more details
    # # if conversation is resumable, then attempt to apply the previous conversation transcript to the chat_ctx
    # if has_previous_conversation:
    #     last_interaction = resumption_data.last_interaction_time
    #     if last_interaction:
    #         current_time = time.time()
    #         time_since_last_conversation = current_time - last_interaction
        
    #     # Load the conversation history into the chat context
    #     chat_ctx: ChatContext = None

    #     # Get the resumption message if available
    #     if is_resumable:
    #         resumption_message = resumption_data.resumption_message
    #         logger.debug(f"Resumption message: {resumption_message}")
            
    #         # attempt to load the conversation history
    #         messages: List[BasicChatMessage] = resumption_data.chat_messages
    #         if messages:
    #             chat_ctx = agent.chat_ctx.copy()
    #             for message in messages:
    #                 chat_ctx.add_message(
    #                     role=message.role,
    #                     content=message.content
    #                 )
    #     else:
    #         resumption_message = None

    #     user_state_message = resumption_data.user_state_message
    #     if user_state_message:
    #         if not chat_ctx:
    #             chat_ctx = agent.chat_ctx.copy()

    #         # append the user state message to the chat context
    #         chat_ctx.add_message(
    #             role="system",
    #             content=user_state_message
    #         )

    #     if chat_ctx:
    #         await agent.update_chat_ctx(chat_ctx=chat_ctx)
            
    # # # load up the previous conversation into the chat context
    # # # and say the response message
    # # has_previous_conversation = False
    # # try:
    # #     conversation_json = get_user_latest_conversation(
    # #         account=account,
    # #         user_id=user_id
    # #     )
        
    # #     if conversation_json:
    # #         # Get the conversation history from the JSON
    # #         conversation_history = conversation_json.get("messages", [])
    # #         if conversation_history:
    # #             # Create a new chat context with the conversation history
    # #             chat_ctx = agent.chat_ctx.copy()
    # #             for message in conversation_history:
    # #                 if message.get('type') == "message" and message.get("role") in ["user", "assistant"]:
    # #                     has_previous_conversation = True
    # #                     chat_ctx.add_message(
    # #                         role=message["role"],
    # #                         content=message["content"]
    # #                     )

    # #             if has_previous_conversation:
    # #                 await agent.update_chat_ctx(chat_ctx=chat_ctx)
    # # except Exception as e:
    # #     logger.error(f"Error loading conversation history: {e}")
    # #     # Fallback to default behavior if conversation history cannot be loaded
    # #     has_previous_conversation = False

    # # Prepare welcome message - use resumption message if available
    # logger.info(f"Time to first utterance: {time.monotonic() - start_total:.2f} seconds")
    # response_message = participant.attributes.get("responseMessage") or metadata.get("responseMessage")
    # if time_since_last_conversation and time_since_last_conversation < 120:
    #     logger.info(f"Generating resumption message for user {user_id}")
    #     session.generate_reply(instructions="the user just reconnected after a short break so say something appropriate based on the context")
    #     # await speech
    # elif response_message:        
    #     logger.info(f"Using response message: {response_message}")
    #     welcome_message = response_message
    #     # Say welcome immediately - this happens in parallel with setup below
    #     # print time to first response
    #     session.say(text=welcome_message, allow_interruptions=True)
    # elif resumption_message:
    #     logger.info(f"Using resumption message: {resumption_message}")
    #     session.generate_reply(instructions=f"The user just reconnected after a short break so say something appropriate based on the context. The following was the resumption message after their last converstion. However, keep in mind that they may or may not be in the same state (for example, they may be looking at a different product now):\n\n{resumption_message}")
    # else:
    #     # if there is a previous conversation, then generate a reply based on the context
    #     if has_previous_conversation:
    #         logger.info(f"Previous conversation found, generating reply")
    #         session.generate_reply(instructions="The user just reconnected after a short break so say something appropriate based on the context")
    #     else:
    #         welcome_message = account_manager.get_default_greeting()
    #         logger.info(f"Using default greeting: {welcome_message}")
    #         # Say welcome immediately - this happens in parallel with setup below
    #         # print time to first response
    #         # await session.say(text=welcome_message, allow_interruptions=True)
    #         session.say(text=welcome_message, allow_interruptions=True)

    welcome_message = account_manager.get_default_greeting()
    logger.info(f"Using default greeting: {welcome_message}")
    # Say welcome immediately - this happens in parallel with setup below
    # print time to first response
    # await session.say(text=welcome_message, allow_interruptions=True)
    session.say(text=welcome_message, allow_interruptions=True)
        
    # PHASE 2: Full setup happens in parallel with welcome message
    phase2_start = time.monotonic()
    await _complete_agent_setup(
        ctx=ctx, agent=agent, participant=participant
    )
    logger.debug(f"Phase 2 (complete setup) completed in {time.monotonic() - phase2_start:.2f} seconds")
    
    # PHASE 3: Agent is fully operational
    # ==================================
    logger.info(f"Agent fully initialized and ready in {time.monotonic() - start_total:.2f} seconds")
    
    # # Stay connected until session ends
    # await ctx.wait_until_disconnected()

async def _complete_agent_setup(ctx, agent, participant):
    """Complete the full agent setup in the background while initial greeting plays"""
    try:
        # # 1. Set up the agent with core components 
        # agent, llm_model, initial_ctx = await setup_agent(ctx)
        
        # 2. Set up user state and get user_id
        # This already includes basic SIP handling
        user_id, log_file_name = await setup_user_state(ctx, agent)
        
        # Enhanced SIP handling from the original function
        if participant.kind == 3:  # "SIP"
            current_date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            phone_number = participant.attributes.get("sip.phoneNumber")
            # Extract just the digits from the phone number
            phone_digits = ''.join(filter(str.isdigit, phone_number))
            # Ensure logs directory exists
            os.makedirs(f"logs/{phone_digits}", exist_ok=True)
            
            log_file_name = f"logs/{phone_digits}/{current_date_time_string}.txt"
            user_id = phone_digits
            ctx.proc.userdata["user_id"] = user_id  # Make sure this is updated
            
            call_id = participant.attributes.get("sip.callID")
            # Write the call id to the log file
            async with aiofiles.open(log_file_name, "w") as file:
                await file.write(f"Call ID: {call_id}\nPhone Number: {phone_number}\nDate and Time: {current_date_time_string}\nType: Interview\n\n")
        
        # # 3. Set up analyzers (conversation, sentiment)
        # conversation_persistor, conversation_analyzer, sentiment_analyzer = setup_analyzers(
        #     ctx=ctx, agent=agent, user_id=user_id, log_file_name=log_file_name, conversation_analysis_model=analysis_model_name, sentiment_analysis_model=analysis_model_name
        # )
        
        # # 4. Handle conversation history from persistor
        # messages = conversation_persistor.get_conversation()
        
        # # 5. Handle resumption data specially
        # if resumption_data:
        #     # Clear resumption data since we're using it
        #     SessionStateManager.clear_conversation_resumption(user_id)
            
        #     # If we have resumption data and no recent conversation, inject the resumption context
        #     if not messages:
        #         # Add a system message about the resumed conversation
        #         recent_topics = ", ".join(resumption_data.get('topics', []))
        #         if recent_topics:
        #             resumption_context = f"This is a continuation of a previous conversation about {recent_topics}."
        #             messages.append(ChatMessage("system", resumption_context))
                    
        #     # Pass conversation filters to the analyzer if available
        #     if "topics" in resumption_data or "bike_type" in resumption_data or "price_range" in resumption_data:
        #         conversation_filters = {
        #             "product_categories": resumption_data.get('topics', []),
        #             "bike_type": resumption_data.get('bike_type', []),
        #             "price_range": resumption_data.get('price_range', '')
        #         }
        #         SessionStateManager.update_user_state(user_id, {
        #             "conversation_filters": conversation_filters
        #         })
                
        #     # Add resumption context to the system prompt
        #     personalized_prompt = initial_ctx.messages[0].content
        #     personalized_prompt += "\n\nThis is a continuation of a previous conversation. "
            
        #     if resumption_data.get('topics'):
        #         topics_str = ", ".join(resumption_data.get('topics', []))
        #         personalized_prompt += f"The user previously discussed: {topics_str}."
                
        #     # Update the system message
        #     initial_ctx.messages[0].content = personalized_prompt
        
        # # 6. Set up the complete context with history
        # full_context = llm.ChatContext().append(
        #     role="system",
        #     text=initial_ctx.messages[0].content
        # )
        
        # # Add conversation history
        # if messages:
        #     for message in messages:
        #         full_context.messages.append(message)
                
        # # # 7. Update the initial agent with all the enhanced capabilities
        # # async with _chat_ctx_lock:
        # #     # Replace the simple context with the full, personalized one
        # #     initial_agent.chat_ctx = full_context
        # #     initial_agent.fnc_ctx = agent.fnc_ctx
            
        # #     # Add RAG callback (should be preserved from setup_agent, but ensure it's set)
        # #     initial_agent._opts.before_llm_cb = _enrich_system_prompt_with_rag
            
        logger.debug(f"Agent setup completed for {user_id}")
        
    except Exception as e:
        logger.error(f"Error in complete agent setup: {e}")
        logger.exception(e)
        # Even with error, agent will continue with basic functionality


# ==============================
# TESTING
# ==============================

def test():
    import asyncio
    agent = Assistant()
    result = asyncio.run(agent.display_products(product_ids=["218669"]))
    print(result)
    print("Test completed successfully")

# ==============================
# MAIN ENTRY POINT
# ==============================

if __name__ == "__main__":
    from livekit.agents.utils.hw import get_cpu_monitor
    from livekit.agents.worker import _WorkerEnvOption
    import math
    from livekit.agents.cli import run_app
    from livekit.agents.worker import WorkerOptions

    # Fallback to normal operation
    run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            request_fnc=request_fnc,
            port=os.getenv("LIVEKIT_PORT", 8081),  # Use a different port for the agent
            # Performance optimizations
            # num_idle_processes=2,  # Keep 2 warm processes ready
            num_idle_processes=_WorkerEnvOption(
                dev_default=1, prod_default=math.ceil(get_cpu_monitor().cpu_count()),  # Keep 2 warm processes ready
            ),
            initialize_process_timeout=15.0,  # Give more time for heavy imports
            # multiprocessing_context="forkserver",  # Faster on Linux
        ),
    )
