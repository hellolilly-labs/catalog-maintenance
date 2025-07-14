import asyncio
import json
import logging
import os
import time

from dotenv import load_dotenv

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
    BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip,
)
from livekit.plugins import (
    assemblyai,
    deepgram, 
    google,
    silero, 
    elevenlabs, 
    noise_cancellation
)
from livekit import rtc
from google.cloud import texttospeech
from liddy_voice.llm_service import LlmService
from liddy_voice.assistant import Assistant
from liddy.model import UserState
from liddy_voice.user_manager import UserManager
from liddy_voice.account_manager import ElevenLabsTtsProviderSettings
from liddy import AccountManager, get_account_manager
from liddy_voice.session_state_manager import SessionStateManager
from liddy.models.product import Product
from liddy.models.product_manager import ProductManager
from liddy.account_config_cache import AccountConfigCache
from liddy.account_config_loader import get_account_config_loader

# Get logger for this module
logger = logging.getLogger(__name__)


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
    """Preload critical components for faster startup"""
    import asyncio
    
    async def _prewarm_components():
        prewarm_start = time.time()
        logger.info("🔥 Starting prewarm: Loading critical components...")
        
        tasks = []
        
        # 1. Prewarm VAD model (critical for turn detection)
        async def prewarm_vad():
            try:
                vad_start = time.time()
                vad_model = silero.VAD.load()
                logger.info(f"✅ VAD model loaded in {time.time() - vad_start:.3f}s")
                return ("vad", vad_model)
            except Exception as e:
                logger.error(f"Failed to load VAD: {e}")
                return ("vad", None)
        
        tasks.append(prewarm_vad())
        
        async def prewarm_account_config_loader():
            try:
                account_config_loader_start = time.time()
                account_config_loader = get_account_config_loader()
                logger.info(f"✅ Account config loader loaded in {time.time() - account_config_loader_start:.3f}s")
                return ("account_config_loader", account_config_loader)
            except Exception as e:
                logger.error(f"Failed to load account config loader: {e}")
                return ("account_config_loader", None)
        
        tasks.append(prewarm_account_config_loader())
        
        # # 2. Prewarm TTS voices (if using cloud providers)
        # async def prewarm_tts():
        #     try:
        #         tts_start = time.time()
        #         # Prewarm OpenAI TTS (fast default)
        #         openai_tts = openai.TTS(voice="nova")
                
        #         # Prewarm ElevenLabs if API key exists
        #         elevenlabs_api_key = os.getenv("ELEVEN_API_KEY")
        #         elevenlabs_tts = None
        #         if elevenlabs_api_key:
        #             elevenlabs_tts = elevenlabs.TTS(
        #                 voice_id="8sGzMkj2HZn6rYwGx6G0",  # Default voice
        #                 api_key=elevenlabs_api_key
        #             )
                
        #         logger.info(f"✅ TTS models loaded in {time.time() - tts_start:.3f}s")
        #         return ("tts", {"openai": openai_tts, "elevenlabs": elevenlabs_tts})
        #     except Exception as e:
        #         logger.error(f"Failed to load TTS: {e}")
        #         return ("tts", None)
        
        # tasks.append(prewarm_tts())
        
        # # 3. Prewarm STT models
        # async def prewarm_stt():
        #     try:
        #         stt_start = time.time()
        #         # Prewarm primary STT
        #         deepgram_stt = deepgram.STT()
                
        #         # Prewarm AssemblyAI as backup
        #         assemblyai_stt = assemblyai.STT()
                
        #         logger.info(f"✅ STT models loaded in {time.time() - stt_start:.3f}s")
        #         return ("stt", {"deepgram": deepgram_stt, "assemblyai": assemblyai_stt})
        #     except Exception as e:
        #         logger.error(f"Failed to load STT: {e}")
        #         return ("stt", None)
        
        # tasks.append(prewarm_stt())
        
        # Account prewarming removed - using Redis for all account configs
        # Accounts are loaded via `load-data` command before starting the agent
        
        # # 5. Prewarm LLM connections
        # async def prewarm_llm():
        #     try:
        #         llm_start = time.time()
        #         # Create LLM clients to establish connections
        #         openai_llm = openai.LLM(model="gpt-4.1-mini")
                
        #         # You could make a simple test call here if needed
        #         # await openai_llm.complete("Hello")
                
        #         logger.info(f"✅ LLM connections established in {time.time() - llm_start:.3f}s")
        #         return ("llm", openai_llm)
        #     except Exception as e:
        #         logger.error(f"Failed to establish LLM connections: {e}")
        #         return ("llm", None)
        
        # tasks.append(prewarm_llm())
        
        # Run all prewarm tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store prewarmed components in process userdata for reuse
        for result in results:
            if isinstance(result, tuple) and not isinstance(result, Exception):
                component_name, component = result
                if component:
                    proc.userdata[f"prewarmed_{component_name}"] = component
        
        total_time = time.time() - prewarm_start
        logger.info(f"🎯 Prewarm completed in {total_time:.3f}s")
        
        # Log summary
        prewarmed_components = [r[0] for r in results if isinstance(r, tuple) and r[1] is not None]
        logger.info(f"✅ Prewarmed components: {', '.join(prewarmed_components)}")
    
    # Run the async prewarm function with timeout protection
    try:
        # Use a shorter timeout to ensure we don't exceed process init timeout
        asyncio.run(asyncio.wait_for(_prewarm_components(), timeout=12.0))
    except asyncio.TimeoutError:
        logger.warning("Prewarm timed out after 12s - continuing with partial prewarm")
    except Exception as e:
        logger.error(f"Prewarm error: {e}")
        # Don't fail the process, just log the error
    
    logger.info("Prewarm phase completed")

# ==============================
# REQUEST FUNCTION
# ==============================

async def request_fnc(ctx: JobProcess):
    await ctx.accept()

# ==============================
# MAIN ENTRYPOINT
# ==============================

async def setup_agent(account: str, room_name: str):
    """Configure and return the voice agent"""
    # Try to get prewarmed components from the current process
    try:
        from livekit.agents import get_job_context
        job_ctx = get_job_context()
        proc_userdata = job_ctx.proc.userdata if hasattr(job_ctx, 'proc') else {}
    except:
        proc_userdata = {}
    
    # Configure STT
    credentials_file = None
    
    # Check if we have prewarmed STT models
    prewarmed_stt = proc_userdata.get("prewarmed_stt", {})
    if prewarmed_stt and isinstance(prewarmed_stt, dict):
        logger.debug("✅ Using prewarmed STT models")
        primary_stt = prewarmed_stt.get("assemblyai") or prewarmed_stt.get("deepgram")
        secondary_stt = prewarmed_stt.get("deepgram") if primary_stt != prewarmed_stt.get("deepgram") else None
        if secondary_stt:
            stt_model = stt.FallbackAdapter([primary_stt, secondary_stt])
        else:
            stt_model = primary_stt
    else:
        # Fallback to creating new STT models
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
                    buffer_size_seconds=0.05,
                    format_turns=False,
                    end_of_turn_confidence_threshold=0.7,
                    min_end_of_turn_silence_when_confident=160,
                    max_turn_silence=2400
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
        account_manager: AccountManager = await get_account_manager(account)
        # Check if get_tts_settings is async
        tts_settings = await account_manager.get_tts_settings()
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
                    # auto_mode=True,
                    # chunk_length_schedule=[80, 120, 200],  # Smaller initial chunks for faster first byte
                    chunk_length_schedule=[50, 100, 150],  # Smaller initial chunks for faster first byte
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
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        print(f"ELEVENLABS_API_KEY: {elevenlabs_api_key}")
        import traceback
        traceback.print_exc()
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
    primary_llm_model = LlmService.fetch_model_service_from_model(primary_model_name, account=account, session_id=room_name, model_use="voice_assistant", parallel_tool_calls=True)
    fallback_model_name = os.getenv("FALLBACK_MODEL_NAME", "gpt-4.1-mini")
    fallback_model = LlmService.fetch_model_service_from_model(fallback_model_name, account=account, session_id=room_name, model_use="voice_assistant", parallel_tool_calls=True)
    
    
    llm_model = llm.FallbackAdapter([
        primary_llm_model,
        fallback_model
    ])
        
    # Get prewarmed VAD or create new one
    prewarmed_vad = proc_userdata.get("prewarmed_vad")
    if prewarmed_vad:
        logger.debug("✅ Using prewarmed VAD model")
        vad_model = prewarmed_vad
    else:
        logger.debug("📦 Loading new VAD model")
        vad_model = silero.VAD.load()
    
    # Create the agent session
    session = AgentSession[UserState](
        # userdata=user_state,
        user_away_timeout=30.0,
        stt=stt_model,
        llm=llm_model,
        tts=tts_model,
        vad=vad_model,
        turn_detection="stt"
        # turn_detection=EnglishModel() # MultilingualModel()
        # turn_detection=MultilingualModel()
    )
    
    return session


# async def setup_user_state(ctx, agent: Assistant):
#     """Set up user state and personalization"""
#     # user_id = ctx.room.name
#     user_id = agent.get_user_id()
    
#     # Update user state to record this voice interaction
#     user_state = agent.session.userdata or UserManager.get_user_state(user_id)
#     if user_state is None:
#         user_state = UserState(user_id=user_id)
#     agent.session.userdata = user_state

#     user_state.interaction_start_time = time.time()    
#     user_state.last_interaction_time = time.time()
#     user_state.voice_session_id = ctx.room.name
#     user_state.conversation_exit_state = None
#     # # now clear the user_state transcript
#     # SessionStateManager.clear_conversation_resumption(user_id)

#     # Handle SIP calls specially
#     if participant.kind == 3:  # "SIP"
#         current_date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         phone_number = participant.attributes.get("sip.phoneNumber")
#         # Extract just the digits from the phone number
#         phone_digits = ''.join(filter(str.isdigit, phone_number))
#         user_id = phone_digits
#         call_id = participant.attributes.get("sip.callID")
        
#     await agent._update_user_state_prompt()
        
#     return user_id

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice agent with optimized greeting delivery
    
    Key optimization: All heavy initialization happens BEFORE room connection,
    so greeting can be delivered immediately when participant joins.
    """
    start_total = time.monotonic()
    timing_breakdown = {}
    global participant, agent
    
    # Extract account from room name pattern: "{account}_{uuid}"
    account = ctx.room.name.split("_")[0]
    logger.info(f"🚀 Starting voice agent for account: {account}")
    
    # ============================================================
    # PRE-CONNECTION SETUP - Do all heavy lifting before connecting
    # ============================================================
    
    # 1. Account manager setup
    account_mgr_start = time.monotonic()
    account_manager = await get_account_manager(account)
    timing_breakdown['account_manager'] = time.monotonic() - account_mgr_start
    
    # 3. Create agent session with all models
    session_start = time.monotonic()
    session = await setup_agent(account=account, room_name=ctx.room.name)
    timing_breakdown['session_setup'] = time.monotonic() - session_start
    
    # 4. Initialize Assistant
    primary_model_name = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
    assistant_start = time.monotonic()
    agent = Assistant(ctx=ctx, primary_model=primary_model_name, account=account_manager.account)
    timing_breakdown['assistant_init'] = time.monotonic() - assistant_start
    
    # 5. Configure room options
    options_start = time.monotonic()
    use_noise_cancellation = os.getenv("USE_NOISE_CANCELLATION", "false").lower() == "true"
    room_input_options = RoomInputOptions(
        close_on_disconnect=False,
        noise_cancellation=noise_cancellation.BVC() if use_noise_cancellation else None
    )
    timing_breakdown['room_options'] = time.monotonic() - options_start
    
    # 6. Start session with agent
    session_start_time = time.monotonic()
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=room_input_options
    )
    timing_breakdown['session_start'] = time.monotonic() - session_start_time
    
    # ============================================================
    # ROOM CONNECTION - Connect and wait for participant
    # ============================================================
    
    # 7. Connect to room
    start_connect = time.monotonic()
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    timing_breakdown['room_connect'] = time.monotonic() - start_connect
    logger.info(f"✅ Connected to room in {timing_breakdown['room_connect']:.3f}s")
    
    # 8. Wait for participant to join
    logger.info("⏳ Waiting for participant...")
    start_wait = time.monotonic()
    participant = await ctx.wait_for_participant()
    timing_breakdown['wait_participant'] = time.monotonic() - start_wait
    logger.info(f"✅ Participant joined in {timing_breakdown['wait_participant']:.3f}s")
    
    # ============================================================
    # POST-CONNECTION - Deliver greeting and complete setup
    # ============================================================
    
    # 9. Set up user state
    user_state_start = time.monotonic()
    metadata_str = participant.metadata
    user_id = participant.identity
    current_url = None
    current_url_title = None
    current_url_timestamp = None
    greeting = None
    
    if metadata_str:
        metadata = json.loads(metadata_str)
        
        try:
            user_id = metadata.get("userId") if metadata.get("userId") else metadata.get("user_id") if metadata.get("user_id") else participant.identity
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            user_id = participant.identity
        
        try:
            current_url = metadata.get("currentUrl") if metadata.get("currentUrl") else metadata.get("current_url") if metadata.get("current_url") else None
            current_url_title = metadata.get("currentTitle") if metadata.get("currentTitle") else metadata.get("current_title") if metadata.get("current_title") else None
            current_url_timestamp = metadata.get("currentUrlTimestamp") if metadata.get("currentUrlTimestamp") else metadata.get("current_url_timestamp") if metadata.get("current_url_timestamp") else None
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            current_url = None
            current_url_title = None
            current_url_timestamp = None
            
        try:
            greeting = metadata.get("greeting") if metadata.get("greeting") else metadata.get("greeting_message") if metadata.get("greeting_message") else None
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            greeting = None
    
    user_state = UserManager.get_user_state(user_id)
    if user_state is None:
        user_state = UserState(user_id=user_id)
    user_state.account = account_manager.account
    session.userdata = user_state
    agent.set_user_id(user_id=user_id)
    timing_breakdown['user_state_retrieval'] = time.monotonic() - user_state_start
    
    # 2. Pre-fetch greeting early to minimize latency
    if not greeting:
        greeting_fetch_start = time.monotonic()
        greeting = await account_manager.get_default_greeting()
        timing_breakdown['greeting_prefetch'] = time.monotonic() - greeting_fetch_start
        logger.info(f"📝 Pre-fetched greeting: {greeting[:50]}...")

    # 10. Deliver greeting immediately (non-blocking)
    greeting_start = time.monotonic()
    logger.info(f"🎙️ Delivering greeting: {greeting}")
    
    # async def deliver_greeting():
    tts_start = time.monotonic()
    timing_breakdown['time_to_first_utterance'] = time.monotonic() - start_total
    # await session.say(text=welcome_message, allow_interruptions=False)
    
    speech_instructions = f"Briefly greet the user with something like: {greeting}"
    
    # if the user is currently looking at a product, then add the product to the speech instructions
    try:
        # if not current_url:
        #     user_history = SessionStateManager.get_user_recent_history(user_id=user_id)
        #     if user_history and len(user_history) > 0:
        #         most_recent_history = user_history[0]
        #         current_url = most_recent_history.url
        #         current_url_title = most_recent_history.title if most_recent_history.title else current_url
        
        if current_url:
            # Use centralized smart product extraction
            product = None
            
            # Get product using smart extraction (no fallback to URL lookup for performance)
            product_manager: ProductManager = await agent.get_product_manager()
            product = await product_manager.find_product_from_url_smart(url=current_url, fallback_to_url_lookup=False)
            
            if product and isinstance(product, Product):
                speech_instructions += f"\n\nAs a reference, the user is currently looking at the following product. However, be sure to welcome the user and ask them how you can help them!\n\n{Product.to_markdown(product=product, depth=0, obfuscatePricing=True)}\n\n"
            else:
                speech_instructions += f"\n\nAs a reference, the user is currently looking at the following page. However, be sure to welcome the user and ask them how you can help them!\n\n[{current_url_title if current_url_title else current_url}]({current_url})\n\n"
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error getting user history: {e}")
    
    use_generated_greeting = os.getenv("USE_GENERATED_GREETING", "true").lower() == "true"
    
    if use_generated_greeting:
        logger.info(f"🎙️ Delivering generated greeting: {speech_instructions}")
        await session.generate_reply(
            instructions=speech_instructions
        )
    else:
        logger.info(f"🎙️ Delivering pre-fetched greeting: {greeting}")
        await session.say(text=greeting, allow_interruptions=False)
        
    tts_time = time.monotonic() - tts_start
    timing_breakdown['tts_generation'] = tts_time
    timing_breakdown['greeting_delivery'] = time.monotonic() - greeting_start
    logger.info(f"⚡ Time to first utterance: {timing_breakdown['time_to_first_utterance']:.3f}s (TTS: {tts_time:.3f}s)")
    
    # # greeting_task = asyncio.create_task(deliver_greeting())
    # await deliver_greeting()
    
    # 11. Start deferred prewarming in background
    deferred_prewarming_task = None
    if hasattr(agent, 'start_deferred_prewarming'):
        deferred_prewarming_task = asyncio.create_task(
            agent.start_deferred_prewarming(prewarm_llm=(not use_generated_greeting))
        )
    
    # 12. Set up metrics collection
    metrics_setup_start = time.monotonic()
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
                logger.info(f"📊 Metrics - EOU: {eou_metric.end_of_utterance_delay:.3f}s, "
                           f"LLM TTFT: {llm_metric.ttft:.3f}s, TTS TTFB: {tts_metric.ttfb:.3f}s, "
                           f"Total: {total_latency:.3f}s")
                eou_metric = None
                llm_metric = None
                tts_metric = None
        usage_collector.collect(ev.metrics)
        # asyncio.create_task(agent.on_metrics_collected(ev))
    session.on("metrics_collected", handle_metrics_collected)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"📊 Usage summary: {summary}")

    # At shutdown, generate and log the summary from the usage collector
    ctx.add_shutdown_callback(log_usage)
    
    timing_breakdown['metrics_setup'] = time.monotonic() - metrics_setup_start

    # 13. Add cleanup callback
    async def cleanup_assistant():
        try:
            await agent.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up Assistant: {e}")
    
    ctx.add_shutdown_callback(cleanup_assistant)
    
    # 14. Background audio setup (if enabled)
    bg_audio_start = time.monotonic()
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
    timing_breakdown['background_audio'] = time.monotonic() - bg_audio_start

    # 15. Set up event handlers
    handlers_start = time.monotonic()
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
    
    # room on data_received
    def handle_data_received(data: rtc.DataPacket):
        asyncio.create_task(agent.on_data_received(payload=data.data, participant=data.participant, kind=data.kind, topic=data.topic))
    ctx.room.on("data_received", handle_data_received)
    
    timing_breakdown['event_handlers'] = time.monotonic() - handlers_start
    
    # 16. Complete user setup
    user_id_start = time.monotonic()
    logger.debug(f"Starting voice assistant for participant {participant.identity}")
    user_id = participant.identity
    agent.set_user_id(user_id)
    await agent._update_user_state_prompt()
    timing_breakdown['user_id_setup'] = time.monotonic() - user_id_start

    # TODO: Add conversation resumption (see ROADMAP/CONVERSATION_RESUMPTION_STRATEGY.md)
    # # Check if there's a recent conversation to resume
    # resumption_data: ConversationResumptionState = SessionStateManager.get_conversation_resumption(user_state=user_state)
    # is_resumable = resumption_data is not None and resumption_data.is_resumable
    # resumption_message = None
    # has_previous_conversation = resumption_data is not None
    
    # time_since_last_conversation = None
    
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
    #         welcome_message = await account_manager.get_default_greeting()
    #         logger.info(f"Using default greeting: {welcome_message}")
    #         # Say welcome immediately - this happens in parallel with setup below
    #         # print time to first response
    #         # await session.say(text=welcome_message, allow_interruptions=True)
    #         session.say(text=welcome_message, allow_interruptions=True)
    
    # ============================================================
    # FINALIZATION - Wait for async tasks and log timing
    # ============================================================
    
    # # Wait for greeting delivery to complete for accurate timing
    # if 'greeting_task' in locals() and not greeting_task.done():
    #     await greeting_task
    
    # Wait for deferred prewarming to complete
    if deferred_prewarming_task and not deferred_prewarming_task.done():
        await deferred_prewarming_task
        
    # Calculate total initialization time
    timing_breakdown['total_time'] = time.monotonic() - start_total
    
    # Log comprehensive timing summary
    logger.info(f"\n🎯 ENTRYPOINT TIMING SUMMARY 🎯\n"
                f"{'='*50}\n"
                f"Time to first utterance:     {timing_breakdown.get('time_to_first_utterance', 0):.3f}s\n"
                f"Total initialization:   {timing_breakdown['total_time']:.3f}s\n"
                f"{'='*50}\n"
                f"PRE-CONNECTION SETUP:\n"
                f"  Account manager:      {timing_breakdown['account_manager']:.3f}s\n"
                f"  Greeting prefetch:    {timing_breakdown.get('greeting_prefetch', 0):.3f}s\n"
                f"  Session setup:        {timing_breakdown['session_setup']:.3f}s\n"
                f"  Assistant init:       {timing_breakdown['assistant_init']:.3f}s\n"
                f"  Room options:         {timing_breakdown['room_options']:.3f}s\n"
                f"  Session start:        {timing_breakdown['session_start']:.3f}s\n"
                f"\nROOM CONNECTION:\n"
                f"  Room connect:         {timing_breakdown['room_connect']:.3f}s\n"
                f"  Wait participant:     {timing_breakdown['wait_participant']:.3f}s\n"
                f"\nPOST-CONNECTION:\n"
                f"  User state setup:     {timing_breakdown['user_state_retrieval']:.3f}s\n"
                f"  Event handlers:       {timing_breakdown['event_handlers']:.3f}s\n"
                f"  User ID setup:        {timing_breakdown['user_id_setup']:.3f}s\n"
                f"  TTS generation:       {timing_breakdown.get('tts_generation', 0):.3f}s\n"
                f"{'='*50}")
    
    # Identify slowest components
    sorted_timings = sorted(timing_breakdown.items(), key=lambda x: x[1], reverse=True)
    slowest = [f"{k}: {v:.3f}s" for k, v in sorted_timings[:5] if k not in ['total_time', 'phase1_total', 'phase2_total', 'time_to_first_word']]
    logger.info(f"🐌 Slowest components: {', '.join(slowest)}")
    
    # Performance warnings
    if timing_breakdown['time_to_first_utterance'] > 2.0:
        logger.warning(f"⚠️ Time to first utterance exceeded 2s target: {timing_breakdown['time_to_first_utterance']:.3f}s")
    if timing_breakdown['total_time'] > 5.0:
        logger.warning(f"⚠️ Total initialization exceeded 5s target: {timing_breakdown['total_time']:.3f}s")
    logger.info(f"🎯 Total initialization: {timing_breakdown['total_time']:.3f}s")

# async def _complete_agent_setup(ctx, agent, participant):
#     """Complete the full agent setup in the background while initial greeting plays"""
#     try:
#         # # 1. Set up the agent with core components 
#         # agent, llm_model, initial_ctx = await setup_agent(ctx)
        
#         # 2. Set up user state and get user_id
#         # This already includes basic SIP handling
#         user_id = await setup_user_state(ctx, agent)
        
#         # Enhanced SIP handling from the original function
#         if participant.kind == 3:  # "SIP"
#             current_date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             phone_number = participant.attributes.get("sip.phoneNumber")
#             # Extract just the digits from the phone number
#             phone_digits = ''.join(filter(str.isdigit, phone_number))
#             # Ensure logs directory exists
#             user_id = phone_digits
#             ctx.proc.userdata["user_id"] = user_id  # Make sure this is updated
            
#             call_id = participant.attributes.get("sip.callID")
        
#         # # 3. Set up analyzers (conversation, sentiment)
#         # conversation_persistor, conversation_analyzer, sentiment_analyzer = setup_analyzers(
#         #     ctx=ctx, agent=agent, user_id=user_id, log_file_name=log_file_name, conversation_analysis_model=analysis_model_name, sentiment_analysis_model=analysis_model_name
#         # )
        
#         # # 4. Handle conversation history from persistor
#         # messages = conversation_persistor.get_conversation()
        
#         # # 5. Handle resumption data specially
#         # if resumption_data:
#         #     # Clear resumption data since we're using it
#         #     SessionStateManager.clear_conversation_resumption(user_id)
            
#         #     # If we have resumption data and no recent conversation, inject the resumption context
#         #     if not messages:
#         #         # Add a system message about the resumed conversation
#         #         recent_topics = ", ".join(resumption_data.get('topics', []))
#         #         if recent_topics:
#         #             resumption_context = f"This is a continuation of a previous conversation about {recent_topics}."
#         #             messages.append(ChatMessage("system", resumption_context))
                    
#         #     # Pass conversation filters to the analyzer if available
#         #     if "topics" in resumption_data or "bike_type" in resumption_data or "price_range" in resumption_data:
#         #         conversation_filters = {
#         #             "product_categories": resumption_data.get('topics', []),
#         #             "bike_type": resumption_data.get('bike_type', []),
#         #             "price_range": resumption_data.get('price_range', '')
#         #         }
#         #         SessionStateManager.update_user_state(user_id, {
#         #             "conversation_filters": conversation_filters
#         #         })
                
#         #     # Add resumption context to the system prompt
#         #     personalized_prompt = initial_ctx.messages[0].content
#         #     personalized_prompt += "\n\nThis is a continuation of a previous conversation. "
            
#         #     if resumption_data.get('topics'):
#         #         topics_str = ", ".join(resumption_data.get('topics', []))
#         #         personalized_prompt += f"The user previously discussed: {topics_str}."
                
#         #     # Update the system message
#         #     initial_ctx.messages[0].content = personalized_prompt
        
#         # # 6. Set up the complete context with history
#         # full_context = llm.ChatContext().append(
#         #     role="system",
#         #     text=initial_ctx.messages[0].content
#         # )
        
#         # # Add conversation history
#         # if messages:
#         #     for message in messages:
#         #         full_context.messages.append(message)
                
#         # # # 7. Update the initial agent with all the enhanced capabilities
#         # # async with _chat_ctx_lock:
#         # #     # Replace the simple context with the full, personalized one
#         # #     initial_agent.chat_ctx = full_context
#         # #     initial_agent.fnc_ctx = agent.fnc_ctx
            
#         # #     # Add RAG callback (should be preserved from setup_agent, but ensure it's set)
#         # #     initial_agent._opts.before_llm_cb = _enrich_system_prompt_with_rag
            
#         logger.debug(f"Agent setup completed for {user_id}")
        
#     except Exception as e:
#         logger.error(f"Error in complete agent setup: {e}")
#         logger.exception(e)
#         # Even with error, agent will continue with basic functionality


# ==============================
# TESTING
# ==============================

def test():
    import asyncio
    agent = Assistant()
    result = asyncio.run(agent.display_products(product_ids=["218669"]))
    print(result)
    print("Test completed successfully")

# def test_voice_search(account: str = "sundayriley.com", query: str = "Complete evening anti-aging starter kit containing six dermatologist-loved minis—ideal for travel, trial, or gifting. 1) Ceramic Slip Cleanser 30 mL / 1 fl oz: French green, bentonite, and white clays plus rice oil esters dissolve makeup while balancing pH. 2) Pink Drink Firming Peptide Essence 10 mL / 0.3 fl oz: EGFs + fermented honey + acetyl-hexapeptide-8 support microbiome resilience and visible firmness. 3) Good Genes Lactic Acid Treatment 8 mL / 0.2 fl oz: 7 % purified lactic acid with licorice and lemongrass resurfaces, brightens, and refines pores overnight. 4) A+ High-Dose Retinoid Serum 8 mL / 0.27 fl oz: 6.5 % stabilized retinoid blend (retinol ester, encapsulated retinol, blue-green algae) smooths wrinkles with minimal irritation. 5) Luna Sleeping Night Oil 5 mL / 0.17 fl oz: trans-retinoic acid ester in cold-pressed chia, evening primrose, and blue tansy visibly plumps while you sleep. 6) Ice Ceramide Moisturizing Cream 8 g / 0.3 fl oz: plant-derived ceramides, cholesterol, and vitamin F lock in hydration and repair barrier. All formulas are sulfate-, paraben-, gluten-, soy-, phthalate-, and fragrance-free, vegetarian, and pregnancy consult recommended for retinoids. Use sequence: cleanse → essence → lactic acid (2–3x/week) → retinoid or oil → cream. Delivers smoother texture, refined lines, and lasting moisture in one TSA-approved pouch."):
async def test_voice_search(account: str = "specialized.com", query: str = "top of the line mountain bike"):
    from liddy_voice.voice_search_wrapper import VoiceSearchService
    from liddy.search.service import SearchService
    from liddy.account_manager import get_account_manager as get_liddy_account_manager
    
    logger.info(f"🧪 Starting voice search test for {account}")
    
    # Phase 1: Prewarm search instance with full prewarming
    prewarm_start = time.time()
    logger.info("🔥 Prewarming search instance...")
    
    # Get account manager and prewarm search
    liddy_account_manager = await get_liddy_account_manager(account)
    search_config = liddy_account_manager.get_search_config()
    
    # Prewarm based on index type
    if search_config.unified_index:
        await SearchService._get_search_instance(liddy_account_manager, prewarm_level="full")
        logger.info(f"✅ Unified search instance prewarmed")
    else:
        from liddy.search.pinecone import get_search_pinecone
        await get_search_pinecone(
            brand_domain=account,
            dense_index_name=search_config.dense_index,
            sparse_index_name=search_config.sparse_index,
            prewarm_level="full"
        )
        logger.info(f"✅ Separate search instances prewarmed")
    
    prewarm_time = time.time() - prewarm_start
    logger.info(f"✅ Prewarming completed in {prewarm_time:.3f}s")
    
    # Phase 2: Setup test context
    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="assistant",
        content="Hey! This is Spence! How can I help you today?"
    )
    chat_ctx.add_message(
        role="user",
        content="show me a top of the line mountain bike"
    )
    
    product_knowledge = """
    
    """
    
    params={
        "include_metadata": True,
        # "enhancer_params": {
        #     "chat_ctx": chat_ctx,
        #     "product_knowledge": product_knowledge
        # }
    }
    
    # Phase 3: Perform search with timing
    search_start = time.time()
    logger.info(f"🔍 Performing search: '{query[:50]}...'")
    
    result = await VoiceSearchService.perform_search(
        query=query,
        search_function=VoiceSearchService.search_products_rag,
        search_params=params,
        user_state=UserState(account=account, user_id="test"),
        # chat_ctx=chat_ctx,
    )
    
    search_time = time.time() - search_start
    
    # Phase 4: Display results
    logger.info(f"✅ Search completed in {search_time:.3f}s")
    
    if isinstance(result, dict) and "results" in result:
        logger.info(f"Found {len(result['results'])} results")
        for i, product in enumerate(result['results'], 1):
            if hasattr(product, 'name'):
                logger.info(f"  {i}. {product.get('score')} ==> {product.name}")
            elif product.get('metadata'):
                if product.get('metadata').get('name'):
                    logger.info(f"  {i}. {product.get('score')} ==> {product.get('metadata').get('name')} {product.get('metadata').get('price')}")
                elif product.get('metadata').get('descriptor'):
                    logger.info(f"  {i}. {product.get('score')} ==> {product.get('metadata').get('descriptor')}")
                else:
                    logger.info(f"  {i}. {product.get('score')} ==> {product.get('metadata')}")
            else:
                logger.info(f"  {i}. {product.get('score')} ==> {product}")
    else:
        logger.info(f"Found {len(result)} results")
        for i, product in enumerate(result[:3], 1):
            if hasattr(product, 'name'):
                logger.info(f"  {i}. {product.name}")
    
    logger.info(f"\n📊 Performance Summary:")
    logger.info(f"  - Prewarm time: {prewarm_time:.3f}s")
    logger.info(f"  - Search time:  {search_time:.3f}s")
    logger.info(f"  - Total time:   {prewarm_time + search_time:.3f}s")
    
    print("\n✅ Test completed successfully")

# async def entrypoint_v2(ctx: JobContext):
#     """
#     Optimized entrypoint that prioritizes getting the agent talking ASAP.
    
#     Strategy:
#     1. Connect and start minimal session immediately
#     2. Greet user while loading features in background
#     3. Progressive enhancement as components load
#     """
#     timing = {}
#     start_time = time.monotonic()
    
#     logger.info("🚀 Starting optimized entrypoint_v2")
    
#     # PHASE 1: Critical Path - Get agent talking ASAP
#     critical_start = time.monotonic()
    
#     # 1. Connect to room immediately
#     await ctx.connect()
#     timing['connect'] = time.monotonic() - critical_start
    
#     # 2. Get participant info if available (non-blocking check)
#     participant = None
#     account = "default"
#     user_id = "default"
    
#     # Check if participant already connected (don't wait)
#     if ctx.room.remote_participants:
#         participant = list(ctx.room.remote_participants.values())[0]
#         user_id = participant.identity
        
#         # Quick metadata parse for account
#         if participant.metadata:
#             try:
#                 metadata = json.loads(participant.metadata)
#                 account = metadata.get("account", "default")
#             except:
#                 pass
    
#     # 3. Create minimal session with fallbacks
#     session_start = time.monotonic()
    
#     # Create basic user state
#     user_state = UserState(user_id=user_id, account=account)
    
#     # Get credentials for TTS/STT
#     elevenlabs_api_key = os.getenv("ELEVEN_API_KEY")
    
#     # Create minimal session
#     session = AgentSession(
#         userdata=user_state,
#         stt=deepgram.STT(model="nova-3-websocket"),  # Fastest STT
#         llm=openai.LLM(model="gpt-4.1-mini"),  # Fast LLM
#         tts=openai.TTS(voice="nova"),  # Fast TTS to start
#         vad=silero.VAD.load(),
#         turn_detection="stt",  # Fastest turn detection
#         user_away_timeout=30.0
#     )
#     timing['session_create'] = time.monotonic() - session_start
    
#     # 4. Create minimal assistant
#     assistant_start = time.monotonic()
    
#     # Start with minimal Assistant that will enhance itself
#     agent = Assistant(
#         ctx=ctx,
#         primary_model="gpt-4.1-mini",
#         user_id=user_id,
#         account=account
#     )
#     timing['assistant_create'] = time.monotonic() - assistant_start
    
#     # 5. Start session with noise cancellation
#     start_session = time.monotonic()
#     use_noise_cancellation = os.getenv("USE_NOISE_CANCELLATION", "false").lower() == "true"
    
#     await session.start(
#         room=ctx.room,
#         agent=agent,
#         room_input_options=RoomInputOptions(
#             close_on_disconnect=False,
#             noise_cancellation=noise_cancellation.BVC() if use_noise_cancellation else None
#         )
#     )
#     timing['session_start'] = time.monotonic() - start_session
    
#     timing['critical_path_total'] = time.monotonic() - critical_start
#     logger.info(f"✅ Critical path completed in {timing['critical_path_total']:.3f}s - Agent is ready to talk!")
    
#     # PHASE 2: Progressive Enhancement (all in background)
#     enhancement_start = time.monotonic()
    
#     # Create enhancement tasks that run in parallel
#     enhancement_tasks = []
    
#     # Task 1: Wait for participant if we don't have one yet
#     async def ensure_participant():
#         nonlocal participant, user_id, account
#         if not participant:
#             logger.info("⏳ Waiting for participant in background...")
#             participant = await ctx.wait_for_participant()
#             user_id = participant.identity
            
#             # Update metadata
#             if participant.metadata:
#                 try:
#                     metadata = json.loads(participant.metadata)
#                     new_account = metadata.get("account", account)
            
#             # Update user state
#             user_state.user_id = user_id
#             user_state.account = account
#             session.userdata = user_state
            
#             logger.info(f"✅ Participant {user_id} connected with account {account}")
    
#     enhancement_tasks.append(ensure_participant())
    
#     # Task 2: Upgrade to better TTS if configured
#     async def upgrade_tts():
#         try:
#             # Get account manager
#             account_manager = await get_account_manager(account)
#             tts_settings = account_manager.get_tts_settings()
            
#             if tts_settings and tts_settings.providers:
#                 logger.info("🔄 Upgrading TTS to account-specific voice...")
                
#                 # Create new TTS models
#                 tts_models = []
#                 for provider in tts_settings.providers:
#                     if isinstance(provider, ElevenLabsTtsProviderSettings) and elevenlabs_api_key:
#                         tts_model = elevenlabs.TTS(
#                             voice_id=provider.voice_id,
#                             model=provider.voice_model,
#                             api_key=elevenlabs_api_key,
#                             voice_settings=elevenlabs.VoiceSettings(
#                                 stability=provider.voice_stability,
#                                 similarity_boost=provider.voice_similarity_boost,
#                                 style=provider.voice_style,
#                                 use_speaker_boost=provider.voice_use_speaker_boost,
#                                 speed=provider.voice_speed if provider.voice_speed else 1.0
#                             )
#                         )
#                         tts_models.append(tts_model)
#                     elif provider.voice_provider == "google" and credentials_file:
#                         tts_model = google.TTS(
#                             credentials_file=credentials_file,
#                             voice_name=provider.voice_id,
#                             language="en-US"
#                         )
#                         tts_models.append(tts_model)
                
#                 # Update session TTS
#                 if tts_models:
#                     new_tts = tts.FallbackAdapter(tts_models) if len(tts_models) > 1 else tts_models[0]
#                     session._tts = new_tts
#                     logger.info("✅ TTS upgraded to account-specific voice")
#         except Exception as e:
#             logger.error(f"Error upgrading TTS: {e}")
    
#     enhancement_tasks.append(upgrade_tts())
    
#     # Task 3: Setup event handlers
#     async def setup_handlers():
#         # Metrics collection
#         usage_collector = metrics.UsageCollector()
        
#         def handle_metrics_collected(ev: MetricsCollectedEvent):
#             metrics.log_metrics(ev.metrics)
#             usage_collector.collect(ev.metrics)
        
#         session.on("metrics_collected", handle_metrics_collected)
        
#         # Agent event handlers
#         session.on("agent_state_changed", lambda ev: asyncio.create_task(agent.on_agent_state_changed(ev)))
#         session.on("user_state_changed", lambda ev: asyncio.create_task(agent.on_user_state_changed(ev)))
        
#         # Room event handlers
#         ctx.room.on("participant_disconnected", lambda p: asyncio.create_task(agent.on_participant_disconnected(p)))
#         ctx.room.on("participant_connected", lambda p: asyncio.create_task(agent.on_participant_connected(p)))
#         ctx.room.on("disconnected", lambda ev: asyncio.create_task(agent.on_room_disconnected(ev)))
        
#         # Shutdown callback
#         async def log_usage():
#             summary = usage_collector.get_summary()
#             logger.info(f"📊 Usage summary: {summary}")
        
#         ctx.add_shutdown_callback(log_usage)
        
#         # Add assistant cleanup
#         async def cleanup_assistant():
#             try:
#                 await agent.cleanup()
#             except Exception as e:
#                 logger.error(f"Error cleaning up Assistant: {e}")
        
#         ctx.add_shutdown_callback(cleanup_assistant)
#         logger.info("✅ Event handlers configured")
    
#     enhancement_tasks.append(setup_handlers())
    
#     # Task 4: Setup user state properly
#     async def configure_user_state():
#         if participant:
#             stored_state = UserManager.get_user_state(participant.identity)
#             if stored_state:
#                 # Merge stored state
#                 user_state.voice_count = stored_state.voice_count
#                 user_state.chat_count = stored_state.chat_count
#                 user_state.last_chat_time = stored_state.last_chat_time
#                 user_state.last_voice_time = stored_state.last_voice_time
                
#             # Update for this session
#             user_state.interaction_start_time = time.time()
#             user_state.last_interaction_time = time.time()
#             user_state.voice_session_id = ctx.room.name
            
#             # Handle SIP calls
#             if participant.kind == 3:  # SIP
#                 phone_number = participant.attributes.get("sip.phoneNumber", "")
#                 phone_digits = ''.join(filter(str.isdigit, phone_number))
#                 if len(phone_digits) >= 10:
#                     user_state.phone_number = phone_digits[-10:]
            
#             # Save updated state
#             UserManager.save_user_state(user_state)
#             await UserManager.update_user_room_reconnect_time(user_state.user_id, ctx.room.name)
            
#             # Update agent's user tracking
#             agent._user_id = user_id
#             agent.session.userdata = user_state
#             logger.info("✅ User state configured")
    
#     enhancement_tasks.append(configure_user_state())
    
#     # Task 5: Background audio (if enabled)
#     async def setup_background_audio():
#         use_background_audio = os.getenv("USE_BACKGROUND_AUDIO", "false").lower() == "true"
#         if use_background_audio:
#             background_audio = BackgroundAudioPlayer(
#                 ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.05),
#                 thinking_sound=[
#                     AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.125),
#                     AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.1),
#                 ]
#             )
#             await background_audio.start(room=ctx.room, agent_session=session)
#             logger.info("✅ Background audio started")
    
#     enhancement_tasks.append(setup_background_audio())
    
#     # Run all enhancement tasks in parallel
#     await asyncio.gather(*enhancement_tasks, return_exceptions=True)
    
#     timing['enhancement_total'] = time.monotonic() - enhancement_start
#     timing['total'] = time.monotonic() - start_time
    
#     # Log final timing summary
#     logger.info("📊 Entrypoint V2 Timing Summary:")
#     logger.info(f"  - Critical Path: {timing['critical_path_total']:.3f}s")
#     logger.info(f"    - Connect: {timing['connect']:.3f}s")
#     logger.info(f"    - Session: {timing['session_create']:.3f}s")
#     logger.info(f"    - Assistant: {timing['assistant_create']:.3f}s")
#     logger.info(f"    - Start: {timing['session_start']:.3f}s")
#     logger.info(f"  - Enhancement: {timing['enhancement_total']:.3f}s (background)")
#     logger.info(f"  - Total Time: {timing['total']:.3f}s")
    
#     # PHASE 3: Generate initial greeting (if appropriate)
#     if participant and not agent._planned_reconnect:
#         # Check conversation history for resumption
#         resumption_state = SessionStateManager.get_conversation_resumption(user_id)
        
#         if resumption_state and resumption_state.suggested_message:
#             # Resume conversation
#             agent.chat_ctx.add_message(
#                 role="user",
#                 content=[resumption_state.suggested_message]
#             )
#             await session.generate_reply(
#                 instructions="The user has just returned. Respond to their message naturally."
#             )
#         else:
#             # Fresh greeting - get account manager for greeting
#             try:
#                 account_mgr = await get_account_manager(account)
#                 default_greeting = await account_mgr.get_default_greeting()
#                 await session.say(default_greeting, add_to_chat_ctx=True)
#             except:
#                 # Fallback greeting
#                 await session.say("Hello! How can I help you today?", add_to_chat_ctx=True)



# ==============================
# MAIN ENTRY POINT
# ==============================

async def precache_account_configs(force_refresh=False):
    """Pre-download all account.json files and cache them in Redis
    
    Args:
        force_refresh: If True, refresh cache even if entries already exist
    """
    logger.info("📥 Pre-caching account configurations to Redis...")
    cache_start = time.time()
    
    try:
        account_config_loader = get_account_config_loader()
        # from liddy.storage import get_account_storage_provider
        # storage = get_account_storage_provider()
        
        
        redis_cache = AccountConfigCache()
        # redis_cache.clear_all()
        
        accounts = []
        
        try:
            accounts = await account_config_loader.get_accounts()
            logger.info(f"Found {len(accounts)} accounts in Redis cache")
            # accounts = await redis_cache.get_accounts_async()
            # logger.info(f"Found {len(accounts)} accounts in Redis cache")
        except Exception as e:
            logger.warning(f"Could not get accounts from Redis: {e}")
        
        if len(accounts) == 0:
            logger.warning("No accounts found in Redis cache, using fallback list")
            accounts = [
                "specialized.com",
                "sundayriley.com",
                "flexfits.com",
                "gucci.com",
                "balenciaga.com"
            ]

        # Download and cache account.json files in parallel
        cache_tasks = []
        skipped = 0
        
        for account in accounts:
            async def cache_account(acc):
                try:
                    # Check if already in Redis (skip if not forcing refresh)
                    if not force_refresh:
                        existing = await redis_cache.get_config_async(acc)
                        if existing:
                            logger.debug(f"✅ {acc} already in Redis cache")
                            return acc, True, True  # account, success, skipped
                    
                    # Download from GCS
                    config_data = await account_config_loader.get_account_config(account=acc)
                    return acc, True, False
                    
                    # if config_data:
                    #     # Store in Redis with 24-hour TTL
                    #     await redis_cache.set_config_async(acc, config_data, ttl=86400)
                    #     logger.debug(f"📝 Cached {acc} to Redis")
                    #     return acc, True, False  # account, success, not skipped
                except Exception as e:
                    logger.warning(f"Failed to cache {acc}: {e}")
                    return acc, False, False
            
            cache_tasks.append(cache_account(account))
        
        results = await asyncio.gather(*cache_tasks, return_exceptions=True)
        
        # Count successes
        successes = sum(1 for r in results if isinstance(r, tuple) and r[1])
        skipped = sum(1 for r in results if isinstance(r, tuple) and len(r) > 2 and r[2])
        failures = len(results) - successes
        
        logger.info(f"✅ Pre-cached {successes} account configs to Redis in {time.time() - cache_start:.3f}s "
                   f"({skipped} already cached, {failures} failures)")
            
    except Exception as e:
        logger.error(f"Failed to pre-cache account configs: {e}")
        # Don't fail the process, continue without cache


if __name__ == "__main__":
    from livekit.agents.utils.hw import get_cpu_monitor
    from livekit.agents.worker import _WorkerEnvOption
    import math
    from livekit.agents.cli import run_app
    from livekit.agents.worker import WorkerOptions
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "help":
            print("""
Liddy Voice Agent - Command Line Options

Usage: python voice_agent.py [command] [options]

Commands:
  help                    Show this help message
  test [account]          Test voice search (default: specialized.com)
  cache-accounts          Pre-cache all account configurations in Redis (deprecated - use load-data)
  load-data [accounts]    Load products, account configs, and STT word boost terms into Redis (comma-separated or from env)
  load-products [accounts] Alias for load-data (kept for backwards compatibility)
  download-files          Download required files (skips account pre-caching)
  
Without command:         Start the voice agent server

Environment Variables:
  USE_REDIS_PRODUCTS      Enable Redis product storage (true/false, default: true)
  REDIS_URL               Redis connection URL
  VOICE_ACCOUNTS          Comma-separated list of accounts
  LIVEKIT_URL             LiveKit server URL
  LIVEKIT_API_KEY         LiveKit API key
  LIVEKIT_API_SECRET      LiveKit API secret

Examples:
  python voice_agent.py load-data specialized.com
  python voice_agent.py load-data "specialized.com,trek.com,giant-bicycles.com"
  python voice_agent.py load-data  # Uses VOICE_ACCOUNTS env var or loads all accounts
  python voice_agent.py test specialized.com
""")
            exit()
        elif sys.argv[1] == "test":
            asyncio.run(test_voice_search(account=sys.argv[2] if len(sys.argv) > 2 else "specialized.com"))
            exit()
        elif sys.argv[1] == "cache-accounts":
            # Deprecated - redirect to load-data
            logger.warning("⚠️  cache-accounts is deprecated. Using load-data instead...")
            sys.argv[1] = "load-data"  # Redirect to load-data command
            # Fall through to load-data handler
        
        if sys.argv[1] == "load-data" or sys.argv[1] == "load-products":  # Support legacy command
            # Load both products and accounts into Redis
            async def load_data():
                """Load products and account configs for specified accounts into Redis"""
                # Parse accounts - either from args or environment
                accounts = None
                if len(sys.argv) > 2:
                    # Support comma-separated accounts
                    accounts = [a.strip() for a in sys.argv[2].split(',') if a.strip()]
                
                all_success = True
                
                # Load products
                logger.info("📦 Loading products into Redis...")
                from liddy.models.redis_product_loader import RedisProductLoader
                
                product_loader = RedisProductLoader()
                try:
                    await product_loader.connect()
                    
                    # Use load_all_accounts which handles multiple accounts
                    product_results = await product_loader.load_all_accounts(accounts)
                    
                    # Verify loading for each account
                    for account in product_results:
                        if product_results[account] > 0:
                            verified = await product_loader.verify_loading(account)
                            if not verified:
                                logger.error(f"❌ Product verification failed for {account}")
                                all_success = False
                        else:
                            all_success = False
                    
                except Exception as e:
                    logger.error(f"❌ Error loading products: {e}")
                    all_success = False
                finally:
                    await product_loader.disconnect()
                
                # Load account configs
                logger.info("🔧 Loading account configurations into Redis...")
                try:
                    account_loader = get_account_config_loader()
                    cache = AccountConfigCache()
                    
                    # Get accounts to load (use same list as products)
                    if not accounts:
                        accounts = await account_loader.get_accounts()
                    
                    config_successes = 0
                    for account in accounts:
                        try:
                            # Load from GCS and cache to Redis
                            config = await account_loader.get_account_config(account)
                            if config:
                                await cache.set_config_async(account, config)
                                config_successes += 1
                                logger.debug(f"✅ Cached config for {account}")
                            else:
                                logger.warning(f"⚠️ No config found for {account}")
                        except Exception as e:
                            logger.error(f"❌ Failed to cache config for {account}: {e}")
                            all_success = False
                    
                    logger.info(f"✅ Cached {config_successes}/{len(accounts)} account configs")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading account configs: {e}")
                    all_success = False
                
                # Load STT word boost terms
                logger.info("🎤 Loading STT word boost terms into Redis...")
                redis_client = None
                try:
                    from liddy.storage import get_account_storage_provider
                    import redis.asyncio as redis
                    
                    storage_manager = get_account_storage_provider()
                    
                    # Get Redis client
                    redis_host = os.getenv('REDIS_HOST', 'localhost')
                    redis_port = os.getenv('REDIS_PORT', '6379')
                    redis_password = os.getenv('REDIS_PASSWORD', '')
                    
                    if redis_password:
                        redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"
                    else:
                        redis_url = f"redis://{redis_host}:{redis_port}"
                    
                    redis_client = await redis.from_url(
                        redis_url,
                        encoding="utf-8",
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5
                    )
                    
                    word_boost_successes = 0
                    for account in accounts:
                        try:
                            # Try optimized file first
                            word_boost_data = None
                            pronunciation_guide = {}
                            
                            try:
                                content = await storage_manager.read_file(
                                    account=account,
                                    file_path="stt_word_boost.json"
                                )
                                data = json.loads(content)
                                word_boost_data = data.get('word_boost', [])
                                
                                # Also try to load pronunciation guide
                                try:
                                    vocab_content = await storage_manager.read_file(
                                        account=account,
                                        file_path="stt_vocabulary.json"
                                    )
                                    vocab_data = json.loads(vocab_content)
                                    pronunciation_guide = vocab_data.get('pronunciation_guide', {})
                                except:
                                    pass
                                    
                            except:
                                # Fallback to product catalog research
                                try:
                                    research_content = await storage_manager.read_file(
                                        account=account,
                                        file_path="research/product_catalog/research.md"
                                    )
                                    # Extract word boost section using same logic as Assistant
                                    part_e_start = research_content.find("## Part E: STT Word Boost Vocabulary")
                                    if part_e_start != -1:
                                        part_e_end = research_content.find("\n## ", part_e_start + 1)
                                        if part_e_end == -1:
                                            part_e_end = len(research_content)
                                        word_boost_section = research_content[part_e_start:part_e_end]
                                        
                                        # Try to parse JSON
                                        import re
                                        json_match = re.search(r'```json\s*({[^}]+})\s*```', word_boost_section, re.DOTALL)
                                        if json_match:
                                            data = json.loads(json_match.group(1))
                                            word_boost_data = data.get('word_boost', [])
                                except:
                                    pass
                            
                            if word_boost_data:
                                # Store in Redis
                                word_boost_key = f"stt_word_boost:{account}"
                                await redis_client.set(word_boost_key, json.dumps({
                                    'word_boost': word_boost_data,
                                    'pronunciation_guide': pronunciation_guide,
                                    'loaded_at': time.time()
                                }))
                                word_boost_successes += 1
                                logger.debug(f"✅ Cached {len(word_boost_data)} word boost terms for {account}")
                            else:
                                logger.debug(f"⚠️ No word boost terms found for {account}")
                                
                        except Exception as e:
                            logger.error(f"❌ Failed to cache word boost for {account}: {e}")
                            all_success = False
                    
                    logger.info(f"✅ Cached word boost terms for {word_boost_successes}/{len(accounts)} accounts")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading word boost terms: {e}")
                    all_success = False
                finally:
                    # Ensure Redis connection is properly closed
                    if redis_client:
                        try:
                            await redis_client.aclose()
                        except Exception as e:
                            logger.warning(f"Error closing Redis client: {e}")
                
                return all_success
            
            # Run load_data with proper event loop handling
            success = asyncio.run(load_data())
            exit(0 if success else 1)
    
    # Account pre-caching is now handled by the load-data command
    # Run `python voice_agent.py load-data` before starting the agent to ensure Redis has all data
    if len(sys.argv) > 1 and sys.argv[1] in ["download-files", "load-data", "load-products"]:
        logger.info("🚀 Skipping automatic account precache - handled by command")
    else:
        logger.info("💡 Tip: Run `python voice_agent.py load-data` to pre-load all products and accounts into Redis")
    
    # Run the application
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
