# Voice Security Components

Enhanced security features for LiveKit voice assistants, including voice-only mode for maximum security.

## 🔒 Voice-Only Security Mode

Voice-only mode disables text input to eliminate text-based prompt injection attacks while providing enhanced monitoring of voice interactions.

### Configuration

Set the following environment variables:

```bash
# Enable voice-only mode (disables text input)
VOICE_ONLY_MODE=true

# Enable enhanced voice monitoring (default: true)
VOICE_ENHANCED_MONITORING=true

# Set security level: minimal, standard, strict (default: standard)
VOICE_SECURITY_LEVEL=standard
```

### Quick Integration

```python
from liddy_voice.security import get_voice_security_config, create_security_manager

# Get configuration
config = get_voice_security_config()

# Create security manager
security_manager = create_security_manager()

# Use in your voice agent
if config.is_voice_only_mode():
    print("🔒 Voice-only mode enabled - enhanced security active")
```

## 🛡️ Security Components

### PromptSanitizer
- Real-time injection detection with pattern matching
- Risk scoring system (0.0-1.0) with severity thresholds
- Content sanitization for medium-risk inputs
- Enhanced patterns with look-behind regex for jailbreaks

### EchoMonitor
- Multi-method similarity calculation for echo detection
- Dynamic intervention with temporary LLM parameter adjustment
- Conversation tracking with history management
- <0.04ms average response time

### RuntimeSecurityManager
- Unified coordination of security components
- Voice-specific anomaly detection
- Conversation lifecycle management
- Security dashboard with health monitoring

## 🎯 Voice-Specific Security Features

When `VOICE_ONLY_MODE=true` is enabled:

1. **Text Input Disabled**: `RoomInputOptions(text_enabled=False)`
2. **Voice Anomaly Detection**:
   - Unusually long transcriptions
   - Rapid-fire input detection
   - Speech-to-text manipulation patterns
   - Silent period monitoring

3. **Enhanced Monitoring**:
   - Voice session tracking
   - Input rate analysis
   - Pattern recognition for STT artifacts
   - Conversation flow validation

## 📊 Security Levels

### Minimal
- Lower thresholds for testing environments
- Reduced false positives
- Basic monitoring

### Standard (Default)
- Balanced security and usability
- Production-ready thresholds
- Comprehensive monitoring

### Strict
- Maximum security settings
- Higher sensitivity to anomalies
- Enhanced monitoring and logging

## 🔧 Integration Examples

### Basic Voice Agent Setup

```python
import os
from livekit.agents import RoomInputOptions
from liddy_voice.security import get_voice_security_config

# Load configuration
config = get_voice_security_config()

# Configure room options
room_input_options = RoomInputOptions(
    close_on_disconnect=False,
    text_enabled=not config.is_voice_only_mode(),  # Disable text in voice-only mode
    noise_cancellation=your_noise_cancellation_setup
)

# Log security status
if config.is_voice_only_mode():
    print("🔒 Voice-Only Security Mode: Text input disabled")
else:
    print("💬 Mixed Mode: Voice and text input enabled")
```

### Security Manager Integration

```python
from liddy_voice.security import create_security_manager, apply_security_thresholds

# Create and configure security manager
security_manager = create_security_manager()
apply_security_thresholds(security_manager)

# Process user input
async def process_user_input(user_input, conversation_context, conversation_id):
    # Security check
    security_result = await security_manager.process_conversation_turn(
        user_input, conversation_context, conversation_id
    )
    
    if not security_result['proceed']:
        return f"Security check failed: {security_result['reason']}"
    
    # Use sanitized input for processing
    sanitized_input = security_result['sanitized_input']
    
    # ... process with your LLM ...
    ai_response = await your_llm_processing(sanitized_input)
    
    # Check for echo behavior
    echo_result = await security_manager.process_response(
        user_input, ai_response, conversation_id
    )
    
    if not echo_result['allow_response']:
        return echo_result['intervention_message']
    
    return ai_response
```

### Security Dashboard

```python
# Get security metrics
dashboard = security_manager.get_security_dashboard()

print(f"Security Status: {dashboard['health_status']['status']}")
print(f"Echo Rate: {dashboard['echo_monitoring']['echo_rate']:.1%}")
print(f"Voice-Only Mode: {dashboard['voice_security']['voice_only_mode']}")
print(f"Voice Anomalies: {dashboard['voice_security']['voice_anomalies']}")
```

## 🚨 Security Alerts

The system monitors for various security threats:

### Text-Based Attacks (All Modes)
- Prompt injection attempts
- Role override commands
- System command injection
- Suspicious content patterns

### Voice-Specific Threats (Voice-Only Mode)
- STT manipulation attempts
- Rapid-fire automated inputs
- Unusual transcription patterns
- Silent session monitoring

## 📈 Performance Metrics

Target performance metrics:
- Echo rate: <2% (critical: >5%)
- Security false positives: <0.1% (critical: >0.5%)
- Input processing: <1ms average
- Echo detection: <0.2ms average

## 🔍 Monitoring and Logging

Security events are logged with appropriate severity levels:

```python
# Example log output
INFO:voice_security:🔒 Voice-Only Security Mode: Text input disabled for enhanced security
WARNING:voice_security:🔊 Voice anomaly: Rapid input rate in conversation conv_123 (12.3/sec)
ERROR:voice_security:🚨 Echo intervention triggered after 3 consecutive echoes in conversation conv_123
```

## 🛠️ Environment Configuration

Complete environment variable reference:

```bash
# Core voice security
VOICE_ONLY_MODE=true                    # Enable voice-only mode
VOICE_ENHANCED_MONITORING=true          # Enable enhanced monitoring
VOICE_SECURITY_LEVEL=standard          # Security level: minimal/standard/strict

# LiveKit configuration (existing)
USE_NOISE_CANCELLATION=true            # Enable noise cancellation
MODEL_NAME=openai/gpt-4o-mini          # LLM model to use

# Security tuning (optional)
VOICE_ECHO_THRESHOLD=0.75              # Echo detection threshold
VOICE_INJECTION_THRESHOLD=0.65         # Injection detection threshold
```

This configuration provides comprehensive voice security while maintaining excellent performance and usability.