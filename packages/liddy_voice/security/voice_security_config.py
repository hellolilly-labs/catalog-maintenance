"""
Voice Security Configuration

Provides configuration utilities for voice-only security mode and
integration with the LiveKit voice assistant.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VoiceSecurityConfig:
    """Configuration management for voice security features"""
    
    def __init__(self):
        self.voice_only_mode = self._get_voice_only_mode()
        self.enhanced_monitoring = self._get_enhanced_monitoring()
        self.security_level = self._get_security_level()
        
        # Log current configuration
        self._log_configuration()
    
    def _get_voice_only_mode(self) -> bool:
        """Check if voice-only mode is enabled"""
        return os.getenv("VOICE_ONLY_MODE", "false").lower() == "true"
    
    def _get_enhanced_monitoring(self) -> bool:
        """Check if enhanced voice monitoring is enabled"""
        return os.getenv("VOICE_ENHANCED_MONITORING", "true").lower() == "true"
    
    def _get_security_level(self) -> str:
        """Get security level configuration"""
        return os.getenv("VOICE_SECURITY_LEVEL", "standard").lower()
    
    def _log_configuration(self):
        """Log current security configuration"""
        if self.voice_only_mode:
            logger.info("🔒 Voice-Only Security Mode: ENABLED")
            logger.info("   - Text input disabled for enhanced security")
            logger.info("   - Voice-specific monitoring enabled")
            logger.info("   - Enhanced protection against text-based attacks")
        else:
            logger.info("💬 Mixed Input Mode: Voice and text input enabled")
        
        if self.enhanced_monitoring:
            logger.info("🔍 Enhanced Voice Monitoring: ENABLED")
        
        logger.info(f"🛡️ Security Level: {self.security_level.upper()}")
    
    def get_room_input_options_config(self) -> Dict[str, Any]:
        """Get configuration for LiveKit RoomInputOptions"""
        return {
            'text_enabled': not self.voice_only_mode,
            'close_on_disconnect': False
        }
    
    def get_security_manager_config(self) -> Dict[str, Any]:
        """Get configuration for RuntimeSecurityManager"""
        return {
            'voice_only_mode': self.voice_only_mode,
            'enhanced_monitoring': self.enhanced_monitoring,
            'security_level': self.security_level
        }
    
    def get_security_thresholds(self) -> Dict[str, Any]:
        """Get security thresholds based on security level"""
        thresholds = {
            'minimal': {
                'echo_threshold': 0.85,
                'injection_high_threshold': 0.8,
                'voice_anomaly_threshold': 0.7
            },
            'standard': {
                'echo_threshold': 0.75,
                'injection_high_threshold': 0.65,
                'voice_anomaly_threshold': 0.5
            },
            'strict': {
                'echo_threshold': 0.65,
                'injection_high_threshold': 0.45,
                'voice_anomaly_threshold': 0.3
            }
        }
        
        return thresholds.get(self.security_level, thresholds['standard'])
    
    def is_voice_only_mode(self) -> bool:
        """Check if voice-only mode is enabled"""
        return self.voice_only_mode
    
    def should_enable_enhanced_monitoring(self) -> bool:
        """Check if enhanced monitoring should be enabled"""
        return self.enhanced_monitoring
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get summary of current environment configuration"""
        return {
            'voice_only_mode': self.voice_only_mode,
            'enhanced_monitoring': self.enhanced_monitoring,
            'security_level': self.security_level,
            'environment_variables': {
                'VOICE_ONLY_MODE': os.getenv("VOICE_ONLY_MODE", "false"),
                'VOICE_ENHANCED_MONITORING': os.getenv("VOICE_ENHANCED_MONITORING", "true"),
                'VOICE_SECURITY_LEVEL': os.getenv("VOICE_SECURITY_LEVEL", "standard")
            }
        }


# Global configuration instance
_voice_security_config: Optional[VoiceSecurityConfig] = None

def get_voice_security_config() -> VoiceSecurityConfig:
    """Get the global voice security configuration instance"""
    global _voice_security_config
    if _voice_security_config is None:
        _voice_security_config = VoiceSecurityConfig()
    return _voice_security_config


def create_security_manager() -> 'RuntimeSecurityManager':
    """Create a configured RuntimeSecurityManager instance"""
    from .runtime_security_manager import RuntimeSecurityManager
    
    config = get_voice_security_config()
    return RuntimeSecurityManager(
        voice_only_mode=config.is_voice_only_mode()
    )


def apply_security_thresholds(security_manager: 'RuntimeSecurityManager') -> None:
    """Apply security level thresholds to security manager"""
    config = get_voice_security_config()
    thresholds = config.get_security_thresholds()
    
    # Apply echo monitor thresholds
    security_manager.echo_monitor.adjust_thresholds(
        echo_threshold=thresholds['echo_threshold']
    )
    
    # Apply sanitizer thresholds
    security_manager.prompt_sanitizer.severity_thresholds['high'] = thresholds['injection_high_threshold']
    
    logger.info(f"✅ Applied {config.security_level} security thresholds")