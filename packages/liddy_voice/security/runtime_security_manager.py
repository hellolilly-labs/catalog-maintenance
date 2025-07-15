"""
RuntimeSecurityManager - Coordinates runtime security components

Provides a unified interface for managing prompt sanitization and echo monitoring
during voice assistant conversations.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from .prompt_sanitizer import PromptSanitizer
from .echo_monitor import EchoMonitor

logger = logging.getLogger(__name__)


class RuntimeSecurityManager:
    """Coordinates runtime security components for the voice assistant"""
    
    def __init__(self, voice_only_mode: bool = False):
        self.prompt_sanitizer = PromptSanitizer()
        self.echo_monitor = EchoMonitor()
        self.voice_only_mode = voice_only_mode
        
        # Aggregate security metrics
        self.security_metrics = {
            'blocked_inputs': 0,
            'sanitized_inputs': 0,
            'echo_detections': 0,
            'interventions': 0,
            'total_conversations': 0,
            'voice_anomalies': 0,
            'start_time': datetime.now()
        }
        
        # Track active conversations
        self.active_conversations = set()
        
        # Voice-specific security tracking
        self.voice_sessions = {}  # conversation_id -> voice session data
        
        mode_str = "Voice-Only" if voice_only_mode else "Mixed Mode"
        logger.info(f"🛡️ RuntimeSecurityManager initialized in {mode_str}")
        
        if voice_only_mode:
            logger.info("🔒 Enhanced voice security monitoring enabled")
    
    async def process_conversation_turn(
        self, 
        user_input: str, 
        conversation_context: List[Dict], 
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Process a complete conversation turn with security checks
        
        Args:
            user_input: The user's input message
            conversation_context: Previous conversation messages
            conversation_id: Unique conversation identifier
            
        Returns:
            Dictionary with security analysis results and actions
        """
        if not user_input:
            return {
                'proceed': True,
                'reason': 'Empty input allowed',
                'sanitized_input': user_input,
                'security_action': 'allowed'
            }
        
        # Track conversation
        if conversation_id not in self.active_conversations:
            self.active_conversations.add(conversation_id)
            self.security_metrics['total_conversations'] += 1
            
            # Initialize voice session tracking
            if self.voice_only_mode:
                self.voice_sessions[conversation_id] = {
                    'start_time': datetime.now(),
                    'total_inputs': 0,
                    'avg_input_length': 0,
                    'suspicious_patterns': 0,
                    'last_input_time': datetime.now()
                }
            
            logger.debug(f"🆕 New conversation tracked: {conversation_id}")
        
        # Voice-specific monitoring (if in voice-only mode)
        if self.voice_only_mode and conversation_id in self.voice_sessions:
            voice_anomaly = await self._analyze_voice_input(user_input, conversation_id)
            if voice_anomaly:
                self.security_metrics['voice_anomalies'] += 1
        
        # Step 1: Input sanitization
        sanitization_result = await self.prompt_sanitizer.sanitize_input(
            user_input, conversation_context
        )
        
        # Update metrics
        if not sanitization_result['allowed']:
            self.security_metrics['blocked_inputs'] += 1
            logger.warning(f"🚫 Input blocked for conversation {conversation_id}: {sanitization_result['reason']}")
            return {
                'proceed': False,
                'reason': sanitization_result['reason'],
                'sanitized_input': None,
                'security_action': 'blocked',
                'risk_score': sanitization_result.get('risk_score', 0.0),
                'conversation_id': conversation_id
            }
        
        if sanitization_result['risk_level'] == 'medium':
            self.security_metrics['sanitized_inputs'] += 1
            logger.info(f"🧹 Input sanitized for conversation {conversation_id}")
        
        return {
            'proceed': True,
            'reason': sanitization_result['reason'],
            'sanitized_input': sanitization_result['sanitized_input'],
            'security_action': 'allowed' if sanitization_result['risk_level'] == 'low' else 'sanitized',
            'risk_score': sanitization_result.get('risk_score', 0.0),
            'conversation_id': conversation_id
        }
    
    async def process_response(
        self, 
        user_input: str, 
        ai_response: str, 
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Process AI response for echo detection and intervention
        
        Args:
            user_input: The original user input
            ai_response: The AI's response
            conversation_id: Unique conversation identifier
            
        Returns:
            Dictionary with echo analysis and potential interventions
        """
        if not user_input or not ai_response:
            return {
                'allow_response': True,
                'intervention_message': None,
                'echo_detected': False,
                'llm_adjustments': {}
            }
        
        # Check for echo behavior
        echo_result = await self.echo_monitor.check_response(
            user_input, ai_response, conversation_id
        )
        
        # Update metrics
        if echo_result['is_echo']:
            self.security_metrics['echo_detections'] += 1
            logger.info(f"🔄 Echo detected in conversation {conversation_id}: score={echo_result['echo_score']:.3f}")
        
        if echo_result['requires_intervention']:
            self.security_metrics['interventions'] += 1
            logger.error(f"🚨 Echo intervention required for conversation {conversation_id}")
            return {
                'allow_response': False,
                'intervention_message': echo_result['suggested_action'],
                'echo_detected': True,
                'llm_adjustments': echo_result.get('temp_penalty_adjustment', {}),
                'echo_score': echo_result['echo_score'],
                'consecutive_count': echo_result['consecutive_count'],
                'conversation_id': conversation_id
            }
        
        return {
            'allow_response': True,
            'intervention_message': None,
            'echo_detected': echo_result['is_echo'],
            'llm_adjustments': echo_result.get('temp_penalty_adjustment', {}),
            'echo_score': echo_result['echo_score'],
            'consecutive_count': echo_result['consecutive_count'],
            'conversation_id': conversation_id
        }
    
    async def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Clean up and provide summary when a conversation ends
        
        Args:
            conversation_id: The conversation that ended
            
        Returns:
            Summary statistics for the conversation
        """
        if conversation_id in self.active_conversations:
            self.active_conversations.remove(conversation_id)
            logger.debug(f"🏁 Conversation ended: {conversation_id}")
        
        # Get conversation-specific statistics
        echo_stats = self.echo_monitor.get_conversation_echo_stats(conversation_id)
        
        # Get voice session stats if available
        voice_stats = {}
        if conversation_id in self.voice_sessions:
            voice_stats = self.voice_sessions[conversation_id].copy()
            del self.voice_sessions[conversation_id]  # Clean up
        
        # Reset conversation history to prevent memory leaks
        self.echo_monitor.reset_conversation_history(conversation_id)
        
        return {
            'conversation_id': conversation_id,
            'echo_stats': echo_stats,
            'voice_stats': voice_stats,
            'ended_at': datetime.now().isoformat()
        }
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data"""
        runtime_minutes = (datetime.now() - self.security_metrics['start_time']).total_seconds() / 60
        
        # Get component statistics
        sanitizer_stats = self.prompt_sanitizer.get_security_stats()
        echo_stats = self.echo_monitor.get_global_echo_stats()
        
        # Calculate aggregate rates
        total_inputs = sanitizer_stats['total_inputs']
        total_responses = echo_stats['total_responses']
        
        dashboard = {
            'runtime': {
                'uptime_minutes': runtime_minutes,
                'active_conversations': len(self.active_conversations),
                'total_conversations': self.security_metrics['total_conversations']
            },
            'input_security': {
                'total_inputs': total_inputs,
                'blocked_inputs': self.security_metrics['blocked_inputs'],
                'sanitized_inputs': self.security_metrics['sanitized_inputs'],
                'block_rate': self.security_metrics['blocked_inputs'] / max(total_inputs, 1),
                'sanitization_rate': self.security_metrics['sanitized_inputs'] / max(total_inputs, 1),
                'false_positive_rate': sanitizer_stats['false_positive_rate']
            },
            'echo_monitoring': {
                'total_responses': total_responses,
                'echo_detections': self.security_metrics['echo_detections'],
                'interventions': self.security_metrics['interventions'],
                'echo_rate': self.security_metrics['echo_detections'] / max(total_responses, 1),
                'intervention_rate': self.security_metrics['interventions'] / max(self.security_metrics['echo_detections'], 1),
                'false_positive_rate': echo_stats['false_positive_rate']
            },
            'performance': {
                'inputs_per_minute': total_inputs / max(runtime_minutes, 0.1),
                'responses_per_minute': total_responses / max(runtime_minutes, 0.1),
                'conversations_per_hour': self.security_metrics['total_conversations'] / max(runtime_minutes / 60, 0.1)
            },
            'voice_security': {
                'voice_only_mode': self.voice_only_mode,
                'active_voice_sessions': len(self.voice_sessions),
                'voice_anomalies': self.security_metrics['voice_anomalies'],
                'voice_anomaly_rate': self.security_metrics['voice_anomalies'] / max(total_inputs, 1) if self.voice_only_mode else 0
            },
            'health_status': self._calculate_health_status()
        }
        
        return dashboard
    
    def _calculate_health_status(self) -> Dict[str, Any]:
        """Calculate overall security system health"""
        sanitizer_stats = self.prompt_sanitizer.get_security_stats()
        echo_stats = self.echo_monitor.get_global_echo_stats()
        
        # Define health thresholds based on success criteria
        thresholds = {
            'max_echo_rate': 0.02,  # <2%
            'max_block_rate': 0.001,  # <0.1% false positives
            'max_intervention_rate': 0.05  # <5% interventions
        }
        
        # Calculate health scores
        echo_health = 1.0 - min(echo_stats['echo_rate'] / thresholds['max_echo_rate'], 1.0)
        block_health = 1.0 - min(sanitizer_stats['false_positive_rate'] / thresholds['max_block_rate'], 1.0)
        
        overall_health = (echo_health + block_health) / 2
        
        # Determine status
        if overall_health >= 0.9:
            status = 'excellent'
        elif overall_health >= 0.7:
            status = 'good'
        elif overall_health >= 0.5:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'overall_score': overall_health,
            'status': status,
            'echo_health': echo_health,
            'block_health': block_health,
            'current_echo_rate': echo_stats['echo_rate'],
            'current_false_positive_rate': sanitizer_stats['false_positive_rate'],
            'thresholds': thresholds
        }
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get security summary for a specific conversation"""
        echo_stats = self.echo_monitor.get_conversation_echo_stats(conversation_id)
        
        return {
            'conversation_id': conversation_id,
            'is_active': conversation_id in self.active_conversations,
            'echo_statistics': echo_stats,
            'security_alerts': self._get_conversation_alerts(conversation_id, echo_stats)
        }
    
    def _get_conversation_alerts(self, conversation_id: str, echo_stats: Dict[str, Any]) -> List[str]:
        """Generate security alerts for a conversation"""
        alerts = []
        
        if echo_stats['echo_rate'] > 0.1:  # >10% echo rate
            alerts.append(f"High echo rate detected: {echo_stats['echo_rate']:.1%}")
        
        if echo_stats['max_consecutive'] >= 3:
            alerts.append(f"Consecutive echoing detected: {echo_stats['max_consecutive']} in a row")
        
        if echo_stats['avg_echo_score'] > 0.7:
            alerts.append(f"High average echo similarity: {echo_stats['avg_echo_score']:.2f}")
        
        return alerts
    
    def report_false_positive(self, conversation_id: str, input_text: str = None, response_text: str = None, 
                            false_positive_type: str = "both", reason: str = "Manual review"):
        """Report false positives to improve the security components"""
        if false_positive_type in ["sanitizer", "both"] and input_text:
            self.prompt_sanitizer.report_false_positive(input_text, reason)
        
        if false_positive_type in ["echo", "both"] and input_text and response_text:
            self.echo_monitor.report_false_positive(input_text, response_text, reason)
        
        logger.info(f"🔍 False positive reported for conversation {conversation_id}: type={false_positive_type}, reason={reason}")
    
    def adjust_security_settings(self, 
                                echo_threshold: float = None,
                                intervention_threshold: int = None,
                                sanitizer_thresholds: Dict[str, float] = None):
        """Adjust security component thresholds (use with caution)"""
        if echo_threshold is not None or intervention_threshold is not None:
            self.echo_monitor.adjust_thresholds(echo_threshold, intervention_threshold)
        
        if sanitizer_thresholds:
            for level, threshold in sanitizer_thresholds.items():
                if level in self.prompt_sanitizer.severity_thresholds:
                    old_value = self.prompt_sanitizer.severity_thresholds[level]
                    self.prompt_sanitizer.severity_thresholds[level] = threshold
                    logger.info(f"🔧 Sanitizer {level} threshold adjusted: {old_value:.3f} -> {threshold:.3f}")
        
        logger.warning("⚠️ Security settings have been manually adjusted - monitor performance carefully")
    
    async def _analyze_voice_input(self, user_input: str, conversation_id: str) -> bool:
        """
        Analyze voice input for anomalies and suspicious patterns
        
        Args:
            user_input: The transcribed voice input
            conversation_id: The conversation identifier
            
        Returns:
            True if anomaly detected, False otherwise
        """
        if conversation_id not in self.voice_sessions:
            return False
        
        session = self.voice_sessions[conversation_id]
        current_time = datetime.now()
        
        # Update session statistics
        session['total_inputs'] += 1
        session['last_input_time'] = current_time
        
        # Calculate running average input length
        input_length = len(user_input)
        if session['total_inputs'] == 1:
            session['avg_input_length'] = input_length
        else:
            # Exponential moving average
            alpha = 0.3
            session['avg_input_length'] = (1 - alpha) * session['avg_input_length'] + alpha * input_length
        
        anomaly_detected = False
        
        # Check for voice-specific anomalies
        
        # 1. Unusually long input (possible speech-to-text manipulation)
        if input_length > session['avg_input_length'] * 3 and input_length > 500:
            logger.warning(f"🔊 Voice anomaly: Unusually long input in conversation {conversation_id} ({input_length} chars)")
            anomaly_detected = True
        
        # 2. Rapid-fire inputs (possible automated attack)
        session_duration = (current_time - session['start_time']).total_seconds()
        if session_duration > 0:
            input_rate = session['total_inputs'] / session_duration
            if input_rate > 10:  # More than 10 inputs per second is suspicious
                logger.warning(f"🔊 Voice anomaly: Rapid input rate in conversation {conversation_id} ({input_rate:.1f}/sec)")
                anomaly_detected = True
        
        # 3. Check for patterns that suggest STT manipulation
        suspicious_patterns = [
            r'[A-Z]{10,}',  # Long sequences of capital letters (STT artifacts)
            r'(.)\1{20,}',  # Repeated characters (audio manipulation)
            r'[0-9]{50,}',  # Long number sequences (unlikely in natural speech)
            r'[^\w\s]{10,}',  # Long sequences of special characters
        ]
        
        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, user_input):
                logger.warning(f"🔊 Voice anomaly: Suspicious STT pattern in conversation {conversation_id}: {pattern}")
                anomaly_detected = True
                break
        
        # 4. Silent periods detection (if input is empty but session is active)
        if not user_input.strip():
            time_since_last = (current_time - session['last_input_time']).total_seconds()
            if time_since_last > 300:  # 5 minutes of silence
                logger.info(f"🔊 Voice session: Extended silence in conversation {conversation_id}")
        
        if anomaly_detected:
            session['suspicious_patterns'] += 1
        
        return anomaly_detected