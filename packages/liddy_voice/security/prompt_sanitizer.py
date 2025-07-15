"""
PromptSanitizer - Runtime protection against prompt injection attacks

Provides real-time analysis and filtering of user inputs to detect and prevent
prompt injection attempts while minimizing false positives.
"""

import re
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class PromptSanitizer:
    """Runtime protection against prompt injection and malicious instructions"""
    
    def __init__(self):
        # Enhanced injection patterns with look-behind for jailbreaks
        self.injection_patterns = [
            r'ignore previous instructions',
            r'forget everything',
            r'(?i)(?<![\\w])system (prompt|message)',  # Look-behind for jailbreaks
            r'you are now',
            r'roleplay as',
            r'pretend to be',
            r'act as if',
            r'⟦.*⟧',  # Block attempts to inject INTERNAL markers
            r'override.*settings',
            r'new.*instructions',
            r'change.*behavior',
            r'ignore.*rules',
            r'bypass.*security',
            r'admin.*mode',
            r'developer.*mode',
            r'debug.*mode',
            r'maintenance.*mode'
        ]
        
        # Risk severity thresholds
        self.severity_thresholds = {
            'low': 0.15,
            'medium': 0.35, 
            'high': 0.65
        }
        
        # Statistics tracking
        self.stats = {
            'total_inputs': 0,
            'blocked_inputs': 0,
            'sanitized_inputs': 0,
            'false_positives': 0,  # Manually tracked
            'start_time': datetime.now()
        }
    
    async def sanitize_input(self, user_input: str, conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """
        Sanitize user input and detect potential injection attempts
        
        Args:
            user_input: The raw user input to analyze
            conversation_context: Previous conversation messages for context analysis
            
        Returns:
            Dictionary with sanitization results and actions
        """
        if not user_input:
            return {
                'allowed': True,
                'reason': 'Empty input',
                'sanitized_input': user_input,
                'risk_level': 'low',
                'log_alert': False
            }
        
        self.stats['total_inputs'] += 1
        
        # Check for injection patterns
        injection_score = self._calculate_injection_risk(user_input)
        
        # Context analysis for sophisticated attacks
        context_risk = self._analyze_conversation_context(conversation_context or [])
        
        # Combined risk assessment
        total_risk = max(injection_score, context_risk)
        
        logger.debug(f"Input risk assessment: injection={injection_score:.3f}, context={context_risk:.3f}, total={total_risk:.3f}")
        
        if total_risk > self.severity_thresholds['high']:
            self.stats['blocked_inputs'] += 1
            logger.warning(f"🚨 HIGH RISK INPUT BLOCKED: {user_input[:100]}... (risk={total_risk:.3f})")
            return {
                'allowed': False,
                'reason': 'High risk prompt injection detected',
                'sanitized_input': None,
                'risk_level': 'high',
                'log_alert': True,
                'risk_score': total_risk,
                'injection_score': injection_score,
                'context_score': context_risk
            }
        elif total_risk > self.severity_thresholds['medium']:
            # Allow but sanitize
            sanitized = self._sanitize_suspicious_content(user_input)
            self.stats['sanitized_inputs'] += 1
            logger.info(f"⚠️ Medium risk input sanitized: {user_input[:50]}... -> {sanitized[:50]}... (risk={total_risk:.3f})")
            return {
                'allowed': True,
                'reason': 'Medium risk - content sanitized',
                'sanitized_input': sanitized,
                'risk_level': 'medium',
                'log_alert': True,
                'risk_score': total_risk,
                'injection_score': injection_score,
                'context_score': context_risk
            }
        else:
            return {
                'allowed': True,
                'reason': 'Input appears safe',
                'sanitized_input': user_input,
                'risk_level': 'low',
                'log_alert': False,
                'risk_score': total_risk,
                'injection_score': injection_score,
                'context_score': context_risk
            }
    
    def _calculate_injection_risk(self, text: str) -> float:
        """Calculate injection risk score 0.0-1.0"""
        if not text:
            return 0.0
            
        risk_score = 0.0
        text_lower = text.lower()
        
        # Pattern matching with weighted scores
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                # System-related patterns are higher risk
                if 'system' in pattern or 'admin' in pattern or 'developer' in pattern:
                    risk_score += 0.4
                else:
                    risk_score += 0.2
        
        # Check for excessive special characters (obfuscation attempts)
        special_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        if special_ratio > 0.3:
            risk_score += 0.3
        elif special_ratio > 0.2:
            risk_score += 0.1
        
        # Check for suspiciously long inputs (injection payloads are often verbose)
        if len(text) > 2000:
            risk_score += 0.3
        elif len(text) > 1000:
            risk_score += 0.1
        
        # Check for repeated patterns (common in jailbreak attempts)
        words = text_lower.split()
        if len(words) > 5:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))
            if repetition_ratio > 0.5:
                risk_score += 0.2
        
        # Check for base64 or hex encoded content (payload hiding)
        if re.search(r'[a-zA-Z0-9+/]{20,}={0,2}', text) or re.search(r'[0-9a-fA-F]{32,}', text):
            risk_score += 0.3
        
        # Check for multiple newlines or formatting attempts
        if text.count('\\n') > 5 or text.count('\n') > 5:
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def _analyze_conversation_context(self, conversation_context: List[Dict]) -> float:
        """Analyze conversation context for sophisticated attack patterns"""
        if not conversation_context:
            return 0.0
        
        context_risk = 0.0
        
        # Look for escalating injection attempts
        recent_messages = conversation_context[-5:] if len(conversation_context) > 5 else conversation_context
        injection_attempts = 0
        
        for message in recent_messages:
            message_text = message.get('content', '') or message.get('text', '')
            if message.get('role') == 'user' and message_text:
                message_risk = self._calculate_injection_risk(message_text)
                if message_risk > 0.3:
                    injection_attempts += 1
        
        # Escalating pattern detection
        if injection_attempts >= 3:
            context_risk += 0.4
        elif injection_attempts >= 2:
            context_risk += 0.2
        
        # Check for conversation manipulation attempts
        for message in recent_messages:
            message_text = message.get('content', '') or message.get('text', '')
            if message.get('role') == 'user' and message_text:
                manipulation_patterns = [
                    r'let.?s start over',
                    r'forget.*conversation',
                    r'new.*session',
                    r'restart.*chat',
                    r'clear.*history'
                ]
                for pattern in manipulation_patterns:
                    if re.search(pattern, message_text.lower()):
                        context_risk += 0.1
        
        return min(context_risk, 1.0)
    
    def _sanitize_suspicious_content(self, text: str) -> str:
        """Sanitize content by removing or replacing suspicious elements"""
        sanitized = text
        
        # Remove obvious injection patterns but preserve legitimate content
        sanitized = re.sub(r'(?i)ignore\s+previous\s+instructions?', '[content filtered]', sanitized)
        sanitized = re.sub(r'(?i)forget\s+everything', '[content filtered]', sanitized)
        sanitized = re.sub(r'(?i)(?<![\w])system\s+(prompt|message)', '[content filtered]', sanitized)
        sanitized = re.sub(r'(?i)you\s+are\s+now', 'you are', sanitized)
        sanitized = re.sub(r'(?i)roleplay\s+as', 'discuss', sanitized)
        sanitized = re.sub(r'(?i)pretend\s+to\s+be', 'imagine being', sanitized)
        
        # Remove INTERNAL markers
        sanitized = re.sub(r'⟦.*?⟧', '[content filtered]', sanitized)
        
        # Remove excessive special characters while preserving normal punctuation
        sanitized = re.sub(r'[^\w\s.,!?;:\'"()-]{3,}', '[symbols filtered]', sanitized)
        
        # Limit length to prevent overwhelming the system
        if len(sanitized) > 1500:
            sanitized = sanitized[:1500] + '... [truncated for length]'
        
        return sanitized.strip()
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get current security statistics"""
        runtime_minutes = (datetime.now() - self.stats['start_time']).total_seconds() / 60
        
        return {
            'total_inputs': self.stats['total_inputs'],
            'blocked_inputs': self.stats['blocked_inputs'],
            'sanitized_inputs': self.stats['sanitized_inputs'],
            'false_positives': self.stats['false_positives'],
            'block_rate': self.stats['blocked_inputs'] / max(self.stats['total_inputs'], 1),
            'sanitization_rate': self.stats['sanitized_inputs'] / max(self.stats['total_inputs'], 1),
            'false_positive_rate': self.stats['false_positives'] / max(self.stats['blocked_inputs'] + self.stats['sanitized_inputs'], 1),
            'runtime_minutes': runtime_minutes,
            'inputs_per_minute': self.stats['total_inputs'] / max(runtime_minutes, 0.1)
        }
    
    def report_false_positive(self, user_input: str, reason: str = "Manual review"):
        """Report a false positive for improving the system"""
        self.stats['false_positives'] += 1
        logger.info(f"🔍 False positive reported: {user_input[:100]}... Reason: {reason}")
        
        # Log for analysis and pattern improvement
        logger.info(f"False positive analysis needed for input: {user_input}")
    
    def update_patterns(self, new_patterns: List[str], remove_patterns: List[str] = None):
        """Update injection patterns (for system maintenance)"""
        if new_patterns:
            self.injection_patterns.extend(new_patterns)
            logger.info(f"Added {len(new_patterns)} new injection patterns")
        
        if remove_patterns:
            for pattern in remove_patterns:
                if pattern in self.injection_patterns:
                    self.injection_patterns.remove(pattern)
            logger.info(f"Removed {len(remove_patterns)} injection patterns")