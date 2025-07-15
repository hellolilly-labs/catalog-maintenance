"""
EchoMonitor - Real-time echo behavior detection and intervention

Monitors AI responses for echo behavior (repeating user input) and provides
dynamic intervention with temporary LLM parameter adjustments.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class EchoMonitor:
    """Monitor and prevent echo behavior in real-time"""
    
    def __init__(self):
        self.recent_responses = []
        self.max_history = 10
        self.echo_threshold = 0.75  # Similarity threshold for echo detection
        self.intervention_threshold = 3  # Consecutive echoes before intervention
        
        # Auto-reset tracking
        self.pending_reset = False          # Flag for frequency penalty reset
        self.prev_freq_penalty = 0.0        # Remembers baseline frequency penalty
        
        # Statistics tracking
        self.stats = {
            'total_responses': 0,
            'echo_detections': 0,
            'interventions': 0,
            'false_positives': 0,
            'start_time': datetime.now()
        }
    
    async def check_response(self, user_input: str, ai_response: str, conversation_id: str) -> Dict[str, Any]:
        """
        Check if AI response exhibits echo behavior
        
        Args:
            user_input: The user's input message
            ai_response: The AI's response message
            conversation_id: Unique conversation identifier
            
        Returns:
            Dictionary with echo detection results and suggested actions
        """
        if not user_input or not ai_response:
            return {
                'is_echo': False,
                'echo_score': 0.0,
                'consecutive_count': 0,
                'requires_intervention': False,
                'suggested_action': None
            }
        
        self.stats['total_responses'] += 1
        
        # Calculate similarity between input and response
        echo_score = self._calculate_echo_similarity(user_input, ai_response)
        
        # Track in conversation history
        response_data = {
            'input': user_input,
            'response': ai_response,
            'echo_score': echo_score,
            'conversation_id': conversation_id,
            'timestamp': datetime.now()
        }
        
        self.recent_responses.append(response_data)
        
        # Maintain history size
        if len(self.recent_responses) > self.max_history:
            self.recent_responses.pop(0)
        
        # Check for echo pattern
        is_echo = echo_score > self.echo_threshold
        consecutive_echoes = self._count_consecutive_echoes()
        
        if is_echo:
            self.stats['echo_detections'] += 1
            logger.warning(f"🔄 Echo detected: similarity={echo_score:.3f}, input='{user_input[:50]}...', response='{ai_response[:50]}...'")
        
        result = {
            'is_echo': is_echo,
            'echo_score': echo_score,
            'consecutive_count': consecutive_echoes,
            'requires_intervention': consecutive_echoes >= self.intervention_threshold,
            'suggested_action': None,
            'conversation_id': conversation_id
        }
        
        if result['requires_intervention']:
            self.stats['interventions'] += 1
            result['suggested_action'] = self._generate_intervention_note()
            # Add frequency penalty while keeping existing presence penalty
            result['temp_penalty_adjustment'] = {'frequency_penalty': 0.2}
            self.pending_reset = True  # Mark for reset after next turn
            logger.info("🔄 EchoMonitor: frequency_penalty=0.2 for next turn")
            logger.error(f"🚨 Echo intervention triggered after {consecutive_echoes} consecutive echoes in conversation {conversation_id}")
        
        return result
    
    def _calculate_echo_similarity(self, input_text: str, response_text: str) -> float:
        """Calculate similarity score between input and response using multiple methods"""
        if not input_text or not response_text:
            return 0.0
        
        # Normalize text for comparison
        input_normalized = self._normalize_text(input_text)
        response_normalized = self._normalize_text(response_text)
        
        # Method 1: SequenceMatcher (overall similarity)
        matcher = SequenceMatcher(None, input_normalized, response_normalized)
        sequence_similarity = matcher.ratio()
        
        # Method 2: Word overlap analysis
        input_words = set(input_normalized.split())
        response_words = set(response_normalized.split())
        
        if not input_words or not response_words:
            word_overlap = 0.0
        else:
            intersection = input_words.intersection(response_words)
            word_overlap = len(intersection) / len(input_words.union(response_words))
        
        # Method 3: Longest common subsequence
        lcs_length = self._longest_common_subsequence_length(input_normalized, response_normalized)
        lcs_similarity = lcs_length / max(len(input_normalized), len(response_normalized), 1)
        
        # Combine methods with weights
        # SequenceMatcher is most reliable for exact echoing
        # Word overlap catches partial echoing
        # LCS catches structured echoing
        final_similarity = (
            sequence_similarity * 0.6 +
            word_overlap * 0.3 +
            lcs_similarity * 0.1
        )
        
        logger.debug(f"Echo similarity: sequence={sequence_similarity:.3f}, words={word_overlap:.3f}, lcs={lcs_similarity:.3f}, final={final_similarity:.3f}")
        
        return final_similarity
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for similarity comparison"""
        import re
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common filler words that don't indicate echoing
        filler_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = normalized.split()
        words = [word for word in words if word not in filler_words]
        normalized = ' '.join(words)
        
        # Remove punctuation except for important structural elements
        normalized = re.sub(r'[^\w\s\-\']', ' ', normalized)
        
        # Clean up extra spaces again
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _longest_common_subsequence_length(self, text1: str, text2: str) -> int:
        """Calculate the length of the longest common subsequence"""
        m, n = len(text1), len(text2)
        
        # Create a 2D array to store LCS lengths
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the dp array
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _count_consecutive_echoes(self) -> int:
        """Count consecutive echo occurrences from most recent responses"""
        count = 0
        for response_data in reversed(self.recent_responses):
            if response_data['echo_score'] > self.echo_threshold:
                count += 1
            else:
                break
        return count
    
    def _generate_intervention_note(self) -> str:
        """Generate intervention message for persistent echo behavior"""
        intervention_messages = [
            "🔄 I notice I might be repeating your words. Let me focus on providing helpful information about our products instead. What specific product category interests you?",
            "🔄 Let me refocus on helping you find what you're looking for. What type of product can I help you with today?",
            "🔄 I want to make sure I'm giving you useful information. Could you tell me more about what you're shopping for?",
            "🔄 Let me provide better assistance. What specific products or information would be most helpful right now?"
        ]
        
        # Rotate through different messages to avoid repetition
        import random
        return random.choice(intervention_messages)
    
    def get_conversation_echo_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get echo statistics for a specific conversation"""
        conversation_responses = [
            r for r in self.recent_responses 
            if r['conversation_id'] == conversation_id
        ]
        
        if not conversation_responses:
            return {
                'total_responses': 0,
                'echo_count': 0,
                'echo_rate': 0.0,
                'max_consecutive': 0,
                'avg_echo_score': 0.0
            }
        
        echo_count = sum(1 for r in conversation_responses if r['echo_score'] > self.echo_threshold)
        echo_scores = [r['echo_score'] for r in conversation_responses]
        
        # Calculate max consecutive echoes
        max_consecutive = 0
        current_consecutive = 0
        for response in conversation_responses:
            if response['echo_score'] > self.echo_threshold:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return {
            'total_responses': len(conversation_responses),
            'echo_count': echo_count,
            'echo_rate': echo_count / len(conversation_responses),
            'max_consecutive': max_consecutive,
            'avg_echo_score': sum(echo_scores) / len(echo_scores),
            'recent_scores': echo_scores[-5:]  # Last 5 scores for trend analysis
        }
    
    def get_global_echo_stats(self) -> Dict[str, Any]:
        """Get overall echo statistics across all conversations"""
        runtime_minutes = (datetime.now() - self.stats['start_time']).total_seconds() / 60
        
        return {
            'total_responses': self.stats['total_responses'],
            'echo_detections': self.stats['echo_detections'],
            'interventions': self.stats['interventions'],
            'false_positives': self.stats['false_positives'],
            'echo_rate': self.stats['echo_detections'] / max(self.stats['total_responses'], 1),
            'intervention_rate': self.stats['interventions'] / max(self.stats['echo_detections'], 1),
            'false_positive_rate': self.stats['false_positives'] / max(self.stats['echo_detections'], 1),
            'runtime_minutes': runtime_minutes,
            'responses_per_minute': self.stats['total_responses'] / max(runtime_minutes, 0.1)
        }
    
    def report_false_positive(self, user_input: str, ai_response: str, reason: str = "Manual review"):
        """Report a false positive echo detection for system improvement"""
        self.stats['false_positives'] += 1
        logger.info(f"🔍 Echo false positive reported: input='{user_input[:50]}...', response='{ai_response[:50]}...', reason='{reason}'")
    
    def reset_conversation_history(self, conversation_id: str):
        """Reset echo history for a specific conversation (useful for new sessions)"""
        original_count = len(self.recent_responses)
        self.recent_responses = [
            r for r in self.recent_responses 
            if r['conversation_id'] != conversation_id
        ]
        removed_count = original_count - len(self.recent_responses)
        if removed_count > 0:
            logger.info(f"🔄 Reset echo history for conversation {conversation_id}: removed {removed_count} entries")
    
    def adjust_thresholds(self, echo_threshold: float = None, intervention_threshold: int = None):
        """Adjust detection thresholds for tuning (use carefully)"""
        if echo_threshold is not None:
            old_threshold = self.echo_threshold
            self.echo_threshold = max(0.1, min(1.0, echo_threshold))
            logger.info(f"🔧 Echo threshold adjusted: {old_threshold:.3f} -> {self.echo_threshold:.3f}")
        
        if intervention_threshold is not None:
            old_threshold = self.intervention_threshold
            self.intervention_threshold = max(1, min(10, intervention_threshold))
            logger.info(f"🔧 Intervention threshold adjusted: {old_threshold} -> {self.intervention_threshold}")
    
    def should_reset_frequency_penalty(self) -> bool:
        """Check if frequency penalty should be reset after this turn"""
        if self.pending_reset:
            self.pending_reset = False
            logger.info("🔄 EchoMonitor: frequency_penalty should be reset to baseline")
            return True
        return False