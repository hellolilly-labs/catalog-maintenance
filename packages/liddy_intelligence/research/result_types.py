"""
Result types for research modules

Provides structured data classes for research outputs to enable
clean API access without markdown parsing.
"""

from dataclasses import dataclass
from typing import List, Tuple

# Type alias for a term with its confidence score
Term = Tuple[str, float]  # (term_name, confidence_score)


@dataclass(frozen=True)
class PriceTerminology:
    """
    Structured price terminology data with confidence scores.
    
    Each tier contains a tuple of (term, confidence) pairs where:
    - term: The actual terminology (e.g., "s-works", "pro", "base")
    - confidence: 0.0-1.0 score based on web hits, product coverage, and provenance
    
    Attributes:
        premium_terms: High-end/professional tier indicators (immutable tuple)
        mid_terms: Mid-range tier indicators (immutable tuple)
        budget_terms: Entry-level/budget tier indicators (immutable tuple)
    """
    premium_terms: Tuple[Term, ...]
    mid_terms: Tuple[Term, ...]
    budget_terms: Tuple[Term, ...]
    
    # Backward compatibility properties for legacy code expecting lists
    @property
    def premium_terms_list(self) -> List[Term]:
        """Legacy compatibility: return premium_terms as list. TODO deprecate after Q3."""
        return list(self.premium_terms)
    
    @property
    def mid_terms_list(self) -> List[Term]:
        """Legacy compatibility: return mid_terms as list. TODO deprecate after Q3."""
        return list(self.mid_terms)
    
    @property
    def budget_terms_list(self) -> List[Term]:
        """Legacy compatibility: return budget_terms as list. TODO deprecate after Q3."""
        return list(self.budget_terms)
    
    def all_terms(self) -> List[Term]:
        """Get all terms across all tiers."""
        return list(self.premium_terms) + list(self.mid_terms) + list(self.budget_terms)
    
    def term_names_only(self, tier: str = None) -> List[str]:
        """Get just the term names without confidence scores."""
        if tier == "premium":
            return [t for t, _ in self.premium_terms]
        elif tier == "mid":
            return [t for t, _ in self.mid_terms]
        elif tier == "budget":
            return [t for t, _ in self.budget_terms]
        else:
            # Return all terms
            return [t for t, _ in self.all_terms()]