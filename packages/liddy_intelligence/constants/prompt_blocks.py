"""Prompt blocks and constants for relevance guardrails and self-audit."""

RELEVANCE_GUARDRAILS = """
### Relevance Guardrails (ALWAYS read before answering)

1. **Domain Check**
   • This brand's primary industry = "{{industry}}".  
   • Reject terms that do **not** appear in either  
       – the supplied product-catalog text **or**  
       – ≥ 2 independent {{industry}}-related web sources you cite.  
   • If you cannot find {{industry}} evidence for a term, DISCARD it.

2. **Specificity Test**
   • Keep only language that differentiates **this** product.  
   • Remove generic tech buzz-phrases (e.g. cloud-native, AI persona, gloves-level).

3. **Self-Audit**  
   • After writing your draft, list any token/phrase that violates Rules 1-2.  
   • If the list is non-empty, revise the draft until the list is empty.  
   • Output only the final clean draft; do **not** output the list.
"""

SENTINEL = "**SELF-CHECK PASSED**"

# Additional prompt for terminology extraction
TERMINOLOGY_SELECTION_CRITERIA = """
### Selection Criteria
• A candidate term must (a) appear ≥ 3 times in the supplied catalog OR (b) be cited by ≥ 2 {{industry}} sources (include those sources).  
• Discard all others.
"""