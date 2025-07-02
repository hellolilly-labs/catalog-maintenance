# Voice Assistant Search Comparison Testing Guide

This guide explains how to test and compare the enhanced search functionality against a baseline approach, with special focus on voice-realistic scenarios.

## Overview

The comparison framework tests two search approaches:

1. **Enhanced Search**: Uses separate dense/sparse indexes with neural reranking, ProductCatalogResearcher knowledge, and filter extraction
2. **Baseline Search**: Uses a single dense index with `Product.to_markdown()` for text generation

### Testing Approaches

We provide two testing frameworks:

1. **Standard Testing** (`test_search_comparison.py`) - Direct query comparison
2. **Voice-Realistic Testing** (`voice_search_comparison.py`) - Simulates actual voice conversations with tool calls

## Setup

### 1. Install Required Dependencies

```bash
pip install rich numpy
```

### 2. Ensure Your Enhanced Indexes Are Ready

Make sure you've already ingested products using the enhanced approach:

```bash
python ingest_product_catalog.py --brand specialized.com
```

### 3. Set Up the Baseline Index

The baseline index needs to be created and populated:

```bash
# Interactive mode will prompt you to set up the baseline
python run_search_comparison.py --account specialized.com --setup-baseline
```

Or run the test script directly:

```bash
python test_search_comparison.py --account specialized.com --ingest-baseline --max-products 1000
```

## Running Tests

### Voice-Realistic Testing (Recommended for Production Validation)

The voice-realistic framework simulates actual voice conversations with natural speech patterns:

```bash
python voice_search_comparison.py --account specialized.com
```

This framework:
- Uses the actual system prompt from Langfuse (e.g., `specialized.com/full_instructions`)
- Simulates natural voice queries with speech patterns (um, uh, like)
- Uses the same LLM as the voice assistant (gpt-4.1)
- Handles tool calls (`product_search`, `display_product`, `knowledge_search`)
- Evaluates results in conversational context

#### Voice Test Scenarios:
- **Beginner Cyclist**: "Um, I'm looking to get into road cycling but I don't really know where to start"
- **Parent Shopping**: "I need a bike for my daughter, she's like 10 years old"
- **Upgrade Search**: "So I have this older bike and I'm thinking about upgrading"
- **Comparison Shopping**: "I saw the Specialized Tarmac online, do you have that or something similar?"
- **Budget Conscious**: "What's your best road bike for under fifteen hundred?"

### Interactive Mode (Good for Development)

The interactive mode provides a user-friendly interface for testing:

```bash
python run_search_comparison.py --account specialized.com
```

#### Interactive Commands:

- **`search <query>`** - Compare results for a specific query
  ```
  >>> search mountain bike under $2000
  ```

- **`chat`** - Enter conversational mode
  - Have a natural conversation
  - Use `/search` to compare the last search-like query
  - Use `/exit` to leave chat mode

- **`scenarios`** - Run predefined test scenarios
  - Tests common search patterns
  - Generates comprehensive metrics

- **`report`** - Generate comparison report

- **`help`** - Show detailed help

- **`exit`** - Exit the program

### Batch Mode

Run all test scenarios at once:

```bash
python test_search_comparison.py --account specialized.com
```

This will:
1. Run predefined test scenarios
2. Compare results for each query
3. Generate a comprehensive report
4. Save results to `search_comparison_results/`

## Understanding the Results

### Voice-Realistic Results

The voice comparison report includes:

1. **Conversation Context**: Shows the actual dialogue flow
2. **Tool Call Analysis**: Which tools were invoked and when
3. **Natural Language Understanding**: How well each approach handles spoken queries
4. **Contextual Relevance**: Results quality considering the full conversation

Example voice report section:
```
## Scenario: Parent shopping for child's bike

Voice Query: "bike for 10 year old daughter who likes pink and purple good for neighborhood rides"

### Conversation Context:
- USER: I need a bike for my daughter, she's like 10 years old
- ASSISTANT: I'd be happy to help you find the perfect bike for your daughter! 
- USER: She really likes pink and purple colors
- ASSISTANT: Great! Let me search for bikes that would be suitable for a 10-year-old...

### Search Performance:
- Enhanced Search: 8 results in 0.823s (understood age, color preference, use case)
- Baseline Search: 12 results in 0.342s (matched keywords but missed context)
```

### Standard Performance Metrics

The comparison shows several key metrics:

1. **Response Time**: How long each search approach takes
2. **Results Count**: Number of results returned
3. **Overlap Ratio**: How many products appear in both result sets
4. **Rank Correlation**: How similar the ranking order is

### Result Quality

Each comparison includes:
- Top results from each approach
- Relevance explanations (for enhanced search)
- LLM evaluation comparing result quality

### Example Output

```
Performance Metrics
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric          ┃ Enhanced ┃ Baseline ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Response Time   │ 0.823s   │ 0.342s   │
│ Results Count   │ 10       │ 10       │
│ Overlap Ratio   │ 60.00%   │ 60.00%   │
│ Rank Correlation│ 0.452    │ 0.452    │
└─────────────────┴──────────┴──────────┘
```

## Test Scenarios

The framework includes several predefined scenarios:

1. **Basic Category Search**: "I'm looking for a mountain bike"
2. **Price Constraint**: "I need a road bike under $2000"
3. **Feature Specific**: "Do you have any bikes with electronic shifting?"
4. **Conversational Refinement**: Multi-turn dialogue with context
5. **Similarity Search**: "Something similar to the Specialized Tarmac SL7"

## Interpreting LLM Evaluations

The LLM evaluation considers:
- Relevance to query and conversation context
- Result diversity and coverage
- Ranking quality (best matches first)
- Suitability for voice interaction

## Best Practices

1. **Test with Real Queries**: Use actual customer queries for realistic results
2. **Consider Context**: Test multi-turn conversations to see context handling
3. **Multiple Runs**: Run tests multiple times to account for variability
4. **Different Accounts**: Test with different brands to see generalization

## Troubleshooting

### "Baseline index appears empty"

Run with `--setup-baseline` flag:
```bash
python run_search_comparison.py --account specialized.com --setup-baseline
```

### "Could not load system prompt from Langfuse"

The system will use a fallback prompt. To use brand-specific prompts:
1. Ensure Langfuse is configured
2. Check prompt name format: `{account}/full_instructions`

### Slow Performance

- Reduce `--max-products` for faster baseline ingestion
- Use smaller test scenarios
- Check your Pinecone connection

## Output Files

Results are saved to different directories based on test type:

### Voice-Realistic Tests
`voice_search_results/`:
- **`voice_comparison_results.json`**: Full conversation transcripts with search metrics
- **`voice_comparison_report.md`**: Analysis of voice interaction quality

### Standard Tests
`search_comparison_results/`:
- **`comparison_results.json`**: Detailed results in JSON format
- **`comparison_report.md`**: Human-readable comparison report

## Advanced Usage

### Custom Test Scenarios

Edit `_create_test_scenarios()` in `test_search_comparison.py` to add custom scenarios:

```python
SearchTestScenario(
    scenario_id="custom_test",
    description="Your custom test",
    conversation=[
        ConversationTurn("assistant", "How can I help?"),
        ConversationTurn("user", "Your test query")
    ],
    expected_product_types=["expected results"],
    evaluation_criteria={"focus": "your criteria"}
)
```

### Adjusting Comparison Parameters

Modify search parameters in `compare_searches()`:

```python
# Enhanced search parameters
enhanced_results, enhanced_metrics = await SearchService.unified_product_search(
    query=query,
    top_k=50,      # Increase candidate pool
    top_n=20,      # Return more results
    enable_reranking=True,
    # ... other parameters
)
```

## Next Steps

1. Run initial comparison tests
2. Analyze which approach works better for your use case
3. Fine-tune parameters based on results
4. Consider A/B testing in production
5. Monitor real user satisfaction metrics