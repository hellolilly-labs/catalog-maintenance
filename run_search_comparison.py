#!/usr/bin/env python3
"""
Interactive Search Comparison Runner

This script provides an interactive way to test and compare search approaches
for the voice assistant, with real-time results and visualizations.
"""

import asyncio
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import readline  # For better input experience

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from test_search_comparison import (
    SearchTestRunner, 
    BaselineSearchIndex, 
    ConversationSimulator,
    SearchComparator,
    ConversationTurn
)
from src.models.product_manager import ProductManager
from src.storage import get_account_storage_provider
from src.search.search_service import SearchService
from dataclasses import dataclass

# Define UserState locally for this test
@dataclass
class UserState:
    """Simple user state for testing."""
    user_id: str = "test_user"
    session_id: str = "test_session"
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}

console = Console()


class InteractiveSearchTester:
    """Interactive interface for search comparison testing."""
    
    def __init__(self, account: str):
        self.account = account
        self.storage_manager = get_account_storage_provider()
        self.baseline_index = BaselineSearchIndex(account)
        self.comparator = SearchComparator(account)
        self.simulator = ConversationSimulator(account)
        self.conversation_active = False
        
    async def setup(self):
        """Initialize all components."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing search systems...", total=3)
            
            # Initialize comparator
            await self.comparator.initialize()
            progress.update(task, advance=1, description="Initialized search comparator")
            
            # Load system prompt
            await self.simulator.load_system_prompt()
            progress.update(task, advance=1, description="Loaded system prompt")
            
            # Check baseline index
            await self.baseline_index.create_index()
            progress.update(task, advance=1, description="Setup complete!")
        
        console.print("[green]✅ All systems ready![/green]")
    
    async def check_baseline_index(self) -> bool:
        """Check if baseline index has products."""
        try:
            # Do a simple search to check if index has data
            results = await self.baseline_index.search("bike", top_k=1)
            return len(results) > 0
        except:
            return False
    
    async def ingest_baseline_products(self, max_products: int = 1000):
        """Ingest products into baseline index."""
        console.print(f"[yellow]Loading products for {self.account}...[/yellow]")
        
        product_manager = ProductManager(self.storage_manager)
        products = await product_manager.fetch_products(self.account, num_products=max_products)
        
        console.print(f"[green]Loaded {len(products)} products[/green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Ingesting products to baseline index...", total=1)
            await self.baseline_index.ingest_products(products)
            progress.update(task, advance=1, description="Ingestion complete!")
        
        console.print("[green]✅ Baseline index ready![/green]")
    
    async def run_interactive_mode(self):
        """Run interactive search comparison mode."""
        console.print(Panel.fit(
            f"[bold blue]Interactive Search Comparison[/bold blue]\n"
            f"Account: [yellow]{self.account}[/yellow]\n\n"
            f"Commands:\n"
            f"  [green]search <query>[/green] - Compare search results\n"
            f"  [green]chat[/green] - Start conversational search\n"
            f"  [green]scenarios[/green] - Run predefined test scenarios\n"
            f"  [green]report[/green] - Generate comparison report\n"
            f"  [green]help[/green] - Show this help\n"
            f"  [green]exit[/green] - Exit the program"
        ))
        
        while True:
            try:
                command = Prompt.ask("\n[bold blue]>>>[/bold blue]").strip()
                
                if not command:
                    continue
                
                if command.lower() == "exit":
                    break
                elif command.lower() == "help":
                    await self.show_help()
                elif command.lower().startswith("search "):
                    query = command[7:].strip()
                    await self.compare_single_search(query)
                elif command.lower() == "chat":
                    await self.run_chat_mode()
                elif command.lower() == "scenarios":
                    await self.run_test_scenarios()
                elif command.lower() == "report":
                    await self.generate_report()
                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    async def compare_single_search(self, query: str):
        """Compare search results for a single query."""
        console.print(f"\n[bold]Comparing search results for:[/bold] '{query}'")
        
        # Get current chat context
        chat_context = self.simulator.get_chat_context()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running searches...", total=1)
            
            comparison = await self.comparator.compare_searches(
                query=query,
                chat_context=chat_context,
                user_state=self.simulator.user_state
            )
            
            progress.update(task, advance=1, description="Complete!")
        
        # Display results
        self.display_comparison_results(comparison)
    
    def display_comparison_results(self, comparison):
        """Display comparison results in a formatted table."""
        # Performance metrics
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Enhanced", style="green")
        perf_table.add_column("Baseline", style="yellow")
        
        perf_table.add_row(
            "Response Time",
            f"{comparison.enhanced_time:.3f}s",
            f"{comparison.baseline_time:.3f}s"
        )
        perf_table.add_row(
            "Results Count",
            str(len(comparison.enhanced_results)),
            str(len(comparison.baseline_results))
        )
        perf_table.add_row(
            "Overlap Ratio",
            f"{comparison.overlap_ratio:.2%}",
            f"{comparison.overlap_ratio:.2%}"
        )
        perf_table.add_row(
            "Rank Correlation",
            f"{comparison.ranking_correlation:.3f}",
            f"{comparison.ranking_correlation:.3f}"
        )
        
        console.print(perf_table)
        
        # Top results comparison
        results_table = Table(title="Top 5 Results Comparison")
        results_table.add_column("#", style="dim", width=3)
        results_table.add_column("Enhanced Search", style="green", width=40)
        results_table.add_column("Score", style="green", width=8)
        results_table.add_column("Baseline Search", style="yellow", width=40)
        results_table.add_column("Score", style="yellow", width=8)
        
        for i in range(5):
            enhanced_name = ""
            enhanced_score = ""
            baseline_name = ""
            baseline_score = ""
            
            if i < len(comparison.enhanced_results):
                r = comparison.enhanced_results[i]
                enhanced_name = f"{r.name[:37]}..." if len(r.name) > 40 else r.name
                enhanced_score = f"{r.score:.3f}"
                
            if i < len(comparison.baseline_results):
                r = comparison.baseline_results[i]
                baseline_name = f"{r.name[:37]}..." if len(r.name) > 40 else r.name
                baseline_score = f"{r.score:.3f}"
            
            results_table.add_row(
                str(i + 1),
                enhanced_name,
                enhanced_score,
                baseline_name,
                baseline_score
            )
        
        console.print(results_table)
        
        # LLM Evaluation
        if comparison.llm_evaluation:
            console.print("\n[bold]LLM Evaluation:[/bold]")
            console.print(Panel(comparison.llm_evaluation['evaluation'], border_style="blue"))
    
    async def run_chat_mode(self):
        """Run conversational search mode."""
        console.print("\n[bold green]Entering chat mode[/bold green]")
        console.print("[dim]Type your messages, use /search to trigger comparison, /exit to leave chat[/dim]\n")
        
        self.conversation_active = True
        self.simulator.conversation_history = []
        
        # Start with assistant greeting
        greeting = "Hello! How can I help you find the perfect product today?"
        self.simulator.add_turn("assistant", greeting)
        console.print(f"[blue]Assistant:[/blue] {greeting}")
        
        while self.conversation_active:
            try:
                user_input = Prompt.ask("[green]You[/green]")
                
                if user_input.lower() == "/exit":
                    break
                elif user_input.lower() == "/search":
                    # Find last user message that looks like a search
                    search_query = self._extract_search_query()
                    if search_query:
                        await self.compare_single_search(search_query)
                    else:
                        console.print("[yellow]No clear search query found in conversation[/yellow]")
                else:
                    # Add to conversation
                    self.simulator.add_turn("user", user_input)
                    
                    # Simulate assistant response (simplified)
                    if self._is_search_query(user_input):
                        response = "I'll help you find that. Let me search our catalog..."
                        console.print(f"[blue]Assistant:[/blue] {response}")
                        self.simulator.add_turn("assistant", response)
                        
                        # Auto-trigger search comparison
                        await self.compare_single_search(user_input)
                    else:
                        response = "I understand. Could you tell me more about what you're looking for?"
                        console.print(f"[blue]Assistant:[/blue] {response}")
                        self.simulator.add_turn("assistant", response)
                        
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /exit to leave chat[/yellow]")
        
        self.conversation_active = False
        console.print("[dim]Left chat mode[/dim]")
    
    def _is_search_query(self, message: str) -> bool:
        """Check if message is a search query."""
        search_indicators = [
            'looking for', 'need', 'want', 'show me', 'find',
            'search', 'recommend', 'suggest', 'what about',
            'do you have', 'bike', 'product', 'under', 'over'
        ]
        return any(ind in message.lower() for ind in search_indicators)
    
    def _extract_search_query(self) -> Optional[str]:
        """Extract the most recent search query from conversation."""
        for turn in reversed(self.simulator.conversation_history):
            if turn.speaker == "user" and self._is_search_query(turn.message):
                return turn.message
        return None
    
    async def run_test_scenarios(self):
        """Run predefined test scenarios."""
        console.print("\n[bold]Running test scenarios...[/bold]")
        
        runner = SearchTestRunner(self.account)
        await runner.setup()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running scenarios...", total=len(runner.test_scenarios))
            
            results = []
            for scenario in runner.test_scenarios:
                progress.update(task, description=f"Testing: {scenario.description}")
                
                # Run scenario
                simulation = await runner.simulator.simulate_conversation(scenario)
                
                for search_info in simulation['search_queries']:
                    comparison = await runner.comparator.compare_searches(
                        query=search_info['query'],
                        chat_context=search_info['context'],
                        user_state=runner.simulator.user_state
                    )
                    
                    results.append({
                        'scenario': scenario,
                        'comparison': comparison
                    })
                
                progress.update(task, advance=1)
        
        # Display summary
        console.print("\n[bold green]Scenario Test Results:[/bold green]")
        
        summary_table = Table(title="Test Scenario Summary")
        summary_table.add_column("Scenario", style="cyan")
        summary_table.add_column("Query", style="white")
        summary_table.add_column("Enhanced Time", style="green")
        summary_table.add_column("Baseline Time", style="yellow")
        summary_table.add_column("Overlap", style="magenta")
        
        for result in results:
            scenario = result['scenario']
            comparison = result['comparison']
            
            summary_table.add_row(
                scenario.scenario_id,
                comparison.query[:40] + "..." if len(comparison.query) > 40 else comparison.query,
                f"{comparison.enhanced_time:.3f}s",
                f"{comparison.baseline_time:.3f}s",
                f"{comparison.overlap_ratio:.2%}"
            )
        
        console.print(summary_table)
        
        # Save results
        runner.results = results
        await runner.save_results()
        console.print("\n[green]Results saved to search_comparison_results/[/green]")
    
    async def generate_report(self):
        """Generate and display comparison report."""
        console.print("[yellow]Generating report...[/yellow]")
        
        # This would aggregate all results and generate a comprehensive report
        # For now, just show a message
        console.print("[green]Report saved to search_comparison_results/comparison_report.md[/green]")
    
    async def show_help(self):
        """Show detailed help information."""
        help_text = """
# Search Comparison Tool Help

## Commands:

### search <query>
Compare search results for a specific query using both enhanced and baseline approaches.
Example: `search mountain bike under $2000`

### chat
Enter conversational mode to test search in a natural dialogue context.
- Type messages naturally
- Use `/search` to trigger comparison of the last search-like query
- Use `/exit` to leave chat mode

### scenarios
Run predefined test scenarios that cover common search patterns:
- Basic category search
- Price-constrained search  
- Feature-specific search
- Multi-turn conversations
- Similarity search

### report
Generate a comprehensive comparison report with all test results.

### exit
Exit the program.

## Tips:
- The enhanced search uses separate dense/sparse indexes with reranking
- The baseline search uses a single dense index with Product.to_markdown()
- Watch for differences in result relevance, diversity, and response time
"""
        console.print(Markdown(help_text))


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive search comparison tool")
    parser.add_argument("--account", default="specialized.com", help="Brand account")
    parser.add_argument("--setup-baseline", action="store_true", help="Set up baseline index")
    parser.add_argument("--max-products", type=int, default=1000, help="Max products for baseline")
    
    args = parser.parse_args()
    
    console.print(f"[bold cyan]Voice Assistant Search Comparison Tool[/bold cyan]")
    console.print(f"Account: [yellow]{args.account}[/yellow]\n")
    
    tester = InteractiveSearchTester(args.account)
    
    try:
        # Setup
        await tester.setup()
        
        # Check if we need to set up baseline
        has_baseline = await tester.check_baseline_index()
        
        if not has_baseline or args.setup_baseline:
            if not has_baseline:
                console.print("[yellow]Baseline index appears empty[/yellow]")
            
            if Confirm.ask("Set up baseline index with products?"):
                await tester.ingest_baseline_products(args.max_products)
        
        # Run interactive mode
        await tester.run_interactive_mode()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise
    
    console.print("\n[green]Thank you for using the search comparison tool![/green]")


if __name__ == "__main__":
    asyncio.run(main())