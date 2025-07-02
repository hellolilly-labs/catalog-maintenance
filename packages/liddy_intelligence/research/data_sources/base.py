"""
Base Data Source Strategy Interface

Defines the contract for all data source strategies in the research pipeline.
Each strategy handles a specific type of data collection (web search, product catalog, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
from dataclasses import dataclass


@dataclass
class DataGatheringContext:
    """Context information for data gathering operations"""
    brand_domain: str
    researcher_name: str
    phase_name: str
    additional_context: Dict[str, Any] = None
    
    @property
    def brand_name(self) -> str:
        """Extract brand name from domain"""
        return self.brand_domain.replace('.com', '').replace('.', ' ').title()


@dataclass 
class DataGatheringResult:
    """Result of data gathering operation"""
    results: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    successful_searches: int = 0
    failed_searches: int = 0
    ssl_errors: int = 0
    metadata: Dict[str, Any] = None


class DataSource(ABC):
    """
    Abstract base class for data source strategies.
    
    Each concrete implementation handles a specific type of data collection
    while providing a consistent interface for the research pipeline.
    """
    
    @abstractmethod
    async def gather(self, queries: List[Union[str, Dict[str, Any]]], context: DataGatheringContext) -> DataGatheringResult:
        """
        Gather data based on queries and context.
        
        Args:
            queries: List of search queries or data identifiers
            context: Context information for the data gathering operation
            
        Returns:
            DataGatheringResult containing collected data and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this data source is available and properly configured.
        
        Returns:
            True if the data source can be used, False otherwise
        """
        pass
    
    def get_source_type(self) -> str:
        """
        Get the type identifier for this data source.
        
        Returns:
            String identifier for this data source type
        """
        return self.__class__.__name__.lower().replace('datasource', '')