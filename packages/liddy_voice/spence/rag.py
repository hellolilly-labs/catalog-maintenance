"""
Pinecone RAG (Retrieval-Augmented Generation) System

This module provides a production-ready RAG implementation using Pinecone as the vector store.
It supports multi-namespace search, advanced filtering, and automatic reranking of results.
"""

import os
import re
import asyncio
import json
import logging
import time
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Union, Any, Awaitable, Tuple, cast

# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone, SearchQuery, SearchRerank, RerankModel
from liddy_voice.spence.account_manager import get_account_manager

logger = logging.getLogger(__name__)

class PineconeRAG:
    """
    Pinecone-based RAG system for retrieving relevant information based on user queries.
    
    This implementation supports:
    - Multi-namespace search across products, information, and brand content
    - Intent-based query handling (informational vs. product queries)
    - Advanced metadata filtering with array field support
    - Automatic reranking of results based on relevance
    - Detailed instrumentation and performance tracking
    """
    
    def __init__(
        self, 
        account: str,
        index_name: str, 
        model_name: str = "llama-text-embed-v2",  # Use integrated Pinecone embedding model
        namespace: str = "specialized",
        connection_timeout: float = 10.0,
        debug: bool = False
    ):
        """
        Initialize the RAG system with Pinecone
        
        Args:
            index_name: Name of the Pinecone index
            model_name: Embedding model to use
            namespace: Default Pinecone namespace
            connection_timeout: Timeout in seconds for connection attempts
            
        Raises:
            ValueError: If PINECONE_API_KEY is missing
            ConnectionError: If connection to Pinecone fails or times out
        """
        api_key = os.getenv('PINECONE_API_KEY')
        
        if not api_key:
            logger.error("PINECONE_API_KEY environment variable is required")
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.account = account
        self.index_name = index_name
        self.namespace = namespace
        self.model = model_name
        self.api_key = api_key
        self.connection_timeout = connection_timeout
        
        # kick off the init asyncio task
        asyncio.create_task(self.init())

    async def init(self):
        account_manager = await get_account_manager(account=self.account)
        index_name, embedding_model = account_manager.get_rag_details()
            
        try:
            # Initialize Pinecone client
            start_time = time.time()
            self.pc = Pinecone(api_key=self.api_key)
            self.connection_time = datetime.now().isoformat()
            self.initialization_timestamp = int(time.time())
            
            # Test connection and index availability
            self.index = self.pc.Index(index_name)
            
            # Check for timeout
            if time.time() - start_time > self.connection_timeout:
                raise ConnectionError(f"Connection to Pinecone timed out after {self.connection_timeout}s")
                
            stats = self.index.describe_index_stats()
            conn_time = time.time() - start_time
            
            # Validate namespaces
            self.available_namespaces = []
            if hasattr(stats, 'namespaces') and stats.namespaces:
                self.available_namespaces = list(stats.namespaces.keys())
                logger.debug(f"Available namespaces: {', '.join(self.available_namespaces)}")
                
            # Check vector count for data health
            total_vectors = sum(ns.vector_count for ns in stats.namespaces.values()) if hasattr(stats, 'namespaces') else 0
            logger.debug(f"Connected to Pinecone index '{index_name}', {total_vectors} total vectors, response time: {conn_time:.2f}s")
            
            # Save index statistics for health checks
            self.stats = {
                "total_vectors": total_vectors,
                "namespaces": self.available_namespaces,
                "connection_time": conn_time
            }
            
            # Cache common intents and patterns for faster processing
            self._info_keywords = [
                "how to", "what is", "explain", "guide", "help", "info", "information", 
                "description", "tell me about", "learn", "understand", "details"
            ]
            
            self._sizing_keywords = [
                "size", "sizing", "fit", "measurement", "inches", "cm", "centimeters",
                "height", "inseam", "small", "medium", "large", "xlarge"
            ]
            
            # Set up request tracking 
            self._requests_processed = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._total_query_time = 0.0
            
            # Create asyncio lock for concurrent operations
            self._lock = asyncio.Lock()
            
            # Set verbose logging flag
            self._verbose_logging = logger.level <= logging.DEBUG
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {str(e)}")
            raise ConnectionError(f"Failed to connect to Pinecone: {str(e)}") from e

    async def encode(self, text: Union[str, List[str]]) -> List[float]:
        """
        Convert text to embeddings using Pinecone's integrated embedding service
        
        Args:
            text: String or list of strings to encode
            
        Returns:
            List of embeddings (list of floats)
        """
        if isinstance(text, str):
            text = [text]
            
        try:
            start_time = time.time()
            
            # Use Pinecone's integrated inference API with correct parameter name
            response = self.pc.inference.embed(
                model=self.model,
                inputs=text,  # Correct parameter name
                parameters={
                    "input_type": "query",  # For search queries
                    "dimension": 2048
                }
            )
            
            # Extract embeddings from response
            embeddings = response.data
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Encoded {len(text)} texts in {elapsed_time:.2f}s")
            
            # Track request stats
            async with self._lock:
                self._requests_processed += 1
                self._successful_requests += 1
                self._total_query_time += elapsed_time
            
            return embeddings[0] if len(text) == 1 else embeddings
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            # Track failed request
            async with self._lock:
                self._requests_processed += 1
                self._failed_requests += 1
                
            # Return empty embedding in case of error (will lead to no matches)
            return [] if len(text) == 1 else [[] for _ in text]



    async def search(
        self, 
        query: str, 
        top_k: int = 10, 
        top_n: int = 5, 
        min_score: float = 0.35, 
        min_n: int = 0,
        namespace: Optional[str] = None,
        timeout: float = 15.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity
        
        Args:
            query: Search query text
            top_k: Number of initial results to retrieve
            top_n: Number of results to return after reranking
            min_score: Minimum relevance score (0-1)
            namespace: Namespace to search (defaults to self.namespace)
            timeout: Maximum time in seconds to wait for search completion
            
        Returns:
            List of document dictionaries with id, text, metadata, and score
            
        Raises:
            ValueError: If query is empty
            TimeoutError: If search takes too long
            RuntimeError: If search fails for other reasons
        """
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Empty query provided to search")
            return []
            
        ns = namespace or self.namespace
        logger.debug(f"Searching in namespace '{ns}' for: {query[:50]}...")
        
        # Convert the query into a numerical vector
        try:
            overall_start_time = time.time()
            # query_embedding = await self.encode(query)
            
            # if not query_embedding:
            #     logger.warning(f"Failed to encode query: {query[:50]}")
            #     return []
                
            # # Check for timeout
            # if time.time() - overall_start_time > timeout:
            #     logger.error(f"Search timed out after encoding: {timeout}s")
            #     raise TimeoutError(f"Search timed out after {timeout}s")

            # Search the index
            search_start = time.time()
            results = self.index.search_records(
                namespace=ns, 
                query=SearchQuery(
                    inputs={
                        "text": query
                    }, 
                    top_k=top_k
                ),
                rerank=SearchRerank(
                    model=RerankModel.Cohere_Rerank_3_5, #  Bge_Reranker_V2_M3, # : "bge-reranker-v2-m3",
                    rank_fields=["text"],
                    top_n=top_n
                ),
                fields=["metadata"],
            )

            # results = self.index.query(
            #     namespace=ns,
            #     vector=query_embedding,
            #     top_k=top_k,
            #     include_values=False,
            #     include_metadata=True
            # )
            search_time = time.time() - search_start

            # Check for timeout 
            if time.time() - overall_start_time > timeout:
                logger.error(f"Search timed out after vector search: {timeout}s")
                raise TimeoutError(f"Search timed out after {timeout}s")

            # Process search results
            filtered_results = []
            # documents = []
            for result in results.result.hits:
                # Extract text from metadata source field
                metadata = result.fields.get('metadata')
                if result._score >= min_score or len(filtered_results) < min_n:
                    filtered_results.append({
                        "id": result._id, 
                        "metadata": metadata,
                        "score": result._score
                    })
                
                # if 'source' in result.metadata:
                #     documents.append({
                #         "id": result.id, 
                #         "text": result.metadata['source'],
                #         "metadata": result.metadata,
                #         "score": result.score if hasattr(result, 'score') else 1.0
                #     })
                # elif 'text' in result.metadata:
                #     documents.append({
                #         "id": result.id, 
                #         "text": result.metadata['text'],
                #         "metadata": result.metadata,
                #         "score": result.score if hasattr(result, 'score') else 1.0
                #     })

            # Only rerank if we have enough results
            # if documents and len(documents) > 1:
            if filtered_results and len(filtered_results) > 1:
                # rerank_start = time.time()
                # reranked_results = self.pc.inference.rerank(
                #     model="bge-reranker-v2-m3",
                #     query=query,
                #     documents=[{"id": doc["id"], "text": doc["text"]} for doc in documents],
                #     top_n=min(top_n, len(documents)),
                #     return_documents=True,
                #     parameters={"truncate": "END"}
                # )
                # rerank_time = time.time() - rerank_start
                    
                # # Check for timeout
                # if time.time() - overall_start_time > timeout:
                #     logger.error(f"Search timed out after reranking: {timeout}s")
                #     raise TimeoutError(f"Search timed out after {timeout}s")
                
                # # Filter out results below threshold
                # filtered_results = []
                # for match in reranked_results.data:
                #     if match.score >= min_score:
                #         # Find the original document with metadata
                #         for doc in documents:
                #             if doc["id"] == match.id:
                #                 filtered_results.append({
                #                     'id': match.id,
                #                     'score': match.score,
                #                     'text': match.document['text'],
                #                     'metadata': doc.get('metadata', {})
                #                 })
                #                 break
                
                total_time = time.time() - overall_start_time
                # logger.debug(f"RAG pipeline times: embed={total_time-search_time-rerank_time:.2f}s, search={search_time:.2f}s, rerank={rerank_time:.2f}s, total={total_time:.2f}s")
                logger.debug(f"RAG pipeline times: embed={total_time-search_time:.2f}s, search={search_time:.2f}s, total={total_time:.2f}s")
                # logger.debug(f"Found {len(filtered_results)}/{len(documents)} relevant documents after reranking")
                logger.debug(f"Found {len(filtered_results)} relevant documents after reranking")
                
                # Update stats
                async with self._lock:
                    self._requests_processed += 1
                    self._successful_requests += 1
                    self._total_query_time += total_time
                
                return filtered_results
            
            # If no reranking, return original results
            total_time = time.time() - overall_start_time
            logger.debug(f"RAG pipeline times: embed={total_time-search_time:.2f}s, search={search_time:.2f}s, total={total_time:.2f}s")
            logger.debug(f"Found {len(filtered_results)} documents (no reranking)")
            
            # Update stats
            async with self._lock:
                self._requests_processed += 1
                self._successful_requests += 1
                self._total_query_time += total_time

            return filtered_results
            # return [
            #     {
            #         'id': doc["_id"],
            #         'score': doc.get("_score", 1.0),
            #         # 'text': doc["text"],
            #         'metadata': doc.get('metadata', {})
            #     }
            #     for doc in filtered_results
            # ]
        except TimeoutError:
            # Already logging timeout errors above
            async with self._lock:
                self._requests_processed += 1
                self._failed_requests += 1
            return []
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update stats
            async with self._lock:
                self._requests_processed += 1
                self._failed_requests += 1
            
            return []
    
    async def search_with_filter(
        self, 
        query: str, 
        filter_dict: Optional[Dict[str, Any]] = None, 
        top_k: int = 10, 
        top_n: int = 5, 
        min_n: int = 0,
        min_score: float = 0.7,
        namespace: Optional[str] = None,
        alpha: float = 0.5,  # Hybrid search weighting parameter
        timeout: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with metadata filters using hybrid search
        
        Args:
            query: Search query text
            filter_dict: Dictionary of metadata filters
            top_k: Number of initial results to retrieve
            top_n: Number of results to return after reranking
            min_n: Minimum number of results to return
            min_score: Minimum relevance score (0-1)
            namespace: Namespace to search (defaults to self.namespace)
            alpha: Weight between dense (0) and sparse (1) vectors
            timeout: Maximum time in seconds to wait for search completion
            
        Returns:
            List of document dictionaries with id, text, metadata, and score
        """
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Empty query provided to search_with_filter")
            return []
            
        ns = namespace or self.namespace
        logger.debug(f"Filtered search in namespace '{ns}' for: {query[:50]}...")
        
        # Convert the query into a numerical vector
        try:
            overall_start_time = time.time()
            
            # Set a deadline for the entire operation
            deadline = overall_start_time + timeout
            
            # Determine query intent - information or product
            has_info_intent = any(keyword in query.lower() for keyword in self._info_keywords)
            has_sizing_intent = any(keyword in query.lower() for keyword in self._sizing_keywords)
            
            # Build filter expression
            filter_expr = None
            if filter_dict:
                filter_conditions = []
                
                # Handle content type filtering based on query intent
                if has_info_intent or has_sizing_intent:
                    # If asking for information, don't restrict content type
                    pass
                elif "content_type" not in filter_dict and ns == "products":
                    # Default to products for product-like queries in the products namespace
                    filter_conditions.append({"content_type": {"$eq": "product"}})
                
                # Special handling for sizing queries
                if has_sizing_intent and "category" not in filter_dict:
                    filter_conditions.append({"category": {"$eq": "sizing_guide"}})
                    
                # Add all other filters
                for key, value in filter_dict.items():
                    if value is None:
                        continue
                        
                    if isinstance(value, list):
                        if not value:  # Skip empty lists
                            continue
                        # For array fields, use $in operator
                        filter_conditions.append({key: {"$in": value}})
                    elif key in ["colors", "subcategories", "all_categories", "sizes"] and isinstance(value, str):
                        # For array fields with partial match
                        filter_conditions.append({key: {"$in": [value]}})
                    elif isinstance(value, (str, int, float, bool)):
                        # For scalar values
                        filter_conditions.append({key: {"$eq": value}})
                
                if filter_conditions:
                    filter_expr = {"$and": filter_conditions}
            
            # Log the filter expression for debugging
            if self._verbose_logging and filter_expr:
                logger.debug(f"Using filter expression: {filter_expr}")
            
            # Check if we're already past the deadline
            if time.time() > deadline:
                logger.error(f"Search timed out after filter creation: {timeout}s")
                raise TimeoutError(f"Search timed out after {timeout}s")
    
            # Calculate remaining time for the search operation
            remaining_time = max(0.1, deadline - time.time())
            
            # Search with both dense and sparse vectors (hybrid search)
            search_start = time.time()
            
            # Wrap search in wait_for with remaining timeout
            try:
                search_task = self.index.search_records(
                    namespace=namespace, 
                    query=SearchQuery(
                        inputs={"text": query}, 
                        top_k=top_k,
                        filter=filter_expr,
                    ),
                    rerank=SearchRerank(
                        model=RerankModel.Cohere_Rerank_3_5,
                        rank_fields=["text"],
                        top_n=top_n
                    ),
                    fields=["metadata","text"],
                )
                
                # For synchronous operations, we can't use asyncio.wait_for
                # Instead, check if we're past deadline after the operation
                results = search_task
                
            except Exception as e:
                logger.error(f"Search operation failed: {e}")
                raise
                
            search_time = time.time() - search_start
            
            # Check if we've exceeded the deadline
            if time.time() > deadline:
                logger.error(f"Search timed out after vector search: {timeout}s")
                raise TimeoutError(f"Search timed out after {timeout}s")
            
            # Extract documents for reranking
            filtered_results = []
            documents = []
            for result in results.result.hits:
                # Extract text from metadata or source field
                text = result.fields.get('text', '')
                if not text:  # Fallback to text field
                    text = result.get('metadata')
                metadata = json.loads(result.fields.get('metadata') or '{}')
                    
                if result._score >= min_score or len(filtered_results) < min_n:
                    documents.append({
                        "id": result._id, 
                        "text": text,
                        "metadata": metadata,
                        "score": result._score
                    })
                    filtered_results.append({
                        "id": result._id, 
                        "text": text,
                        "metadata": metadata,
                        "score": result._score
                    })
            
            # Calculate times and log
            total_time = time.time() - overall_start_time
            logger.debug(f"Filtered RAG pipeline times: embed={total_time-search_time:.2f}s, search={search_time:.2f}s, total={total_time:.2f}s")
            logger.debug(f"Found {len(filtered_results)} relevant documents")
            
            # Update stats
            async with self._lock:
                self._requests_processed += 1
                self._successful_requests += 1
                self._total_query_time += total_time
            
            return filtered_results
            
        except TimeoutError:
            # Already logged above
            async with self._lock:
                self._requests_processed += 1
                self._failed_requests += 1
            return []
            
        except asyncio.CancelledError:
            logger.warning("Search operation was cancelled")
            async with self._lock:
                self._requests_processed += 1
                self._failed_requests += 1
            return []
            
        except Exception as e:
            logger.error(f"Error in filtered search: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update stats
            async with self._lock:
                self._requests_processed += 1
                self._failed_requests += 1
            
            return []

    async def multi_namespace_search(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespaces: Optional[List[str]] = None,
        top_k: int = 5,
        min_score: float = 0.3,
        timeout: float = 30.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple namespaces in parallel
        
        Args:
            query: Search query text
            filter_dict: Dictionary of metadata filters (applied to products namespace)
            namespaces: List of namespaces to search (defaults to ["products", "information", "brand"])
            top_k: Number of results to retrieve per namespace
            min_score: Minimum relevance score (0-1)
            timeout: Maximum time to wait for all searches
            
        Returns:
            Dictionary mapping namespace to search results
        """
        if not namespaces:
            namespaces = ["products", "information", "brand"]
            
        # Create search tasks for each namespace
        search_tasks = []
        for namespace in namespaces:
            # Use specific filter for products namespace only
            ns_filter = filter_dict if namespace == "products" else {}
            task = self.search_with_filter(
                query=query, 
                filter_dict=ns_filter,
                namespace=namespace, 
                top_k=top_k,
                min_score=min_score
            )
            search_tasks.append((namespace, task))
        
        # Execute all searches in parallel with timeout
        results = {}
        try:
            # Run all tasks with timeout
            pending_tasks = {asyncio.create_task(task): ns for ns, task in search_tasks}
            done_tasks = set()
            
            start_time = time.time()
            while pending_tasks and time.time() - start_time < timeout:
                done, pending = await asyncio.wait(
                    pending_tasks.keys(), 
                    timeout=min(1.0, timeout - (time.time() - start_time)),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    namespace = pending_tasks[task]
                    try:
                        results[namespace] = task.result()
                    except Exception as e:
                        logger.error(f"Error searching {namespace} namespace: {str(e)}")
                        results[namespace] = []
                    
                    # Remove from pending
                    del pending_tasks[task]
                    done_tasks.add(task)
            
            # Handle any remaining tasks that timed out
            for task, namespace in pending_tasks.items():
                task.cancel()
                logger.warning(f"Search in {namespace} namespace timed out")
                results[namespace] = []
                
        except asyncio.CancelledError:
            logger.warning("Multi-namespace search was canceled")
        except Exception as e:
            logger.error(f"Error in multi-namespace search: {str(e)}")
        
        # Ensure all namespaces have entries in results
        for namespace in namespaces:
            if namespace not in results:
                results[namespace] = []
                
        return results
    
    async def search_namespaces(
        self,
        query: str,
        namespaces: Optional[List[str]] = None,
        top_k: int = 10,
        top_n: int = 5,
        min_n: int = 0,
        min_score: float = 0.0,
        timeout: float = 5.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Simple parallel search across multiple namespaces using basic search with strict timeout
        """
        if not namespaces:
            namespaces = ["products", "information", "brand"]
            
        # Create tasks for each namespace
        tasks = {}
        for namespace in namespaces:
            # Create the search task but don't start it yet
            search_task = self.search(
                query=query, 
                namespace=namespace, 
                top_k=top_k,
                top_n=top_n,
                min_n=min_n,
                min_score=min_score,
                timeout=timeout/2  # Give each search less time than the total timeout
            )
            # Wrap with a timeout to force cancellation
            task = asyncio.create_task(
                asyncio.wait_for(search_task, timeout=timeout)
            )
            tasks[namespace] = task
        
        # Execute all searches truly in parallel with proper timeout handling
        results = {}
        start_time = time.time()
        
        # Initialize all namespace results to empty lists
        for namespace in namespaces:
            results[namespace] = []
        
        try:
            # Wait for all tasks with a timeout
            completed_tasks, pending_tasks = await asyncio.wait(
                tasks.values(), 
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Process completed tasks
            for namespace, task in tasks.items():
                if task in completed_tasks:
                    try:
                        results[namespace] = task.result()
                    except asyncio.TimeoutError:
                        logger.warning(f"Search in namespace '{namespace}' timed out")
                    except Exception as e:
                        logger.error(f"Error searching namespace '{namespace}': {str(e)}")
                else:
                    # Cancel any pending tasks
                    if not task.done():
                        task.cancel()
                        logger.warning(f"Search in namespace '{namespace}' was cancelled due to timeout")
        
        except asyncio.TimeoutError:
            logger.warning(f"Multi-namespace search timed out after {timeout}s")
            # Cancel all tasks
            for namespace, task in tasks.items():
                if not task.done():
                    task.cancel()
                    
        except Exception as e:
            logger.error(f"Error in multi-namespace search: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Log timing
        total_time = time.time() - start_time
        result_counts = {ns: len(res) for ns, res in results.items()}
        logger.debug(f"Multi-namespace search completed in {total_time:.2f}s with results: {result_counts}")
        
        return results
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get health information about the RAG system
        
        Returns:
            Dict with health metrics and statistics
        """
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            
            # Calculate metrics
            avg_query_time = 0
            if self._requests_processed > 0:
                avg_query_time = self._total_query_time / self._requests_processed
                
            success_rate = 0
            if self._requests_processed > 0:
                success_rate = (self._successful_requests / self._requests_processed) * 100
                
            # Compile health report
            return {
                "status": "healthy",
                "index_name": self.index_name,
                "default_namespace": self.namespace,
                "available_namespaces": self.available_namespaces,
                "uptime_seconds": int(time.time()) - self.initialization_timestamp,
                "requests": {
                    "total": self._requests_processed,
                    "successful": self._successful_requests,
                    "failed": self._failed_requests,
                    "success_rate": success_rate,
                    "avg_query_time": avg_query_time
                },
                "index_stats": {
                    "total_vectors": sum(ns.vector_count for ns in stats.namespaces.values()) if hasattr(stats, 'namespaces') else 0,
                    "namespaces": {name: ns.vector_count for name, ns in stats.namespaces.items()} if hasattr(stats, 'namespaces') else {}
                }
            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "requests_processed": self._requests_processed,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests
            }
    
    @property
    def verbose_logging(self) -> bool:
        """Check if verbose logging is enabled"""
        return self._verbose_logging

    def _generate_sparse_vector(self, text: str):
        """Generate a sparse vector representation for the query text"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Count token frequencies
        token_counts = {}
        for token in tokens:
            if len(token) > 2:  # Skip very short tokens
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Create sparse indices and values
        indices = []
        values = []
        
        vocabulary_size = 100000  # Large sparse dimension size
        for token, count in token_counts.items():
            index = hash(token) % vocabulary_size
            indices.append(index)
            values.append(float(count))
        
        return {"indices": indices, "values": values}

    async def search_ranked(
        self,
        query: str,
        namespaces: Optional[List[str]] = None,
        top_k: int = 15,
        top_n: int = 5,
        min_score: float = 0.3,
        min_n: int = 0,
        timeout: float = 15.0
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple namespaces and rank all results by score
        
        Args:
            query: Search query text
            namespaces: List of namespaces to search (defaults to ["products", "information", "brand"])
            top_k: Number of results to retrieve per namespace
            top_n: Total number of results to return after ranking
            min_score: Minimum relevance score (0-1)
            min_n: Minimum number of results to return if available
            timeout: Maximum time to wait for all searches
            
        Returns:
            Combined and ranked list of search results from all namespaces
        """
        if not namespaces:
            namespaces = ["products", "information", "brand"]
            
        # Search all namespaces
        namespace_results = await self.search_namespaces(
            query=query,
            namespaces=namespaces,
            top_k=top_k,
            top_n=top_n,
            min_score=min_score,
            min_n=min_n,
            timeout=timeout
        )
        
        # Combine and tag all results with their namespace
        all_results = []
        for namespace, results in namespace_results.items():
            for result in results:
                result['namespace'] = namespace
                all_results.append(result)
        
        # Sort all results by score (highest first)
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Apply min_score filter
        filtered_results = [r for r in all_results if r.get('score', 0) >= min_score]
        
        # Apply top_n filter but ensure we have min_n results if possible
        if min_n > 0 and len(filtered_results) < min_n and len(all_results) >= min_n:
            # If we don't have enough results after filtering, take the top min_n results
            # regardless of min_score
            final_results = all_results[:min_n]
        else:
            # Otherwise just apply top_n to filtered results
            final_results = filtered_results[:top_n]
        
        # Print detailed results
        print(f"\n===== RANKED SEARCH RESULTS =====")
        print(f"Query: '{query}'")
        print(f"Parameters: top_k={top_k}, top_n={top_n}, min_score={min_score}, min_n={min_n}")
        print(f"Found {len(all_results)} total results, {len(filtered_results)} above min_score")
        print(f"Returning top {len(final_results)} results\n")
        
        # # Print each result with details
        # for i, result in enumerate(final_results):
        #     namespace = result.get('namespace', 'unknown')
        #     score = result.get('score', 0)
            
        #     print(f"[{i+1}] Score: {score:.4f} - Namespace: {namespace}")
            
        #     # Parse metadata into a Python dict if it's a JSON string
        #     metadata = result.get('metadata', {})
        #     if isinstance(metadata, str):
        #         try:
        #             metadata = json.loads(metadata)
        #         except:
        #             pass
                    
        #     # Print different details based on namespace
        #     if namespace == "products":
        #         print(f"    Product {metadata.get('id', 'N/A')}: {metadata.get('name', 'Unknown')}")
        #         print(f"    Category: {', '.join(metadata.get('categories', ['Unknown']))}")
        #         print(f"    Price: {metadata.get('original_price', 'Unknown')}")
        #     else:
        #         print(f"    Title: {metadata.get('title', 'Unknown')}")
        #         print(f"    Category: {metadata.get('category', 'Unknown')}")
                
        #     # Add a separator between results
        #     print("")
        
        return final_results

    @staticmethod
    async def rag_query_pinecone(query: str, account:str=None, index_name: Optional[str]=None, embedding_model: Optional[str]=None, namespaces: Optional[List[str]]=None, top_k: int = 25, top_n: int = 10, min_score: float = 0.015, min_n: int = 0, use_ranked: bool = False) -> List[dict]:
        """ Perform a RAG query using Pinecone. """
        if not hasattr(PineconeRAG, "_instance"):
            account_manager = await get_account_manager(account=account)
            index_name, embedding_model = account_manager.get_rag_details()
            
            if not index_name:
                raise ValueError("No index name provided. Please set up the Pinecone index name in your account settings.")
            
            rag = PineconeRAG(index_name=index_name, model_name=embedding_model, debug=False, account=account)
            PineconeRAG._instance = rag
        else:
            rag = PineconeRAG._instance

        try:
            if use_ranked:
                # Use the enhanced query for search
                start_time = time.time()
                search_ranked_results = await rag.search_ranked(
                    query=query,
                    top_k=top_k,
                    top_n=top_n,
                    min_score=min_score,
                    min_n=min_n
                )
                end_time = time.time()
                logger.debug(f"Time taken to perform RAG query: {end_time - start_time} seconds")
                # print(f"\n  Found {len(search_ranked_results)} total results across all namespaces")
                
                if search_ranked_results:
                    # map the metadata from each result to a dictionary
                    for i, result in enumerate(search_ranked_results):
                        # get the metadata from the result
                        metadata = result.get('metadata', {})
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                pass
                        # check to see if the metadata is a string and try to parse it as JSON
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                pass
                        # update the result with the parsed metadata
                        result['metadata'] = metadata
                
                return search_ranked_results

            # Use the standard query for search        
            start_time = time.time()
            search_results = await rag.search(
                query=query,
                namespace="products",
                top_k=top_k,
                top_n=top_n,
                min_score=min_score,
                min_n=min_n
            )
            end_time = time.time()
            logger.debug(f"Time taken to perform RAG search: {end_time - start_time} seconds")
            
            # compare the results
            print(f"\n  Found {len(search_results)} total results across all namespaces")
            if search_results:
                # map the metadata from each result to a dictionary
                for i, result in enumerate(search_results):
                    # get the metadata from the result
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            pass
                    # check to see if the metadata is a string and try to parse it as JSON
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            pass
                    # update the result with the parsed metadata
                    result['metadata'] = metadata

            # # print out the results with the: _id, _score, metadata.get("name")
            # for i, result in enumerate(search_results):
            #     print(f"  {i+1}. ID: {result['id']}, Score: {result['score']:.4f}")
            #     metadata = result.get('metadata', {})
            #     if isinstance(metadata, str):
            #         metadata = json.loads(metadata)
                
            #     if metadata.get('content_type') == 'product':
            #         print(f"     Product: {metadata.get('name', 'Unknown')}")
            #         categories = metadata.get('categories', [])
            #         if categories:
            #             print(f"     Categories: {', '.join(categories)}")
            #         print(f"     Price: {metadata.get('original_price', 'Unknown')}")
            #     else:
            #         print(f"     Title: {metadata.get('title', 'Unknown')}")
            
            return search_results

        except Exception as e:
            print(f"  Error testing query: {e}")
            import traceback
            traceback.print_exc()




# async def _enrich_system_prompt_with_rag(agent: Agent) -> None:
#     """Enrich system prompt with RAG results from multiple specialized datastores"""
#     # global user_id
    
#     chat_ctx = agent.chat_ctx.copy()
#     async with _chat_ctx_lock:
#         messages = chat_ctx.items.copy()
    
#     # Get last user message
#     user_msg = next((m for m in reversed(messages) if m.role == "user"), None)
#     if not user_msg:
#         return
    
#     # Get conversation analyzer
#     conversation_analyzer = chat_ctx._metadata.get("conversation_analyzer")
#     user_id = chat_ctx._metadata.get("user_id") or uuid.uuid4().hex
    
#     # Get filters and search context
#     filters = {}
#     search_context = ""
#     if conversation_analyzer:
#         conversation_filters = await conversation_analyzer.get_filters()
        
#         # Extract useful filters
#         for filter_key in ["all_categories", "price_range", "colors", "sizes"]:
#             if filter_key in conversation_filters and conversation_filters[filter_key]:
#                 filters[filter_key] = conversation_filters[filter_key]
                
#         # Get search context if available
#         search_context = conversation_filters.get("search_context", "")
    
#     # ============ ENHANCED QUERY BUILDING ============
    
#     # Build a richer query using conversation context and domain-specific terms
#     query_parts = [user_msg.content]
    
#     # Add context from recent conversation history (up to 2 previous messages)
#     if len(messages) > 2:
#         previous_msgs = [m.content for m in messages[-4:-1] if m.role == "user"]
#         if previous_msgs:
#             query_parts.append(f"Previous messages: {' '.join(previous_msgs)}")
    
#     # Add explicit terms for common queries to improve matching
#     if any(term in user_msg.content.lower() for term in ["mountain", "trail", "downhill", "off-road"]):
#         query_parts.append("mountain bike MTB trail riding off-road rough terrain stumpjumper")
#     elif any(term in user_msg.content.lower() for term in ["road", "race", "racing", "pavement"]):
#         query_parts.append("road bike racing tarmac aethos roubaix pavement speed performance")
#     elif any(term in user_msg.content.lower() for term in ["gravel", "all-road", "mixed terrain"]):
#         query_parts.append("gravel bike diverge mixed terrain all-road adventure")
        
#     # Add search context if available
#     if search_context:
#         query_parts.append(f"Context from conversation: {search_context}")
    
#     # Complete enhanced query
#     enhanced_query = " ".join(query_parts)
    
#     # Log the enhanced query for debugging
#     logger.debug(f"Enhanced RAG query: '{enhanced_query[:100]}...'")
#     logger.debug(f"Using filters: {filters}")
    
#     # Initialize RAG client once
#     rag = PineconeRAG(index_name="specialized-detailed")
    
#     # ============ IMPROVED SEARCH PARAMETERS ============
    
#     # Query all namespaces in parallel with much lower thresholds
#     search_tasks = [
#         ("products", rag.search_with_filter(enhanced_query, filter_dict=filters, 
#                                           namespace="products", top_k=10, min_score=0.0, timeout=4.0)),
#         ("information", rag.search_with_filter(enhanced_query, filter_dict={}, 
#                                              namespace="information", top_k=5, min_score=0.1, timeout=4.0)),
#         ("brand", rag.search_with_filter(enhanced_query, filter_dict={}, 
#                                        namespace="brand", top_k=2, min_score=0.2, timeout=4.0))
#     ]
    
#     # Execute all searches truly in parallel with timeout
#     results = {}
#     tasks = {namespace: asyncio.create_task(task) for namespace, task in search_tasks}
    
#     # Use wait with timeout instead of gather to handle timeouts properly
#     overall_timeout = 5.0  # Overall timeout for all searches
#     try:
#         done, pending = await asyncio.wait(
#             tasks.values(), 
#             timeout=overall_timeout,
#             return_when=asyncio.ALL_COMPLETED
#         )
        
#         # Cancel any tasks that didn't complete within the timeout
#         for task in pending:
#             task.cancel()
            
#         # Process completed results
#         for namespace, task in tasks.items():
#             if task in done:
#                 try:
#                     results[namespace] = task.result()
#                     logger.debug(f"RAG results for {namespace}: {len(results[namespace])} documents")
#                     # Log top result details if available
#                     if results[namespace]:
#                         top_result = results[namespace][0]
#                         if namespace == "products":
#                             logger.debug(f"Top {namespace} result: {top_result.get('metadata', {}).get('name', 'Unknown')} - Score: {top_result.get('score', 0):.3f}")
#                         else:
#                             logger.debug(f"Top {namespace} result: {top_result.get('text', '')[:50]}... - Score: {top_result.get('score', 0):.3f}")
#                 except Exception as e:
#                     logger.error(f"Error retrieving results for {namespace}: {e}")
#                     results[namespace] = []
#             else:
#                 logger.warning(f"Search in {namespace} namespace timed out")
#                 results[namespace] = []
                
#     except asyncio.TimeoutError:
#         # This should not happen with the wait pattern above, but just in case
#         logger.error(f"Overall timeout waiting for RAG search results after {overall_timeout}s")
#         # Cancel all pending tasks
#         for task in tasks.values():
#             if not task.done():
#                 task.cancel()
                
#     except Exception as e:
#         logger.error(f"Unexpected error in RAG search: {e}")
#         import traceback
#         traceback.print_exc()
    
    # # ============ FORMAT RESULTS THE SAME WAY ============
    
    # # Format results by namespace priority
    # context = ""
    
    # # Information first (if available and relevant)
    # if "information" in results and results["information"]:
    #     context += "HELPFUL INFORMATION:\n\n"
    #     for result in results["information"][:2]:  # Limit to top 2
    #         title = result.get('metadata', {}).get('title', 'Information Guide')
    #         context += f"# {title}\n{result.get('text', '')}\n\n"
    
    # # Brand content second (if available and relevant)
    # if "brand" in results and results["brand"]:
    #     context += "ABOUT SPECIALIZED:\n\n"
    #     for result in results["brand"][:1]:  # Limit to top 1
    #         title = result.get('metadata', {}).get('title', 'Brand Information')
    #         context += f"# {title}\n{result.get('text', '')}\n\n"
    
    # # if there are no products, then add the default one:
    # if "products" not in results or not results["products"]:
    #     results["products"] = []
    
    # # Products last (often most common)
    # valid_product_urls = [r.get('metadata', {}).get('product_url') for r in results["products"]] if "products" in results else []
    
    # if "products" in results and results["products"]:
    #     if context:
    #         context += "RELEVANT PRODUCT INFORMATION:\n\n"
    #     else:
    #         context += "Here's information about products that might help answer the customer's question:\n\n"
        
    #     for i, result in enumerate(results["products"][:5]):  # Limit to top 5
    #         metadata = result.get('metadata', {})
    #         # if metadata is a Dict, then convert it to a string
    #         if isinstance(metadata, dict):
    #             # metadata = obj_to_markdown(metadata, title=f"Product {i+1} Metadata", level=2, convert_keys=True)
    #             metadata = json.dumps(metadata)
    #         context += f"{metadata}\n"
    #         # # Get product info
    #         # name = metadata.get('name', 'Unknown Product')
    #         # price = metadata.get('price', 'N/A')
    #         # category = metadata.get('main_category', 'Product')
            
    #         # context += f"PRODUCT {i+1}: {name} ({category}, {price})\n"
            
    #         # # Add key product details
    #         # if sizes := metadata.get('sizes', []):
    #         #     context += f"Available sizes: {', '.join(sizes)}\n"
                
    #         # if colors := metadata.get('colors', []):
    #         #     context += f"Available colors: {', '.join(colors)}\n"
                
    #         # if url := metadata.get('url'):
    #         #     context += f"Product URL: {url}\n"
            
    #         # # Add product text, but limit length
    #         # context += f"{result.get('text', '')[:300]}...\n\n"
    
    # # Update system prompt with RAG content
    # if context:
    #     prompt_manager.update_rag_content(context)
    # if search_context:
    #     prompt_manager.update_conversation_analysis(search_context)
    # if valid_product_urls:
    #     prompt_manager.update_valid_product_urls(valid_product_urls)
    
    # prompt_additions = SessionStateManager.generate_prompt_additions(user_id)
    # prompt_manager.update_prompt_additions(prompt_additions=prompt_additions)
    # updated_prompt = prompt_manager.build_prompt()
    
    # # Update the system message
    # async with _chat_ctx_lock:
    #     current_system_msg = next((m for m in chat_ctx.items if m.role == "system"), None)
    #     if current_system_msg:
    #         current_system_msg.content = updated_prompt
    #     else:
    #         chat_ctx.items.insert(0, ChatMessage("system", updated_prompt))

