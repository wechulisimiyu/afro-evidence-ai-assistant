
"""
Vector store utilities for the Medical RAG system.
"""
from typing import List, Dict, Optional
from config import embeddings, guidelines_store, journals_store
import google.generativeai as genai
from loguru import logger

class VectorStoreManager:
    def __init__(self):
        """Initialize the vector store manager."""
        self.embeddings = embeddings
        self._load_stores()

    def _load_stores(self):
        """Load both vector stores."""
        try:
            self.guidelines_store = guidelines_store
            logger.info("Guidelines vector store received successfully")
        except Exception as e:
            logger.error(f"Error loading guidelines store: {str(e)}")
            self.guidelines_store = None

        try:
            self.journals_store = journals_store
            logger.info("Journals vector store received successfully")
        except Exception as e:
            logger.error(f"Error loading journals store: {str(e)}")
            self.journals_store = None

    async def search_guidelines(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict]:
        """
        Search the guidelines vector store.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of relevant documents with metadata
        """
        if not self.guidelines_store:
            logger.error("Guidelines store not available")
            return []
            
        results = await self.guidelines_store.asimilarity_search_with_score(
            query,
            k=k
        )
        
        return [{
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score
        } for doc, score in results]

    async def search_journals(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Search the journals vector store.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of relevant documents with metadata
        """
        if not self.journals_store:
            logger.error("Journals store not available")
            return []
            
        results = await self.journals_store.asimilarity_search_with_score(
            query,
            k=k
        )
        
        return [{
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score
        } for doc, score in results]

    async def search_all(
        self,
        query: str,
        k_guidelines: int = 3,
        k_journals: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Search both vector stores.
        
        Args:
            query (str): The search query
            k_guidelines (int): Number of guideline results
            k_journals (int): Number of journal results
            
        Returns:
            Dict[str, List[Dict]]: Combined search results
        """
        guidelines_results = await self.search_guidelines(query, k_guidelines)
        journals_results = await self.search_journals(query, k_journals)
        
        return {
            "guidelines": guidelines_results,
            "journals": journals_results
        }
