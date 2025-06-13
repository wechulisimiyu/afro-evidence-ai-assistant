from typing import Dict, Any, List, Optional
from vector_tools import VectorStoreManager
from config import (
    HMIS_API_URL,
    HMIS_API_KEY,
    HMIS_API_ENABLED
)
import aiohttp
import logging

logger = logging.getLogger(__name__)

class RelatedCasesManager:
    def __init__(self):
        """Initialize the related cases manager."""
        self.vector_manager = VectorStoreManager()
        self.hmis_api_url = HMIS_API_URL
        self.hmis_api_key = HMIS_API_KEY
        self.hmis_enabled = HMIS_API_ENABLED

    async def get_related_cases(
        self,
        diagnosis: str,
        user_id: str,
        vector_manager: Optional[VectorStoreManager] = None
    ) -> List[Dict[str, Any]]:
        """Get related cases from HMIS database or journals based on diagnosis.
        
        Args:
            diagnosis: The diagnosis to find related cases for
            user_id: The ID of the current patient to exclude from results
            vector_manager: Optional VectorStoreManager instance for journal search
            
        Returns:
            List of related cases with consistent format
        """
        try:
            # First try to get related cases from HMIS if enabled
            if self.hmis_enabled:
                try:
                    async with aiohttp.ClientSession() as session:
                        headers = {
                            "Authorization": f"Bearer {self.hmis_api_key}",
                            "Content-Type": "application/json"
                        }
                        params = {
                            "diagnosis": diagnosis,
                            "limit": 5,
                            "exclude_patient": user_id  # Exclude current patient
                        }
                        
                        async with session.get(
                            f"{self.hmis_api_url}/api/cases/related",
                            headers=headers,
                            params=params
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                hmis_cases = data.get("cases", [])
                                if hmis_cases:
                                    logger.info(f"Found {len(hmis_cases)} related cases in HMIS")
                                    return hmis_cases
                            else:
                                logger.warning(f"HMIS API error: {response.status}, falling back to journal search")
                except Exception as e:
                    logger.warning(f"Error accessing HMIS for related cases: {str(e)}, falling back to journal search")

            # Fall back to journal search if HMIS is not available or fails
            vector_mgr = vector_manager or self.vector_manager
            journal_results = await vector_mgr.search_journals(
                f"Similar cases to: {diagnosis}",
                k=5
            )
            
            # Format journal results to match HMIS case format
            formatted_cases = []
            for result in journal_results:
                formatted_case = {
                    "case_id": f"journal_{result.get('id', 'unknown')}",
                    "source": "journal",
                    "diagnosis": result.get("diagnosis", ""),
                    "symptoms": result.get("symptoms", []),
                    "treatment": result.get("treatment", ""),
                    "outcome": result.get("outcome", ""),
                    "reference": {
                        "title": result.get("title", ""),
                        "authors": result.get("authors", []),
                        "journal": result.get("journal", ""),
                        "year": result.get("year", "")
                    }
                }
                formatted_cases.append(formatted_case)
            
            logger.info(f"Found {len(formatted_cases)} related cases from journals")
            return formatted_cases
            
        except Exception as e:
            logger.error(f"Error getting related cases: {str(e)}")
            return [] 