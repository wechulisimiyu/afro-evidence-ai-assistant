"""
Researcher agent for medical research and literature analysis.
"""
from langgraph.graph import StateGraph, END
from src.config import (
    model, MAX_OUTPUT_TOKENS, HMIS_API_KEY, 
    HMIS_API_ENABLED, HMIS_API_URL
)
from src.vector_tools import VectorStoreManager
import pandas as pd
from loguru import logger
from typing import TypedDict, List, Dict, Any, Optional
import aiohttp
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def safe_json_dumps(obj: Any) -> str:
    """Safely convert object to JSON string, handling numpy types."""
    return json.dumps(obj, cls=NumpyEncoder, indent=2)

def extract_references(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract reference information from search results."""
    references = []
    for result in results:
        if "metadata" in result:
            ref = {
                "title": result["metadata"].get("title", "Unknown"),
                "source": result["metadata"].get("source", "Unknown"),
                "date": result["metadata"].get("date", "Unknown"),
                "relevance_score": result.get("relevance_score", 0)
            }
            references.append(ref)
    return references

class ResearchState(TypedDict):
    """State schema for the research workflow."""
    query: str
    local_summary: List[Dict[str, Any]]
    international_comparison: List[Dict[str, Any]]
    gaps: List[Dict[str, Any]]
    report: Dict[str, Any]

class ResearcherAgent:
    def __init__(self):
        """Initialize the research agent."""
        self.vector_manager = VectorStoreManager()
        self.model = model
        self.hmis_api_url = HMIS_API_URL
        self.hmis_api_key = HMIS_API_KEY
        self.hmis_enabled = HMIS_API_ENABLED

    async def summarize_local_cases(self, state: ResearchState) -> ResearchState:
        """Summarize similar local cases from HMIS database."""
        query = state["query"]
        
        if not self.hmis_enabled:
            logger.warning("HMIS API is not enabled. Skipping local case search.")
            state["local_summary"] = []
            return state

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.hmis_api_key}",
                    "Content-Type": "application/json"
                }
                params = {
                    "query": query,
                    "limit": 5
                }
                
                async with session.get(
                    f"{self.hmis_api_url}/api/cases/search",
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        state["local_summary"] = data.get("cases", [])
                    else:
                        logger.error(f"HMIS API error: {response.status}")
                        state["local_summary"] = []
        except Exception as e:
            logger.error(f"Error accessing HMIS API: {str(e)}")
            state["local_summary"] = []
        
        return state

    async def compare_international_sources(self, state: ResearchState) -> ResearchState:
        """Compare with international and local sources using vector search."""
        query = state["query"]
        try:
            # Search for international journals
            international_results = await self.vector_manager.search_journals(
                f"International research and case studies on: {query}",
                k=3
            )

            # Search for local journals
            local_results = await self.vector_manager.search_journals(
                f"Local research and case studies on: {query}",
                k=3
            )

            # Extract references
            references = {
                "international": extract_references(international_results),
                "local": extract_references(local_results)
            }

            # Generate detailed comparison
            prompt = f"""Based on the following research findings, provide a detailed comparison (max {MAX_OUTPUT_TOKENS} tokens):

International Research References:
{safe_json_dumps(references['international'])}

Local Research References:
{safe_json_dumps(references['local'])}

Please provide a comprehensive comparison focusing on:

1. Study Design
   - Compare study types (RCT, observational, etc.)
   - Methodological approaches
   - Study duration and timeline

2. Participants
   - Sample sizes
   - Demographics
   - Inclusion/exclusion criteria
   - Recruitment methods

3. Interventions or Exposures
   - Treatment protocols
   - Control groups
   - Dosage and duration
   - Follow-up procedures

4. Outcomes
   - Primary outcomes
   - Secondary outcomes
   - Measurement methods
   - Statistical significance

5. Data Sources
   - Electronic medical records
   - Clinical trials databases
   - Surveys or questionnaires
   - Other data collection methods

6. Principal Findings
   - Key discoveries
   - Statistical results
   - Clinical significance
   - Novel insights

7. Comparison with Existing Literature
   - Similarities with previous studies
   - Differences and innovations
   - Conflicting findings
   - Knowledge gaps addressed

8. Strengths & Limitations
   - Methodological strengths
   - Study limitations
   - Potential biases
   - Generalizability

9. Implications for Practice
   - Clinical applications
   - Policy recommendations
   - Implementation challenges
   - Future research needs

Please structure the comparison to highlight differences between international and local research in each category.
"""

            response = self.model.generate_content(prompt)
            analysis = response.text

            state["international_comparison"] = {
                "references": references,
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error comparing sources: {str(e)}")
            state["international_comparison"] = {
                "references": {"international": [], "local": []},
                "analysis": "Error occurred during comparison"
            }
        return state

    async def identify_research_gaps(self, state: ResearchState) -> ResearchState:
        """Identify research gaps using guidelines vector search."""
        query = state["query"]
        try:
            results = await self.vector_manager.search_guidelines(
                f"Identify research gaps in the topic: {query}",
                k=3
            )
            state["gaps"] = results
        except Exception as e:
            logger.error(f"Error identifying research gaps: {str(e)}")
            state["gaps"] = []
        return state

    async def generate_research_report(self, state: ResearchState) -> ResearchState:
        """Generate final research report."""
        try:
            # Extract references from all sources
            references = {
                "local_cases": extract_references(state["local_summary"]),
                "international": extract_references(state["international_comparison"]["references"]["international"]),
                "local_research": extract_references(state["international_comparison"]["references"]["local"]),
                "guidelines": extract_references(state["gaps"])
            }

            # Generate comprehensive report
            prompt = f"""You are a JSON-only response bot. Based on the following research findings, generate a comprehensive research report in JSON format (max {MAX_OUTPUT_TOKENS} tokens):

References:
{safe_json_dumps(references)}

Key Findings:
- Local Cases: {safe_json_dumps(state['local_summary'])}
- International Comparison: {state['international_comparison']['analysis']}
- Research Gaps: {safe_json_dumps(state['gaps'])}

Required JSON response format:
{{
    "summary": "Executive summary of the research findings and implications",
    "references": {{
        "diagnosis": [
            {{
                "title": "Reference title",
                "source": "Source of reference",
                "relevance": "How it relates to the research"
            }}
        ],
        "guidelines": [
            {{
                "title": "Guideline title",
                "source": "Source of guideline",
                "relevance": "How it relates to the research"
            }}
        ],
        "case_comparisons": {{
            "similar_cases": [
                {{
                    "description": "Case description",
                    "outcome": "Case outcome",
                    "relevance": "How it relates to the research"
                }}
            ]
        }},
        "related_cases": [
            {{
                "title": "Case title",
                "description": "Case description",
                "relevance": "How it relates to the research"
            }}
        ]
    }}
}}

Remember: Respond with ONLY the JSON object, no other text or explanation."""

            response = self.model.generate_content(prompt)
            
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            # Remove any markdown code block indicators if present
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                report = json.loads(response_text)
                state["report"] = report
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error in research report: {str(je)}")
                logger.error(f"Raw response: {response_text}")
                # Return a default empty report structure
                state["report"] = {
                    "summary": "",
                    "references": {
                        "diagnosis": [],
                        "guidelines": [],
                        "case_comparisons": {},
                        "related_cases": []
                    }
                }
        except Exception as e:
            logger.error(f"Error generating research report: {str(e)}")
            state["report"] = {
                "summary": "",
                "references": {
                    "diagnosis": [],
                    "guidelines": [],
                    "case_comparisons": {},
                    "related_cases": []
                }
            }
        return state

    def create_workflow(self):
        """Create the research workflow graph."""
        # Define the state schema
        workflow = StateGraph(state_schema=ResearchState)
        
        # Add nodes
        workflow.add_node("SummarizeLocal", self.summarize_local_cases)
        workflow.add_node("CompareInternational", self.compare_international_sources)
        workflow.add_node("FindGaps", self.identify_research_gaps)
        workflow.add_node("GenerateReport", self.generate_research_report)

        # Set edges
        workflow.set_entry_point("SummarizeLocal")
        workflow.add_edge("SummarizeLocal", "CompareInternational")
        workflow.add_edge("CompareInternational", "FindGaps")
        workflow.add_edge("FindGaps", "GenerateReport")
        workflow.add_edge("GenerateReport", END)

        return workflow.compile()

    async def run_research(self, query: str) -> Dict[str, Any]:
        """Run the research workflow."""
        initial_state: ResearchState = {
            "query": query,
            "local_summary": [],
            "international_comparison": [],
            "gaps": [],
            "report": {}
        }
        graph = self.create_workflow()
        result = await graph.ainvoke(initial_state)
        return result["report"]


