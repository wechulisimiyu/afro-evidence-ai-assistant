"""
Clinician research agent for research-based clinical analysis.
"""
from langgraph.graph import StateGraph, END
from src.config import (
    model, MAX_OUTPUT_TOKENS, MAX_INPUT_TOKENS,
    REQUESTS_PER_MINUTE, TOKENS_PER_MINUTE
)
from src.vector_tools import VectorStoreManager
from loguru import logger
from typing import TypedDict, List, Dict, Any, Optional
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float64, np.float32, np.float16)):  # Updated for NumPy 2.0
            return float(obj)
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):  # Updated for NumPy 2.0
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
        if isinstance(result, dict) and "metadata" in result:
            ref = {
                "title": result["metadata"].get("title", "Unknown"),
                "source": result["metadata"].get("source", "Unknown"),
                "date": result["metadata"].get("date", "Unknown"),
                "relevance_score": result.get("relevance_score", 0)
            }
            references.append(ref)
        elif isinstance(result, str):
            references.append({
                "title": result,
                "source": "Unknown",
                "date": "Unknown",
                "relevance_score": 0
            })
    return references

class ClinicianResearchState(TypedDict):
    """State schema for the clinician research workflow."""
    prompt: str
    journal_findings: List[Dict[str, Any]]
    guidelines: List[Dict[str, Any]]
    report: Dict[str, Any]

class ClinicianResearchAgent:
    def __init__(self):
        """Initialize the clinician research agent."""
        self.vector_manager = VectorStoreManager()
        self.model = model

    async def search_journals(self, state: ClinicianResearchState) -> ClinicianResearchState:
        """Search medical journals for relevant information."""
        try:
            results = await self.vector_manager.search_journals(
                f"Clinical research and case studies on: {state['prompt']}",
                k=5
            )
            state["journal_findings"] = results
        except Exception as e:
            logger.error(f"Error searching journals: {str(e)}")
            state["journal_findings"] = []
        return state

    async def get_guidelines(self, state: ClinicianResearchState) -> ClinicianResearchState:
        """Get clinical guidelines based on the prompt."""
        try:
            results = await self.vector_manager.search_guidelines(
                f"Clinical guidelines for: {state['prompt']}",
                k=3
            )
            state["guidelines"] = results
        except Exception as e:
            logger.error(f"Error getting guidelines: {str(e)}")
            state["guidelines"] = []
        return state

    async def generate_research_report(self, state: ClinicianResearchState) -> ClinicianResearchState:
        """Generate a comprehensive clinical research report."""
        try:
            # Prepare the research report prompt
            report_prompt = f"""You are a JSON-only response bot. Generate a clinical research report based on the following information and respond with ONLY a JSON object, no other text:

Clinical Query:
{state['prompt']}

Journal Findings:
{json.dumps(state['journal_findings'], indent=2, cls=NumpyEncoder)}

Clinical Guidelines:
{json.dumps(state['guidelines'], indent=2, cls=NumpyEncoder)}

Required JSON response format:
{{
    "summary": "Executive summary of findings and recommendations",
    "references": {{
        "diagnosis": [
            {{
                "title": "Reference title",
                "source": "Source of reference",
                "relevance": "How it relates to the case"
            }}
        ],
        "guidelines": [
            {{
                "title": "Guideline title",
                "source": "Source of guideline",
                "relevance": "How it relates to the case"
            }}
        ],
        "case_comparisons": {{
            "similar_cases": [
                {{
                    "description": "Case description",
                    "outcome": "Case outcome",
                    "relevance": "How it relates to current case"
                }}
            ]
        }},
        "related_cases": [
            {{
                "title": "Case title",
                "description": "Case description",
                "relevance": "How it relates to current case"
            }}
        ]
    }}
}}

Remember: Respond with ONLY the JSON object, no other text or explanation."""

            # Generate the research report
            response = self.model.generate_content(report_prompt)
            
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
        """Create the clinician research workflow graph."""
        workflow = StateGraph(state_schema=ClinicianResearchState)
        
        # Add nodes
        workflow.add_node("SearchJournals", self.search_journals)
        workflow.add_node("GetGuidelines", self.get_guidelines)
        workflow.add_node("GenerateReport", self.generate_research_report)

        # Set edges
        workflow.set_entry_point("SearchJournals")
        workflow.add_edge("SearchJournals", "GetGuidelines")
        workflow.add_edge("GetGuidelines", "GenerateReport")
        workflow.add_edge("GenerateReport", END)

        return workflow.compile()

    async def run_research_analysis(self, prompt: str) -> Dict[str, Any]:
        """Run the clinician research workflow."""
        initial_state: ClinicianResearchState = {
            "prompt": prompt,
            "journal_findings": [],
            "guidelines": [],
            "report": {}
        }
        graph = self.create_workflow()
        result = await graph.ainvoke(initial_state)
        return result["report"]