"""
Clinician agent for medical diagnosis and treatment recommendations.
"""
from langgraph.graph import StateGraph, END
from src.config import (
    model,
    MAX_OUTPUT_TOKENS,
    HMIS_API_URL,
    HMIS_API_KEY,
    HMIS_API_ENABLED
)
from src.vector_tools import VectorStoreManager
from src.get_hmis_tool import RelatedCasesManager
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

class ClinicianState(TypedDict):
    """State schema for the clinician workflow."""
    user_id: str
    clinician_prompt: str
    patient_data: Dict[str, Any]
    clinical_context: Optional[str]  # For storing error messages or additional context
    diagnosis: Dict[str, Any]
    guidelines: List[Dict[str, Any]]
    case_comparisons: Dict[str, Any]
    related_cases: List[Dict[str, Any]]
    report: Dict[str, Any]

class ClinicianAgent:
    def __init__(self):
        """Initialize the clinician agent."""
        self.vector_manager = VectorStoreManager()
        self.model = model
        self.hmis_enabled = HMIS_API_ENABLED
        self.hmis_api_url = HMIS_API_URL
        self.hmis_api_key = HMIS_API_KEY
        self.related_cases_manager = RelatedCasesManager()

    async def generate_diagnosis(self, state: ClinicianState) -> ClinicianState:
        """Generate initial diagnosis using patient data and journal references."""
        try:
            # Include clinical context in the prompt if available
            context_info = f"\nClinical Context: {state['clinical_context']}" if state.get('clinical_context') else ""
            
            # Search journals for similar cases
            journal_results = await self.vector_manager.search_journals(
                f"Patient symptoms and history: {safe_json_dumps(state['patient_data'])}{context_info}",
                k=5
            )

            # Generate diagnosis with references
            prompt = f"""Based on the following patient data and similar cases, provide a diagnosis (max {MAX_OUTPUT_TOKENS} tokens):

Patient Data:
{safe_json_dumps(state['patient_data'])}{context_info}

Similar Cases from Literature:
{safe_json_dumps(journal_results)}

Please provide:
1. Primary diagnosis with confidence level
2. Differential diagnoses
3. Key findings supporting the diagnosis
4. Relevant references from the literature
"""

            response = self.model.generate_content(prompt)
            diagnosis = response.text

            state["diagnosis"] = {
                "analysis": diagnosis,
                "references": extract_references(journal_results)
            }
        except Exception as e:
            logger.error(f"Error generating diagnosis: {str(e)}")
            state["diagnosis"] = {
                "analysis": "Error generating diagnosis",
                "error": str(e),
                "references": []
            }
        return state

    async def get_guidelines(self, state: ClinicianState) -> ClinicianState:
        """Get clinical guidelines based on diagnosis."""
        try:
            diagnosis = state["diagnosis"]["analysis"]
            results = await self.vector_manager.search_guidelines(
                f"Clinical guidelines for: {diagnosis}",
                k=3
            )
            state["guidelines"] = results
        except Exception as e:
            logger.error(f"Error getting guidelines: {str(e)}")
            state["guidelines"] = []
        return state

    async def get_case_comparisons(self, state: ClinicianState) -> ClinicianState:
        """Get case comparisons from journals."""
        try:
            # Search for international cases
            international_results = await self.vector_manager.search_journals(
                f"International case studies for: {state['diagnosis']['analysis']}",
                k=3
            )

            # Search for local cases
            local_results = await self.vector_manager.search_journals(
                f"Local case studies for: {state['diagnosis']['analysis']}",
                k=3
            )

            state["case_comparisons"] = {
                "international": extract_references(international_results),
                "local": extract_references(local_results)
            }
        except Exception as e:
            logger.error(f"Error getting case comparisons: {str(e)}")
            state["case_comparisons"] = {"international": [], "local": []}
        return state

    async def get_related_cases(self, state: ClinicianState) -> ClinicianState:
        """Get related cases from HMIS database or journals based on diagnosis."""
        try:
            diagnosis = state["diagnosis"]["analysis"]
            related_cases = await self.related_cases_manager.get_related_cases(
                diagnosis=diagnosis,
                user_id=state["user_id"],
                vector_manager=self.vector_manager
            )
            state["related_cases"] = related_cases
        except Exception as e:
            logger.error(f"Error getting related cases: {str(e)}")
            state["related_cases"] = []
        return state

    async def generate_clinical_report(self, state: ClinicianState) -> ClinicianState:
        """Generate final clinical report."""
        try:
            prompt = f"""Based on the following information, generate a comprehensive clinical report (max {MAX_OUTPUT_TOKENS} tokens):

Patient Data:
{safe_json_dumps(state['patient_data'])}

Diagnosis:
{state['diagnosis']['analysis']}

Clinical Guidelines:
{safe_json_dumps(state['guidelines'])}

Case Comparisons:
{safe_json_dumps(state['case_comparisons'])}

Related Cases:
{safe_json_dumps(state['related_cases'])}

Please provide a structured report with:
1. Patient Summary
2. Diagnosis and Differential Diagnoses
3. Treatment Recommendations
4. Prognosis
5. References (use the provided reference metadata)
"""

            response = self.model.generate_content(prompt)
            report = response.text

            state["report"] = {
                "summary": report,
                "references": {
                    "diagnosis": state["diagnosis"]["references"],
                    "guidelines": extract_references(state["guidelines"]),
                    "case_comparisons": state["case_comparisons"],
                    "related_cases": state["related_cases"]
                }
            }
        except Exception as e:
            logger.error(f"Error generating clinical report: {str(e)}")
            state["report"] = {"error": "Failed to generate report"}
        return state

    def create_workflow(self):
        """Create the clinician workflow graph."""
        workflow = StateGraph(state_schema=ClinicianState)
        
        # Add nodes
        workflow.add_node("GenerateDiagnosis", self.generate_diagnosis)
        workflow.add_node("GetGuidelines", self.get_guidelines)
        workflow.add_node("GetCaseComparisons", self.get_case_comparisons)
        workflow.add_node("GetRelatedCases", self.get_related_cases)
        workflow.add_node("GenerateReport", self.generate_clinical_report)

        # Set edges
        workflow.set_entry_point("GenerateDiagnosis")
        workflow.add_edge("GenerateDiagnosis", "GetGuidelines")
        workflow.add_edge("GetGuidelines", "GetCaseComparisons")
        workflow.add_edge("GetCaseComparisons", "GetRelatedCases")
        workflow.add_edge("GetRelatedCases", "GenerateReport")
        workflow.add_edge("GenerateReport", END)

        return workflow.compile()

    async def run_clinical_analysis(
        self,
        clinician_prompt: str,
        user_id: str,
        extracted_patient_data: Optional[Dict[str, Any]] = None,
        clinical_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the clinical analysis workflow."""
        try:
            initial_state: ClinicianState = {
                "user_id": user_id,
                "clinician_prompt": clinician_prompt,
                "patient_data": extracted_patient_data or {},
                "clinical_context": clinical_context,
                "diagnosis": {},
                "guidelines": [],
                "case_comparisons": {},
                "related_cases": [],
                "report": {}
            }
            
            # Validate patient data
            if not initial_state["patient_data"]:
                logger.warning("No patient data provided")
                initial_state["clinical_context"] = "No patient data available. Analysis will be based on clinical prompt only."
            
            graph = self.create_workflow()
            result = await graph.ainvoke(initial_state)
            return result["report"]
        except Exception as e:
            logger.error(f"Error in clinical analysis workflow: {str(e)}")
            return {
                "error": f"Clinical analysis failed: {str(e)}",
                "summary": "Unable to generate clinical report due to an error.",
                "references": {}
            }


