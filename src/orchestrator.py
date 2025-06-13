# orchestrator.py

from langgraph.graph import StateGraph, END
from typing import Dict, Optional, TypedDict, Any
from agents.clinician_agent import ClinicianAgent
from agents.researcher_agent import ResearcherAgent
from agents.clinician_research_agent import ClinicianResearchAgent

import json
from config import model
import aiohttp
from config import HMIS_API_URL, HMIS_API_KEY, HMIS_API_ENABLED
import logging
import asyncio

logger = logging.getLogger(__name__)

class OrchestratorState(TypedDict):
    """State schema for the orchestrator workflow."""
    prompt: str
    role: str
    user_id: Optional[str]
    target: str
    agent_output: Dict[str, Any]
    patient_data: Optional[Dict[str, Any]]  # Added to store extracted patient data
    error: Optional[str]  # Added to store error messages

def extract_patient_data(prompt: str) -> Dict[str, Any]:
    """Extract structured patient data from clinician's prompt."""
    try:
        extraction_prompt = f"""You are a JSON-only response bot. Extract structured patient information from the following clinical prompt and respond with ONLY a JSON object, no other text:

{prompt}

Required JSON response format (respond with ONLY this JSON, no other text):
{{
    "symptoms": [],
    "history": "",
    "vitals": {{}},
    "lab_results": [],
    "imaging": [],
    "current_medications": [],
    "allergies": [],
    "family_history": ""
}}

If no patient information is found, return the empty structure above. Remember: Respond with ONLY the JSON object, no other text or explanation."""

        response = model.generate_content(extraction_prompt)
        
        # Clean the response text to ensure it's valid JSON
        response_text = response.text.strip()
        if not response_text:
            return None
            
        # Remove any markdown code block indicators if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Try to parse JSON
        extracted_data = json.loads(response_text)
        return extracted_data
        
    except json.JSONDecodeError as je:
        print(f"JSON parsing error in patient data extraction: {str(je)}")
        print(f"Raw response: '{response.text if 'response' in locals() else 'No response'}'")
        return None
    except Exception as e:
        print(f"Error extracting patient data: {str(e)}")
        return None

async def fetch_patient_data(user_id: str) -> Dict[str, Any]:
    """Fetch patient data from HMIS database."""
    if not HMIS_API_ENABLED:
        logger.warning("HMIS API is not enabled. Cannot fetch patient data.")
        raise ValueError("HMIS API is not available")

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {HMIS_API_KEY}",
                "Content-Type": "application/json"
            }
            
            async with session.get(
                f"{HMIS_API_URL}/api/patients/{user_id}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 404:
                    raise ValueError(f"Patient with ID {user_id} does not exist")
                else:
                    logger.error(f"HMIS API error: {response.status}")
                    raise ValueError(f"Failed to fetch patient data: {response.status}")
    except Exception as e:
        logger.error(f"Error accessing HMIS API: {str(e)}")
        raise

# --- Step 1: Route Based on Role and User ID ---
async def route_request(state: OrchestratorState) -> OrchestratorState:
    role = state.get("role", "").lower()
    user_id = state.get("user_id")
    prompt = state["prompt"]
    
    # Initialize patient data
    patient_data = None
    has_patient_info = False
    
    # First, extract patient data from prompt
    extracted_data = extract_patient_data(prompt)
    if extracted_data:
        has_patient_info = any([
            isinstance(extracted_data.get("symptoms"), list) and len(extracted_data["symptoms"]) > 0,
            isinstance(extracted_data.get("history"), str) and extracted_data["history"].strip() != "",
            isinstance(extracted_data.get("vitals"), dict) and len(extracted_data["vitals"]) > 0,
            isinstance(extracted_data.get("lab_results"), list) and len(extracted_data["lab_results"]) > 0,
            isinstance(extracted_data.get("imaging"), list) and len(extracted_data["imaging"]) > 0,
            isinstance(extracted_data.get("current_medications"), list) and len(extracted_data["current_medications"]) > 0,
            isinstance(extracted_data.get("allergies"), list) and len(extracted_data["allergies"]) > 0,
            isinstance(extracted_data.get("family_history"), str) and extracted_data["family_history"].strip() != ""
        ])
        patient_data = extracted_data
    
    # If user_id is provided, try to fetch from HMIS (this takes precedence)
    if user_id and HMIS_API_ENABLED:
        try:
            hmis_data = await fetch_patient_data(user_id)
            patient_data = hmis_data
            has_patient_info = True  # HMIS data means we have patient info
            logger.info(f"Successfully fetched patient data from HMIS for user_id: {user_id}")
        except ValueError as e:
            if "does not exist" in str(e):
                logger.warning(f"Patient {user_id} not found in HMIS")
                state["error"] = str(e)
                # Continue with extracted data if available
            else:
                logger.warning(f"HMIS error: {str(e)}, using extracted patient data")
                state["error"] = str(e)
        except Exception as e:
            logger.error(f"Unexpected HMIS error: {str(e)}")
            state["error"] = str(e)
    
    # Store final patient data in state
    state["patient_data"] = patient_data
    
    # Routing logic based on corrected patient info detection
    if role == "researcher":
        state["target"] = "Researcher"
        logger.info("→ Routing to Researcher")
    elif role == "clinician":
        if user_id or has_patient_info:
            state["target"] = "Clinician"
            logger.info("→ Routing to Clinician (has user_id or patient info)")
        else:
            state["target"] = "ClinicianResearch"
            logger.info("→ Routing to ClinicianResearch (no user_id and no patient info)")
    else:
        # Default routing
        if has_patient_info or user_id:
            state["target"] = "Clinician"
            logger.info("→ Routing to Clinician (default with data)")
        elif any(k in prompt.lower() for k in ["research", "literature", "study", "analysis"]):
            state["target"] = "Researcher"
            logger.info("→ Routing to Researcher (default research keywords)")
        else:
            state["target"] = "ClinicianResearch"
            logger.info("→ Routing to ClinicianResearch (default)")
    
    return state

# --- Step 2: Call Clinician Agent ---
async def call_clinician_agent(state: OrchestratorState) -> OrchestratorState:
    print("🩺 Routing to Clinician Agent...")
    agent = ClinicianAgent()
    
    # Check if the agent method supports the new parameters
    try:
        # Try new interface first
        output = await agent.run_clinical_analysis(
            clinician_prompt=state["prompt"],
            user_id=state["user_id"] or "unknown",
            extracted_patient_data=state["patient_data"],
            clinical_context=state.get("error")
        )
    except TypeError:
        # Fall back to original interface if new parameters aren't supported
        logger.warning("Agent doesn't support new parameters, using original interface")
        output = await agent.run_clinical_analysis(
            clinician_prompt=state["prompt"],
            user_id=state["user_id"] or "unknown"
        )
    
    state["agent_output"] = output
    return state

# --- Step 3: Call Researcher Agent ---
async def call_researcher_agent(state: OrchestratorState) -> OrchestratorState:
    print("🔬 Routing to Researcher Agent...")
    agent = ResearcherAgent()
    output = await agent.run_research(state["prompt"])
    state["agent_output"] = output
    return state

# --- Step 4: Call Clinician Research Agent ---
async def call_clinician_research_agent(state: OrchestratorState) -> OrchestratorState:
    print("📚 Routing to Clinician Research Agent...")
    agent = ClinicianResearchAgent()
    output = await agent.run_research_analysis(state["prompt"])
    state["agent_output"] = output
    return state

# --- Step 5: Final Output ---
def final_output(state: OrchestratorState) -> OrchestratorState:
    print("\n🎯 Final Agent Output:")
    if isinstance(state["agent_output"], dict):
        if "summary" in state["agent_output"]:
            print("\nSummary:")
            print(state["agent_output"]["summary"])
            if "references" in state["agent_output"]:
                print("\nReferences:")
                for category, refs in state["agent_output"]["references"].items():
                    print(f"\n{category.upper()}:")
                    for ref in refs:
                        if isinstance(ref, dict):
                            print(f"- {ref.get('title', 'Unknown')} ({ref.get('source', 'Unknown')}, {ref.get('date', 'Unknown')})")
                        else:
                            print(f"- {ref}")
    else:
        print(state["agent_output"])
    return state

# --- LangGraph Orchestration Graph ---
workflow = StateGraph(state_schema=OrchestratorState)
workflow.add_node("Route", route_request)
workflow.add_node("Clinician", call_clinician_agent)
workflow.add_node("Researcher", call_researcher_agent)
workflow.add_node("ClinicianResearch", call_clinician_research_agent)
workflow.add_node("Output", final_output)

workflow.set_entry_point("Route")

# Branch logic
workflow.add_conditional_edges(
    "Route",
    lambda state: state["target"],
    {
        "Clinician": "Clinician",
        "Researcher": "Researcher",
        "ClinicianResearch": "ClinicianResearch"
    }
)

workflow.add_edge("Clinician", "Output")
workflow.add_edge("Researcher", "Output")
workflow.add_edge("ClinicianResearch", "Output")
workflow.add_edge("Output", END)

graph = workflow.compile()

# --- Run the Orchestrator ---
if __name__ == "__main__":
    async def main():
        prompt = input("Enter your prompt: ")
        role = input("Enter your role (clinician/researcher): ")
        user_id = input("Enter user ID (or press Enter to skip): ").strip() or None
        
        initial_state: OrchestratorState = {
            "prompt": prompt,
            "role": role,
            "user_id": user_id,
            "target": "",
            "agent_output": {},
            "patient_data": None,
            "error": None
        }
        
        result = await graph.ainvoke(initial_state)
        print("\nResult:", result)

    asyncio.run(main())