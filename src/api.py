"""
FastAPI application for medical research and clinical analysis system.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
from .orchestrator import graph
from loguru import logger

app = FastAPI(
    title="Medical Research and Clinical Analysis API",
    description="API for medical research and clinical analysis using AI agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Reference(BaseModel):
    """Model for a single reference."""
    title: str
    source: str
    date: str
    relevance_score: float = 0.0

class References(BaseModel):
    """Model for grouped references."""
    diagnosis: List[Reference] = []
    guidelines: List[Reference] = []
    case_comparisons: Dict[str, List[Reference]] = {}
    related_cases: List[Reference] = []

class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""
    prompt: str
    role: str
    user_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    """Response model for analysis endpoint."""
    summary: str
    references: References
    agent_type: str

class PatientInfo(BaseModel):
    """Model for patient information."""
    user_id: str
    demographics: Dict[str, Any]
    medical_history: List[str]
    current_medications: List[str]
    allergies: List[str]
    vitals: Dict[str, Any]
    lab_results: List[Dict[str, Any]]
    imaging: List[Dict[str, Any]]
    family_history: str
    last_updated: str

def format_references(references: Dict[str, Any]) -> References:
    """Format references into the expected structure."""
    formatted = References()
    
    # Helper function to convert a reference dict to Reference model
    def to_reference(ref: Dict[str, Any]) -> Reference:
        if isinstance(ref, dict):
            return Reference(
                title=ref.get("title", "Unknown"),
                source=ref.get("source", "Unknown"),
                date=ref.get("date", "Unknown"),
                relevance_score=float(ref.get("relevance_score", 0.0))
            )
        return Reference(title=str(ref), source="Unknown", date="Unknown")

    # Process each category of references
    if "diagnosis" in references:
        formatted.diagnosis = [to_reference(ref) for ref in references["diagnosis"]]
    
    if "guidelines" in references:
        formatted.guidelines = [to_reference(ref) for ref in references["guidelines"]]
    
    if "case_comparisons" in references:
        formatted.case_comparisons = {
            key: [to_reference(ref) for ref in refs]
            for key, refs in references["case_comparisons"].items()
        }
    
    if "related_cases" in references:
        formatted.related_cases = [to_reference(ref) for ref in references["related_cases"]]
    
    return formatted

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "orchestrator": "operational",
            "agents": "operational"
        }
    }

@app.get("/patient/{user_id}", response_model=PatientInfo)
async def get_patient_info(user_id: str):
    """Get patient information by user ID."""
    try:
        # Prepare initial state for patient data extraction
        initial_state = {
            "prompt": f"Get patient information for user ID: {user_id}",
            "role": "clinician",
            "user_id": user_id,
            "target": "Clinician",
            "agent_output": {},
            "extracted_patient_data": None,
            "clinical_context": None
        }

        # Run analysis to get patient data
        result = await graph.ainvoke(initial_state)
        
        # Extract patient data from the result
        output = result["agent_output"]
        if not isinstance(output, dict):
            raise HTTPException(
                status_code=500,
                detail="Invalid agent output format"
            )

        # Format patient information
        try:
            patient_data = output.get("patient_data", {})
            patient_info = PatientInfo(
                user_id=user_id,
                demographics=patient_data.get("demographics", {}),
                medical_history=patient_data.get("medical_history", []),
                current_medications=patient_data.get("current_medications", []),
                allergies=patient_data.get("allergies", []),
                vitals=patient_data.get("vitals", {}),
                lab_results=patient_data.get("lab_results", []),
                imaging=patient_data.get("imaging", []),
                family_history=patient_data.get("family_history", ""),
                last_updated=patient_data.get("last_updated", "")
            )
            return patient_info
        except Exception as e:
            logger.error(f"Error formatting patient information: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error formatting patient information: {str(e)}"
            )

    except Exception as e:
        logger.error(f"Error fetching patient information: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching patient information: {str(e)}"
        )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Analyze medical query using appropriate agent."""
    try:
        # Prepare initial state
        initial_state = {
            "prompt": request.prompt,
            "role": request.role,
            "user_id": request.user_id,
            "target": "",
            "agent_output": {},
            "extracted_patient_data": None,
            "clinical_context": None
        }

        # Run analysis
        result = await graph.ainvoke(initial_state)
        
        # Extract output
        output = result["agent_output"]
        if not isinstance(output, dict):
            raise HTTPException(
                status_code=500,
                detail="Invalid agent output format"
            )

        # Format response
        try:
            response = AnalysisResponse(
                summary=output.get("summary", ""),
                references=format_references(output.get("references", {})),
                agent_type=result["target"]
            )
            return response
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error formatting response: {str(e)}"
            )

    except Exception as e:
        logger.error(f"Error processing analysis request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


