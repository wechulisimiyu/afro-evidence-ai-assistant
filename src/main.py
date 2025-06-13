"""
Main entry point for the medical research and clinical analysis system.
"""
import sys
import uvicorn
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.config import APP_HOST, APP_PORT

def start():
    """Start the FastAPI server."""
    uvicorn.run(
        "src.api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True,
        reload_dirs=[project_root]  # Watch the entire project directory for changes
    )

if __name__ == "__main__":
    start() 