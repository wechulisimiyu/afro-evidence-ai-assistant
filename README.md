# Clinical Decision Support System

An AI-powered clinical decision support system that combines patient data analysis, medical research, and clinical guidelines to assist healthcare professionals in diagnosis and treatment planning.

## Features

- Patient data extraction and analysis
- Clinical diagnosis generation
- Medical research integration
- HMIS (Health Management Information System) integration
- Vector-based semantic search
- Multi-agent architecture for specialized tasks

## Project Structure

```
.
├── src/                      # Source code
│   ├── agents/              # AI agents
│   │   ├── clinician_agent.py
│   │   ├── researcher_agent.py
│   │   └── clinician_research_agent.py
│   ├── data/               # Data storage
│   │   ├── journals_vector_store/  # Vector DB for journals
│   │   └── guidelines_vector_store/ # Vector DB for guidelines
│   ├── api.py              # FastAPI routes
│   ├── config.py           # Configuration
│   ├── get_hmis_tool.py    # HMIS integration
│   ├── main.py             # Application entry point
│   ├── orchestrator.py     # Workflow orchestration
│   └── vector_tools.py     # Vector search utilities
├── logs/                   # Application logs
├── tests/                  # Test files
├── scripts/                # Utility scripts
│   ├── pdfs/               # Raw PDF documents
│   └── build_vector_db.py  # Script to build vector databases
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Project metadata
```

## Setup

### Local Development

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key
APP_HOST=127.0.0.1
APP_PORT=8004
APP_DEBUG=true
LLM_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=models/embedding-001
HMIS_API_URL=your_hmis_api_url
HMIS_API_KEY=your_hmis_api_key
HMIS_API_ENABLED=true
```

### Docker Setup

1. Build and start the containers:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8004`

## Usage

### API Endpoints

1. Clinical Analysis:
```bash
curl -X POST "http://localhost:8004/analyze'" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Patient symptoms and history",
           "user_id": "patient123",
           "role": "clinician"
         }'
```

2. Research Query:
```bash
curl -X POST "http://localhost:8004/analyze'" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Research query",
           "role": "researcher"
         }'
```

### Response Format

```json
{
    "report": {
        "summary": "Clinical analysis or research findings",
        "references": {
            "diagnosis": [],
            "guidelines": [],
            "case_comparisons": {},
            "related_cases": []
        }
    }
}
```

## Development

### Building Vector Databases

The system maintains separate vector databases for journals and guidelines:

1. Place your PDF documents in the appropriate directories:
   - Medical journals: `scripts/pdfs/journals/`
   - Clinical guidelines: `scripts/pdfs/guidelines/`

2. Run the vector database builder script:
```bash
python scripts/build_vector_db.py
```

This script will:
- Process PDFs from both directories
- Create separate vector stores for journals and guidelines
- Store the vector databases in their respective directories:
  - `src/data/journals_vector_store/`
  - `src/data/guidelines_vector_store/`

### Running Tests

```bash
pytest tests/
```

## Docker Commands

- Start the application:
```bash
docker-compose up
```

- Start in detached mode:
```bash
docker-compose up -d
```

- Stop the application:
```bash
docker-compose down
```

- View logs:
```bash
docker-compose logs -f
```

- Rebuild containers:
```bash
docker-compose up --build
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
