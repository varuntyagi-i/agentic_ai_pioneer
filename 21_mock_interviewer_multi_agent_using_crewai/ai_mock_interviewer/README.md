# ai-mock-interviewer
A smart mock interview assistant built with CrewAI, simulating realistic tech interviews with multi-agent conversations and feedback.

## Installation

1. Make sure you have [Poetry](https://python-poetry.org/docs/#installation) installed on your system.

2. Clone this repository:
```bash
git clone https://github.com/yourusername/ai-mock-interviewer.git
cd ai-mock-interviewer
```

3. Install dependencies:
```bash
poetry install
```

4. Activate the virtual environment:
```bash
poetry shell
```

## Development

### Run the CLI version
```bash
cd ai-mock-interviewer
poetry run ai_mock_interviewer/interview_practice_system.py
```

### Run the Streamlit Web App
```bash
cd ai-mock-interviewer
poetry run streamlit run ai_mock_interviewer/chatbot_ui.py
```

### Linting and Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. To run the linter:

```bash
# Check for linting issues
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .
```