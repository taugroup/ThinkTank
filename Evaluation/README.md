# Meeting Transcript Compilation and Evaluation System

A comprehensive system for compiling multi-agent meeting transcripts from individual JSON response files and evaluating meeting quality using AI-powered analysis.

## Overview

This system provides a complete workflow for:
1. **Compiling** individual agent response files into structured transcript JSON files
2. **Evaluating** meeting transcripts using multiple AI evaluators to assess quality, collaboration, and effectiveness

## System Architecture

```
📁 Evaluation/
├── 📄 compile_transcript.py      # Transcript compilation tool
├── 📄 evaluate_meeting.py        # Meeting evaluation CLI
├── 📁 core/                      # Core evaluation components
├── 📁 evaluators/                # AI evaluator implementations
├── 📁 metrics/                   # Quality assessment metrics
└── 📁 utils/                     # Helper utilities
```

## Prerequisites

- Python 3.11.11+
- API keys for at least one evaluator:
  - `GEMINI_API_KEY` or `GOOGLE_API_KEY` (for Gemini evaluator)
  - `GROQ_API_KEY` (for Groq evaluator)
  - Local Ollama installation (for Qwen evaluator)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ThinkTank/Evaluation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   ```bash
   # For Gemini
   export GEMINI_API_KEY="your_gemini_api_key"
   # Or
   export GOOGLE_API_KEY="your_google_api_key"
   
   # For Groq
   export GROQ_API_KEY="your_groq_api_key"
   ```

## Quick Start

### Step 1: Compile Transcript from Individual Response Files

Use the `compile_transcript.py` script to convert individual agent response JSON files into a structured transcript.

```bash
# Basic usage - output will be saved as folder_name.json
python compile_transcript.py path/to/response/folder

# Specify custom output location
python compile_transcript.py path/to/response/folder -o compiled_transcript.json
```

**Example:**
```bash
python compile_transcript.py AI_in_Sustainable_AGriculture/
# Creates: AI_in_Sustainable_AGriculture.json
```

### Step 2: Evaluate the Compiled Transcript

Use the `evaluate_meeting.py` script to analyze the compiled transcript and generate an evaluation report.

```bash
# Basic evaluation
python evaluate_meeting.py compiled_transcript.json

# Specify custom output location for the report
python evaluate_meeting.py compiled_transcript.json --output evaluation_report.json
```

**Example:**
```bash
python evaluate_meeting.py AI_in_Sustainable_AGriculture.json
# Creates: meeting_evaluation_report.json
```

## Detailed Usage

### Transcript Compilation

The `compile_transcript.py` script processes folders containing individual JSON response files from multi-agent meetings.

#### Expected Input Structure
```
meeting_folder/
├── Agent_1_response_20231115_153649.json
├── Agent_2_response_20231115_154358.json
├── Coordinator_opening_20231115_153525.json
├── Coordinator_synthesis_20231115_154215.json
├── Critical_Thinker_critique_20231115_154018.json
└── Coordinator_final_summary_20231115_160304.json
```

#### Features
- **Automatic Agent Detection**: Extracts agent names and roles from file names and content
- **Round Organization**: Automatically assigns round numbers based on chronological order
- **Response Type Classification**: Identifies opening statements, expert responses, synthesis, critiques, and summaries
- **Metadata Extraction**: Captures timestamps, agent expertise, and goals

#### Output Format
```json
{
  "meeting_topic": "Advancing Protein Function & Therapeutics Using AlphaFold",
  "project_name": "alzheimers_2",
  "experts": [
    {
      "title": "Dr. Smith",
      "expertise": "Structural Biology",
      "role": "domain_expert",
      "goal": "Contribute protein structure insights"
    }
  ],
  "transcript": [
    {
      "name": "Coordinator",
      "content": "Welcome to our meeting...",
      "round": 0,
      "type": "opening"
    }
  ],
  "summary": "Multi-agent expert meeting...",
  "metadata": {
    "total_responses": 15,
    "total_agents": 4,
    "compilation_date": "2023-11-15T16:03:04"
  }
}
```

### Meeting Evaluation

The `evaluate_meeting.py` script uses multiple AI evaluators to assess meeting quality across various dimensions.

#### Evaluation Metrics
- **Quality Metrics**: Content depth, accuracy, relevance
- **Collaboration Metrics**: Turn-taking, idea building, respectful disagreement
- **Progress Metrics**: Goal achievement, actionable outcomes, consensus building

#### Available Commands

1. **Basic Evaluation**
   ```bash
   python evaluate_meeting.py transcript.json
   ```

2. **Custom Output**
   ```bash
   python evaluate_meeting.py transcript.json --output custom_report.json
   ```

3. **System Status**
   ```bash
   python evaluate_meeting.py --status
   ```

4. **Generate Sample Transcript**
   ```bash
   python evaluate_meeting.py --sample
   ```

5. **Quiet Mode**
   ```bash
   python evaluate_meeting.py transcript.json --quiet
   ```

#### Evaluation Report Structure
```json
{
  "overall_assessment": {
    "average_score": 0.85,
    "quality_rating": "high",
    "meeting_effectiveness_score": 0.82
  },
  "agent_performance": {
    "Dr. Smith": {
      "overall_average": 0.88,
      "total_responses": 3
    }
  },
  "detailed_metrics": {
    "content_quality": {
      "average": 0.86,
      "scores": [0.8, 0.9, 0.88]
    }
  },
  "recommendations": [
    "Encourage more specific examples in responses",
    "Improve time management for discussions"
  ]
}
```

## Example Workflow

Here's a complete example using the provided sample data:

```bash
# Step 1: Compile transcript from individual response files
python compile_transcript.py AI_in_Sustainable_AGriculture/
# Output: ✅ Compiled transcript saved to: AI_in_Sustainable_AGriculture.json

# Step 2: Evaluate the compiled transcript
python evaluate_meeting.py AI_in_Sustainable_AGriculture.json
# Output: ✅ Report saved to: meeting_evaluation_report.json

# Step 3: View the evaluation results
cat meeting_evaluation_report.json
```

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: For Gemini evaluator
- `GROQ_API_KEY`: For Groq evaluator
- `OLLAMA_HOST`: For local Ollama (optional, defaults to localhost:11434)

### System Status Check
```bash
python evaluate_meeting.py --status
```
This will show:
- Available evaluators
- API key status
- System configuration
- Version information

## Troubleshooting

### Common Issues

1. **No JSON files found**
   ```
   ValueError: No JSON files found in path/to/folder
   ```
   **Solution**: Ensure the folder contains JSON response files with proper naming convention.

2. **API key not configured**
   ```
   ❌ Gemini: Not configured
   ```
   **Solution**: Set the appropriate environment variable for your chosen evaluator.

3. **File not found**
   ```
   ❌ Error: Transcript file not found: transcript.json
   ```
   **Solution**: Check the file path and ensure the transcript was compiled successfully.

### Debug Mode
For detailed error information, remove the `--quiet` flag when running evaluations:
```bash
python evaluate_meeting.py transcript.json
```

## File Naming Conventions

The system expects specific file naming patterns for proper compilation:

- **Agent responses**: `AgentName_roundNumber_response_YYYYMMDD_HHMMSS.json`
- **Opening statements**: `AgentName_opening_YYYYMMDD_HHMMSS.json`
- **Synthesis**: `AgentName_synthesis_YYYYMMDD_HHMMSS.json`
- **Critiques**: `AgentName_critique_YYYYMMDD_HHMMSS.json`
- **Final summaries**: `AgentName_final_summary_YYYYMMDD_HHMMSS.json`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run `python evaluate_meeting.py --status` to verify system configuration
3. Create an issue in the repository with detailed error information
