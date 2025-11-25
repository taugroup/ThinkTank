# ThinkTank Meeting Evaluation System

An AI-powered evaluation framework for analyzing multi-agent collaborative meetings from the ThinkTank project. This system compiles individual agent response files into structured transcripts and evaluates meeting quality, collaboration effectiveness, and research outcomes using multiple LLM evaluators.

## Overview

The Evaluation system is a critical component of the **ThinkTank** framework - a platform for orchestrating collaborative AI agent meetings in scientific research and domain-specific problem-solving. This module provides:

1. **Transcript Compilation**: Converts individual agent response JSON files from ThinkTank meetings into structured, chronologically-ordered transcripts
2. **Multi-Evaluator Assessment**: Uses Gemini, Groq, and Qwen LLMs to evaluate meeting quality across multiple dimensions
3. **Comprehensive Reporting**: Generates detailed reports on agent performance, collaboration metrics, and meeting effectiveness

### Integration with ThinkTank

ThinkTank orchestrates meetings between specialized AI agents (Coordinator, Critical Thinker, and Domain Experts) to solve complex problems. This Evaluation system analyzes those meetings to:
- Assess the quality of agent contributions
- Measure collaboration and idea-building effectiveness
- Track progress toward meeting goals
- Identify areas for improvement in agent interactions

## System Architecture

```
📁 Evaluation/
├── 📄 __init__.py                    # Package initialization & exports
├── 📄 main.py                        # Main evaluation system entry point
├── 📄 compile_transcript.py          # Transcript compilation CLI tool
├── 📄 evaluate_meeting.py            # Meeting evaluation CLI tool
│
├── 📁 core/                          # Core evaluation engine
│   ├── __init__.py
│   ├── evaluation_manager.py        # Orchestrates multi-evaluator assessment
│   ├── transcript_parser.py         # Parses and validates transcript data
│   └── report_generator.py          # Generates comprehensive evaluation reports
│
├── 📁 evaluators/                    # LLM evaluator implementations
│   ├── __init__.py
│   ├── base_evaluator.py            # Abstract base class for evaluators
│   ├── gemini_evaluator.py          # Google Gemini evaluator
│   ├── groq_evaluator.py            # Groq evaluator
│   └── qwen_evaluator.py            # Qwen (Ollama) evaluator
│
├── 📁 metrics/                       # Evaluation metric definitions
│   ├── __init__.py
│   ├── quality_metrics.py           # Content quality, depth, accuracy
│   ├── collaboration_metrics.py     # Turn-taking, idea building, respect
│   └── progress_metrics.py          # Goal achievement, consensus, outcomes
│
└── 📁 utils/                         # Helper utilities
    ├── __init__.py
    ├── config.py                     # Configuration management
    └── helpers.py                    # API validation, file loading
```

### Key Components

- **EvaluationSystem**: Main interface for running evaluations
- **EvaluationManager**: Coordinates multiple evaluators in parallel
- **TranscriptParser**: Validates and structures meeting transcript data
- **ReportGenerator**: Aggregates evaluator results into comprehensive reports
- **Evaluators**: Pluggable LLM implementations (Gemini, Groq, Qwen)
- **Metrics**: Standardized evaluation criteria across quality, collaboration, and progress dimensions

## Prerequisites

- **Python 3.11+** (tested with 3.11.11)
- **At least one LLM evaluator** configured:
  - **Gemini**: `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable
  - **Groq**: `GROQ_API_KEY` environment variable
  - **Qwen**: Local Ollama installation with `qwen2.5:32b` model
- **ThinkTank meeting transcripts**: Generated from ThinkTank agent meetings (or use the compilation tool)

## Installation

### 1. Install ThinkTank (if not already installed)

```bash
git clone https://github.com/taugroup/ThinkTank.git
cd ThinkTank
```

### 2. Set up Python environment

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure LLM Evaluators

Choose at least one evaluator to use:

**Option A: Gemini (Recommended)**
```bash
export GEMINI_API_KEY="your_gemini_api_key"
# Or alternatively:
export GOOGLE_API_KEY="your_google_api_key"
```

**Option B: Groq**
```bash
export GROQ_API_KEY="your_groq_api_key"
```

**Option C: Qwen (Local/Offline)**
```bash
# Install Ollama
brew install ollama  # Or visit https://ollama.com/download

# Pull the Qwen model
ollama pull qwen2.5:32b

# Start Ollama server
ollama serve
```

### 4. Verify Setup

```bash
cd Evaluation
python evaluate_meeting.py --status
```

This will show which evaluators are available and properly configured.

## Quick Start

### Complete Workflow Example

```bash
cd Evaluation

# Step 1: Compile individual agent responses into a structured transcript
python compile_transcript.py AI_in_Sustainable_AGriculture/
# Output: ✅ Compiled transcript saved to: AI_in_Sustainable_AGriculture.json

# Step 2: Evaluate the meeting transcript
python evaluate_meeting.py AI_in_Sustainable_AGriculture.json
# Output: ✅ Report saved to: meeting_evaluation_report.json

# Step 3: Review the evaluation results
cat meeting_evaluation_report.json
```

### Using with ThinkTank Meetings

After running a ThinkTank meeting through the main application:

```bash
# 1. Navigate to the Evaluation directory
cd Evaluation

# 2. Compile the meeting responses (stored in ../Transcript Folders/)
python compile_transcript.py "../Transcript Folders/YourMeetingFolder/"

# 3. Evaluate the compiled transcript
python evaluate_meeting.py YourMeetingFolder.json

# 4. View results
python evaluate_meeting.py YourMeetingFolder.json --output detailed_report.json
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

## Understanding the Evaluation Process

### What Gets Evaluated

The system analyzes ThinkTank meetings across three key dimensions:

**1. Quality Metrics**
- Content depth and technical accuracy
- Relevance to meeting goals
- Evidence-based reasoning
- Clarity and coherence

**2. Collaboration Metrics**
- Turn-taking balance
- Idea building and synthesis
- Respectful disagreement
- Active listening and engagement

**3. Progress Metrics**
- Goal achievement
- Actionable outcomes
- Consensus building
- Decision clarity

### Evaluation Output

Each evaluation generates a comprehensive report including:

- **Overall Assessment**: Aggregate quality score and rating (low/medium/high/excellent)
- **Agent Performance**: Individual scores for each agent (Coordinator, Critical Thinker, Domain Experts)
- **Detailed Metrics**: Breakdown by quality, collaboration, and progress dimensions
- **Recommendations**: Actionable suggestions for improving future meetings
- **Meeting Metadata**: Duration, participant count, round progression

### Example Output

```json
{
  "overall_assessment": {
    "average_score": 0.85,
    "quality_rating": "high",
    "meeting_effectiveness_score": 0.82
  },
  "agent_performance": {
    "Coordinator": {
      "overall_average": 0.88,
      "total_responses": 5,
      "quality_average": 0.87,
      "collaboration_average": 0.89,
      "progress_average": 0.88
    },
    "Critical Thinker": {
      "overall_average": 0.84,
      "total_responses": 3
    }
  },
  "recommendations": [
    "Encourage more specific examples in technical discussions",
    "Improve time management for consensus building"
  ]
}
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

## Advanced Usage

### Programmatic API

You can use the evaluation system programmatically in your own Python scripts:

```python
from Evaluation import evaluate_meeting_transcript, EvaluationSystem

# Method 1: Direct file evaluation
report = evaluate_meeting_transcript(
    transcript_file_path="meeting_transcript.json"
)

# Method 2: Using transcript data directly
transcript_data = {
    "meeting_topic": "AI in Healthcare",
    "project_name": "HealthTech Initiative",
    "experts": [...],
    "transcript": [...]
}
report = evaluate_meeting_transcript(transcript_data=transcript_data)

# Method 3: Using EvaluationSystem class
system = EvaluationSystem()
status = system.get_system_status()
report = system.evaluate_transcript_file("meeting_transcript.json")
```

### Custom Configuration

Create a custom configuration for evaluation parameters:

```python
from Evaluation import EvaluationSystem, Config

config = Config()
config.set("evaluation.max_workers", 5)
config.set("evaluation.timeout_per_evaluation", 120)

system = EvaluationSystem(config=config)
report = system.evaluate_transcript_file("transcript.json")
```

## Integration with ThinkTank Workflow

The Evaluation system is designed to work seamlessly with ThinkTank's meeting output:

1. **ThinkTank Meeting** → Generates individual agent response JSON files
2. **Compile Transcript** → Aggregates responses into structured transcript
3. **Evaluate Meeting** → Analyzes transcript quality and effectiveness
4. **Review & Iterate** → Use insights to improve agent configurations

This feedback loop helps optimize:
- Agent prompt engineering
- Domain expert selection
- Meeting structure and flow
- Collaboration dynamics

## Contributing

Contributions are welcome! To contribute:

1. Fork the ThinkTank repository
2. Create a feature branch (`git checkout -b feature/evaluation-improvement`)
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Areas for Contribution
- Additional LLM evaluator implementations
- New evaluation metrics
- Performance optimizations
- Documentation improvements

## Citation

If you use the ThinkTank Evaluation system in your research, please cite:

```bibtex
@article{surabhi2025thinktank,
  title={ThinkTank: A Framework for Generalizing Domain-Specific AI Agent Systems into Universal Collaborative Intelligence Platforms},
  author={Surabhi, Praneet Sai Madhu and Mudireddy, Dheeraj Reddy and Tao, Jian},
  journal={arXiv preprint arXiv:2506.02931},
  year={2025}
}
```

## Team

- **Developers**: Praneet Sai Madhu Surabhi, Dheeraj Mudireddy (MS Data Science '25, TAU Group)
- **Advisor**: Prof. Dr. Jian Tao (Asst. Dir. of TAMIDS, Dir. of Digital Twin Lab)

## License

This project is part of the ThinkTank framework and is licensed under the MIT License.

## Support

For issues and questions:

1. **Check System Status**: Run `python evaluate_meeting.py --status`
2. **Review Troubleshooting**: See the troubleshooting section above
3. **GitHub Issues**: Create an issue at https://github.com/taugroup/ThinkTank
4. **Documentation**: Review the main ThinkTank README for context

## Related Links

- [ThinkTank Main Repository](https://github.com/taugroup/ThinkTank)
- [ThinkTank Paper (arXiv)](https://arxiv.org/abs/2506.02931)
- [Virtual Lab Paper (bioRxiv)](https://www.biorxiv.org/content/10.1101/2024.11.11.623004v1) - Inspiration for multi-agent collaboration
