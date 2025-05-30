# ThinkTank – Agentic Research Simulation.

**ThinkTank** is a fully local, agentic research platform that is inspired by the team-based architecture described in the paper:

> *The Virtual Lab: AI Agents Design New SARS-CoV-2 Nanobodies with Experimental Validation*  
> [Read the paper](https://www.biorxiv.org/content/10.1101/2024.11.11.623004v1)

This implementation simulates a team made up of local LLM-based agents/experts, running fully offline via [Ollama](https://ollama.com) and powered by the `Llama3.1` model.

---

## 💡 What It Does

ThinkTank orchestrates collaborative meetings between the following agents:

- **Generalized Coordinator**  
  Synthesizes all ideas, resolves debates, and produces final meeting summaries and project decisions.

- **Critic Thinker**  
  Points out flaws, inconsistencies, or gaps in reasoning, and demands scientific rigor.

- **Domain Experts**  
  Experts are created on-the-fly based on your project description. Each has a different role, expertise, and contribution style.

### Meeting Types Supported

- **Team Meetings**: All agents interact in multi-round discussions on a shared agenda.
- **Individual Meetings**: One expert is paired with the critic for focused work.
- **Parallel + Merge**: Multiple creative meetings are run in parallel, and their results are merged into a single synthesis.

All meeting discussions and summaries are **stored in a local memory database** for future reference and summarization.

---

## Requirements

1. Python version >= 3.11
2. git

---

## Setup

### 1. Install Ollama and pull the model

```bash
brew install ollama               # Or follow https://ollama.com/download
ollama pull llama3.1              # This will download the model
ollama serve                      # Starts the Ollama server on localhost:11434
```

### 2. Clone and install Python dependencies

```bash
git clone https://github.com/taugroup/ThinkTank.git
cd ThinkTank

# Create a clean environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the script

```bash
python main.py
```

### 4. `meeting_transcript.txt` contains the responses from all the agents.

---

## Designer, Developer, Architect
- Praneet Sai Madhu Surabhi, MS. Data Science '25, TAU Group
  > Advisor: Prof. Dr. Jian Tao, Asst. Dir. of TAMIDS, Dir. of Digital Twin Lab
