# ThinkTank – A Framework for Generalizing Domain-Specific AI Agent Systems into Universal Collaborative Intelligence Platforms

The ThinkTank framework draws inspiration from the collaborative AI methodologies demonstrated in [*The Virtual Lab: AI Agents Design New SARS-CoV-2 Nanobodies with Experimental Validation*](https://www.biorxiv.org/content/10.1101/2024.11.11.623004v1), which showcased the effectiveness of multi-agent systems in scientific research and hypothesis generation.



The implementation of ThinkTank simulates a scientific team made up of local LLM-based agents enhanced with documents provided by the user, running fully offline via [Ollama](https://ollama.com) and any LLM models supported by [Agno](https://www.agno.com/).

---

## 💡 What It Does

ThinkTank orchestrates collaborative research meetings between the following agents:

- **Coordinator**  
  Synthesizes all ideas, resolves debates, and produces final meeting summaries and project decisions.

- **Critical Thinker**  
  Points out flaws, inconsistencies, or gaps in reasoning, and demands scientific rigor.

- **Domain Expert Agents**  
  User can input any number of experts with their own vector stores which act as RAG based tool for the agent. Each has a different role, expertise, and contribution style.

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

# To handle mermaid code from LLMs
npm install -g @mermaid-js/mermaid-cli
```

### 3. Run the script

```bash
streamlit run app.py
```

### 4. Meeting transcript can be downloaded as a DOCX file from the app.

---

## Designer, Developer, Architect
- Praneet Sai Madhu Surabhi, MS. Data Science '25, TAU Group
  > Advisor: Prof. Dr. Jian Tao, Asst. Dir. of TAMIDS, Dir. of Digital Twin Lab
