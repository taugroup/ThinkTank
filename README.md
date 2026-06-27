# Policy Think Tank — Local-First Multi-Agent Policy Analysis

> **This repository is being refactored** from TAU Group's scientific **ThinkTank**
> into a **local-first multi-agent policy think tank**. You enter a broad policy
> question (e.g. *"Should Boston implement congestion pricing downtown?"*) and the
> system delegates research to stakeholder agents, synthesizes a recommendation,
> red-teams it, and forecasts effects with deterministic Python.
>
> - **Run the app:** `streamlit run app.py` (runs in mock mode with no model needed).
> - **Run tests:** `python -m pytest tests/`
> - **Run evals:** `python evals/run_evals.py`
> - **Architecture:** see [`ARCHITECTURE.md`](ARCHITECTURE.md),
>   [`MODEL_SELECTION.md`](MODEL_SELECTION.md), [`ADLC.md`](ADLC.md).
>
> **Attribution & license.** Adapted from **TAU Group's ThinkTank** (Texas A&M
> University), MIT-licensed — see [`LICENSE`](LICENSE). The original scientific
> virtual-lab documentation is preserved below for reference; the meeting-loop
> entrypoints are being retired in favor of the LangGraph policy workflow.

---

# (Original) ThinkTank – A Framework for Generalizing Domain-Specific AI Agent Systems into Universal Collaborative Intelligence Platforms

The ThinkTank framework draws inspiration from the collaborative AI methodologies demonstrated in [*The Virtual Lab: AI Agents Design New SARS-CoV-2 Nanobodies with Experimental Validation*](https://www.biorxiv.org/content/10.1101/2024.11.11.623004v1), which showcased the effectiveness of multi-agent systems in scientific research and hypothesis generation.

The implementation of ThinkTank simulates a scientific team made up of local LLM-based agents enhanced with documents provided by the user, running fully offline via [Ollama](https://ollama.com) and any LLM models supported by [Agno](https://www.agno.com/).

Please find the paper here: https://arxiv.org/abs/2506.02931

---

## 💡 What It Does

ThinkTank orchestrates collaborative research meetings between the following agents:

- **Coordinator**  
  Synthesizes all ideas, resolves debates, and produces final meeting summaries and project decisions.

- **Critical Thinker**  
  Points out flaws, inconsistencies, or gaps in reasoning, and demands scientific rigor.

- **Domain Expert Agents**  
Users can create any number of Domain Expert agents, each equipped with their own specialized collection in the vector store that serves as a RAG-based tool. Each expert agent is configured with distinct roles, areas of expertise, and unique contribution styles tailored to specific domains or organizational needs.

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

### 4. Meeting transcript can be downloaded as a Word(.docx) file from the app.

---

## Citation
```text
@article{surabhi2025thinktank,
  title={ThinkTank: A Framework for Generalizing Domain-Specific AI Agent Systems into Universal Collaborative Intelligence Platforms},
  author={Surabhi, Praneet Sai Madhu and Mudireddy, Dheeraj Reddy and Tao, Jian},
  journal={arXiv preprint arXiv:2506.02931},
  year={2025}
}
```

## Team
- (In no particular order) Praneet Sai Madhu Surabhi, Dheeraj Mudireddy, Sujith Julakanti {MS. Data Science '25, TAU Group}
- Advisor: Prof. Dr. Jian Tao, Asst. Dir. of TAMIDS, Dir. of Digital Twin Lab
