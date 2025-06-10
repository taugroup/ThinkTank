from typing import Optional
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.memory.v2.memory import Memory
from agno.storage.sqlite import SqliteStorage

def build_local_agent(
    *,
    name: str,
    description: str,
    role: str,
    temperature: float = 0.2,
    memory: Optional[Memory] = None,
    storage: Optional[SqliteStorage] = None,
    enable_agentic_memory: bool = False,
    **extra_agent_kwargs,
) -> Agent:
    """Return an Agent backed by a *local* llama3.1:8b Ollama model."""

    model = Ollama(
        id="qwen3:8b",
        options={"temperature": temperature},
    )

    return Agent(
        name=name,
        description=description,
        role=role,
        model=model,
        markdown=True,
        memory=memory,
        enable_agentic_memory=enable_agentic_memory,
        storage=storage,
        add_history_to_messages=True,
        num_history_runs=3,
        **extra_agent_kwargs,
    )
