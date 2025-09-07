import json
import textwrap

from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
from typing import Dict, List

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.memory.v2.memory import Memory
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2 import UserMemory

from utils import now, indent
from agent_builder import build_local_agent

@dataclass
class Message:
    role: str   # "pi" | "scientist" | "critic" | "human"
    author: str
    content: str
    timestamp: str = field(default_factory = now)

    def chatml(self) -> Dict[str, str]:
        return {"role": "user" if self.role == "human" else "assistant", "content": self.content}
    
class ThinkTank:
    """Exact meeting loop replica running on local ollama."""

    def __init__(self, project_description: str, db_path: str = "ThinkTank.db", initialize = False) -> None:
        self.project_description = project_description

        # ── Persistent state ────────────────────────────────────────────────
        db = Path(db_path)
        db.parent.mkdir(parents=True, exist_ok=True)
        self._storage = SqliteStorage(table_name="sessions", db_file=str(db))
        self._memory = Memory(
            model=Ollama(id="llama3.1", options={"temperature": 0.0}),
            db=SqliteMemoryDb(table_name="memories", db_file=str(db)),
        )

        # ── Core agents ─────────────────────────────────────────────────────
        self.pi = build_local_agent(
            name="Coordinator",
            description="Leads the project, resolves conflicts, synthesises insights.",
            role="Provide final recommendations and meeting summaries.",
            temperature=0.2,
            memory=self._memory,
            storage=self._storage,
            enable_agentic_memory=True,
        )

        self.critic = build_local_agent(
            name="Critical Thinker",
            description="Spot logical flaws and methodological weaknesses.",
            role="Offer rigorous but constructive criticism.",
            temperature=0.3,
            memory=self._memory,
            storage=self._storage,
        )

        self.scientists: List[Agent] = []  # filled below
        self._messages: List[Message] = []

        # Build initial scientists from the PI’s suggestion ------------------
        if initialize:
            self._init_scientists()

    # ------------------------------------------------------------------
    # Agent generation
    # ------------------------------------------------------------------
    def _init_scientists(self) -> None:
        prompt = textwrap.dedent(
            f"""
            Project description: {self.project_description}

            You are a lab director creating a new interdisciplinary research team.

            List **exactly three** additional scientist agents best suited for this project.
            For each, return a JSON dictionary with the keys:
              - "title" (e.g. Computational Biologist)
              - "expertise"
              - "goal"
              - "role"

            Return **only** a JSON array. Do not add commentary.

            Example format:
            [
              {{
                "title": "Computational Biologist",
                "expertise": "Protein folding, antibody optimization",
                "goal": "Improve nanobody binding affinity",
                "role": "Use simulations to refine designs"
              }},
              ...
            ]
            """)
        resp = self.pi.run(prompt, stream=False).content
        try:
            defs: List[Dict[str, str]] = json.loads(resp)
        except Exception as e:
            print(resp)
            raise ValueError("PI did not output valid JSON for scientist definitions") from e

        for d in defs:
            agent = build_local_agent(
                name=d["title"],
                description=f"Expertise: {d['expertise']}. Goal: {d['goal']}",
                role=d["role"],
                memory=self._memory,
                storage=self._storage,
            )
            self.scientists.append(agent)

    # ------------------------------------------------------------------
    # Transcript helpers
    # ------------------------------------------------------------------
    def _log(self, role: str, author: str, content: str) -> None:
        self._messages.append(Message(role=role, author=author, content=content))
        with open("meeting_transcript.txt", "a", encoding="utf-8") as f:
            f.write(f"\n----- {author} -----\n{content}\n")
        with open("meeting_responses.jsonl", "a", encoding="utf-8") as jf:
            json.dump({
                "timestamp": msg.timestamp,
                "role": role,
                "author": author,
                "content": content
            }, jf)
            jf.write("\n")

    def _context(self) -> str:
        return "\n".join(f"[{m.timestamp}] {m.author}: {m.content}" for m in self._messages)

    # ------------------------------------------------------------------
    # Core meeting routines (unchanged flow)
    # ------------------------------------------------------------------
    def run_team_meeting(self, agenda: str, rounds: int = 3) -> str:
        print(f"\n=== Team Meeting | {agenda} ===")

        pi_open = self.pi.run(
            f"You are convening a team meeting. Agenda: {agenda}. Share initial guidance.",
            stream=False,
        ).content
        self._log("pi", self.pi.name, pi_open)
        print(indent(pi_open))

        for r in range(1, rounds + 1):
            print(f"\n--- Round {r}/{rounds} ---")
            for sci in self.scientists:
                resp = sci.run(
                    f"Context so far:\n{self._context()}\n\nYour contribution for round {r}:",
                    stream=False,
                ).content
                self._log("scientist", sci.name, resp)
                print(indent(f"{sci.name}: {resp}"))

            critique = self.critic.run(
                f"Context so far:\n{self._context()}\n\nCritique round {r}",
                stream=False,
            ).content
            self._log("critic", self.critic.name, critique)
            print(indent(f"{self.critic.name}: {critique}"))

            synth = self.pi.run(
                f"Context so far:\n{self._context()}\n\nSynthesise round {r} and pose follow-ups.",
                stream=False,
            ).content
            self._log("pi", self.pi.name, synth)
            print(indent(synth))

        summary = self.pi.run(
            f"Context so far:\n{self._context()}\n\nProvide the final detailed meeting summary and recommendations.",
            stream=False,
        ).content
        self._log("pi", self.pi.name, summary)
        print("\n=== Meeting Complete ===\n")

        self._memory.add_user_memory(UserMemory(memory = summary), user_id="think_tank")
        return summary

    # ------------------------------------------------------------------
    # Individual meeting (Scientist ↔ Critic)                              
    # ------------------------------------------------------------------
    def run_individual_meeting(
        self,
        agent: Agent,
        agenda: str,
        rounds: int = 2,
    ) -> str:
        """Hold a focused meeting with **one scientist + the critic**, mirroring the paper."""
        title = f"Individual Meeting | {agent.name} | Agenda: {agenda}"
        print(f"=== {title} ===")

        # Scientist initial answer
        answer = agent.run(agenda, stream=False).content
        self._log("scientist", agent.name, answer)
        print(indent(answer))

        for r in range(1, rounds + 1):
            critique = self.critic.run(
                f"Scientist answer:{answer}\nProvide constructive critique (round {r}/{rounds}).",
                stream=False,
            ).content
            self._log("critic", self.critic.name, critique)
            print(indent(f"{self.critic.name}: {critique}"))

            answer = agent.run(
                f"Critique:{critique} Revise your answer accordingly (round {r}/{rounds}).",
                stream=False,
            ).content
            self._log("scientist", agent.name, answer)
            print(indent(answer))

        self._memory.add_user_memory(UserMemory(memory = answer),"think_tank")
        print("=== Individual Meeting Complete ===")
        return answer

    # ------------------------------------------------------------------
    # Parallel creative meetings + merge                                  
    # ------------------------------------------------------------------
    def run_parallel_team_meetings(
        self,
        agenda: str,
        rounds: int = 3,
        parallel_runs: int = 4,
        ) -> str:
        """Run several *creative* meetings (higher temperature) and merge them."""
        summaries: List[str] = []

        # Temporarily crank temperature for creativity
        agents = [self.pi, self.critic, *self.scientists]
        original_temps = {a.name: a.model.options.get("temperature", 0.2) for a in agents}

        for i in range(parallel_runs):
            print(f"Parallel Meeting {i+1}/{parallel_runs}")
            for a in agents:
                a.model.options["temperature"] = 0.8
            summaries.append(self.run_team_meeting(agenda, rounds))

        # Reset temperatures
        for a in agents:
            a.model.options["temperature"] = original_temps[a.name]

        # Merge‑meeting conducted by PI
        merge_prompt = textwrap.dedent(
            f"""
            We held {parallel_runs} parallel meetings on the agenda below.  Combine their **best ideas** into a
            single, coherent recommendation.  Remove contradictions, keep arguments well-reasoned and cite the
            originating meeting when helpful.

            Agenda: {agenda}
            --------------------------------------------------------------------
            {chr(10).join(f'### SUMMARY {j+1}{s}' for j, s in enumerate(summaries))}
            """
        )
        merged = self.pi.run(merge_prompt, stream=False).content
        self._log("pi", self.pi.name, merged)
        self._memory.add_user_memory(UserMemory(memory = merged),"think_tank")
        return merged


if __name__ == "__main__":
    DESC = (
        "Use machine learning to develop nanobodies that bind to KP.3 variant "
        "of SARS-CoV-2 spike protein while retaining cross-reactivity."
    )
    lab = ThinkTank(DESC)
    lab.run_team_meeting("Choose initial nanobody design strategy")