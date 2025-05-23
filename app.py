from __future__ import annotations

from typing import Dict, List
import streamlit as st

from agent_builder import build_local_agent
from think_tank import ThinkTank

from agno.memory.v2 import UserMemory

def build_custom_thinktank(project_desc: str, scientists_def: List[Dict[str, str]]) -> ThinkTank:
    """Instantiate ThinkTank and replace scientists with custom definitions."""
    lab = ThinkTank(project_desc)

    # Wipe auto‑generated scientists & add user ones
    lab.scientists.clear()
    for sd in scientists_def:
        lab.scientists.append(
            build_local_agent(
                name=sd["title"],
                description=f"Expertise: {sd['expertise']}. Goal: {sd['goal']}",
                role=sd["role"],
                memory=lab._memory,
                storage=lab._storage,
            )
        )
    return lab

st.set_page_config(page_title="Virtual Lab - Think Tank", layout="wide")
st.title("🧪 ThankTank - Lab Simulator")

st.sidebar.header("Project Setup")
project_desc = st.sidebar.text_area(
    "Main problem / project statement",
    placeholder="Describe the scientific problem you want the AI team to tackle…",
    height=150,
)

num_scientists = st.sidebar.number_input(
    "Number of scientists", min_value=1, max_value=8, value=3, step=1
)

# Prepare an editable table for scientist definitions
DEFAULT_ROWS = [
    {
        "title": f"Scientist {i+1}",
        "expertise": "",
        "goal": "",
        "role": ""
    }
    for i in range(num_scientists)
]

df = st.sidebar.data_editor(
    DEFAULT_ROWS,
    num_rows="dynamic",
    use_container_width=True,
    key="scientist_table",
)

st.sidebar.header("Team Meeting")
meeting_topic = st.sidebar.text_input("Meeting title / agenda", value="First strategy discussion")
rounds = st.sidebar.number_input("Number of rounds", min_value=1, max_value=5, value=3)

run_btn = st.sidebar.button("Run Team Meeting", disabled=not (project_desc and meeting_topic))

if run_btn:
    # Validate scientist rows (filter empty titles)
    sci_defs = [row for row in df if row["title"].strip()]
    if not sci_defs:
        st.error("Please provide at least one scientist with a title.")
        st.stop()

    lab = build_custom_thinktank(project_desc, sci_defs)

    # Containers for streaming output
    st.subheader("Team Meeting: " + meeting_topic)

    def write_agent(name: str, content: str):
        st.markdown(f"#### —— {name} ——")
        st.markdown(content)

    # 0. PI opening ----------------------------------------------------------
    pi_open = lab.pi.run(
        f"You are convening a team meeting. Agenda: {meeting_topic}. Share initial guidance.",
        stream=False,
    ).content
    lab._log("pi", lab.pi.name, pi_open)
    write_agent(lab.pi.name, pi_open)

    # Main discussion rounds -------------------------------------------------
    for r in range(1, rounds + 1):
        st.markdown(f"### 🔄 Round {r}/{rounds}")
        # Scientists speak
        for sci in lab.scientists:
            resp = sci.run(
                f"Context so far:\n{lab._context()}\n\nYour contribution for round {r}:",
                stream=False,
            ).content
            lab._log("scientist", sci.name, resp)
            write_agent(sci.name, resp)

        # Critic feedback
        crit = lab.critic.run(
            f"Context so far:\n{lab._context()}\n\nCritique round {r}",
            stream=False,
        ).content
        lab._log("critic", lab.critic.name, crit)
        write_agent(lab.critic.name, crit)

        # PI synthesis / follow‑ups
        synth = lab.pi.run(
            f"Context so far:\n{lab._context()}\n\nSynthesise round {r} and pose follow‑ups.",
            stream=False,
        ).content
        lab._log("pi", lab.pi.name, synth)
        write_agent(lab.pi.name + " (synthesis)", synth)

    # Final summary ----------------------------------------------------------
    summary = lab.pi.run(
        f"Context so far:\n{lab._context()}\n\nProvide the final detailed meeting summary and recommendations.",
        stream=False,
    ).content
    lab._log("pi", lab.pi.name, summary)

    st.subheader("📝 Final Meeting Summary")
    st.markdown(summary)

    # Persist summary
    
    lab._memory.add_user_memory(
        UserMemory(memory=summary), user_id="think_tank"
    )

    st.success("Meeting complete and saved to memory database.")