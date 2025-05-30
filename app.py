from __future__ import annotations
import concurrent.futures

from pathlib import Path
from typing import Dict, List, Any
import json
import time
import base64

import streamlit as st

from agent_builder import build_local_agent
from ingestion import process_documents
from retrieval import retrieve_documents
from think_tank import ThinkTank
from utils import export_meeting, clean_name
from agno.memory.v2 import UserMemory

DB_FILE = Path("projects_db.json")
TEMPLATE_FILE = Path("scientist_templates.json")

def img_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

img_b64 = img_to_base64("assets/Logo_tau.png")

if "rows" not in st.session_state:
    st.session_state.rows = []

if "selected_template" not in st.session_state:
    st.session_state.selected_template = "<select>"

if "markdown_log" not in st.session_state:
    st.session_state.markdown_log = []

rows = st.session_state.rows

def download_function(project_name: str,
                            project_desc: str,
                            project_data: Dict[str, Any],
                            meeting: Dict[str, Any],
                            transcript) -> None:
    
    files = export_meeting(project_name, project_desc, project_data["scientists"], meeting, transcript)
    st.download_button("⬇️  DOCX", files["docx"],
                    file_name=f"{meeting['topic']}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

#  project DB 

def _load_projects() -> Dict[str, Any]:
    if DB_FILE.exists():
        return json.loads(DB_FILE.read_text())
    return {}


def _save_projects(data: Dict[str, Any]):
    DB_FILE.write_text(json.dumps(data, indent=2))

#  scientist templates DB

def _load_templates() -> List[Dict[str, str]]:
    if TEMPLATE_FILE.exists():
        return json.loads(TEMPLATE_FILE.read_text())
    # seed with three defaults on first run
    defaults = [
        {"title": "Immunologist", "expertise": "Immunopathology, antibody‑antigen interactions", "goal": "Guide immune‑targeting strategies", "role": "Analyse epitope selection and immune response"},
        {"title": "Machine Learning Expert", "expertise": "Deep learning, protein sequence modelling", "goal": "Develop predictive models for design", "role": "Build & chain ML models to rank candidates"},
        {"title": "Computational Biologist", "expertise": "Protein folding simulation, molecular dynamics", "goal": "Validate structural stability", "role": "Simulate docking & refine structures"},
    ]
    TEMPLATE_FILE.write_text(json.dumps(defaults, indent=2))
    return defaults


def _save_templates(templates: List[Dict[str, str]]) -> None:
    """Persist scientist templates to disk."""
    TEMPLATE_FILE.write_text(json.dumps(templates, indent=2))

def build_custom_thinktank(project_desc: str, scientists: List[Dict[str, str]]) -> ThinkTank:
    lab = ThinkTank(project_desc)
    lab.scientists.clear()
    tools = [retrieve_documents]
    for sd in scientists:
        lab.scientists.append(
            build_local_agent(
                name=sd["title"],
                description=f"Expertise: {sd['expertise']}. Goal: {sd['goal']}",
                role=sd["role"],
                memory=lab._memory,
                storage=lab._storage,
                tools=tools,
            )
        )
    return lab

def write(name: str, content: str):
        st.markdown(f"## 🧑‍🔬 {name} \n")
        st.session_state.markdown_log.append(f"## 🧑‍🔬 {name} \n")
        st.markdown(content)
        st.session_state.markdown_log.append(content)

def run_thinktank_meeting(
    project_name: str,
    project_desc: str,
    scientists: List[Dict[str, str]],
    meeting_topic: str,
    rounds: int,
    projects_db: Dict[str, Any],
):
    """Execute a team meeting and write transcript + summary back to database."""

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for scientist in st.session_state.rows:
            print(f"Processing scientist: {scientist['title']}")
            files = scientist.get('files', [])
            print(f"Files for {scientist['title']}: {files}")
            file_bytes_list = [(f.name, f.getvalue()) for f in files]

            future = executor.submit(process_documents, file_bytes_list, clean_name(scientist['title']))
            futures.append(future)

        # Optionally wait for all to complete and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    lab = build_custom_thinktank(project_desc, scientists)

    st.subheader(f"🧑‍🔬 Team Meeting - {meeting_topic}")
    st.session_state.markdown_log.append(f"# 🧑‍🔬 Team Meeting - {meeting_topic}")
    def log(name: str, content: str):
        write(name, content)

    # PI opening
    pi_open = lab.pi.run(
        f"You are convening a team meeting. Agenda: {meeting_topic}. Share initial guidance to the scientists..",
        stream=False,
    ).content
    log(lab.pi.name, pi_open)

    # Discussion rounds 
    for r in range(1, rounds + 1):
        st.markdown(f"### 🔄 Round {r}/{rounds}")
        st.session_state.markdown_log.append(f"### 🔄 Round {r}/{rounds}")
        for sci in lab.scientists:

            tool_prompt = f"""
                You have access to the following tool:

                1.Tool: `retrieve_documents`
                    - Purpose: Retrieve relevant document chunks from the knowledge database using natural language queries.
                    - Usage:
                        1. Analyze the current task or context and formulate meaningful queries.
                        2. Call: retrieve_documents(queries: List[str], collection_name: str) -> List[str]
                        3. Use collection_name = {clean_name(sci.name)}

                    Instructions:
                    - First, think about what information is needed to accomplish your task.
                    - Generate targeted, specific queries based on your expertise.
                    - Use `retrieve_documents` to fetch supporting content.
                    - Incorporate retrieved content directly into your reasoning or task output.
                    - **Do not output the summary or paraphrase the retrieved content — use it as-is.**

                Your goal is to leverage the retrieved knowledge to solve the task accurately and completely.
            """

            resp = sci.run(
                f"{tool_prompt}\nContext so far:\n{lab._context()}\n\nYour contribution for round {r}:",
                stream=False,
            ).content
            lab._log("scientist", sci.name, resp)
            log(sci.name, resp)

        crit = lab.critic.run(
            f"Context so far:\n{lab._context()}\n\nCritique round {r}",
            stream=False,
        ).content
        lab._log("critic", lab.critic.name, crit)
        log(lab.critic.name, crit)

        synth = lab.pi.run(
            f"Context so far:\n{lab._context()}\n\nSynthesise round {r} and pose follow‑ups.",
            stream=False,
        ).content
        lab._log("pi", lab.pi.name, synth)
        log(lab.pi.name + " (synthesis)", synth)

    # Final summary
    summary = lab.pi.run(
        f"Context so far:\n{lab._context()}\n\nProvide the final detailed meeting summary and recommendations.",
        stream=False,
    ).content
    lab._log("pi", lab.pi.name, summary)

    st.subheader("📝 Final Meeting Summary")
    st.session_state.markdown_log.append("📝 Final Meeting Summary")
    st.markdown(summary)
    st.session_state.markdown_log.append(summary)

    lab._memory.add_user_memory(memory=UserMemory(memory=summary), user_id=project_name)


    # Save to DB
    proj = projects_db.setdefault(project_name, {"description": project_desc, "scientists": scientists, "meetings": []})
    for sci in scientists:
        sci.pop("files", None)  # remove files from scientists to avoid bloating the DB
    proj["description"] = project_desc
    proj["scientists"] = scientists
    proj["meetings"].append({
        "timestamp": int(time.time()),
        "topic": meeting_topic,
        "rounds": rounds,
        "transcript": st.session_state.markdown_log,
        "summary": summary,
    })
    _save_projects(projects_db)
    st.success("Meeting complete and saved 🎉")
    return proj

st.set_page_config(page_title="Think Tank", layout="wide")
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            width: 500px;
            max-width: 1500px;
        }
        [data-testid="stSidebar"] + div .block-container {
            padding-left: 300px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='text-align:center;'>🧠 Think Tank</h1>",
    unsafe_allow_html=True,
)

projects_db = _load_projects()
project_names = sorted(projects_db.keys())

# ── Project selection / creation 
st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_b64}" width="40" style="margin-right:10px;">
        <span style="font-size: 14px;">Developed by TAU Group</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.header("Project Manager")
proj_choice = st.sidebar.selectbox("Select a Project", ["➕ New project"] + project_names)

if proj_choice == "➕ New project":
    project_name = st.sidebar.text_input("New project name")
    project_data = {"description": "", "scientists": [], "meetings": []}
    if not project_name:
        st.stop()
else:
    project_name = proj_choice
    project_data = projects_db[project_name]

# ── Project description --
project_desc = st.sidebar.text_area("Project description", value=project_data.get("description", ""), height=120)

# ── Scientist roster --

# Scientist templates loaded from disk 
TEMPLATES: List[Dict[str, str]] = _load_templates()

st.sidebar.subheader("Scientist Manager")

# Template management
with st.sidebar.expander("Manage Scientists and templates", expanded=False):
    # Select existing template to load
    selected_tpl_title = st.selectbox("Load template to edit", ["<new>"] + [t["title"] for t in TEMPLATES], key="tpl_select")

    # Populate fields depending on selection
    if selected_tpl_title == "<new>":
        tpl_data = {"title": "", "expertise": "", "goal": "", "role": ""}
    else:
        tpl_data = next(t for t in TEMPLATES if t["title"] == selected_tpl_title)

    new_t = st.text_input("Title", value=tpl_data["title"], key="tpl_title")
    col1, col2 = st.columns(2)
    with col1:
        new_e = st.text_area("Expertise", value=tpl_data["expertise"], height=70, key="tpl_exp")
    with col2:
        new_g = st.text_area("Goal", value=tpl_data["goal"], height=70, key="tpl_goal")
    new_r = st.text_area("Role", value=tpl_data["role"], height=70, key="tpl_role")

    # Save / update template
    if st.button("Save template") and new_t.strip():
        TEMPLATES = [t for t in TEMPLATES if t["title"] != new_t]
        TEMPLATES.append({"title": new_t, "expertise": new_e, "goal": new_g, "role": new_r})
        _save_templates(TEMPLATES)
        (st.rerun if hasattr(st, 'rerun') else st.experimental_rerun)()

    # delete templates
    del_tpl = st.multiselect("Delete templates", [t["title"] for t in TEMPLATES])
    if del_tpl and st.button("Remove selected templates"):
        TEMPLATES = [t for t in TEMPLATES if t["title"] not in del_tpl]
        _save_templates(TEMPLATES)
        (st.rerun if hasattr(st, 'rerun') else st.experimental_rerun)()

    #  Load or initialise project rows  project rows 
    if proj_choice == '➕ New project':
        if "rows" not in st.session_state:
            st.session_state.rows = project_data.get("scientists", [])
    else:
        if len(st.session_state.rows) < 1 or 'files' not in st.session_state.rows[-1]:
            st.session_state.rows = project_data.get("scientists", [])

    def _add_template():
        sel = st.session_state.tpl_selectbox
        if sel != "<select>" and sel not in [r["title"] for r in st.session_state.rows]:
            st.session_state.rows.append(next(t for t in TEMPLATES if t["title"] == sel))
            _save_projects(projects_db)
        st.session_state.tpl_selectbox = "<select>"

    #  Add from template --
    st.selectbox(
        "Add scientist from template",
        ["<select>"] + [t["title"] for t in TEMPLATES],
        key="tpl_selectbox",
        on_change=_add_template,
    )

    #  Manual create scientist 
    if st.button("Add blank scientist"):
        st.session_state.rows.append({"title": f"Scientist {len(st.session_state.rows)+1}", "expertise": "", "goal": "", "role": ""})
        st.rerun()

    #  Editable table 
    st.session_state.rows = st.session_state.rows[:8]  # limit to 8
    display_rows = []
    real_rows = st.session_state.rows
    for r in real_rows:
        r_view = r.copy()
        if "files" in r_view:
            # show just the count or filenames
            val = r_view["files"]
            r_view["files"] = len(val) if isinstance(val, (list, tuple)) else (val or 0)
        display_rows.append(r_view)

    scientist_table = st.data_editor(display_rows, num_rows="dynamic", use_container_width=True, key="scientist_table")

    def _files_changed(i: int):
        uploaded = st.session_state[f"files_{i}"]      # new value after change
        st.session_state.rows[i]["files"] = uploaded
        print(f"Files for scientist {i}: {[f.name for f in uploaded]}")

    for i, scientist in enumerate(st.session_state.rows):
        c = st.container()
        c.markdown(f"### Files for {scientist['title'] or f'Scientist {i+1}'}")
        c.file_uploader(
            "Choose files for the vector store",
            accept_multiple_files=True,
            key=f"files_{i}",
            on_change=_files_changed,
            args=(i,),        # pass only the index
        )

# ── Meeting selection / creation -- / creation --
st.sidebar.subheader("Team Meeting")
meetings = project_data.get("meetings", [])
meeting_labels = [f"{i+1}. {m['topic']}" for i, m in enumerate(meetings)]
meeting_choice = st.sidebar.selectbox("Select a meeting", ["New meeting"] + meeting_labels)

if meeting_choice == "New meeting":
    meeting_topic = st.sidebar.text_input("New meeting topic / title")
    rounds = int(st.sidebar.number_input("Rounds", min_value=1, value=3, step=1))
    run_btn = st.sidebar.button("Run Team Meeting")
    if run_btn:
        clean_scientists = [row for row in scientist_table if row["title"].strip()]
        if not clean_scientists:
            st.error("Provide at least one scientist")
            st.stop()
        proj = run_thinktank_meeting(project_name, project_desc, clean_scientists, meeting_topic, rounds, projects_db)
        download_function(project_name, project_desc, project_data, proj["meetings"][-1], st.session_state.markdown_log)
else:
    # Load existing meeting 
    sel_index = meeting_labels.index(meeting_choice)
    meeting = meetings[sel_index]
    st.markdown(f"## 🗂️ Meeting Record - {meeting['topic']}")
    st.session_state.markdown_log.append(f"## 🗂️ Meeting Record - {meeting['topic']}")
    for msg in meeting["transcript"]:
        if type(msg) is dict:
            name = msg.get("name", "Unknown")
            content = msg.get("content", "")
            write(name, content)
        else:
            st.markdown(msg)
            st.session_state.markdown_log.append(msg)
    st.markdown("### Summary")
    st.session_state.markdown_log.append("### Summary")
    st.markdown(meeting["summary"])
    st.session_state.markdown_log.append(meeting["summary"])
    download_function(project_name, project_desc, project_data, meeting, meeting['transcript'])