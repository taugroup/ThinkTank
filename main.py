import json
import time
import base64
import concurrent.futures
from typing import List
import asyncio

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import Meeting, Expert, Project
from utils import load_projects, save_projects, load_templates, save_templates

from ingestion import process_documents
from retrieval import retrieve_documents
from think_tank import ThinkTank
from agent_builder import build_local_agent
from utils import clean_name, export_meeting
from agno.memory.v2 import UserMemory

app = FastAPI()

# Allow CORS for your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# In-memory caches (backed by JSON files)
projects_db = load_projects()
TEMPLATES = load_templates()

def clean_think_tags(text: str) -> str:
    """
    Cleans the text by removing 'Think' tags and their content.
    
    Args:
        text (str): The input text to clean.
        
    Returns:
        str: The cleaned text without 'Think' tags.
    """
    cleaned_text = text.replace("Think", "").replace("```", "")
    return cleaned_text

@app.get("/projects")
async def list_projects():
    obj = projects_db
    print(obj)
    return obj


@app.post("/projects")
async def create_project(tpl: Project):
    if tpl.title in projects_db:
        raise HTTPException(400, "Project already exists")
    projects_db[tpl.title] = tpl.serialize()
    save_projects(projects_db)
    return {"msg": "created"}


@app.get("/projects/{name}")
async def get_project(name: str):
    if name not in projects_db:
        raise HTTPException(404, "Not found")
    return projects_db[name]

@app.get("/templates")
async def get_templates():
    return TEMPLATES


@app.post("/templates")
async def upsert_template(tpl: Expert):
    global TEMPLATES
    # remove existing with same title
    TEMPLATES = [t for t in TEMPLATES if t["title"] != tpl.title]
    TEMPLATES.append(tpl.dict())
    save_templates(TEMPLATES)
    return {"msg": "saved"}


@app.delete("/templates/{title}")
async def del_template(title: str):
    global TEMPLATES
    TEMPLATES = [t for t in TEMPLATES if t["title"] != title]
    save_templates(TEMPLATES)
    return {"msg": "deleted"}

@app.websocket("/ws/meeting")
async def meeting_ws(websocket: WebSocket):
    """
    Client must first send a JSON payload with the Meeting fields:
    {
      "project_name": str,
      "experts": [ {title, expertise, goal, role}, ... ],
      "vector_store": [[<base64_file_bytes>, ...]],  # optional, if files are uploaded
      "meeting_topic": str,
      "rounds": int
    }
    Then the server will stream every log line as JSON messages:
    { "name": <agent_name>, "content": <text> }
    """
    await websocket.accept()
    try:
        init = await websocket.receive_json()
        init['timestamp'] = int(time.time())
        init['transcript'] = []
        init['summary'] = ""
        print("Received init JSON:", init)
        req = Meeting(**init)
    except Exception as e:
        print("Error parsing init JSON or Meeting model:", e)
        await websocket.close(code=1003)
        return
    
    transcript = []

    # helper to stream and log
    async def stream(name: str, content: str):
        """Helper to send a message and log it."""
        transcript.append({"name": name, "content": content})
        try:
            print(f"Attempting to stream: {name}") # Add a log here
            await websocket.send_json({"name": name, "content": content})
            print(f"Successfully streamed: {name}") # And a success log
            lab._log("stream", name, content)
        except Exception as e:
            # This will catch ANY error during the send operation
            print(f"!!!!!! FAILED to send stream message for '{name}'. Error: {e} !!!!!!")
    # 1) ingest documents in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, sci in enumerate(req.experts):
            files = req.vector_store[i] if i < len(req.vector_store) else []

            futures.append(executor.submit(process_documents, files, clean_name(sci.title)))
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                await websocket.send_json({"name": "ingestion", "content": f"Error: {e}"})
    project_desc = projects_db[req.project_name]["description"]
    lab = ThinkTank(project_desc)
    lab.scientists.clear()
    tools = [retrieve_documents]
    for sd in req.experts:
        lab.scientists.append(
            build_local_agent(
                name=sd.title,
                description=f"Expertise: {sd.expertise}. Goal: {sd.goal}",
                role=sd.role,
                memory=lab._memory,
                storage=lab._storage,
                tools=tools,
            )
        )
    print('Starting meeting for project:', req.project_name)
    await stream(f"Starting meeting for project: {req.project_name}", "")
    await asyncio.sleep(0.01)
    # transcript.append({"heading": f"🧑‍🔬 Team Meeting - {req.meeting_topic}"})
    await stream('# 🧑‍🔬 Team Meeting', f'## {req.meeting_topic}')
    await asyncio.sleep(0.01)

    # PI opening
    pi_open = lab.pi.run(
        f"You are convening a team meeting. Agenda: {req.meeting_topic}. Share initial guidance to the experts..",
        stream=False,
    ).content
    await stream(lab.pi.name, clean_think_tags(pi_open))
    await asyncio.sleep(0.01)
    # discussion rounds
    for r in range(1, req.rounds + 1):
        await stream(f"## Round {r}/{req.rounds}", "")
        await asyncio.sleep(0.01)
        # transcript.append({"subheading": f"Round {r}/{req.rounds}"})
        for sci in lab.scientists:
            tool_prompt = f"""
                You are an expert in a team meeting. Your task is to contribute to the discussion based on your expertise and the context provided.
                DO NOT summarize or paraphrase the context, but use it to inform your response.
                Generate a new response every time.

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
            resp = sci.run(tool_prompt, stream=False).content
            print(f"Expert {sci.name} response: {resp}")
            resp = clean_think_tags(resp)
            await stream(f'{sci.name}', resp)
            await asyncio.sleep(0.01)
            # transcript.append({"name": sci.name, "content": resp})
        
        crit = lab.critic.run(f"Context so far:\n{lab._context()}\nCritique round {r}", stream=False).content
        crit = clean_think_tags(crit)
        await stream(lab.critic.name, crit)
        await asyncio.sleep(0.01)
        # transcript.append({"name": lab.critic.name, "content": crit})

        synth = lab.pi.run(f"Context so far:\n{lab._context()}\nSynthesise round {r} and pose follow-ups.", stream=False).content
        synth = clean_think_tags(synth)
        await stream(f"{lab.pi.name} (Feedback)", synth)
        await asyncio.sleep(0.01)
        # transcript.append({"name": f"{lab.pi.name} (Feedback)", "content": synth})

        # final summary
        summary = lab.pi.run(f"Context so far:\n{lab._context()}\nProvide the final detailed meeting summary and recommendations.", stream=False).content
        summary = clean_think_tags(summary)
        await stream("** FINAL SUMMARY **", summary)
        await asyncio.sleep(0.01)
        # transcript.append({"name": "FINAL SUMMARY", "content": summary})

        # persist memory & save project
    lab._memory.add_user_memory(memory=UserMemory(memory=summary), user_id=req.project_name)
    proj = projects_db.setdefault(req.project_name, {
        "title": req.project_name,
        "description": project_desc,
        "meetings": []
    })
    proj["description"] = project_desc
    proj["meetings"].append({
        "project_name": req.project_name,
        "experts": [s.dict() for s in req.experts],
        "vector_store": [],
        "meeting_topic": req.meeting_topic,
        "rounds": req.rounds,
        "timestamp": int(time.time()),
        "transcript": transcript,
        "summary": summary,
    })
    save_projects(projects_db)

    # signal end
    await websocket.send_json({"name": "__end__", "content": "Meeting complete"})
    await asyncio.sleep(0.01)
    await websocket.close()
