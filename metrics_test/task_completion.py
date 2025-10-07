import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to import path

from evaluator import AgentMetrics
from agent_builder import build_local_agent  # your LLM-based agent builder
from dotenv import load_dotenv

load_dotenv()  

import google.genai as genai

def make_gemini_llm():
    client=genai.Client(api_key=os.getenv("GENAI_API_KEY"))
    def gemini_call(prompt:str) -> str:
        chat=client.chats.create(model="gemini-2.5-pro")
        resp=chat.send_message(prompt)
        return resp.text
    return gemini_call


def test_task_completion_vs_response():
    # 1️⃣ Build the goal-extraction agent
    goal_agent = build_local_agent(
        name="Goal Extractor",
        description="Extract actionable tasks/goals from meeting transcripts",
        role="Analyze the transcript and output a structured JSON list of tasks with categories",
        temperature=0.0
    )

    # 2️⃣ Load a coordinator transcript to extract tasks
    with open("test_3/Coordinator_opening_20250909_203917.json", "r", encoding="utf-8") as f:
        coordinator_content = json.load(f)["content"]
    gemini_llm=make_gemini_llm()
    # 3️⃣ Use the agent to extract tasks
    metrics = AgentMetrics(similarity_threshold=0.7)
    tasks = metrics.extract_tasks_from_coordinator(coordinator_content, gemini_llm)

    # 4️⃣ Load an agent’s response that we want to check against those tasks
    with open("test_3/Computational Biologist_1_response_20250909_204238.json", "r", encoding="utf-8") as f1, \
         open("test_3/Machine Learning Expert_1_response_20250909_204635.json", "r", encoding="utf-8") as f2:
        agent_response = f1.read() + "\n" + f2.read()

    # 5️⃣ Compute completion fraction and per-task details
    
    completion_fraction, results = metrics.task_completion_vs_response(tasks, agent_response, llm=gemini_llm)
    print(f"Task Completion Fraction: {completion_fraction:.3f}")
    print("Per-task results:")
    for r in results:
        print(r)

    # Optional sanity checks
    # assert isinstance(completion_fraction, float)
    # assert 0.0 <= completion_fraction <= 1.0

# Run the test
test_task_completion_vs_response()