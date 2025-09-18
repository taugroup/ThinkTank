import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # <-- add project root

from evaluator import AgentMetrics
from agent_builder import build_local_agent  # import your agent builder

def test_goal_alignment_full():
    # 1️⃣ Build the goal agent
    goal_agent = build_local_agent(
        name="Goal Extractor",
        description="Extracts actionable tasks and goals from meeting transcripts",
        role="Analyze the transcript and provide a structured JSON list of tasks with categories",
        temperature=0.0
    )

    # 2️⃣ Load sample responses
    with open("test_3/Computational Biologist_2_response_20250909_205716.json", 'r', encoding='utf-8') as f:
        r1 = json.load(f)['content']

    with open("test_3/Machine Learning Expert_2_response_20250909_210124.json", 'r', encoding='utf-8') as f:
        r2 = json.load(f)['content']

    # 3️⃣ Initialize metrics
    metrics = AgentMetrics(similarity_threshold=0.7)

    # 4️⃣ Compute goal alignment
    score = metrics.goal_alignment(r1, r2, goal_agent=goal_agent)
    print(f"Goal Alignment Score: {score:.3f}")

    # Basic sanity checks
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

test_goal_alignment_full()