import re
import numpy as np
import ollama
from typing import List
import json

class AgentMetrics:
    """Evaluate agent responses with multiple metrics."""
    
    def __init__(self, embedding_model: str = "nomic-embed-text", similarity_threshold: float = 0.7):
        self.embedding_model = embedding_model
        self.threshold = similarity_threshold
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Return 1D embedding for text."""
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return np.array(response["embedding"]).flatten()
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = vec1.flatten()
        v2 = vec2.flatten()
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot_product / norm_product if norm_product != 0 else 0.0
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Simple sentence splitter."""
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def evaluate_critique_support(self, responder_text: str, critique_text: str) -> float:
        """
        Score whether critique claims are supported by responder output.
        Returns a value in [0,1].
        """
        claims = self.split_into_sentences(critique_text)
        responder_chunks = self.split_into_sentences(responder_text)
        
        # Precompute embeddings for responder chunks
        responder_embs = [self.get_embedding(chunk) for chunk in responder_chunks]
        
        supported_count = 0
        for claim in claims:
            claim_emb = self.get_embedding(claim)
            similarities = [self.cosine_similarity(claim_emb, r_emb) for r_emb in responder_embs]
            max_sim = max(similarities) if similarities else 0
            if max_sim >= self.threshold:
                # print(f"Claim: {claim} - Max similarity: {max_sim:.3f}")
                supported_count += 1
        
        return supported_count / len(claims) if claims else 0.0

    def extract_tasks_from_coordinator(self,coordinator_json_path: str, goal_agent, stream: bool=False):
        """
        Reads the coordinator's JSON response, extracts tasks/goals,
        and returns a clean list of tasks in JSON format.
        
        Args:
            coordinator_json_path (str): Path to the coordinator JSON file.
            goal_agent: The LLM agent used to extract goals.
            stream (bool): Whether to stream the agent's response.
        
        Returns:
            list[dict]: List of tasks in format [{"category": "...", "task": "..."}]
        """

        # Load coordinator transcript
        with open(coordinator_json_path, "r") as f:
            transcript_json = json.load(f)
        transcript_content = transcript_json["content"]

        # Create prompt
        prompt = f"""
        You are an assistant that extracts actionable tasks from a meeting transcript.
        Extract all tasks or goals for the team members, group them into categories if possible,
        and return as JSON in this format:

        [{{"category": "...", "task": "..."}}]

        Transcript:
        \"\"\"{transcript_content}\"\"\"
        """

        # Run the agent
        response = goal_agent.run(prompt, stream=stream)
        output_text = response.content

        # Extract JSON from agent output
        match = re.search(r'(\[.*\]|\{.*\})', output_text, flags=re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("⚠️ Failed to parse JSON, returning raw text")
                return [output_text]
        else:
            print("⚠️ No JSON found in output")
            return []

    def task_completion_vs_response(self,tasks, agent_response, threshold=0.7):
        """
        Check for each task whether the agent_response contains its content (semantically).
        Returns fraction of tasks 'completed' and per-task results.
        """
        # Split response into chunks (sentences)
        response_chunks = re.split(r'[.!?]\s+', agent_response)
        response_chunks = [r.strip() for r in response_chunks if r.strip()]

        # Precompute embeddings for response chunks
        response_embs = [self.get_embedding(chunk) for chunk in response_chunks]

        results = []
        completed_count = 0

        for task in tasks:
            task_emb = self.get_embedding(task['task'])
            similarities = [self.cosine_similarity_np(task_emb, r_emb) for r_emb in response_embs]
            max_sim = max(similarities) if similarities else 0

            is_done = max_sim >= threshold
            if is_done:
                completed_count += 1

            results.append({
                "task": task["task"],
                "category": task.get("category", None),
                "completed": is_done,
                "similarity": round(max_sim, 3)
            })

        total_tasks = len(tasks)
        completion_fraction = completed_count / total_tasks if total_tasks else 0.0

        return completion_fraction, results


if __name__ == "__main__":
    # from agent_metrics import AgentMetrics  # your class

    # Paths to your "textual JSON" files
    responder_path = "test_3/Coordinator_synthesis_20250909_205344.json"
    critique_path = "test_3/Critical Thinker_critique_20250909_205022.json"

    # Read the files as plain text
    with open(responder_path, 'r', encoding='utf-8') as f:
        responder_text = f.read()

    with open(critique_path, 'r', encoding='utf-8') as f:
        critique_text = f.read()

    # Initialize metrics
    metrics = AgentMetrics(similarity_threshold=0.7)

    # Evaluate critique support
    score = metrics.evaluate_critique_support(responder_text, critique_text)
    print(f"Critique validity score: {score:.3f}")

