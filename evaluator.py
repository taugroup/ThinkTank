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

    def novel_contribution_ratio(
        self,
        prev_round: str,
        curr_round: str,
        threshold: float = 0.7
    ) -> float:
        """
        Compute Novel Contribution Ratio (NCR) between two meeting rounds.

        NCR = new_info_tokens / total_tokens_in_curr_round
        A token is "new" if its sentence embedding similarity with all
        sentences from prev_round is below the threshold.
        """
        # --- split into sentences ---
        split_regex = r"[.!?]\s+"
        prev_sents = [s for s in re.split(split_regex, prev_round) if s.strip()]
        curr_sents = [s for s in re.split(split_regex, curr_round) if s.strip()]

        if not curr_sents:
            return 0.0

        # --- embed once per sentence ---
        prev_embs = np.vstack([self.get_embedding(s) for s in prev_sents]) if prev_sents else np.empty((0, 0))
        curr_embs = [self.get_embedding(s) for s in curr_sents]

        # --- count "new" tokens ---
        def count_tokens(text: str) -> int:
            return len(re.findall(r"\w+", text))

        new_tokens = 0
        total_tokens = sum(count_tokens(s) for s in curr_sents)

        for sent, emb in zip(curr_sents, curr_embs):
            if prev_embs.size == 0:
                new_tokens += count_tokens(sent)
                continue

            # compute max similarity to any previous sentence
            max_sim = max(self.cosine_similarity(emb, p_emb) for p_emb in prev_embs)
            if max_sim < threshold:
                new_tokens += count_tokens(sent)

        return new_tokens / total_tokens if total_tokens else 0.0
    
if __name__ == "__main__":
    responder_path = "test_3/Coordinator_synthesis_20250909_205344.json"
    critique_path  = "test_3/Coordinator_synthesis_20250909_210604.json"

    with open(responder_path, 'r', encoding='utf-8') as f:
        responder_text = f.read()

    with open(critique_path, 'r', encoding='utf-8') as f:
        critique_text = f.read()

    metrics = AgentMetrics(similarity_threshold=0.7)

    # Existing metric
    # score = metrics.evaluate_critique_support(responder_text, critique_text)
    # print(f"Critique validity score: {score:.3f}")

    # ✅ Novel Contribution Ratio (responder vs critique)
    
    ncr = metrics.novel_contribution_ratio(critique_text,
                                           responder_text)
    print(f"Novel Contribution Ratio: {ncr:.3f}")

