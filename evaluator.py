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
    def extract_tasks_from_coordinator(self, content: str, goal_agent, stream: bool = False):
        """
        Extracts tasks/goals from a transcript content string using an LLM agent.

        Args:
            content (str): The transcript content as a string.
            goal_agent: The LLM agent used to extract goals.
            stream (bool): Whether to stream the LLM output.

        Returns:
            list[dict]: List of tasks in format [{"category": "...", "task": "..."}]
        """
        import re, json

        # Create prompt
        prompt = f"""
        You are an assistant that extracts actionable tasks from a meeting transcript.
        Extract all tasks or goals for the team members, group them into categories if possible,
        and return as JSON in this format:

        [{{"category": "...", "task": "..."}}]

        Transcript:
        \"\"\"{content}\"\"\"
        """

        # Run the agent
        response = goal_agent.run(prompt, stream=stream)
        output_text = response.content

        # Try to extract JSON from the LLM output
        match = re.search(r'(\[.*\]|\{.*\})', output_text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                # Validate that each element is a dict with a "task" key
                if isinstance(data, list) and all(isinstance(d, dict) and "task" in d for d in data):
                    return data
            except json.JSONDecodeError:
                pass

        # Fallback: treat each sentence as a task
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', output_text) if s.strip()]
        return [{"category": None, "task": s} for s in sentences]

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
    def collaborative_relevance(
        self,
        agent_a_msgs,
        agent_b_msgs,
        goal_agent,
        stream: bool = False,
        threshold: float = 0.7
    ):
        """
        Computes mutual relevance based on main findings of Agent A (extracted via LLM) 
        and current responses of Agent B.

        Args:
            agent_a_msgs (list[str]): Previous messages from Agent A.
            agent_b_msgs (list[str]): Current messages from Agent B.
            goal_agent: Your Agno/LLM agent instance.
            stream (bool): Whether to stream LLM output.
            threshold (float): Cosine similarity threshold for reference detection.

        Returns:
            float: Fraction of Agent A's main findings referenced by Agent B.
        """
        import numpy as np
        import re
        import json

        def split_sentences(text):
            return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]

        # --- Step 1: Extract main findings from Agent A using LLM ---
        agent_a_text = "\n".join(agent_a_msgs)
        prompt = f"""
        You are an assistant that extracts the key findings or main points from a series of messages.
        Output a JSON array where each element is a main finding (a sentence or short paragraph with clear context, be as precise as possible).

        Messages:
        \"\"\"{agent_a_text}\"\"\"
        """
        response = goal_agent.run(prompt, stream=stream)
        output_text = response.content
        print("Extracted findings:", output_text)
        # --- Step 2: Parse LLM output safely ---
        match = re.search(r'(\[.*\])', output_text, flags=re.DOTALL)
        if match:
            try:
                a_findings_raw = json.loads(match.group(1))
            except json.JSONDecodeError:
                a_findings_raw = split_sentences(agent_a_text)
        else:
            a_findings_raw = split_sentences(agent_a_text)

        # --- Step 3: Ensure all findings are strings ---
        a_findings = []
        for f in a_findings_raw:
            if isinstance(f, dict):
                # join all values into one string
                text = " ".join(str(v) for v in f.values())
            else:
                text = str(f)
            a_findings.append(text)

        if not a_findings:
            return 0.0

        # --- Step 4: Embed Agent A findings ---
        a_embs = [self.get_embedding(f) for f in a_findings]

        # --- Step 5: Embed Agent B sentences ---
        b_sents = [s for msg in agent_b_msgs for s in split_sentences(msg)]
        b_embs = [self.get_embedding(s) for s in b_sents]

        # --- Step 6: Count referenced findings ---
        referenced = 0
        for a_emb in a_embs:
            if b_embs:
                sims = [np.dot(a_emb, b_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(b_emb)) for b_emb in b_embs]
                if max(sims) >= threshold:
                    referenced += 1

        return referenced / len(a_findings)

    def goal_alignment(self, agent_a_json: str, agent_b_json: str, goal_agent, threshold: float = 0.75) -> float:
        """
        Measure semantic overlap of final goals between Agent A and Agent B.

        Parameters
        ----------
        agent_a_json : str
            Raw JSON string (or just the text content) from Agent A’s response file.
        agent_b_json : str
            Raw JSON string (or text content) from Agent B’s response file.
        goal_agent : Agent
            Your Agno agent used to extract tasks/goals.
        threshold : float
            Cosine similarity threshold for considering two goals aligned.

        Returns
        -------
        float
            Fraction of Agent A’s goals that align with some of Agent B’s goals.
        """
        # --- Extract goals from both agents using the goal agent ---
        tasks_a = self.extract_tasks_from_coordinator(agent_a_json, goal_agent)
        tasks_b = self.extract_tasks_from_coordinator(agent_b_json, goal_agent)
        print("Agent A tasks:", tasks_a)
        print("Agent B tasks:", tasks_b)
        goals_a = [t["task"] for t in tasks_a]
        goals_b = [t["task"] for t in tasks_b]

        if not goals_a or not goals_b:
            return 0.0

        # --- Pre-compute embeddings for B goals ---
        b_embs = [self.get_embedding(g) for g in goals_b]

        aligned = 0
        for ga in goals_a:
            ga_emb = self.get_embedding(ga)
            sims = [self.cosine_similarity(ga_emb, gb_emb) for gb_emb in b_embs]
            if sims and max(sims) >= threshold:
                aligned += 1

        return aligned / len(goals_a)

if __name__ == "__main__":

    # ### Testing NCR metric
    # responder_path = "test_3/Coordinator_synthesis_20250909_205344.json"
    # critique_path  = "test_3/Coordinator_synthesis_20250909_210604.json"

    # with open(responder_path, 'r', encoding='utf-8') as f:
    #     responder_text = f.read()

    # with open(critique_path, 'r', encoding='utf-8') as f:
    #     critique_text = f.read()

    # metrics = AgentMetrics(similarity_threshold=0.7)



    # Existing metric
    # score = metrics.evaluate_critique_support(responder_text, critique_text)
    # print(f"Critique validity score: {score:.3f}")


    # ✅ Novel Contribution Ratio (responder vs critique)

    # ncr = metrics.novel_contribution_ratio(critique_text,
    #                                        responder_text)
    # print(f"Novel Contribution Ratio: {ncr:.3f}")


    ### Testing collaborative relevance metric
    from agent_builder import build_local_agent

    # Build your local agent
    goal_agent = build_local_agent(
        name="Goal Extractor",
        description="Extracts actionable tasks and goals from meeting transcripts",
        role="Analyze the transcript and provide a structured JSON list of tasks with categories",
        temperature=0.0
    )

    # 2️⃣ Load responses from files
    import json

    CB1_msgs = "test_3/Computational Biologist_2_response_20250909_205716.json"
    ML2_msgs = "test_3/Machine Learning Expert_1_response_20250909_204635.json"

    with open(CB1_msgs, 'r', encoding='utf-8') as f:
        CB1_json = json.load(f)
        CB1_resp = CB1_json["content"]

    with open(ML2_msgs, 'r', encoding='utf-8') as f:
        ML2_json = json.load(f)
        ML2_resp = ML2_json["content"]

    # 3️⃣ Instantiate your metrics class
    metrics = AgentMetrics(similarity_threshold=0.7)

    # 4️⃣ Compute collaborative relevance
    relevance = metrics.collaborative_relevance(
        agent_b_msgs=[CB1_resp],
        agent_a_msgs=[ML2_resp],
        goal_agent=goal_agent
    )

    print(f"Collaborative Relevance: {relevance:.3f}")
