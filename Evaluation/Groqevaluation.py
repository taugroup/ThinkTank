from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import assert_test, evaluate
import pytest
from typing import List, Dict, Any
import time
import json
import concurrent.futures
import re
import numpy as np
from itertools import permutations
from Groqwrapper import GroqWrapper

class GroqMeetingEvaluationManager:
    """
    Advanced Meeting Evaluation Manager using Groq (openai/gpt-oss-120b) for comprehensive multi-agent evaluation
    """
    
    def __init__(
        self, 
        meeting_topic: str, 
        experts: List[Dict], 
        project_name: str,
        groq_api_key: str = None,
        max_workers: int = 3
    ):
        self.meeting_topic = meeting_topic
        self.experts = experts
        self.project_name = project_name
        self.conversation_history = []
        self.evaluation_results = []
        self.round_evaluation_results = [] # For round-level metrics
        self.round_summaries = {}
        self.agent_performance_tracker = {}
        
        # Initialize thread pool for parallel evaluation
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize Groq evaluator
        self.groq_evaluator = GroqWrapper(
            model_name="openai/gpt-oss-120b",
            api_key=groq_api_key,  # Will use GROQ_API_KEY env var if None
            temperature=0.1,     # Low temperature for consistent evaluation
            max_tokens=2000      # Longer responses for detailed evaluation
        )

        # Initialize evaluation metrics with Groq
        self._setup_evaluation_metrics()
        
        # Performance tracking
        self.start_time = time.time()

    # --- Embedding and Similarity Helpers --- #
    def get_embedding(self, text: str) -> np.ndarray:
        """
        TODO: Implement this with a real sentence embedding model.
        This is a placeholder implementation for demonstration purposes.
        For meaningful results, use sentence-transformers, OpenAI embeddings API, etc.
        """
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        embedding = np.frombuffer(hash_obj.digest(), dtype=np.float32, count=384 // 4)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm != 0 else embedding

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        if v1.size == 0 or v2.size == 0:
            return 0.0
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def novel_contribution_ratio(
        self,
        prev_round: str,
        curr_round: str,
        threshold: float = 0.7
    ) -> float:
        """Compute Novel Contribution Ratio (NCR) between two meeting rounds."""
        split_regex = r"[.!?]\s+"
        prev_sents = [s for s in re.split(split_regex, prev_round) if s.strip()]
        curr_sents = [s for s in re.split(split_regex, curr_round) if s.strip()]
        if not curr_sents: return 0.0

        prev_embs = np.vstack([self.get_embedding(s) for s in prev_sents]) if prev_sents else np.empty((0, 0))
        curr_embs = [self.get_embedding(s) for s in curr_sents]

        def count_tokens(text: str) -> int:
            return len(re.findall(r"\w+", text))

        new_tokens = 0
        total_tokens = sum(count_tokens(s) for s in curr_sents)

        for sent, emb in zip(curr_sents, curr_embs):
            if prev_embs.size == 0:
                new_tokens += count_tokens(sent)
                continue
            max_sim = max(self.cosine_similarity(emb, p_emb) for p_emb in prev_embs)
            if max_sim < threshold:
                new_tokens += count_tokens(sent)

        return new_tokens / total_tokens if total_tokens else 0.0

    def collaborative_relevance(
        self,
        agent_a_msgs: List[str],
        agent_b_msgs: List[str],
        threshold: float = 0.7
    ) -> float:
        """
        Computes mutual relevance based on main findings of Agent A (extracted via LLM) 
        and current responses of Agent B.
        """
        def split_sentences(text):
            return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]

        agent_a_text = "\n".join(agent_a_msgs)
        if not agent_a_text.strip(): return 0.0

        prompt = f"""
        You are an assistant that extracts the key findings or main points from a series of messages.
        Output a JSON array where each element is a main finding (a sentence or short paragraph with clear context, be as precise as possible).

        Messages:
        """{agent_a_text}"""
        """
        response = self.groq_evaluator.client.chat.completions.create(
            model=self.groq_evaluator.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.groq_evaluator.temperature,
            max_tokens=self.groq_evaluator.max_tokens,
        )
        output_text = response.choices[0].message.content

        match = re.search(r'(\[.*\])', output_text, flags=re.DOTALL)
        if match:
            try:
                a_findings_raw = json.loads(match.group(1))
            except json.JSONDecodeError:
                a_findings_raw = split_sentences(agent_a_text)
        else:
            a_findings_raw = split_sentences(output_text)

        a_findings = []
        for f in a_findings_raw:
            text = " ".join(str(v) for v in f.values()) if isinstance(f, dict) else str(f)
            if text.strip(): a_findings.append(text)

        if not a_findings: return 0.0

        a_embs = [self.get_embedding(f) for f in a_findings]
        b_sents = [s for msg in agent_b_msgs for s in split_sentences(msg)]
        if not b_sents: return 0.0
        b_embs = [self.get_embedding(s) for s in b_sents]

        referenced = 0
        for a_emb in a_embs:
            sims = [self.cosine_similarity(a_emb, b_emb) for b_emb in b_embs]
            if max(sims) >= threshold:
                referenced += 1

        return referenced / len(a_findings)

    def _setup_evaluation_metrics(self):
        """Setup comprehensive evaluation metrics using Groq with direct GEval instances"""
        
        self.collaboration_metric = GEval(
            name="Agent Collaboration Quality",
            criteria="See evaluation loop", evaluation_params=[], model=self.groq_evaluator
        )
        self.role_adherence_metric = GEval(
            name="Agent Role Adherence",
            criteria="See evaluation loop", evaluation_params=[], model=self.groq_evaluator
        )
        self.response_quality_metric = GEval(
            name="Agent Response Quality",
            criteria="See evaluation loop", evaluation_params=[], model=self.groq_evaluator
        )
        self.individual_metrics = [self.collaboration_metric, self.role_adherence_metric, self.response_quality_metric]

        self.meeting_progress_metric = GEval(
            name="Meeting Progress Assessment",
            criteria="See evaluation loop", evaluation_params=[], model=self.groq_evaluator
        )
        self.critique_support_metric = GEval(
            name="Critique Support Evaluation",
            criteria="See evaluation loop", evaluation_params=[], model=self.groq_evaluator
        )
        self.task_completion_metric = GEval(
            name="Task Completion",
            criteria="See evaluation loop", evaluation_params=[], model=self.groq_evaluator
        )
        self.round_metrics = [self.meeting_progress_metric, self.critique_support_metric, self.task_completion_metric]

    def add_response(self, agent_name: str, content: str, round_num: int, response_type: str = "expert"):
        """Add a response to the conversation history"""
        response_entry = {
            "agent": agent_name, "content": content, "round": round_num, "type": response_type,
            "timestamp": time.time(), "word_count": len(content.split()), "character_count": len(content)
        }
        self.conversation_history.append(response_entry)
        if agent_name not in self.agent_performance_tracker:
            self.agent_performance_tracker[agent_name] = {
                "total_responses": 0, "total_score": 0, "scores_by_round": {},
                "response_types": [], "strengths": [], "improvement_areas": []
            }
        self.agent_performance_tracker[agent_name]["total_responses"] += 1
        self.agent_performance_tracker[agent_name]["response_types"].append(response_type)
    
    def evaluate_response_async(self, agent_name: str, content: str, round_num: int, response_type: str):
        """Submit individual evaluation task to thread pool"""
        return self.executor.submit(self._evaluate_single_response, agent_name, content, round_num, response_type)

    def _evaluate_single_response(self, agent_name: str, content: str, round_num: int, response_type: str):
        """Comprehensive evaluation of a single agent response using individual metrics"""
        try:
            agent_info = next((exp for exp in self.experts if exp.get('title') == agent_name), {})
            metadata = {"agent_name": agent_name, "agent_info": agent_info, "round": round_num, "response_type": response_type}
            test_case = LLMTestCase(
                input=f"Agent: {agent_name} ({agent_info.get('role')})\nRound: {round_num}",
                actual_output=content, additional_metadata=metadata
            )
            results = {}
            for metric in self.individual_metrics:
                try:
                    metric.measure(test_case)
                    results[metric.name] = {"score": metric.score, "reason": metric.reason}
                    print(f"✅ [Individual] {agent_name} - {metric.name}: {metric.score:.2f}")
                except Exception as e:
                    print(f"❌ {metric.name} failed for {agent_name}: {e}")
            composite_score = self._calculate_composite_score(results)
            evaluation_result = {"agent": agent_name, "round": round_num, "composite_score": composite_score, "results": results}
            self.evaluation_results.append(evaluation_result)
            self._update_agent_performance(agent_name, evaluation_result)
            return evaluation_result
        except Exception as e:
            return {"agent": agent_name, "round": round_num, "error": str(e)}

    def _evaluate_round(self, round_num: int, round_responses: List[Dict]):
        """Perform round-level evaluations."""
        print(f"🔬 Starting evaluations for Round {round_num}...")
        round_results = {"round": round_num, "results": {}}
        
        # --- Novel Contribution Ratio --- #
        try:
            prev_round_resps = [r for r in self.conversation_history if r['round'] == round_num - 1]
            if prev_round_resps:
                prev_round_text = "\n".join(r['content'] for r in prev_round_resps)
                curr_round_text = "\n".join(r['content'] for r in round_responses)
                ncr_score = self.novel_contribution_ratio(prev_round_text, curr_round_text)
                round_results["results"]["Novel Contribution Ratio"] = {"score": ncr_score, "reason": f"{ncr_score:.1%} new info"}
                print(f"✅ [Round {round_num}] Novel Contribution Ratio: {ncr_score:.2f}")
        except Exception as e:
            print(f"❌ [Round {round_num}] NCR failed: {e}")

        # --- Collaborative Relevance --- #
        expert_resps = [r for r in round_responses if r['type'] == 'expert']
        expert_agents = list({r['agent'] for r in expert_resps})
        if len(expert_agents) >= 2:
            relevance_scores = {}
            for agent_a, agent_b in permutations(expert_agents, 2):
                a_msgs = [r['content'] for r in expert_resps if r['agent'] == agent_a]
                b_msgs = [r['content'] for r in expert_resps if r['agent'] == agent_b]
                if a_msgs and b_msgs:
                    score = self.collaborative_relevance(a_msgs, b_msgs)
                    relevance_scores[f"{agent_a} -> {agent_b}"] = score
            if relevance_scores:
                mean_score = np.mean(list(relevance_scores.values()))
                round_results["results"]["Collaborative Relevance"] = {"score": mean_score, "details": relevance_scores}
                print(f"✅ [Round {round_num}] Collaborative Relevance: {mean_score:.2f}")

        # --- DeepEval Round Metrics --- #
        facilitator_resp = next((r for r in round_responses if r['type'] == 'facilitator'), None)
        critic_resp = next((r for r in round_responses if r['type'] == 'critic'), None)
        if expert_resps:
            combined_expert_content = "\n".join(r['content'] for r in expert_resps)
            for metric in self.round_metrics:
                tc = None
                if metric.name == "Task Completion" and facilitator_resp:
                    tc = LLMTestCase(input=facilitator_resp['content'], actual_output=combined_expert_content)
                elif metric.name == "Critique Support Evaluation" and critic_resp:
                    tc = LLMTestCase(input=combined_expert_content, actual_output=critic_resp['content'])
                if tc:
                    try:
                        metric.measure(tc)
                        round_results["results"][metric.name] = {"score": metric.score, "reason": metric.reason}
                        print(f"✅ [Round {round_num}] {metric.name}: {metric.score:.2f}")
                    except Exception as e:
                        print(f"❌ [Round {round_num}] {metric.name} failed: {e}")

        self.round_evaluation_results.append(round_results)
        return round_results

    def _calculate_composite_score(self, results: Dict) -> float:
        """Calculate weighted composite score from individual metrics."""
        weights = {"Agent Collaboration Quality": 0.3, "Agent Role Adherence": 0.4, "Agent Response Quality": 0.3}
        score = sum(results.get(n, {}).get("score", 0) * w for n, w in weights.items())
        return score / sum(weights.values()) if sum(weights.values()) > 0 else 0
    
    def _update_agent_performance(self, agent_name: str, eval_result: Dict):
        tracker = self.agent_performance_tracker[agent_name]
        tracker["total_score"] += eval_result["composite_score"]
        tracker["scores_by_round"][eval_result["round"]] = eval_result["composite_score"]

    def evaluate_transcript(self, transcript_data: Dict) -> Dict[str, Any]:
        """Evaluate a complete meeting transcript."""
        self.meeting_topic = transcript_data["meeting_topic"]
        self.project_name = transcript_data.get("project_name", "Unknown")
        self.experts = transcript_data["experts"]
        self.conversation_history = self._parse_transcript(transcript_data["transcript"])
        self._initialize_agent_trackers()

        # --- Run Evaluations --- #
        individual_futures = [self.evaluate_response_async(r["agent"], r["content"], r["round"], r["type"]) for r in self.conversation_history if r['type'] not in ["system", "header"]]
        concurrent.futures.wait(individual_futures)

        rounds = {r['round']: [] for r in self.conversation_history}
        for r in self.conversation_history: rounds[r['round']].append(r)
        round_futures = [self.executor.submit(self._evaluate_round, num, resps) for num, resps in rounds.items() if num > 0]
        concurrent.futures.wait(round_futures)

        return self.generate_meeting_summary_report()

    def _parse_transcript(self, transcript: List[Dict]) -> List[Dict]:
        return [{"agent": e["name"], "content": e["content"], "round": e.get("round", 1), "type": self._classify_response_type(e["name"])} for e in transcript]

    def _classify_response_type(self, agent_name: str) -> str:
        name = agent_name.lower()
        if any(k in name for k in ["meeting", "round", "summary"]): return "system"
        if "critic" in name: return "critic"
        if any(k in name for k in ["feedback", "pi", "coordinator"]): return "facilitator"
        if name.startswith("#"): return "header"
        return "expert"

    def _initialize_agent_trackers(self):
        for r in self.conversation_history:
            if r["agent"] not in self.agent_performance_tracker: self.agent_performance_tracker[r["agent"]] = {"total_responses": 0, "total_score": 0, "scores_by_round": {}}
            self.agent_performance_tracker[r["agent"]]["total_responses"] += 1

    def generate_meeting_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive meeting summary report."""
        agent_reports = {name: self.get_agent_detailed_report(name) for name in self.agent_performance_tracker.keys()}
        return {
            "meeting_metadata": {"topic": self.meeting_topic, "project": self.project_name, "participants": [e["title"] for e in self.experts]},
            "agent_performance": agent_reports,
            "round_evaluations": sorted(self.round_evaluation_results, key=lambda x: x['round'])
        }

    def get_agent_detailed_report(self, agent_name: str) -> Dict[str, Any]:
        tracker = self.agent_performance_tracker[agent_name]
        avg_score = tracker["total_score"] / tracker["total_responses"] if tracker["total_responses"] > 0 else 0
        return {"agent_name": agent_name, "overall_average_score": avg_score, "total_responses": tracker["total_responses"], "scores_by_round": tracker["scores_by_round"]}

    def shutdown(self):
        self.executor.shutdown(wait=True)

# Example usage would need to be adapted for the new report structure.
if __name__ == "__main__":
    print("GroqMeetingEvaluationManager class loaded. Example usage would need to be updated.")
