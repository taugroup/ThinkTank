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
        # Use a simple hashing to get a deterministic, fixed-size vector
        hash_obj = hashlib.sha256(text.encode())
        # Create a 384-dimensional vector, typical for some sentence transformers
        embedding = np.frombuffer(hash_obj.digest(), dtype=np.float32, count=384 // 4)
        # Normalize to a unit vector for cosine similarity
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

    def _setup_evaluation_metrics(self):
        """Setup comprehensive evaluation metrics using Groq with direct GEval instances"""
        
        # INDIVIDUAL METRICS (per response)
        self.collaboration_metric = GEval(
            name="Agent Collaboration Quality",
            criteria="""
            Evaluate how well agents collaborate in the multi-agent system:
            1. Information Flow: Do agents effectively build upon previous responses?
            2. Role Adherence: Does each agent stay within their expertise area?
            3. Knowledge Integration: Are specialized knowledge bases utilized appropriately?
            4. Coherence: Does the conversation flow logically from one agent to the next?
            5. Redundancy Avoidance: Do agents avoid repeating information unnecessarily?
            
            Rate from 1-100 where 100 is excellent collaboration.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            model=self.groq_evaluator,
            threshold=0.6,
            strict_mode=False
        )
        
        self.role_adherence_metric = GEval(
            name="Agent Role Adherence",
            criteria="""
            Evaluate whether each agent stays within their defined role and expertise:
            1. Expertise Boundaries: Does the agent only provide information within their domain?
            2. Role Consistency: Does the agent maintain their assigned role throughout?
            3. Knowledge Source Usage: Does the agent appropriately reference their expertise?
            4. Goal Alignment: Are the agent's responses aligned with their stated goals?
            
            Rate from 1-100 where 100 is perfect role adherence.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            model=self.groq_evaluator,
            threshold=0.6,
            strict_mode=False
        )
        
        self.response_quality_metric = GEval(
            name="Agent Response Quality",
            criteria="""
            Evaluate the quality of the agent's response in the meeting context:
            1. Expertise Demonstration: Does the response show domain knowledge and expertise?
            2. Context Awareness: Does the agent build upon previous discussion points?
            3. Information Value: Does the response add new, actionable insights?
            4. Clarity and Structure: Is the response well-organized and easy to follow?
            5. Actionability: Does the response provide practical, implementable information?
            
            Rate from 1-100 where 100 is exceptional quality.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            model=self.groq_evaluator,
            threshold=0.6,
            strict_mode=False
        )

        self.individual_metrics = [
            self.collaboration_metric,
            self.role_adherence_metric,
            self.response_quality_metric,
        ]

        # ROUND-LEVEL METRICS
        self.meeting_progress_metric = GEval(
            name="Meeting Progress Assessment",
            criteria="""
            Evaluate how well the meeting is progressing toward its objectives:
            1. Topic Focus: Are responses staying relevant to the meeting agenda?
            2. Collaborative Building: Are agents building on each other's contributions?
            3. Problem Solving: Is the discussion moving toward solutions and decisions?
            4. Knowledge Integration: Are different expertise areas being combined effectively?
            5. Decision Support: Does the response provide information that aids decision-making?
            
            Rate from 1-100 where 100 is excellent progress toward meeting goals.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            model=self.groq_evaluator,
            threshold=0.6,
            strict_mode=False
        )

        self.critique_support_metric = GEval(
            name="Critique Support Evaluation",
            criteria="""
            Evaluate how well the critique aligns with and is supported by the responder's text.
            1. Does the critique accurately reference the responder's statements?
            2. Are the critique's points justified based on what the responder actually said?
            3. Does the critique avoid misinterpreting or fabricating issues?

            Score 0–1:
            - 1.0 → Critique is fully supported by responder output.
            - 0.5 → Partially supported or somewhat misaligned.
            - 0.0 → Critique is mostly unrelated or unsupported.
            """,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,  # critique text
                LLMTestCaseParams.INPUT           # responder text
            ],
            model=self.groq_evaluator,
            threshold=0.6,
            strict_mode=False
        )

        self.task_completion_metric = GEval(
            name="Task Completion",
            criteria="""
            Evaluate how well the "Agent Response" completes the tasks laid out in the "Task-Setting Response".

            1.  First, identify the specific, actionable tasks or goals mentioned in the "Task-Setting Response".
            2.  For each task, check if the "Agent Response" provides a clear and direct attempt to address, answer, or complete it.
            3.  The evaluation should be based on whether the agent's response directly contributes to fulfilling the tasks, not just on mentioning related keywords.

            Score from 0.0 to 1.0, representing the fraction of tasks that were completed or meaningfully addressed. For example, if 3 out of 4 tasks were addressed, the score should be 0.75. If no tasks are given in the "Task-Setting Response", the score should be 1.0 by default.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,      # Task-Setting Response (from Coordinator/System)
                LLMTestCaseParams.ACTUAL_OUTPUT # Agent Response
            ],
            model=self.groq_evaluator,
            threshold=0.5,
            strict_mode=False
        )

        self.round_metrics = [
            self.meeting_progress_metric,
            self.critique_support_metric,
            self.task_completion_metric
        ]

    def add_response(self, agent_name: str, content: str, round_num: int, response_type: str = "expert"):
        """Add a response to the conversation history"""
        response_entry = {
            "agent": agent_name,
            "content": content,
            "round": round_num,
            "type": response_type,
            "timestamp": time.time(),
            "word_count": len(content.split()),
            "character_count": len(content)
        }
        
        self.conversation_history.append(response_entry)
        
        if agent_name not in self.agent_performance_tracker:
            self.agent_performance_tracker[agent_name] = {
                "total_responses": 0,
                "total_score": 0,
                "scores_by_round": {},
                "response_types": [],
                "strengths": [],
                "improvement_areas": []
            }
        
        self.agent_performance_tracker[agent_name]["total_responses"] += 1
        self.agent_performance_tracker[agent_name]["response_types"].append(response_type)
    
    def evaluate_response_async(self, agent_name: str, content: str, round_num: int, response_type: str):
        """Submit individual evaluation task to thread pool"""
        future = self.executor.submit(
            self._evaluate_single_response, 
            agent_name, content, round_num, response_type
        )
        return future

    def _evaluate_single_response(self, agent_name: str, content: str, round_num: int, response_type: str):
        """Comprehensive evaluation of a single agent response using individual metrics"""
        try:
            agent_info = next(
                (exp for exp in self.experts if exp.get('title') == agent_name), 
                {"title": agent_name, "expertise": "Unknown", "role": "Unknown", "goal": "Unknown"}
            )
            
            recent_context = self._get_recent_context(agent_name, 5)
            meeting_stage = self._determine_meeting_stage(round_num)
            
            metadata = {
                "agent_name": agent_name,
                "agent_info": agent_info,
                "round": round_num,
                "response_type": response_type,
                "meeting_topic": self.meeting_topic,
                "meeting_stage": meeting_stage,
                "conversation_context": recent_context,
                "total_responses_so_far": len(self.conversation_history),
                "agent_response_count": self.agent_performance_tracker.get(agent_name, {}).get("total_responses", 0),
                "word_count": len(content.split()),
                "project_context": self.project_name
            }
            
            test_case = LLMTestCase(
                input=f"""
                    Meeting Topic: {self.meeting_topic}
                    Agent: {agent_name} ({agent_info.get('role', 'Unknown')})
                    Expertise: {agent_info.get('expertise', 'Unknown')}
                    Round: {round_num} ({meeting_stage})
                    Response Type: {response_type}
                """.strip(),
                actual_output=content,
                additional_metadata=metadata
            )
            
            results = {}
            evaluation_start = time.time()
            
            for metric in self.individual_metrics:
                try:
                    metric_start = time.time()
                    metric.measure(test_case)
                    metric_duration = time.time() - metric_start
                    
                    results[metric.name] = {
                        "score": metric.score,
                        "reason": metric.reason,
                        "evaluation_time": metric_duration
                    }
                    
                    print(f"✅ [Individual] {agent_name} - {metric.name}: {metric.score:.2f} ({metric_duration:.1f}s)")
                    
                except Exception as e:
                    print(f"❌ {metric.name} failed for {agent_name}: {e}")
                    results[metric.name] = {"score": 0, "reason": f"Evaluation error: {str(e)}", "evaluation_time": 0}
            
            evaluation_duration = time.time() - evaluation_start
            composite_score = self._calculate_composite_score(results)
            
            evaluation_result = {
                "agent": agent_name,
                "round": round_num,
                "response_type": response_type,
                "timestamp": time.time(),
                "evaluation_duration": evaluation_duration,
                "composite_score": composite_score,
                "results": results,
                "metadata": metadata,
                "content_preview": content[:150] + "..."
            }
            
            self.evaluation_results.append(evaluation_result)
            self._update_agent_performance(agent_name, evaluation_result)
            
            print(f"📊 {agent_name} (Round {round_num}) - Composite Score: {composite_score:.2f}")
            return evaluation_result
            
        except Exception as e:
            print(f"❌ Critical evaluation error for {agent_name}: {e}")
            return {"agent": agent_name, "round": round_num, "error": str(e), "timestamp": time.time()}

    def _evaluate_round(self, round_num: int, round_responses: List[Dict]):
        """Perform round-level evaluations for metrics like Task Completion and Critique Support."""
        print(f"🔬 Starting evaluations for Round {round_num}...")
        
        # 1. Identify response types for the round
        facilitator_resp = next((r for r in round_responses if r['type'] == 'facilitator'), None)
        critic_resp = next((r for r in round_responses if r['type'] == 'critic'), None)
        expert_resps = [r for r in round_responses if r['type'] == 'expert']
        
        round_results = {"round": round_num, "results": {}}

        # --- Run Novel Contribution Ratio --- #
        try:
            prev_round_resps = [r for r in self.conversation_history if r['round'] == round_num - 1]
            if prev_round_resps:
                prev_round_text = "\n\n".join([f"{r['agent']}: {r['content']}" for r in prev_round_resps])
                curr_round_text = "\n\n".join([f"{r['agent']}: {r['content']}" for r in round_responses])
                ncr_score = self.novel_contribution_ratio(prev_round_text, curr_round_text)
                round_results["results"]["Novel Contribution Ratio"] = {
                    "score": ncr_score,
                    "reason": f"{ncr_score:.1%} of the content in this round was new information."
                }
                print(f"✅ [Round {round_num}] Novel Contribution Ratio: {ncr_score:.2f}")
        except Exception as e:
            print(f"❌ [Round {round_num}] Novel Contribution Ratio failed: {e}")

        # --- Run DeepEval Round Metrics --- #
        if not expert_resps:
            print(f"🟡 No expert responses in Round {round_num}, skipping some round evaluations.")
            self.round_evaluation_results.append(round_results)
            return

        combined_expert_content = "\n\n".join([f"{r['agent']}: {r['content']}" for r in expert_resps])

        for metric in self.round_metrics:
            tc_to_use = None
            try:
                if metric.name == "Task Completion" and facilitator_resp:
                    tc_to_use = LLMTestCase(
                        input=facilitator_resp['content'],
                        actual_output=combined_expert_content
                    )
                
                elif metric.name == "Critique Support Evaluation" and critic_resp:
                    tc_to_use = LLMTestCase(
                        input=combined_expert_content,
                        actual_output=critic_resp['content']
                    )

                elif metric.name == "Meeting Progress Assessment":
                    all_round_content = "\n\n".join([f"{r['agent']}: {r['content']}" for r in round_responses])
                    tc_to_use = LLMTestCase(
                        input=f"Meeting Topic: {self.meeting_topic}\nRound: {round_num}",
                        actual_output=all_round_content
                    )

                if tc_to_use:
                    metric_start = time.time()
                    metric.measure(tc_to_use)
                    metric_duration = time.time() - metric_start
                    round_results["results"][metric.name] = {
                        "score": metric.score,
                        "reason": metric.reason,
                        "evaluation_time": metric_duration
                    }
                    print(f"✅ [Round {round_num}] {metric.name}: {metric.score:.2f} ({metric_duration:.1f}s)")

            except Exception as e:
                print(f"❌ [Round {round_num}] {metric.name} failed: {e}")
                round_results["results"][metric.name] = {"score": 0, "reason": f"Evaluation error: {str(e)}"}
        
        self.round_evaluation_results.append(round_results)
        return round_results

    def _calculate_composite_score(self, results: Dict) -> float:
        """Calculate weighted composite score from all metrics"""
        weights = {
            "Agent Collaboration Quality": 0.3,
            "Agent Role Adherence": 0.4,
            "Agent Response Quality": 0.3,
        }
        
        total_score = sum(results.get(name, {}).get("score", 0) * weight for name, weight in weights.items())
        total_weight = sum(weight for name, weight in weights.items() if name in results)
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _update_agent_performance(self, agent_name: str, evaluation_result: Dict):
        """Update agent performance tracking"""
        tracker = self.agent_performance_tracker[agent_name]
        composite_score = evaluation_result["composite_score"]
        tracker["total_score"] += composite_score
        tracker["scores_by_round"][evaluation_result["round"]] = composite_score
        
        for metric_name, result in evaluation_result["results"].items():
            if result.get("score", 0) >= 0.8:
                tracker["strengths"].append(f"{metric_name}: {result.get('reason', '')[:100]}...")
            elif result.get("score", 0) <= 0.5:
                tracker["improvement_areas"].append(f"{metric_name}: {result.get('reason', '')[:100]}...")
    
    def _get_recent_context(self, current_agent: str, num_responses: int = 5) -> List[Dict]:
        """Get recent conversation context, excluding current agent's last response"""
        relevant_history = [r for r in self.conversation_history if r["agent"] != current_agent][-num_responses:]
        return [{k: v for k, v in entry.items() if k in ['agent', 'content', 'round', 'type']} for entry in relevant_history]
    
    def _determine_meeting_stage(self, round_num: int) -> str:
        """Determine the current stage of the meeting"""
        total_rounds = max([entry["round"] for entry in self.conversation_history] + [round_num])
        if round_num == 1: return "Opening & Initial Contributions"
        if round_num <= total_rounds * 0.4: return "Exploration & Information Gathering"
        if round_num <= total_rounds * 0.7: return "Analysis & Discussion"
        if round_num <= total_rounds * 0.9: return "Synthesis & Decision Making"
        return "Conclusion & Action Planning"
        
    def evaluate_transcript(self, transcript_data: Dict) -> Dict[str, Any]:
        """Evaluate a complete meeting transcript after the meeting is done"""
        print("🔍 Starting post-meeting transcript evaluation with Groq...")
        
        self.meeting_topic = transcript_data["meeting_topic"]
        self.project_name = transcript_data["project_name"]
        self.experts = transcript_data["experts"]
        
        structured_responses = self._parse_transcript(transcript_data["transcript"])
        self.conversation_history = structured_responses
        self._initialize_agent_trackers()
    
        # --- Part 1: Individual Response Evaluation --- 
        print("🏃‍➡️ Starting individual response evaluations...")
        individual_futures = []
        for response in structured_responses:
            if response["type"] in ["system", "header"] or not response["content"].strip():
                continue
            future = self.evaluate_response_async(
                response["agent"], response["content"], response["round"], response["type"]
            )
            individual_futures.append(future)
        
        concurrent.futures.wait(individual_futures)
        print("✅ Completed individual response evaluations.")

        # --- Part 2: Round-level Evaluation --- 
        print("🔬 Starting round-level evaluations...")
        round_futures = []
        rounds = {}
        for resp in structured_responses:
            r_num = resp['round']
            if r_num not in rounds: rounds[r_num] = []
            rounds[r_num].append(resp)

        for round_num, round_responses in rounds.items():
            if round_num == 0: continue
            future = self.executor.submit(self._evaluate_round, round_num, round_responses)
            round_futures.append(future)

        concurrent.futures.wait(round_futures)
        print("✅ Completed round-level evaluations.")

        # --- Part 3: Reporting --- 
        return self.generate_meeting_summary_report()

    def _parse_transcript(self, transcript: List[Dict]) -> List[Dict]:
        """Parse transcript into structured response format"""
        structured_responses = []
        for entry in transcript:
            response_type = self._classify_response_type(entry["name"])
            round_num = entry.get("round", 1)
            structured_responses.append({
                "agent": entry["name"],
                "content": entry["content"],
                "round": round_num,
                "type": response_type,
                "timestamp": time.time(),
                "word_count": len(entry["content"].split()),
                "character_count": len(entry["content"])
            })
        return structured_responses

    def _classify_response_type(self, agent_name: str) -> str:
        """Classify response type based on agent name"""
        agent_name_lower = agent_name.lower()
        if any(keyword in agent_name_lower for keyword in ["meeting", "round", "summary", "final"]): return "system"
        if "critic" in agent_name_lower: return "critic"
        if any(keyword in agent_name_lower for keyword in ["feedback", "pi", "coordinator"]): return "facilitator"
        if agent_name.startswith("#") or agent_name.startswith("**"): return "header"
        return "expert"

    def _initialize_agent_trackers(self):
        """Initialize agent performance trackers from transcript"""
        for response in self.conversation_history:
            agent_name = response["agent"]
            if agent_name not in self.agent_performance_tracker:
                self.agent_performance_tracker[agent_name] = {
                    "total_responses": 0, "total_score": 0, "scores_by_round": {},
                    "response_types": [], "strengths": [], "improvement_areas": []
                }
            self.agent_performance_tracker[agent_name]["total_responses"] += 1
            self.agent_performance_tracker[agent_name]["response_types"].append(response["type"])

    def _calculate_basic_insights(self) -> Dict[str, Any]:
        """Calculate basic insights from evaluation results"""
        if not self.evaluation_results:
            return {"status": "No evaluations completed yet"}
        
        avg_composite = sum(r["composite_score"] for r in self.evaluation_results) / len(self.evaluation_results)
        
        agent_averages = {}
        for agent_name, tracker in self.agent_performance_tracker.items():
            if tracker["total_responses"] > 0:
                agent_averages[agent_name] = tracker["total_score"] / tracker["total_responses"]
        
        top_performer = max(agent_averages, key=agent_averages.get) if agent_averages else None
        
        return {
            "average_composite_score": avg_composite,
            "performance_trend": "N/A",
            "top_performer": {
                "agent": top_performer,
                "score": agent_averages.get(top_performer, 0)
            } if top_performer else None,
        }

    def _assess_overall_meeting_quality(self, insights: Dict) -> Dict[str, Any]:
        """Assess overall meeting quality"""
        avg_score = insights.get("average_composite_score", 0)
        
        if avg_score >= 0.85:
            rating = "excellent"
            strengths = ["Outstanding collaboration and role adherence"]
            weaknesses = ["Minor optimization opportunities"]
        elif avg_score >= 0.70:
            rating = "good"
            strengths = ["Solid individual performances"]
            weaknesses = ["Some areas for improvement in collaboration"]
        else:
            rating = "needs_improvement"
            strengths = ["Meeting completed"]
            weaknesses = ["Collaboration and role adherence issues identified"]
        
        return {"rating": rating, "strengths": strengths, "weaknesses": weaknesses}

    def _generate_executive_summary(self, insights: Dict, agent_reports: Dict) -> str:
        """Generate executive summary"""
        avg_score = insights.get("average_composite_score", 0)
        return f"Meeting completed with an average individual agent score of {avg_score:.2f}/1.0. Round-level evaluations provide further insight into collaborative outcomes."

    def generate_meeting_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive meeting summary report including round-level results"""
        agent_reports = {name: self.get_agent_detailed_report(name) for name in self.agent_performance_tracker.keys()}
        
        basic_insights = self._calculate_basic_insights()
        overall_quality = self._assess_overall_meeting_quality(basic_insights)
        executive_summary = self._generate_executive_summary(basic_insights, agent_reports)

        return {
            "meeting_metadata": {
                "topic": self.meeting_topic,
                "project": self.project_name,
                "duration_minutes": round((time.time() - self.start_time) / 60, 2),
                "total_responses": len(self.conversation_history),
                "participants": [agent["title"] for agent in self.experts],
            },
            "executive_summary": executive_summary,
            "overall_assessment": {
                "quality_rating": overall_quality["rating"],
                "average_score": basic_insights.get("average_composite_score", 0),
                "performance_trend": basic_insights.get("performance_trend", "unknown"),
                "key_strengths": overall_quality["strengths"],
                "improvement_areas": overall_quality["weaknesses"],
            },
            "agent_performance": agent_reports,
            "key_insights": basic_insights,
            "round_evaluations": self.round_evaluation_results,
            "full_evaluation_log": self.evaluation_results
        }

    def get_agent_detailed_report(self, agent_name: str) -> Dict[str, Any]:
        """Generate detailed performance report for a specific agent"""
        if agent_name not in self.agent_performance_tracker: return {"error": f"No data for {agent_name}"}
        tracker = self.agent_performance_tracker[agent_name]
        avg_score = tracker["total_score"] / tracker["total_responses"] if tracker["total_responses"] > 0 else 0
        return {
            "agent_name": agent_name,
            "overall_average_score": avg_score,
            "total_responses": tracker["total_responses"],
            "scores_by_round": tracker["scores_by_round"],
            "strengths": list(set(tracker["strengths"]) - set(tracker["improvement_areas"])),
            "improvement_areas": list(set(tracker["improvement_areas"]))
        }

    def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        print("✅ Evaluation complete.")

# Example usage and other functions remain largely the same, but would now print a different report structure.

def evaluate_meeting_transcript(transcript_file_path: str = None, transcript_data: Dict = None):
    """
    Evaluate a meeting transcript from file or data using Groq
    """
    
    if transcript_file_path:
        import json
        with open(transcript_file_path, 'r') as f:
            transcript_data = json.load(f)
    
    if not transcript_data:
        raise ValueError("Either transcript_file_path or transcript_data must be provided")
    
    required_fields = ["meeting_topic", "experts", "transcript"]
    for field in required_fields:
        if field not in transcript_data:
            raise ValueError(f"Missing required field: {field}")
    
    eval_manager = GroqMeetingEvaluationManager(
        meeting_topic=transcript_data["meeting_topic"],
        experts=transcript_data["experts"],
        project_name=transcript_data.get("project_name", "Unknown Project")
    )
    
    evaluation_report = eval_manager.evaluate_transcript(transcript_data)
    
    eval_manager.shutdown()
    
    return evaluation_report

def example_transcript_evaluation():
    """Example of how to evaluate a transcript with Groq"""
    
    sample_transcript = {
        "meeting_topic": "Should we implement real-time analytics?",
        "project_name": "Analytics Platform",
        "experts": [
            {"title": "DataAnalyst", "expertise": "Data analysis", "role": "Analyst", "goal": "Provide insights"},
            {"title": "TechArchitect", "expertise": "Architecture", "role": "Technical Lead", "goal": "Ensure feasibility"},
            {"title": "BusinessStrategist", "expertise": "Strategy", "role": "Strategist", "goal": "Business alignment"}
        ],
        "transcript": [
            {"name": "# 🧑‍🔬 Team Meeting", "content": "## Should we implement real-time analytics?", "round": 0},
            {"name": "PI", "content": "Welcome everyone. Let's discuss real-time analytics implementation...", "round": 0},
            {"name": "DataAnalyst", "content": "Based on our user data, real-time analytics would provide significant value...", "round": 1},
            {"name": "TechArchitect", "content": "From a technical perspective, this is feasible with streaming architecture...", "round": 1},
            {"name": "BusinessStrategist", "content": "The business case is compelling with 15-20% conversion improvement...", "round": 1},
            {"name": "Critic", "content": "The proposals are well-founded but we should consider implementation costs...", "round": 1},
            {"name": "PI (Feedback)", "content": "Excellent insights. Let's synthesize these perspectives...", "round": 1},
            {"name": "** FINAL SUMMARY **", "content": "The team recommends proceeding with real-time analytics implementation...", "round": 1}
        ],
        "summary": "Team recommends implementing real-time analytics with streaming architecture",
        "timestamp": int(time.time())
    }
    
    print("🚀 Starting transcript evaluation with Groq (openai/gpt-oss-120b)...")
    evaluation_report = evaluate_meeting_transcript(transcript_data=sample_transcript)
    
    print("\n📊 Evaluation Results:")
    print(f"Overall Meeting Quality: {evaluation_report['overall_assessment']['average_score']:.2f}/1.0")
    print(f"Quality Rating: {evaluation_report['overall_assessment']['quality_rating']}")
    
    print("\n👥 Agent Performance:")
    for agent_name, report in evaluation_report['agent_performance'].items():
        if "error" not in report:
            print(f"  {agent_name}: {report['overall_average_score']:.2f}/1.0 ({report['total_responses']} responses)")
    
    print("\n🔬 Round-level Evaluations:")
    for round_eval in evaluation_report['round_evaluations']:
        print(f"  Round {round_eval['round']}:")
        for metric, result in round_eval['results'].items():
            print(f"    - {metric}: {result['score']:.2f}")

    return evaluation_report

if __name__ == "__main__":
    report = example_transcript_evaluation()
    
    import json
    with open("meeting_evaluation_report_groq.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n✅ Detailed report saved to 'meeting_evaluation_report_groq.json'")