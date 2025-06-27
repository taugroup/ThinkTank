from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import assert_test, evaluate
import pytest
from typing import List, Dict, Any
import time
import json
import concurrent.futures
from LLMWrapper import Qwen3OllamaWrapper

# class AgentCollaborationMetric(GEval):
#     def __init__(self, model):
#         super().__init__(
#             name="Agent Collaboration Quality",
#             criteria="""
#             Evaluate how well agents collaborate in the multi-agent system:
#             1. Information Flow: Do agents effectively build upon previous responses?
#             2. Role Adherence: Does each agent stay within their expertise area?
#             3. Knowledge Integration: Are specialized knowledge bases utilized appropriately?
#             4. Coherence: Does the conversation flow logically from one agent to the next?
#             5. Redundancy Avoidance: Do agents avoid repeating information unnecessarily?
#             """,
#             evaluation_params=[
#                 LLMTestCaseParams.INPUT,
#                 LLMTestCaseParams.ACTUAL_OUTPUT
#             ],
#             model=model,
#             strict_mode=True
#         )

# class AgentRoleAdherenceMetric(GEval):
#     def __init__(self, model):
#         super().__init__(
#             name="Agent Role Adherence",
#             criteria="""
#             Evaluate whether each agent stays within their defined role and expertise:
#             1. Expertise Boundaries: Does the agent only provide information within their domain?
#             2. Role Consistency: Does the agent maintain their assigned role throughout?
#             3. Knowledge Source Usage: Does the agent appropriately reference their vector store?
#             4. Goal Alignment: Are the agent's responses aligned with their stated goals?
#             """,
#             evaluation_params=[
#                 LLMTestCaseParams.INPUT,
#                 LLMTestCaseParams.ACTUAL_OUTPUT
#             ],
#             model=model,
#             strict_mode=True
#         )

# class AgentResponseQualityMetric(GEval):
#     def __init__(self, model):
#         super().__init__(
#             name="Agent Response Quality",
#             criteria="""
#             Evaluate the quality of an individual agent's response in a multi-agent meeting:
#             1. Expertise Relevance: Does the response demonstrate domain expertise?
#             2. Context Awareness: Does the agent build upon previous discussion points?
#             3. Information Value: Does the response add new, valuable insights?
#             4. Clarity: Is the response clear and well-structured?
#             5. Tool Usage: Are knowledge retrieval tools used effectively?
#             """,
#             evaluation_params=[
#                 LLMTestCaseParams.INPUT,
#                 LLMTestCaseParams.ACTUAL_OUTPUT
#             ],
#             model=model,
#             strict_mode=False
#         )

# class MeetingProgressMetric(GEval):
#     def __init__(self, model):
#         super().__init__(
#             name="Meeting Progress Quality",
#             criteria="""
#             Evaluate how well the meeting is progressing toward its objectives:
#             1. Topic Focus: Are responses staying on topic?
#             2. Collaborative Building: Are agents building on each other's contributions?
#             3. Problem Solving: Is the discussion moving toward solutions?
#             4. Knowledge Integration: Are different expertise areas being combined effectively?
#             """,
#             evaluation_params=[
#                 LLMTestCaseParams.INPUT,
#                 LLMTestCaseParams.ACTUAL_OUTPUT
#             ],
#             model=model,
#             strict_mode=False
#         )

class Qwen3MeetingEvaluationManager:
    """
    Advanced Meeting Evaluation Manager using Qwen3:8b for comprehensive multi-agent evaluation
    """
    
    def __init__(
        self, 
        meeting_topic: str, 
        experts: List[Dict], 
        project_name: str,
        ollama_base_url: str = "http://localhost:11434",
        max_workers: int = 3
    ):
        self.meeting_topic = meeting_topic
        self.experts = experts
        self.project_name = project_name
        self.conversation_history = []
        self.evaluation_results = []
        self.round_summaries = {}
        self.agent_performance_tracker = {}
        
        # Initialize thread pool for parallel evaluation
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize Qwen3 evaluator
        self.qwen3_evaluator = Qwen3OllamaWrapper(
            model_name="qwen3:8b",
            base_url=ollama_base_url,
            thinking_mode=True,  # Enable deep reasoning for evaluation
            temperature=0.1,     # Low temperature for consistent evaluation
            max_tokens=2000      # Longer responses for detailed evaluation
        )

        # Initialize evaluation metrics with Qwen3
        self._setup_evaluation_metrics()
        
        # Performance tracking
        self.start_time = time.time()

    def _setup_evaluation_metrics(self):
        """Setup comprehensive evaluation metrics using Qwen3 with direct GEval instances"""
        
        # Agent Collaboration Quality Metric
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
            model=self.qwen3_evaluator,
            threshold=0.6,
            strict_mode=False
        )
        
        # Agent Role Adherence Metric
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
            model=self.qwen3_evaluator,
            threshold=0.6,
            strict_mode=False
        )
        
        # Agent Response Quality Metric
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
            model=self.qwen3_evaluator,
            threshold=0.6,
            strict_mode=False
        )
        
        # Meeting Progress Assessment Metric
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
            model=self.qwen3_evaluator,
            threshold=0.6,
            strict_mode=False
        )
        
        # # Standard DeepEval metrics with Qwen3
        # self.relevancy_metric = AnswerRelevancyMetric(
        #     model=self.qwen3_evaluator,
        #     threshold=0.6
        # )
        
        # self.faithfulness_metric = FaithfulnessMetric(
        #     model=self.qwen3_evaluator,
        #     threshold=0.6
        # )
        
        # Store all metrics
        self.all_metrics = [
            self.collaboration_metric,
            self.role_adherence_metric,
            self.response_quality_metric,
            self.meeting_progress_metric,
            # self.relevancy_metric,
            # self.faithfulness_metric
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
        
        # Initialize agent performance tracking
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
    
    def evaluate_response_async(self, agent_name: str, content: str, round_num: int, response_type: str = "expert"):
        """Submit evaluation task to thread pool"""
        future = self.executor.submit(
            self._evaluate_single_response, 
            agent_name, content, round_num, response_type
        )
        return future
    
    def _evaluate_single_response(self, agent_name: str, content: str, round_num: int, response_type: str):
        """Comprehensive evaluation of a single agent response using Qwen3"""
        try:
            # Get agent information
            agent_info = next(
                (exp for exp in self.experts if exp.get('title') == agent_name), 
                {"title": agent_name, "expertise": "Unknown", "role": "Unknown", "goal": "Unknown"}
            )
            
            # Prepare conversation context
            recent_context = self._get_recent_context(agent_name, 5)
            meeting_stage = self._determine_meeting_stage(round_num)
            
            # Create comprehensive metadata
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
            
            # Create test case
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
            
            # Run all evaluations
            results = {}
            evaluation_start = time.time()
            
            for metric in self.all_metrics:
                try:
                    metric_start = time.time()
                    metric.measure(test_case)
                    metric_duration = time.time() - metric_start
                    
                    results[metric.name] = {
                        "score": metric.score,
                        "reason": metric.reason,
                        "evaluation_time": metric_duration
                    }
                    
                    print(f"✅ {metric.name}: {metric.score:.2f} ({metric_duration:.1f}s)")
                    
                except Exception as e:
                    print(f"❌ {metric.name} failed: {e}")
                    results[metric.name] = {
                        "score": 0,
                        "reason": f"Evaluation error: {str(e)}",
                        "evaluation_time": 0
                    }
            
            evaluation_duration = time.time() - evaluation_start
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(results)
            
            # Store comprehensive evaluation result
            evaluation_result = {
                "agent": agent_name,
                "round": round_num,
                "response_type": response_type,
                "timestamp": time.time(),
                "evaluation_duration": evaluation_duration,
                "composite_score": composite_score,
                "results": results,
                "metadata": metadata,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
            
            self.evaluation_results.append(evaluation_result)
            
            # Update agent performance tracking
            self._update_agent_performance(agent_name, evaluation_result)
            
            print(f"📊 {agent_name} (Round {round_num}) - Composite Score: {composite_score:.2f}")
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ Critical evaluation error for {agent_name}: {e}")
            return {
                "agent": agent_name,
                "round": round_num,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _calculate_composite_score(self, results: Dict) -> float:
        """Calculate weighted composite score from all metrics"""
        weights = {
            "Agent Collaboration Quality": 0.25,
            "Agent Role Adherence": 0.20,
            "Agent Response Quality": 0.25,
            "Meeting Progress Assessment": 0.15,
            "Answer Relevancy": 0.10,
            "Faithfulness": 0.05
        }
        
        total_score = 0
        total_weight = 0
        
        for metric_name, weight in weights.items():
            if metric_name in results and results[metric_name]["score"] > 0:
                total_score += results[metric_name]["score"] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _update_agent_performance(self, agent_name: str, evaluation_result: Dict):
        """Update agent performance tracking"""
        tracker = self.agent_performance_tracker[agent_name]
        
        # Update scores
        composite_score = evaluation_result["composite_score"]
        tracker["total_score"] += composite_score
        tracker["scores_by_round"][evaluation_result["round"]] = composite_score
        
        # Extract strengths and improvement areas from evaluation reasons
        for metric_name, result in evaluation_result["results"].items():
            reason = result.get("reason", "")
            if result["score"] >= 8:
                # High score - extract strengths
                if "excellent" in reason.lower() or "strong" in reason.lower():
                    tracker["strengths"].append(f"{metric_name}: {reason}...")
            elif result["score"] <= 5:
                # Low score - extract improvement areas
                tracker["improvement_areas"].append(f"{metric_name}: {reason}...")
    
    def _get_recent_context(self, current_agent: str, num_responses: int = 5) -> List[Dict]:
        """Get recent conversation context, excluding current agent's last response"""
        relevant_history = []
        
        for entry in reversed(self.conversation_history):
            if len(relevant_history) >= num_responses:
                break
            if entry["agent"] != current_agent:  # Exclude same agent's responses
                relevant_history.append({
                    "agent": entry["agent"],
                    "content": entry["content"][:150] + "..." if len(entry["content"]) > 150 else entry["content"],
                    "round": entry["round"],
                    "type": entry["type"]
                })
        
        return list(reversed(relevant_history))
    
    def _determine_meeting_stage(self, round_num: int) -> str:
        """Determine the current stage of the meeting"""
        total_rounds = max([entry["round"] for entry in self.conversation_history] + [round_num])
        
        if round_num == 1:
            return "Opening & Initial Contributions"
        elif round_num <= total_rounds * 0.4:
            return "Exploration & Information Gathering"
        elif round_num <= total_rounds * 0.7:
            return "Analysis & Discussion"
        elif round_num <= total_rounds * 0.9:
            return "Synthesis & Decision Making"
        else:
            return "Conclusion & Action Planning"
        
    def evaluate_transcript(self, transcript_data: Dict) -> Dict[str, Any]:
        """
        Evaluate a complete meeting transcript after the meeting is done
        
        Args:
            transcript_data: {
                "meeting_topic": str,
                "project_name": str,
                "experts": [{"title": str, "expertise": str, "role": str, "goal": str}],
                "transcript": [{"name": str, "content": str, "round": int, "type": str}],
                "summary": str,
                "timestamp": int
            }
        """
        print("🔍 Starting post-meeting transcript evaluation...")
        
        # Initialize from transcript data
        self.meeting_topic = transcript_data["meeting_topic"]
        self.project_name = transcript_data["project_name"]
        self.experts = transcript_data["experts"]
        
        # Parse and structure the transcript
        structured_responses = self._parse_transcript(transcript_data["transcript"])
        
        # Populate conversation history
        self.conversation_history = structured_responses
        self._initialize_agent_trackers()
    
        evaluation_futures = []
        
        for response in structured_responses:
            # Skip system messages and headers
            if response["type"] in ["system", "header"] or not response["content"].strip():
                continue
                
            future = self.evaluate_response_async(
                response["agent"],
                response["content"],
                response["round"],
                response["type"]
            )
            evaluation_futures.append(future)
        
        # Wait for all evaluations to complete
        print(f"⏳ Processing {len(evaluation_futures)} evaluations...")
        completed_evaluations = 0
        
        for future in evaluation_futures:
            try:
                result = future.result(timeout=3000) 
                if result and "error" not in result:
                    completed_evaluations += 1
            except Exception as e:
                print(f"⚠️ Evaluation error: {e}")
        
        print(f"✅ Completed {completed_evaluations}/{len(evaluation_futures)} evaluations")
        
        # Perform holistic meeting evaluation
        holistic_results = self._evaluate_meeting_holistically(transcript_data)
        
        # Generate comprehensive report
        final_report = self.generate_meeting_summary_report()
        final_report["holistic_evaluation"] = holistic_results
        final_report["transcript_metadata"] = {
            "original_timestamp": transcript_data.get("timestamp", ""),
            "evaluation_timestamp": time.time(),
            "total_responses": len(structured_responses),
            "evaluated_responses": completed_evaluations
        }
        
        return final_report
    
    def _parse_transcript(self, transcript: List[Dict]) -> List[Dict]:
        """Parse transcript into structured response format"""
        structured_responses = []
        
        for entry in transcript:
            # Determine response type based on agent name patterns
            response_type = self._classify_response_type(entry["name"])
            
            # Extract round number (default to 1 if not specified)
            round_num = entry.get("round", 1)
            
            structured_response = {
                "agent": entry["name"],
                "content": entry["content"],
                "round": round_num,
                "type": response_type,
                "timestamp": time.time(),
                "word_count": len(entry["content"].split()),
                "character_count": len(entry["content"])
            }
            
            structured_responses.append(structured_response)
        
        return structured_responses
    
    def _classify_response_type(self, agent_name: str) -> str:
        """Classify response type based on agent name"""
        agent_name_lower = agent_name.lower()
        
        if any(keyword in agent_name_lower for keyword in ["meeting", "round", "summary", "final"]):
            return "system"
        elif "critic" in agent_name_lower:
            return "critic"
        elif "feedback" in agent_name_lower or "pi" in agent_name_lower:
            return "facilitator"
        elif agent_name.startswith("#") or agent_name.startswith("**"):
            return "header"
        else:
            return "expert"
    
    def _initialize_agent_trackers(self):
        """Initialize agent performance trackers from transcript"""
        for response in self.conversation_history:
            agent_name = response["agent"]
            
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
            self.agent_performance_tracker[agent_name]["response_types"].append(response["type"])
    
    def _evaluate_meeting_holistically(self, transcript_data: Dict) -> Dict[str, Any]:
        """Perform holistic evaluation of the entire meeting"""
        print("🔬 Performing holistic meeting evaluation...")
        
        # Create holistic evaluation metric
        holistic_metric = GEval(
            name="Holistic Meeting Quality",
            criteria="""
            Evaluate the overall quality and effectiveness of this multi-agent meeting:
            
            **Meeting Effectiveness (Rate 1-10):**
            1. **Objective Achievement** (25%): Did the meeting achieve its stated goals?
            2. **Collaborative Intelligence** (20%): Did agents effectively build on each other's expertise?
            3. **Decision Quality** (20%): Were well-informed decisions reached?
            4. **Knowledge Integration** (15%): How well were different expertise areas combined?
            5. **Meeting Flow** (10%): Was the discussion logical and well-structured?
            6. **Actionable Outcomes** (10%): Did the meeting produce clear next steps?
            
            **Consider the complete meeting arc:**
            - Opening and agenda setting
            - Information sharing and exploration
            - Analysis and discussion
            - Synthesis and decision-making
            - Conclusion and action planning
            
            Provide specific examples of what worked well and what could be improved.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            model=self.qwen3_evaluator
        )
        
        # Prepare full meeting context
        full_transcript = "\n\n".join([
            f"[{entry['round']}] {entry['agent']}: {entry['content']}"
            for entry in self.conversation_history
            if entry["type"] != "system"
        ])
        
        # Create test case for holistic evaluation
        holistic_test_case = LLMTestCase(
            input=f"""
                Meeting Topic: {self.meeting_topic}
                Project: {self.project_name}
                Participants: {', '.join([exp['title'] for exp in self.experts])}
                Meeting Summary: {transcript_data.get('summary', 'No summary provided')}
            """.strip(),
            actual_output=full_transcript
        )
        
        # Run holistic evaluation
        try:
            holistic_metric.measure(holistic_test_case)
            return {
                "overall_score": holistic_metric.score,
                "detailed_analysis": holistic_metric.reason,
                "evaluation_success": True
            }
        except Exception as e:
            print(f"❌ Holistic evaluation failed: {e}")
            return {
                "overall_score": 0,
                "detailed_analysis": f"Evaluation failed: {str(e)}",
                "evaluation_success": False
            }
    
    def generate_meeting_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive meeting summary report"""
        try:
            # Generate agent reports
            agent_reports = {}
            for agent_name in self.agent_performance_tracker.keys():
                agent_reports[agent_name] = self.get_agent_detailed_report(agent_name)

            
            
            # Calculate basic insights directly from evaluation results
            basic_insights = self._calculate_basic_insights()
            
            # Calculate meeting quality assessment
            overall_quality = self._assess_overall_meeting_quality(basic_insights)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(basic_insights, agent_reports)
            
            # Create comprehensive report structure
            report = {
                "meeting_metadata": {
                    "topic": self.meeting_topic,
                    "project": self.project_name,
                    "duration_minutes": round((time.time() - self.start_time) / 60, 2),
                    "total_responses": len(self.conversation_history),
                    "total_evaluations": len(self.evaluation_results),
                    "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "participants": [agent["title"] for agent in self.experts],
                    "meeting_stages_covered": list(set([self._determine_meeting_stage(r) for r in range(1, max([entry["round"] for entry in self.conversation_history], default=[1]) + 1)]))
                },
                
                "executive_summary": executive_summary,
                
                "overall_assessment": {
                    "quality_rating": overall_quality["rating"],
                    "average_score": basic_insights.get("average_composite_score", 0),
                    "performance_trend": basic_insights.get("performance_trend", "unknown"),
                    "meeting_stage": basic_insights.get("meeting_stage", "unknown"),
                    "key_strengths": overall_quality["strengths"],
                    "improvement_areas": overall_quality["weaknesses"],
                    "meeting_effectiveness_score": self._calculate_meeting_effectiveness()
                },
                
                "detailed_metrics": {
                    "collaboration_quality": self._get_metric_summary("Agent Collaboration Quality"),
                    "role_adherence": self._get_metric_summary("Agent Role Adherence"),
                    "response_quality": self._get_metric_summary("Agent Response Quality"),
                    "meeting_progress": self._get_metric_summary("Meeting Progress Assessment")
                },
                
                "agent_performance": agent_reports,
                
                "meeting_flow_analysis": {
                    "round_progression": self._analyze_round_progression(),
                    "participation_balance": self._analyze_participation_balance(),
                    "topic_coherence": self._analyze_topic_coherence(),
                    "decision_making_quality": self._analyze_decision_making()
                },
                
                "key_insights": basic_insights,
                
                "recommendations": self._generate_comprehensive_recommendations(basic_insights, agent_reports),
                
                "action_items": self._extract_action_items(),
                
                "evaluation_metadata": {
                    "evaluator_model": "qwen3:8b",
                    "total_evaluation_time": sum(r.get("evaluation_duration", 0) for r in self.evaluation_results),
                    "average_evaluation_time": sum(r.get("evaluation_duration", 0) for r in self.evaluation_results) / len(self.evaluation_results) if self.evaluation_results else 0,
                    "evaluation_success_rate": self._calculate_evaluation_success_rate(),
                    "metrics_used": [getattr(metric, 'name', metric.__class__.__name__) for metric in self.all_metrics]
                },
                
                "appendices": {
                    "detailed_conversation_analysis": self._generate_conversation_analysis(),
                    "statistical_summary": self._generate_statistical_summary(),
                    "quality_trends": self._generate_quality_trends()
                }
            }
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating meeting summary report: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
                "partial_data": {
                    "meeting_topic": self.meeting_topic,
                    "project_name": self.project_name,
                    "total_responses": len(self.conversation_history),
                    "total_evaluations": len(self.evaluation_results)
                }
            }

    def _calculate_basic_insights(self) -> Dict[str, Any]:
        """Calculate basic insights from evaluation results without complex analysis"""
        if not self.evaluation_results:
            return {"status": "No evaluations completed yet"}
        
        # Recent performance (last 5 evaluations)
        recent_results = self.evaluation_results[-5:]
        
        # Calculate averages
        avg_composite = sum(r["composite_score"] for r in recent_results) / len(recent_results)
        
        # Agent performance analysis
        agent_averages = {}
        for agent_name, tracker in self.agent_performance_tracker.items():
            if tracker["total_responses"] > 0:
                agent_averages[agent_name] = tracker["total_score"] / tracker["total_responses"]
        
        # Top and bottom performers
        top_performer = max(agent_averages, key=agent_averages.get) if agent_averages else None
        bottom_performer = min(agent_averages, key=agent_averages.get) if agent_averages else None
        
        # Meeting progression analysis
        round_scores = {}
        for result in self.evaluation_results:
            round_num = result["round"]
            if round_num not in round_scores:
                round_scores[round_num] = []
            round_scores[round_num].append(result["composite_score"])
        
        round_averages = {r: sum(scores)/len(scores) for r, scores in round_scores.items()}
        
        # Trend analysis
        if len(round_averages) >= 2:
            recent_rounds = sorted(round_averages.keys())[-2:]
            trend = "improving" if round_averages[recent_rounds[-1]] > round_averages[recent_rounds[0]] else "declining"
        else:
            trend = "stable"
        
        return {
            "meeting_duration": time.time() - self.start_time,
            "total_evaluations": len(self.evaluation_results),
            "average_composite_score": avg_composite,
            "performance_trend": trend,
            "top_performer": {
                "agent": top_performer,
                "score": agent_averages.get(top_performer, 0)
            } if top_performer else None,
            "bottom_performer": {
                "agent": bottom_performer,
                "score": agent_averages.get(bottom_performer, 0)
            } if bottom_performer else None,
            "round_progression": round_averages,
            "agent_performance": agent_averages,
            "meeting_stage": self._determine_meeting_stage(max(round_scores.keys()) if round_scores else 1),
            "evaluation_efficiency": {
                "avg_evaluation_time": sum(r.get("evaluation_duration", 0) for r in recent_results) / len(recent_results),
                "total_evaluation_time": sum(r.get("evaluation_duration", 0) for r in self.evaluation_results)
            }
        }

    def get_agent_detailed_report(self, agent_name: str) -> Dict[str, Any]:
        """Generate detailed performance report for a specific agent"""
        if agent_name not in self.agent_performance_tracker:
            return {"error": f"No data found for agent {agent_name}"}
        
        tracker = self.agent_performance_tracker[agent_name]
        agent_results = [r for r in self.evaluation_results if r["agent"] == agent_name]
        
        if not agent_results:
            return {"error": f"No evaluation results for agent {agent_name}"}
        
        # Calculate detailed metrics
        scores_by_metric = {}
        for result in agent_results:
            for metric_name, metric_result in result["results"].items():
                if metric_name not in scores_by_metric:
                    scores_by_metric[metric_name] = []
                scores_by_metric[metric_name].append(metric_result["score"])
        
        metric_averages = {
            metric: sum(scores) / len(scores) 
            for metric, scores in scores_by_metric.items()
        }
        
        return {
            "agent_name": agent_name,
            "agent_info": next((exp for exp in self.experts if exp.get('title') == agent_name), {}),
            "overall_average": tracker["total_score"] / tracker["total_responses"],
            "total_responses": tracker["total_responses"],
            "scores_by_round": tracker["scores_by_round"],
            "metric_breakdown": metric_averages,
            "strengths": tracker["strengths"][-3:],  # Last 3 strengths
            "improvement_areas": tracker["improvement_areas"][-3:],  # Last 3 areas
            "response_types": list(set(tracker["response_types"])),
            "performance_trend": self._calculate_agent_trend(agent_name)
        }

    def _assess_overall_meeting_quality(self, insights: Dict) -> Dict[str, Any]:
        """Assess overall meeting quality and provide rating"""
        avg_score = insights.get("average_composite_score", 0)
        
        if avg_score >= 0.85:
            rating = "excellent"
            strengths = ["Outstanding collaboration", "High-quality responses", "Effective knowledge integration"]
            weaknesses = ["Minor optimization opportunities"]
        elif avg_score >= 0.70:
            rating = "good"
            strengths = ["Solid collaboration", "Good expertise demonstration", "Clear communication"]
            weaknesses = ["Some areas for improvement in response quality", "Could enhance knowledge integration"]
        elif avg_score >= 0.55:
            rating = "satisfactory"
            strengths = ["Basic collaboration present", "Some valuable insights shared"]
            weaknesses = ["Inconsistent response quality", "Limited knowledge integration", "Room for better collaboration"]
        elif avg_score >= 0.40:
            rating = "needs_improvement"
            strengths = ["Some positive contributions identified"]
            weaknesses = ["Poor collaboration patterns", "Low response quality", "Ineffective knowledge sharing"]
        else:
            rating = "poor"
            strengths = ["Meeting completed"]
            weaknesses = ["Significant collaboration issues", "Very low response quality", "Ineffective communication"]
        
        return {
            "rating": rating,
            "strengths": strengths,
            "weaknesses": weaknesses
        }

    def _generate_executive_summary(self, insights: Dict, agent_reports: Dict) -> str:
        """Generate executive summary of the meeting"""
        try:
            # Key metrics
            avg_score = insights.get("average_composite_score", 0)
            total_responses = len(self.conversation_history)
            duration = round((time.time() - self.start_time) / 60, 2)
            
            # Top performer
            top_performer = insights.get("top_performer", {})
            top_agent = top_performer.get("agent", "N/A")
            top_score = top_performer.get("score", 0)
            
            # Meeting trend
            trend = insights.get("performance_trend", "stable")
            
            summary = f"""
                **Meeting Overview:**
                The {self.meeting_topic} meeting for {self.project_name} concluded with an overall quality score of {avg_score:.2f}/1.0. 
                Over {duration} minutes, {len(self.experts)} participants contributed {total_responses} responses across multiple discussion rounds.

                **Key Performance Highlights:**
                - Top performing agent: {top_agent} (Score: {top_score:.2f})
                - Meeting quality trend: {trend.title()}
                - Total evaluations completed: {len(self.evaluation_results)}

                **Meeting Effectiveness:**
                The discussion demonstrated {'strong' if avg_score > 0.7 else 'moderate' if avg_score > 0.5 else 'limited'} collaborative effectiveness, 
                with agents {'successfully building upon each others expertise' if avg_score > 0.7 else 'showing some collaboration' if avg_score > 0.5 else 'requiring improved coordination'}.

                **Outcome Assessment:**
                {'The meeting achieved its objectives with high-quality collaborative intelligence.' if avg_score > 0.8 else 
                'The meeting made good progress toward its objectives with room for enhancement.' if avg_score > 0.6 else
                'The meeting addressed the topic but would benefit from improved collaboration and focus.'}"""
            
            return summary
            
        except Exception as e:
            return f"Executive summary generation failed: {str(e)}"

    def _get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific metric"""
        metric_results = []
        
        for result in self.evaluation_results:
            if metric_name in result.get("results", {}):
                metric_results.append(result["results"][metric_name]["score"])
        
        if not metric_results:
            return {"average": 0, "count": 0, "trend": "no_data"}
        
        return {
            "average": sum(metric_results) / len(metric_results),
            "maximum": max(metric_results),
            "minimum": min(metric_results),
            "count": len(metric_results),
            "trend": "improving" if len(metric_results) > 1 and metric_results[-1] > metric_results[0] else "stable"
        }

    def _analyze_round_progression(self) -> Dict[str, Any]:
        """Analyze how the meeting progressed through rounds"""
        round_scores = {}
        
        for result in self.evaluation_results:
            round_num = result["round"]
            if round_num not in round_scores:
                round_scores[round_num] = []
            round_scores[round_num].append(result["composite_score"])
        
        round_averages = {r: sum(scores)/len(scores) for r, scores in round_scores.items()}
        
        return {
            "round_averages": round_averages,
            "total_rounds": len(round_averages),
            "best_round": max(round_averages, key=round_averages.get) if round_averages else None,
            "progression_trend": "improving" if len(round_averages) > 1 and 
                            list(round_averages.values())[-1] > list(round_averages.values())[0] else "stable"
        }

    def _analyze_participation_balance(self) -> Dict[str, Any]:
        """Analyze participation balance among agents"""
        participation = {}
        
        for entry in self.conversation_history:
            agent = entry["agent"]
            if entry["type"] == "expert":  # Only count expert responses
                participation[agent] = participation.get(agent, 0) + 1
        
        if not participation:
            return {"balance": "no_data", "distribution": {}}
        
        total_responses = sum(participation.values())
        expected_per_agent = total_responses / len(self.experts)
        
        # Calculate balance score (closer to 1 means better balance)
        balance_score = 1 - (max(participation.values()) - min(participation.values())) / total_responses
        
        return {
            "balance_score": balance_score,
            "distribution": participation,
            "expected_per_agent": expected_per_agent,
            "most_active": max(participation, key=participation.get),
            "least_active": min(participation, key=participation.get)
        }

    def _analyze_topic_coherence(self) -> Dict[str, Any]:
        """Analyze how well the discussion stayed on topic"""
        topic_keywords = self.meeting_topic.lower().split()
        
        on_topic_responses = 0
        total_responses = 0
        
        for entry in self.conversation_history:
            if entry["type"] == "expert":
                total_responses += 1
                content_lower = entry["content"].lower()
                if any(keyword in content_lower for keyword in topic_keywords):
                    on_topic_responses += 1
        
        coherence_score = on_topic_responses / total_responses if total_responses > 0 else 0
        
        return {
            "coherence_score": coherence_score,
            "on_topic_responses": on_topic_responses,
            "total_responses": total_responses,
            "assessment": "high" if coherence_score > 0.8 else "moderate" if coherence_score > 0.6 else "low"
        }

    def _analyze_decision_making(self) -> Dict[str, Any]:
        """Analyze the quality of decision-making in the meeting"""
        decision_indicators = ["recommend", "decide", "conclude", "agree", "propose", "suggest"]
        
        decisions_made = 0
        total_responses = 0
        
        for entry in self.conversation_history:
            if entry["type"] in ["expert", "facilitator"]:
                total_responses += 1
                content_lower = entry["content"].lower()
                if any(indicator in content_lower for indicator in decision_indicators):
                    decisions_made += 1
        
        decision_density = decisions_made / total_responses if total_responses > 0 else 0
        
        return {
            "decision_density": decision_density,
            "decisions_identified": decisions_made,
            "total_responses": total_responses,
            "quality": "high" if decision_density > 0.3 else "moderate" if decision_density > 0.15 else "low"
        }

    def _generate_comprehensive_recommendations(self, insights: Dict, agent_reports: Dict) -> List[str]:
        """Generate comprehensive recommendations based on evaluation results"""
        recommendations = []
        
        avg_score = insights.get("average_composite_score", 0)
        
        # Overall meeting quality recommendations
        if avg_score < 0.6:
            recommendations.append("Consider implementing structured discussion protocols to improve overall meeting quality")
            recommendations.append("Provide pre-meeting briefings to help agents better prepare their contributions")
        
        # Agent-specific recommendations
        for agent_name, report in agent_reports.items():
            if "error" not in report:
                agent_avg = report.get("overall_average", 0)
                if agent_avg < 0.5:
                    recommendations.append(f"{agent_name}: Focus on staying within expertise area and building on team discussions")
                elif agent_avg > 0.8:
                    recommendations.append(f"{agent_name}: Excellent performance - consider mentoring other team members")
        
        # Trend-based recommendations
        trend = insights.get("performance_trend", "stable")
        if trend == "declining":
            recommendations.append("Meeting quality is declining - consider a brief break or refocus on objectives")
        elif trend == "improving":
            recommendations.append("Meeting momentum is positive - maintain current collaborative approach")
        
        return recommendations

    def _extract_action_items(self) -> List[Dict[str, str]]:
        """Extract action items from the meeting discussion"""
        action_indicators = ["action", "todo", "task", "follow up", "next step", "assign", "responsible"]
        action_items = []
        
        for entry in self.conversation_history:
            content_lower = entry["content"].lower()
            if any(indicator in content_lower for indicator in action_indicators):
                action_items.append({
                    "source": entry["agent"],
                    "content": entry["content"][:200] + "..." if len(entry["content"]) > 200 else entry["content"],
                    "round": entry["round"],
                    "type": "identified_action"
                })
        
        return action_items

    def _calculate_meeting_effectiveness(self) -> float:
        """Calculate overall meeting effectiveness score"""
        if not self.evaluation_results:
            return 0.0
        
        # Weighted combination of different effectiveness factors
        collaboration_avg = self._get_metric_summary("Agent Collaboration Quality").get("average", 0)
        progress_avg = self._get_metric_summary("Meeting Progress Assessment").get("average", 0)
        quality_avg = self._get_metric_summary("Agent Response Quality").get("average", 0)
        
        # Participation balance factor
        participation = self._analyze_participation_balance()
        balance_factor = participation.get("balance_score", 0)
        
        # Topic coherence factor
        coherence = self._analyze_topic_coherence()
        coherence_factor = coherence.get("coherence_score", 0)
        
        # Weighted effectiveness score
        effectiveness = (
            collaboration_avg * 0.3 +
            progress_avg * 0.25 +
            quality_avg * 0.25 +
            balance_factor * 0.1 +
            coherence_factor * 0.1
        )
        
        return effectiveness

    def _calculate_evaluation_success_rate(self) -> float:
        """Calculate the success rate of evaluations"""
        if not self.evaluation_results:
            return 0.0
        
        successful_evaluations = sum(1 for result in self.evaluation_results if "error" not in result)
        return successful_evaluations / len(self.evaluation_results)

    def _generate_conversation_analysis(self) -> Dict[str, Any]:
        """Generate detailed conversation analysis"""
        return {
            "total_words": sum(entry["word_count"] for entry in self.conversation_history),
            "average_response_length": sum(entry["word_count"] for entry in self.conversation_history) / len(self.conversation_history) if self.conversation_history else 0,
            "response_types_distribution": self._get_response_type_distribution(),
            "longest_response": max(self.conversation_history, key=lambda x: x["word_count"], default={}).get("word_count", 0),
            "shortest_response": min(self.conversation_history, key=lambda x: x["word_count"], default={}).get("word_count", 0)
        }

    def _get_response_type_distribution(self) -> Dict[str, int]:
        """Get distribution of response types"""
        distribution = {}
        for entry in self.conversation_history:
            response_type = entry["type"]
            distribution[response_type] = distribution.get(response_type, 0) + 1
        return distribution

    def _generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate statistical summary of evaluations"""
        if not self.evaluation_results:
            return {}
        
        all_scores = [result["composite_score"] for result in self.evaluation_results]
        
        return {
            "mean_score": sum(all_scores) / len(all_scores),
            "median_score": sorted(all_scores)[len(all_scores) // 2],
            "std_deviation": self._calculate_std_deviation(all_scores),
            "score_range": max(all_scores) - min(all_scores),
            "total_evaluations": len(all_scores)
        }

    def _calculate_std_deviation(self, scores: List[float]) -> float:
        """Calculate standard deviation of scores"""
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance ** 0.5

    def _generate_quality_trends(self) -> Dict[str, Any]:
        """Generate quality trends over time"""
        if len(self.evaluation_results) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        sorted_results = sorted(self.evaluation_results, key=lambda x: x["timestamp"])
        
        # Calculate trend
        first_half = sorted_results[:len(sorted_results)//2]
        second_half = sorted_results[len(sorted_results)//2:]
        
        first_avg = sum(r["composite_score"] for r in first_half) / len(first_half)
        second_avg = sum(r["composite_score"] for r in second_half) / len(second_half)
        
        return {
            "trend": "improving" if second_avg > first_avg else "declining" if second_avg < first_avg else "stable",
            "first_half_average": first_avg,
            "second_half_average": second_avg,
            "improvement_rate": second_avg - first_avg
        }

    def _calculate_agent_trend(self, agent_name: str) -> str:
        """Calculate performance trend for specific agent"""
        agent_results = [r for r in self.evaluation_results if r["agent"] == agent_name]
        
        if len(agent_results) < 2:
            return "insufficient_data"
        
        # Compare first half vs second half of responses
        mid_point = len(agent_results) // 2
        first_half_avg = sum(r["composite_score"] for r in agent_results[:mid_point]) / mid_point
        second_half_avg = sum(r["composite_score"] for r in agent_results[mid_point:]) / (len(agent_results) - mid_point)
        
        if second_half_avg > first_half_avg + 0.1:
            return "improving"
        elif second_half_avg < first_half_avg - 0.1:
            return "declining"
        else:
            return "stable"

    def shutdown(self):
        """Cleanup resources and generate final report"""
        print("🔄 Shutting down evaluation manager...")
        
        # Wait for pending evaluations
        self.executor.shutdown(wait=True)
        
        # Generate final report
        final_report = self.generate_meeting_summary_report()
        
        print(f"✅ Evaluation complete. Total evaluations: {len(self.evaluation_results)}")
        if final_report.get("overall_assessment"):
            print(f"📊 Average meeting quality: {final_report['overall_assessment']['average_score']:.2f}/1.0")
        
        return final_report


# Usage function for transcript evaluation
def evaluate_meeting_transcript(transcript_file_path: str = None, transcript_data: Dict = None):
    """
    Evaluate a meeting transcript from file or data
    
    Args:
        transcript_file_path: Path to JSON file containing transcript
        transcript_data: Direct transcript data dictionary
    """
    
    if transcript_file_path:
        import json
        with open(transcript_file_path, 'r') as f:
            transcript_data = json.load(f)
    
    if not transcript_data:
        raise ValueError("Either transcript_file_path or transcript_data must be provided")
    
    # Validate transcript data structure
    required_fields = ["meeting_topic", "experts", "transcript"]
    for field in required_fields:
        if field not in transcript_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Initialize evaluation manager
    eval_manager = Qwen3MeetingEvaluationManager(
        meeting_topic=transcript_data["meeting_topic"],
        experts=transcript_data["experts"],
        project_name=transcript_data.get("project_name", "Unknown Project")
    )
    
    # Evaluate the transcript
    evaluation_report = eval_manager.evaluate_transcript(transcript_data)
    
    # Cleanup
    eval_manager.shutdown()
    
    return evaluation_report


# Example usage with your meeting transcript format
def example_transcript_evaluation():
    """Example of how to evaluate a transcript"""
    
    # Example transcript data structure (matches your meeting format)
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
    
    # Evaluate the transcript
    print("🚀 Starting transcript evaluation...")
    evaluation_report = evaluate_meeting_transcript(transcript_data=sample_transcript)
    
    # Display results
    print("\n📊 Evaluation Results:")
    print(f"Overall Meeting Quality: {evaluation_report['overall_assessment']['average_score']:.2f}/1.0")
    print(f"Quality Rating: {evaluation_report['overall_assessment']['quality_rating']}")
    
    # Agent performance summary
    print("\n👥 Agent Performance:")
    for agent_name, report in evaluation_report['agent_performance'].items():
        if "error" not in report:
            print(f"  {agent_name}: {report['overall_average']:.2f}/1.0 ({report['total_responses']} responses)")
    
    return evaluation_report

if __name__ == "__main__":
    # Run example evaluation
    report = example_transcript_evaluation()
    
    # Save detailed report
    import json
    with open("meeting_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n✅ Detailed report saved to 'meeting_evaluation_report.json'")