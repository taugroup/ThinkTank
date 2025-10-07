from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import assert_test, evaluate
import pytest
from typing import List, Dict, Any
import time
import json
import concurrent.futures
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

    def _setup_evaluation_metrics(self):
        """Setup comprehensive evaluation metrics using Groq with direct GEval instances"""
        
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
            model=self.groq_evaluator,
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
            model=self.groq_evaluator,
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
            model=self.groq_evaluator,
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
            model=self.groq_evaluator,
            threshold=0.6,
            strict_mode=False
        )
        
        # Store all metrics
        self.all_metrics = [
            self.collaboration_metric,
            self.role_adherence_metric,
            self.response_quality_metric,
            self.meeting_progress_metric
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
        """Comprehensive evaluation of a single agent response using Groq"""
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
                    tracker["strengths"].append(f"{metric_name}: {reason[:100]}...")
            elif result["score"] <= 5:
                # Low score - extract improvement areas
                tracker["improvement_areas"].append(f"{metric_name}: {reason[:100]}...")
    
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
        print("🔍 Starting post-meeting transcript evaluation with Groq...")
        
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
    
    # [Include all the remaining helper methods from the original code - they remain unchanged]
    # I'll just include the key ones that reference the model:
    # Add these methods to the GroqMeetingEvaluationManager class:

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
            }
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating meeting summary report: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
            }

    def _calculate_basic_insights(self) -> Dict[str, Any]:
        """Calculate basic insights from evaluation results"""
        if not self.evaluation_results:
            return {"status": "No evaluations completed yet"}
        
        recent_results = self.evaluation_results[-5:]
        avg_composite = sum(r["composite_score"] for r in recent_results) / len(recent_results)
        
        agent_averages = {}
        for agent_name, tracker in self.agent_performance_tracker.items():
            if tracker["total_responses"] > 0:
                agent_averages[agent_name] = tracker["total_score"] / tracker["total_responses"]
        
        top_performer = max(agent_averages, key=agent_averages.get) if agent_averages else None
        
        return {
            "average_composite_score": avg_composite,
            "performance_trend": "improving",
            "top_performer": {
                "agent": top_performer,
                "score": agent_averages.get(top_performer, 0)
            } if top_performer else None,
        }

    def get_agent_detailed_report(self, agent_name: str) -> Dict[str, Any]:
        """Generate detailed performance report for a specific agent"""
        if agent_name not in self.agent_performance_tracker:
            return {"error": f"No data found for agent {agent_name}"}
        
        tracker = self.agent_performance_tracker[agent_name]
        
        return {
            "agent_name": agent_name,
            "overall_average": tracker["total_score"] / tracker["total_responses"] if tracker["total_responses"] > 0 else 0,
            "total_responses": tracker["total_responses"],
            "scores_by_round": tracker["scores_by_round"],
        }

    def _assess_overall_meeting_quality(self, insights: Dict) -> Dict[str, Any]:
        """Assess overall meeting quality"""
        avg_score = insights.get("average_composite_score", 0)
        
        if avg_score >= 0.85:
            rating = "excellent"
            strengths = ["Outstanding collaboration"]
            weaknesses = ["Minor optimization opportunities"]
        elif avg_score >= 0.70:
            rating = "good"
            strengths = ["Solid collaboration"]
            weaknesses = ["Some areas for improvement"]
        else:
            rating = "needs_improvement"
            strengths = ["Meeting completed"]
            weaknesses = ["Collaboration issues identified"]
        
        return {"rating": rating, "strengths": strengths, "weaknesses": weaknesses}

    def _generate_executive_summary(self, insights: Dict, agent_reports: Dict) -> str:
        """Generate executive summary"""
        avg_score = insights.get("average_composite_score", 0)
        return f"Meeting completed with quality score of {avg_score:.2f}/1.0"
    def _evaluate_meeting_holistically(self, transcript_data: Dict) -> Dict[str, Any]:
        """Perform holistic evaluation of the entire meeting"""
        print("🔬 Performing holistic meeting evaluation with Groq...")
        
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
            model=self.groq_evaluator
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
    
    # [All other helper methods remain the same - just copy them from the original]
    # Including: generate_meeting_summary_report, _calculate_basic_insights, 
    # _parse_transcript, etc. They don't need changes.

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


# [Copy all remaining helper methods from the original - they're unchanged]
# Just update the function name references:

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
    
    # Validate transcript data structure
    required_fields = ["meeting_topic", "experts", "transcript"]
    for field in required_fields:
        if field not in transcript_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Initialize evaluation manager with Groq
    eval_manager = GroqMeetingEvaluationManager(
        meeting_topic=transcript_data["meeting_topic"],
        experts=transcript_data["experts"],
        project_name=transcript_data.get("project_name", "Unknown Project")
    )
    
    # Evaluate the transcript
    evaluation_report = eval_manager.evaluate_transcript(transcript_data)
    
    # Cleanup
    eval_manager.shutdown()
    
    return evaluation_report


def example_transcript_evaluation():
    """Example of how to evaluate a transcript with Groq"""
    
    # [Same sample_transcript structure as original]
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
    print("🚀 Starting transcript evaluation with Groq (openai/gpt-oss-120b)...")
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
    with open("meeting_evaluation_report_groq.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n✅ Detailed report saved to 'meeting_evaluation_report_groq.json'")