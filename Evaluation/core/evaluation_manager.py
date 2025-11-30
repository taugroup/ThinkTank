"""
Main evaluation manager for coordinating the evaluation process
"""

import os
import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from ..evaluators import GeminiEvaluator, GroqEvaluator, QwenEvaluator
from ..metrics import create_collaboration_metrics, create_quality_metrics, create_progress_metrics
from .transcript_parser import TranscriptParser
from .report_generator import ReportGenerator


class EvaluationManager:
    """
    Main evaluation manager for meeting transcript analysis
    """
    
    def __init__(
        self, 
        meeting_topic: str, 
        experts: List[Dict], 
        project_name: str,
        max_workers: int = 3
    ):
        self.meeting_topic = meeting_topic
        self.experts = experts
        self.project_name = project_name
        self.conversation_history = []
        self.evaluation_results = []
        self.agent_performance_tracker = {}
        
        # Initialize thread pool for parallel evaluation
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize evaluators with fallback options
        self._initialize_evaluators()
        
        # Initialize evaluation metrics
        self._setup_evaluation_metrics()
        
        # Performance tracking
        self.start_time = time.time()
    
    def _initialize_evaluators(self):
        """Initialize evaluators with fallback options"""
        self.primary_evaluator = None
        self.fallback_evaluator = None
        
        # Try to initialize Gemini first (most reliable)
        try:
            self.gemini_evaluator = GeminiEvaluator(temperature=0.1, max_tokens=2000)
            self.primary_evaluator = self.gemini_evaluator
            print("✅ Primary evaluator: Gemini 2.5 Flash")
        except Exception as e:
            print(f"⚠️ Gemini evaluator failed: {e}")
        
        # Try Groq as fallback
        try:
            self.groq_evaluator = GroqEvaluator(
                model_name="llama-3.1-70b-versatile",
                temperature=0.1,
                max_tokens=2000
            )
            if not self.primary_evaluator:
                self.primary_evaluator = self.groq_evaluator
            else:
                self.fallback_evaluator = self.groq_evaluator
            print("✅ Groq evaluator initialized")
        except Exception as e:
            print(f"⚠️ Groq evaluator failed: {e}")
        
        # Try Qwen3 as last resort (requires local setup)
        try:
            self.qwen_evaluator = QwenEvaluator(
                model_name="qwen3:8b",
                thinking_mode=True,
                temperature=0.1,
                max_tokens=2000
            )
            if not self.primary_evaluator:
                self.primary_evaluator = self.qwen_evaluator
            elif not self.fallback_evaluator:
                self.fallback_evaluator = self.qwen_evaluator
            print("✅ Qwen3 evaluator initialized")
        except Exception as e:
            print(f"⚠️ Qwen3 evaluator failed: {e}")
        
        if not self.primary_evaluator:
            raise Exception("No evaluators could be initialized. Please check your API keys and setup.")
        
        print(f"🎯 Using primary evaluator: {type(self.primary_evaluator).__name__}")
        if self.fallback_evaluator:
            print(f"🔄 Fallback evaluator: {type(self.fallback_evaluator).__name__}")
    
    def _setup_evaluation_metrics(self):
        """Setup comprehensive evaluation metrics"""
        # Create metrics using the primary evaluator
        collaboration_metrics = create_collaboration_metrics(self.primary_evaluator)
        quality_metrics = create_quality_metrics(self.primary_evaluator)
        progress_metrics = create_progress_metrics(self.primary_evaluator)
        
        # Store all metrics
        self.all_metrics = [
            collaboration_metrics["collaboration"],
            collaboration_metrics["role_adherence"],
            quality_metrics["response_quality"],
            progress_metrics["meeting_progress"]
        ]
        
        # Store holistic metric separately
        self.holistic_metric = progress_metrics["holistic_meeting"]
    
    def evaluate_transcript(self, transcript_data: Dict) -> Dict[str, Any]:
        """
        Evaluate a complete meeting transcript
        """
        print("🔍 Starting post-meeting transcript evaluation...")
        
        # Validate and parse transcript
        TranscriptParser.validate_transcript(transcript_data)
        structured_responses = TranscriptParser.parse_transcript(transcript_data["transcript"])
        
        # Populate conversation history
        self.conversation_history = structured_responses
        self._initialize_agent_trackers()
        
        # Submit evaluation tasks
        evaluation_futures = []
        for response in structured_responses:
            # Skip system messages and headers
            if response["type"] in ["system", "header"] or not response["content"].strip():
                continue
                
            future = self.executor.submit(
                self._evaluate_single_response,
                response["agent"],
                response["content"],
                response["round"],
                response["type"]
            )
            evaluation_futures.append(future)
        
        # Wait for all evaluations to complete with progress tracking
        self._wait_for_evaluations(evaluation_futures)
        
        # Perform holistic meeting evaluation
        holistic_results = self._evaluate_meeting_holistically(transcript_data)
        
        # Generate comprehensive report
        report_generator = ReportGenerator(
            self.meeting_topic,
            self.project_name,
            self.conversation_history,
            self.evaluation_results,
            self.agent_performance_tracker,
            self.start_time
        )
        
        final_report = report_generator.generate_report()
        final_report["holistic_evaluation"] = holistic_results
        final_report["transcript_metadata"] = {
            "original_timestamp": transcript_data.get("timestamp", ""),
            "evaluation_timestamp": time.time(),
            "total_responses": len(structured_responses),
            "evaluated_responses": len([r for r in self.evaluation_results if "error" not in r])
        }
        
        return final_report
    
    def _wait_for_evaluations(self, evaluation_futures: List):
        """Wait for all evaluations to complete with progress tracking"""
        print(f"⏳ Processing {len(evaluation_futures)} evaluations...")
        completed_evaluations = 0
        failed_evaluations = 0
        
        for i, future in enumerate(evaluation_futures, 1):
            try:
                result = future.result(timeout=300)  # 5 minute timeout per evaluation
                if result and "error" not in result:
                    completed_evaluations += 1
                    print(f"📊 Progress: {completed_evaluations}/{len(evaluation_futures)} completed")
                else:
                    failed_evaluations += 1
                    print(f"⚠️ Evaluation {i} failed")
            except Exception as e:
                failed_evaluations += 1
                print(f"⚠️ Evaluation {i} error: {e}")
        
        print(f"✅ Completed {completed_evaluations}/{len(evaluation_futures)} evaluations")
        if failed_evaluations > 0:
            print(f"⚠️ {failed_evaluations} evaluations failed")
    
    def _evaluate_single_response(self, agent_name: str, content: str, round_num: int, response_type: str):
        """Comprehensive evaluation of a single agent response"""
        try:
            # Get agent information
            agent_info = TranscriptParser.get_agent_info(agent_name, self.experts)
            
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
            
            # Run all evaluations with retry logic
            results = self._run_metrics_with_retry(test_case)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(results)
            
            # Store comprehensive evaluation result
            evaluation_result = {
                "agent": agent_name,
                "round": round_num,
                "response_type": response_type,
                "timestamp": time.time(),
                "composite_score": composite_score,
                "results": results,
                "metadata": metadata,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
            
            self.evaluation_results.append(evaluation_result)
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
    
    def _run_metrics_with_retry(self, test_case: LLMTestCase) -> Dict:
        """Run all metrics with retry logic"""
        results = {}
        
        for metric in self.all_metrics:
            metric_success = False
            for attempt in range(2):  # Try twice
                try:
                    metric_start = time.time()
                    
                    # Use fallback evaluator on second attempt if available
                    if attempt == 1 and self.fallback_evaluator:
                        print(f"🔄 Retrying {metric.name} with fallback evaluator...")
                        original_model = metric.model
                        metric.model = self.fallback_evaluator
                    
                    metric.measure(test_case)
                    metric_duration = time.time() - metric_start
                    
                    results[metric.name] = {
                        "score": metric.score,
                        "reason": metric.reason,
                        "evaluation_time": metric_duration,
                        "evaluator_used": type(metric.model).__name__,
                        "attempt": attempt + 1
                    }
                    
                    print(f"✅ {metric.name}: {metric.score:.2f} ({metric_duration:.1f}s)")
                    metric_success = True
                    
                    # Restore original model if we used fallback
                    if attempt == 1 and self.fallback_evaluator:
                        metric.model = original_model
                    
                    break
                    
                except Exception as e:
                    print(f"❌ {metric.name} attempt {attempt + 1} failed: {e}")
                    
                    # Restore original model if we used fallback
                    if attempt == 1 and self.fallback_evaluator:
                        metric.model = original_model
                    
                    if attempt == 1:  # Last attempt failed
                        results[metric.name] = {
                            "score": 0,
                            "reason": f"Evaluation error after {attempt + 1} attempts: {str(e)}",
                            "evaluation_time": 0,
                            "evaluator_used": "failed",
                            "attempt": attempt + 1
                        }
            
            if not metric_success:
                print(f"⚠️ {metric.name} failed completely, using fallback score")
        
        return results
    
    def _calculate_composite_score(self, results: Dict) -> float:
        """Calculate weighted composite score from all metrics"""
        weights = {
            "Agent Collaboration Quality": 0.25,
            "Agent Role Adherence": 0.20,
            "Agent Response Quality": 0.25,
            "Meeting Progress Assessment": 0.30
        }
        
        total_score = 0
        total_weight = 0
        
        for metric_name, weight in weights.items():
            if metric_name in results and results[metric_name]["score"] > 0:
                total_score += results[metric_name]["score"] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _evaluate_meeting_holistically(self, transcript_data: Dict) -> Dict[str, Any]:
        """Perform holistic evaluation of the entire meeting"""
        print("🔬 Performing holistic meeting evaluation...")
        
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
            self.holistic_metric.measure(holistic_test_case)
            return {
                "overall_score": self.holistic_metric.score,
                "detailed_analysis": self.holistic_metric.reason,
                "evaluation_success": True
            }
        except Exception as e:
            print(f"❌ Holistic evaluation failed: {e}")
            return {
                "overall_score": 0,
                "detailed_analysis": f"Evaluation failed: {str(e)}",
                "evaluation_success": False
            }
    
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
    
    def shutdown(self):
        """Cleanup resources"""
        print("🔄 Shutting down evaluation manager...")
        self.executor.shutdown(wait=True)
        print(f"✅ Evaluation complete. Total evaluations: {len(self.evaluation_results)}")