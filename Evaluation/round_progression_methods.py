# Additional methods for round-by-round progression analysis
# Add these methods to the Qwen3MeetingEvaluationManager class

def _analyze_metrics_by_round(self) -> Dict[str, Any]:
    """Analyze how each metric progresses across rounds"""
    if not self.evaluation_results:
        return {"status": "No evaluation results available"}
    
    # Organize results by round
    rounds_data = {}
    
    for result in self.evaluation_results:
        round_num = result["round"]
        
        if round_num not in rounds_data:
            rounds_data[round_num] = {
                "round_number": round_num,
                "meeting_stage": self._determine_meeting_stage(round_num),
                "agents_participated": [],
                "metrics": {
                    "Agent Collaboration Quality": [],
                    "Agent Role Adherence": [],
                    "Agent Response Quality": [],
                    "Meeting Progress Assessment": []
                },
                "composite_scores": [],
                "response_count": 0
            }
        
        # Add agent to participants list
        if result["agent"] not in rounds_data[round_num]["agents_participated"]:
            rounds_data[round_num]["agents_participated"].append(result["agent"])
        
        # Collect metric scores
        for metric_name, metric_result in result.get("results", {}).items():
            if metric_name in rounds_data[round_num]["metrics"]:
                rounds_data[round_num]["metrics"][metric_name].append(metric_result["score"])
        
        # Collect composite score
        rounds_data[round_num]["composite_scores"].append(result["composite_score"])
        rounds_data[round_num]["response_count"] += 1
    
    # Calculate averages and trends for each round
    round_analysis = {}
    
    for round_num in sorted(rounds_data.keys()):
        round_data = rounds_data[round_num]
        
        # Calculate average for each metric
        metric_averages = {}
        for metric_name, scores in round_data["metrics"].items():
            if scores:
                metric_averages[metric_name] = {
                    "average": sum(scores) / len(scores),
                    "max": max(scores),
                    "min": min(scores),
                    "count": len(scores)
                }
        
        # Calculate overall round statistics
        composite_avg = sum(round_data["composite_scores"]) / len(round_data["composite_scores"]) if round_data["composite_scores"] else 0
        
        round_analysis[f"Round_{round_num}"] = {
            "round_number": round_num,
            "meeting_stage": round_data["meeting_stage"],
            "participants": round_data["agents_participated"],
            "total_responses": round_data["response_count"],
            "overall_quality": {
                "average_composite_score": composite_avg,
                "rating": self._get_quality_rating(composite_avg)
            },
            "metric_breakdown": metric_averages,
            "key_observations": self._generate_round_observations(round_num, metric_averages, composite_avg)
        }
    
    # Add progression analysis
    progression_summary = self._analyze_metric_progression(rounds_data)
    
    return {
        "rounds": round_analysis,
        "progression_summary": progression_summary,
        "total_rounds": len(rounds_data)
    }

def _get_quality_rating(self, score: float) -> str:
    """Convert numeric score to quality rating"""
    if score >= 0.85:
        return "excellent"
    elif score >= 0.70:
        return "good"
    elif score >= 0.55:
        return "satisfactory"
    elif score >= 0.40:
        return "needs_improvement"
    else:
        return "poor"

def _generate_round_observations(self, round_num: int, metrics: Dict, composite: float) -> List[str]:
    """Generate observations for a specific round"""
    observations = []
    
    # Check role adherence
    if "Agent Role Adherence" in metrics:
        role_score = metrics["Agent Role Adherence"]["average"]
        if role_score >= 0.8:
            observations.append(f"Strong role adherence maintained (Score: {role_score:.2f})")
        elif role_score < 0.6:
            observations.append(f"⚠️ Role adherence needs improvement (Score: {role_score:.2f})")
    
    # Check collaboration
    if "Agent Collaboration Quality" in metrics:
        collab_score = metrics["Agent Collaboration Quality"]["average"]
        if collab_score >= 0.8:
            observations.append(f"Excellent collaborative dynamics (Score: {collab_score:.2f})")
        elif collab_score < 0.6:
            observations.append(f"⚠️ Limited collaboration observed (Score: {collab_score:.2f})")
    
    # Check response quality
    if "Agent Response Quality" in metrics:
        quality_score = metrics["Agent Response Quality"]["average"]
        if quality_score >= 0.8:
            observations.append(f"High-quality responses delivered (Score: {quality_score:.2f})")
        elif quality_score < 0.6:
            observations.append(f"⚠️ Response quality could be enhanced (Score: {quality_score:.2f})")
    
    # Check meeting progress
    if "Meeting Progress Assessment" in metrics:
        progress_score = metrics["Meeting Progress Assessment"]["average"]
        if progress_score >= 0.8:
            observations.append(f"Strong progress toward objectives (Score: {progress_score:.2f})")
        elif progress_score < 0.6:
            observations.append(f"⚠️ Limited progress in this round (Score: {progress_score:.2f})")
    
    return observations

def _analyze_metric_progression(self, rounds_data: Dict) -> Dict[str, Any]:
    """Analyze how metrics progress across all rounds"""
    sorted_rounds = sorted(rounds_data.keys())
    
    if len(sorted_rounds) < 2:
        return {"status": "Insufficient rounds for progression analysis"}
    
    # Track each metric's progression
    metric_trends = {
        "Agent Collaboration Quality": [],
        "Agent Role Adherence": [],
        "Agent Response Quality": [],
        "Meeting Progress Assessment": []
    }
    
    composite_progression = []
    
    for round_num in sorted_rounds:
        round_data = rounds_data[round_num]
        
        # Collect average for each metric per round
        for metric_name in metric_trends.keys():
            scores = round_data["metrics"].get(metric_name, [])
            if scores:
                avg_score = sum(scores) / len(scores)
                metric_trends[metric_name].append(avg_score)
        
        # Composite score progression
        if round_data["composite_scores"]:
            composite_progression.append(sum(round_data["composite_scores"]) / len(round_data["composite_scores"]))
    
    # Analyze trends
    progression_analysis = {}
    
    for metric_name, scores in metric_trends.items():
        if len(scores) >= 2:
            trend = self._calculate_trend(scores)
            progression_analysis[metric_name] = {
                "scores_by_round": scores,
                "trend": trend,
                "starting_score": scores[0],
                "ending_score": scores[-1],
                "improvement": scores[-1] - scores[0],
                "average_across_rounds": sum(scores) / len(scores)
            }
    
    # Overall meeting progression
    if len(composite_progression) >= 2:
        overall_trend = self._calculate_trend(composite_progression)
        progression_analysis["overall_meeting_quality"] = {
            "scores_by_round": composite_progression,
            "trend": overall_trend,
            "starting_score": composite_progression[0],
            "ending_score": composite_progression[-1],
            "improvement": composite_progression[-1] - composite_progression[0]
        }
    
    return progression_analysis

def _calculate_trend(self, scores: List[float]) -> str:
    """Calculate trend direction from a series of scores"""
    if len(scores) < 2:
        return "stable"
    
    # Simple linear trend calculation
    first_half_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
    second_half_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
    
    diff = second_half_avg - first_half_avg
    
    if diff > 0.1:
        return "improving"
    elif diff < -0.1:
        return "declining"
    else:
        return "stable"

def _calculate_agent_trend(self, agent_name: str) -> str:
    """Calculate performance trend for a specific agent"""
    scores = self.agent_performance_tracker[agent_name]["scores_by_round"]
    if len(scores) < 2:
        return "insufficient_data"
    
    score_values = [scores[r] for r in sorted(scores.keys())]
    return self._calculate_trend(score_values)
