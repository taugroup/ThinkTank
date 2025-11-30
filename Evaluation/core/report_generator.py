"""
Report generation utilities for meeting evaluations
"""

import time
from typing import Dict, List, Any


class ReportGenerator:
    """
    Generates comprehensive evaluation reports
    """
    
    def __init__(
        self,
        meeting_topic: str,
        project_name: str,
        conversation_history: List[Dict],
        evaluation_results: List[Dict],
        agent_performance_tracker: Dict,
        start_time: float
    ):
        self.meeting_topic = meeting_topic
        self.project_name = project_name
        self.conversation_history = conversation_history
        self.evaluation_results = evaluation_results
        self.agent_performance_tracker = agent_performance_tracker
        self.start_time = start_time
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive meeting summary report"""
        try:
            # Generate agent reports
            agent_reports = {}
            for agent_name in self.agent_performance_tracker.keys():
                agent_reports[agent_name] = self._get_agent_detailed_report(agent_name)
            
            # Calculate basic insights
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
                    "participants": list(self.agent_performance_tracker.keys()),
                    "meeting_stages_covered": self._get_meeting_stages_covered()
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
                    "total_evaluation_time": sum(r.get("evaluation_duration", 0) for r in self.evaluation_results),
                    "average_evaluation_time": self._calculate_average_evaluation_time(),
                    "evaluation_success_rate": self._calculate_evaluation_success_rate(),
                    "metrics_used": ["Agent Collaboration Quality", "Agent Role Adherence", "Agent Response Quality", "Meeting Progress Assessment"]
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
        """Calculate basic insights from evaluation results"""
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
                "avg_evaluation_time": self._calculate_average_evaluation_time(),
                "total_evaluation_time": sum(r.get("evaluation_duration", 0) for r in self.evaluation_results)
            }
        }
    
    def _get_agent_detailed_report(self, agent_name: str) -> Dict[str, Any]:
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
                Over {duration} minutes, {len(self.agent_performance_tracker)} participants contributed {total_responses} responses across multiple discussion rounds.

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
        expected_per_agent = total_responses / len(self.agent_performance_tracker)
        
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
    
    def _calculate_average_evaluation_time(self) -> float:
        """Calculate average evaluation time"""
        if not self.evaluation_results:
            return 0.0
        
        total_time = sum(r.get("evaluation_duration", 0) for r in self.evaluation_results)
        return total_time / len(self.evaluation_results)
    
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
    
    def _get_meeting_stages_covered(self) -> List[str]:
        """Get list of meeting stages covered"""
        stages = set()
        for entry in self.conversation_history:
            stage = self._determine_meeting_stage(entry["round"])
            stages.add(stage)
        return list(stages)
    
    def _determine_meeting_stage(self, round_num: int) -> str:
        """Determine the current stage of the meeting"""
        total_rounds = max([entry["round"] for entry in self.conversation_history], default=[1])
        
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