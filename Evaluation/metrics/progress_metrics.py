"""
Progress-focused evaluation metrics
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def create_progress_metrics(evaluator):
    """Create progress-related metrics"""
    
    meeting_progress_metric = GEval(
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
        model=evaluator,
        threshold=0.65,
        strict_mode=False
    )
    
    holistic_meeting_metric = GEval(
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
        model=evaluator,
        threshold=0.6,
        strict_mode=False
    )
    
    return {
        "meeting_progress": meeting_progress_metric,
        "holistic_meeting": holistic_meeting_metric
    }