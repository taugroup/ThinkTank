"""
Collaboration-focused evaluation metrics
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def create_collaboration_metrics(evaluator):
    """Create collaboration-related metrics"""
    
    collaboration_metric = GEval(
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
        model=evaluator,
        threshold=0.6,
        strict_mode=False
    )
    
    role_adherence_metric = GEval(
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
        model=evaluator,
        threshold=0.6,
        strict_mode=False
    )
    
    return {
        "collaboration": collaboration_metric,
        "role_adherence": role_adherence_metric
    }