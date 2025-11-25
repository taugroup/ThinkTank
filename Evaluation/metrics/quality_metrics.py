"""
Quality-focused evaluation metrics
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def create_quality_metrics(evaluator):
    """Create quality-related metrics"""
    
    response_quality_metric = GEval(
        name="Agent Response Quality",
        criteria="""
        Evaluate the quality of the agent's response in the meeting context:
        1. Expertise Demonstration: Does the response show domain knowledge and expertise?
        2. Context Awareness: Does the agent build upon previous discussion points?
        3. Information Value: Does the response add new, actionable insights?
        4. Clarity and Structure: Is the response well-organized and easy to follow?
        5. Actionability: Does the response provide practical, implementable information?
        
        Provide a score from 0-100 reflecting the overall response quality.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=evaluator,
        threshold=0.7,
        strict_mode=False
    )
    
    return {
        "response_quality": response_quality_metric
    }