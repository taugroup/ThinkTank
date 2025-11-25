"""
Run evaluation on the AI in Sustainable Agriculture transcript
"""

import json
from pathlib import Path
import sys
import os

# Add the Evaluation directory to path
evaluation_dir = Path(__file__).parent
sys.path.insert(0, str(evaluation_dir))

# Set PYTHONPATH to include the Evaluation directory
os.environ['PYTHONPATH'] = str(evaluation_dir) + ':' + os.environ.get('PYTHONPATH', '')

from core.evaluation_manager import EvaluationManager


def evaluate_agriculture_transcript():
    """Evaluate the AI in Sustainable Agriculture transcript"""
    
    # Load the compiled transcript
    transcript_file = Path(__file__).parent / "AI_in_Sustainable_AGriculture.json"
    
    if not transcript_file.exists():
        print(f"❌ Transcript file not found: {transcript_file}")
        return
    
    print("📂 Loading transcript...")
    with open(transcript_file, 'r') as f:
        transcript_data = json.load(f)
    
    # Extract required information
    meeting_topic = transcript_data['meeting_topic']
    experts = transcript_data['experts']
    project_name = transcript_data['project_name']
    
    print(f"\n📋 Meeting Details:")
    print(f"   Topic: {meeting_topic}")
    print(f"   Project: {project_name}")
    print(f"   Experts: {len(experts)}")
    print(f"   Total Responses: {len(transcript_data['transcript'])}")
    
    # Initialize evaluation manager
    print(f"\n🚀 Initializing evaluation manager...")
    
    try:
        eval_manager = EvaluationManager(
            meeting_topic=meeting_topic,
            experts=experts,
            project_name=project_name,
            max_workers=3
        )
        
        # Run evaluation
        print(f"\n🔬 Starting evaluation process...\n")
        evaluation_report = eval_manager.evaluate_transcript(transcript_data)
        
        # Save evaluation report
        output_file = Path(__file__).parent / f"agriculture_evaluation_report.json"
        with open(output_file, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        print(f"\n✅ Evaluation complete!")
        print(f"📄 Report saved to: {output_file}")
        
        # Print summary
        print(f"\n📊 Evaluation Summary:")
        if 'overall_metrics' in evaluation_report:
            metrics = evaluation_report['overall_metrics']
            print(f"   Average Composite Score: {metrics.get('average_composite_score', 'N/A'):.2f}")
            print(f"   Total Evaluations: {metrics.get('total_evaluations', 'N/A')}")
        
        if 'holistic_evaluation' in evaluation_report:
            holistic = evaluation_report['holistic_evaluation']
            print(f"   Overall Meeting Score: {holistic.get('overall_score', 'N/A'):.2f}")
        
        # Cleanup
        eval_manager.shutdown()
        
        return evaluation_report
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("   AI in Sustainable Agriculture Meeting Evaluation")
    print("=" * 70)
    print()
    
    evaluate_agriculture_transcript()
