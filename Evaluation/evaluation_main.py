import json
from evaluation import Qwen3MeetingEvaluationManager
from pathlib import Path

DB = Path('/Users/praneetsurabhi/Desktop/projects/Tao/ThinkTank/projects_db.json')

def call_manager(transcript, experts, project_name, meeting_topic, summary):
    # Initialize evaluation manager
    eval_manager = Qwen3MeetingEvaluationManager(
        meeting_topic=meeting_topic,
        experts=experts,
        project_name=project_name
    )

    transcript_data = {
        "meeting_topic": meeting_topic, 
        "project_name": project_name, 
        "experts": experts, 
        "transcript": transcript, 
        "summary": summary, 
        "timestamp": ""
    }
    
    # Evaluate the transcript
    evaluation_report = eval_manager.evaluate_transcript(transcript_data)
    
    # Cleanup
    eval_manager.shutdown()
    
    return evaluation_report

def evaluate(project_name, meeting_topic):
    """Example of how to evaluate a transcript"""
    
    projects = json.loads(DB.read_text())

    if project_name not in projects:
        print(f'Project "{project_name}" not found')
        return
    
    meetings = projects[project_name]['meetings']
    
    # Initialize variables
    transcript = None
    experts = None
    summary = None
    
    for meeting in meetings:
        if meeting['meeting_topic'] == meeting_topic:
            transcript = meeting['transcript']
            experts = meeting['experts']
            summary = meeting['summary']
            break

    if transcript is None:
        print(f'No meeting found with topic: "{meeting_topic}"')
        return
    
    # Filter transcript to only include entries with round information
    transcript = [x for x in transcript if x.get('round') is not None]

    print(f"Found transcript with {len(transcript)} entries")
    
    # Evaluate the transcript
    print("🚀 Starting transcript evaluation...")
    evaluation_report = call_manager(transcript, experts, project_name, meeting_topic, summary)
    
    # Display results
    print("\n📊 Evaluation Results:")
    if 'overall_assessment' in evaluation_report:
        print(f"Overall Meeting Quality: {evaluation_report['overall_assessment']['average_score']:.2f}/1.0")
        print(f"Quality Rating: {evaluation_report['overall_assessment']['quality_rating']}")
    else:
        print("No overall assessment available")
    
    # Agent performance summary
    print("\n👥 Agent Performance:")
    if 'agent_performance' in evaluation_report:
        for agent_name, report in evaluation_report['agent_performance'].items():
            if "error" not in report and isinstance(report, dict):
                total_responses = report.get('total_responses', 0)
                overall_average = report.get('overall_average', 0)
                print(f"  {agent_name}: {overall_average:.2f}/1.0 ({total_responses} responses)")
            else:
                print(f"  {agent_name}: Error in evaluation")
    else:
        print("No agent performance data available")
    
    return evaluation_report

if __name__ == "__main__":
    try:
        # Run example evaluation
        report = evaluate('Sustainable Urban Energy Transition Initiative', 'Developing a 10-Year Roadmap for Equitable Renewable Energy Transition in Metropolitan Areas')
        
        if report:
            with open("meeting_evaluation_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            print("\n✅ Detailed report saved to 'meeting_evaluation_report.json'")
        else:
            print("\n❌ No evaluation report generated")
            
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()