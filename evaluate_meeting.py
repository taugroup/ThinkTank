#!/usr/bin/env python3

"""
Command-line interface for meeting evaluation system
"""

import argparse
import json
import sys
from pathlib import Path

# Handle imports whether running from parent or Evaluation directory
try:
    from Evaluation import evaluate_meeting_transcript, EvaluationSystem, validate_api_keys
except ModuleNotFoundError:
    from main import evaluate_meeting_transcript, EvaluationSystem
    from utils import validate_api_keys


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate meeting transcripts using AI-powered analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_meeting.py transcript.json
  python evaluate_meeting.py transcript.json --output custom_report.json
  python evaluate_meeting.py --status
  python evaluate_meeting.py --sample
        """
    )
    
    parser.add_argument(
        "transcript_file",
        nargs="?",
        help="Path to the meeting transcript JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for the evaluation report (default: meeting_evaluation_report.json)"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status and available evaluators"
    )
    
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate a sample transcript file for testing"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Handle status command
    if args.status:
        show_status()
        return
    
    # Handle sample generation
    if args.sample:
        generate_sample_transcript()
        return
    
    # Validate transcript file argument
    if not args.transcript_file:
        parser.error("transcript_file is required unless using --status or --sample")
    
    # Check if file exists
    transcript_path = Path(args.transcript_file)
    if not transcript_path.exists():
        print(f"❌ Error: Transcript file not found: {args.transcript_file}")
        sys.exit(1)
    
    try:
        # Run evaluation
        if not args.quiet:
            print(f"🚀 Evaluating transcript: {args.transcript_file}")
        
        report = evaluate_meeting_transcript(transcript_file_path=str(transcript_path))
        
        # Save custom output if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            if not args.quiet:
                print(f"✅ Report saved to: {args.output}")
        
        # Display summary
        if not args.quiet:
            display_summary(report)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def show_status():
    """Show system status"""
    print("🔍 Meeting Evaluation System Status")
    print("=" * 40)
    
    system = EvaluationSystem()
    status = system.get_system_status()
    
    print(f"Version: {status['version']}")
    print(f"Available Evaluators: {', '.join(status['available_evaluators'])}")
    
    print("\n📊 Configuration:")
    config = status['configuration']
    print(f"  Max Workers: {config['max_workers']}")
    print(f"  Timeout per Evaluation: {config['timeout_per_evaluation']}s")
    print(f"  Retry Attempts: {config['retry_attempts']}")
    
    print("\n🔑 API Keys Status:")
    for evaluator, available in status['api_keys_status'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {evaluator.title()}: {'Available' if available else 'Not configured'}")
    
    if not any(status['api_keys_status'].values()):
        print("\n⚠️ No API keys configured. Please set:")
        print("  - GEMINI_API_KEY or GOOGLE_API_KEY for Gemini")
        print("  - GROQ_API_KEY for Groq")
        print("  - Or run Ollama locally for Qwen")


def generate_sample_transcript():
    """Generate a sample transcript file"""
    try:
        from Evaluation.main import create_sample_transcript
    except ModuleNotFoundError:
        from main import create_sample_transcript
    
    sample = create_sample_transcript()
    filename = "sample_transcript.json"
    
    with open(filename, 'w') as f:
        json.dump(sample, f, indent=2)
    
    print(f"✅ Sample transcript generated: {filename}")
    print("\nTo evaluate this sample:")
    print(f"  python evaluate_meeting.py {filename}")


def display_summary(report):
    """Display evaluation summary"""
    print("\n📊 Evaluation Results")
    print("=" * 40)
    
    overall = report['overall_assessment']
    print(f"Overall Quality: {overall['average_score']:.2f}/1.0")
    print(f"Quality Rating: {overall['quality_rating'].title()}")
    print(f"Meeting Effectiveness: {overall['meeting_effectiveness_score']:.2f}/1.0")
    
    print(f"\n👥 Agent Performance:")
    for agent, performance in report['agent_performance'].items():
        if 'error' not in performance:
            print(f"  {agent}: {performance['overall_average']:.2f}/1.0 ({performance['total_responses']} responses)")
    
    print(f"\n📈 Key Metrics:")
    metrics = report['detailed_metrics']
    for metric_name, metric_data in metrics.items():
        print(f"  {metric_name.replace('_', ' ').title()}: {metric_data['average']:.2f}/1.0")
    
    # Show recommendations if available
    if report.get('recommendations'):
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n⏱️ Evaluation completed in {report['meeting_metadata']['duration_minutes']:.1f} minutes")


if __name__ == "__main__":
    main()