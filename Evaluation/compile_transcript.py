"""
Compile individual response JSON files from Alzheimers_2 folder into a complete transcript
"""

import json
import os
from pathlib import Path
from datetime import datetime
import re
import argparse
import sys


def extract_round_number(filename: str) -> int:
    """Extract round number from filename"""
    # Pattern: Agent_X_response where X is the round number
    match = re.search(r'_(\d+)_response', filename)
    if match:
        return int(match.group(1))
    
    # For opening, synthesis, critique, and final_summary
    if 'opening' in filename:
        return 0
    elif 'final_summary' in filename:
        return 999  # Put at the end
    elif 'synthesis' in filename or 'critique' in filename:
        # Extract timestamp and use it for ordering
        match = re.search(r'_(\d{8})_(\d{6})', filename)
        if match:
            timestamp = int(match.group(1) + match.group(2))
            return timestamp
    
    return 0


def determine_response_type(filename: str, content: str) -> str:
    """Determine the type of response based on filename and content"""
    filename_lower = filename.lower()
    
    if 'opening' in filename_lower:
        return 'opening'
    elif 'synthesis' in filename_lower:
        return 'synthesis'
    elif 'critique' in filename_lower:
        return 'critique'
    elif 'final_summary' in filename_lower:
        return 'final_summary'
    else:
        return 'expert'


def extract_agent_name(filename: str) -> str:
    """Extract agent name from filename"""
    # Remove the round number and timestamp
    name = re.sub(r'_\d+_response_\d{8}_\d{6}\.json$', '', filename)
    name = re.sub(r'_(opening|synthesis|critique|final_summary)_\d{8}_\d{6}\.json$', '', name)
    return name


def compile_transcript_from_folder(folder_path: str, output_path: str = None):
    """
    Compile individual JSON response files into a complete transcript
    
    Args:
        folder_path: Path to folder containing individual response JSON files
        output_path: Optional path to save the compiled transcript
    
    Returns:
        Dict containing the compiled transcript data
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Get all JSON files
    json_files = sorted(folder.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {folder_path}")
    
    print(f"📁 Found {len(json_files)} response files")
    
    # Parse all responses
    responses = []
    agent_info = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            agent_name = extract_agent_name(json_file.name)
            
            # Handle both string and dict formats
            if isinstance(data, str):
                content = data
                data = {}  # Convert to empty dict for further processing
            else:
                content = data.get('content', '')
            
            # Extract agent role from system message if available
            if 'messages' in data and len(data['messages']) > 0:
                system_msg = data['messages'][0]
                if system_msg.get('role') == 'system':
                    role_content = system_msg.get('content', '')
                    
                    # Parse role information
                    if agent_name not in agent_info:
                        expertise = ''
                        goal = ''
                        
                        # Extract expertise
                        if 'Expertise:' in role_content:
                            expertise = role_content.split('Expertise:')[1].split('\n')[0].strip()
                        
                        # Extract goal
                        if 'Goal:' in role_content:
                            goal = role_content.split('Goal:')[1].split('\n')[0].strip()
                        elif '<your_role>' in role_content:
                            role_match = re.search(r'<your_role>(.*?)</your_role>', role_content, re.DOTALL)
                            if role_match:
                                goal = role_match.group(1).strip()
                        
                        agent_info[agent_name] = {
                            'title': agent_name,
                            'expertise': expertise if expertise else 'Domain expert',
                            'role': agent_name,
                            'goal': goal if goal else 'Contribute domain expertise'
                        }
            
            round_num = extract_round_number(json_file.name)
            response_type = determine_response_type(json_file.name, content)
            
            responses.append({
                'filename': json_file.name,
                'agent': agent_name,
                'content': content,
                'round': round_num,
                'type': response_type,
                'timestamp': json_file.stat().st_mtime
            })
            
        except Exception as e:
            print(f"⚠️ Error processing {json_file.name}: {e}")
            continue
    
    # Sort responses by timestamp (which corresponds to actual conversation order)
    responses.sort(key=lambda x: x['timestamp'])
    
    # Assign proper round numbers based on sorted order
    current_round = 1
    last_type = None
    
    for i, response in enumerate(responses):
        if response['type'] == 'opening':
            response['round'] = 0
        elif response['type'] == 'final_summary':
            response['round'] = current_round
        else:
            # Increment round when we see a synthesis or critique (marks end of round)
            if last_type in ['synthesis', 'critique'] and response['type'] == 'expert':
                current_round += 1
            response['round'] = current_round
            last_type = response['type']
    
    # Create transcript entries
    transcript = []
    for response in responses:
        transcript.append({
            'name': response['agent'],
            'content': response['content'],
            'round': response['round'],
            'type': response['type']
        })
    
    # Extract meeting topic and project name from content
    meeting_topic = "Advancing Protein Function & Therapeutics Using AlphaFold"
    project_name = folder.name
    
    # Try to extract from first coordinator message
    if responses and 'AlphaFold' in responses[0]['content']:
        meeting_topic = "Advancing Protein Function & Therapeutics Using AlphaFold"
    
    # Create experts list from agent_info
    experts = list(agent_info.values())
    
    # Generate summary
    summary = f"Multi-agent expert meeting on {meeting_topic} with {len(experts)} participants discussing protein structure prediction, therapeutic applications, and validation strategies."
    
    # Compile final transcript data
    transcript_data = {
        'meeting_topic': meeting_topic,
        'project_name': project_name,
        'experts': experts,
        'transcript': transcript,
        'summary': summary,
        'timestamp': int(datetime.now().timestamp()),
        'metadata': {
            'compiled_from': folder_path,
            'total_responses': len(transcript),
            'total_agents': len(experts),
            'compilation_date': datetime.now().isoformat()
        }
    }
    
    # Save to file if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(transcript_data, f, indent=2)
        
        print(f"✅ Compiled transcript saved to: {output_file}")
    
    print(f"\n📊 Transcript Summary:")
    print(f"   - Meeting Topic: {meeting_topic}")
    print(f"   - Project: {project_name}")
    print(f"   - Total Agents: {len(experts)}")
    print(f"   - Total Responses: {len(transcript)}")
    print(f"   - Agents: {', '.join([e['title'] for e in experts])}")
    
    return transcript_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile individual JSON response files into a complete transcript")
    parser.add_argument("folder_path", help="Path to folder containing individual response JSON files")
    parser.add_argument("-o", "--output", help="Output path for the compiled transcript (default: folder_name.json in current directory)")
    
    args = parser.parse_args()
    
    folder_path = args.folder_path
    
    # Generate output path if not provided
    if args.output:
        output_path = args.output
    else:
        folder_name = Path(folder_path).name
        output_path = f"{folder_name}.json"
    
    print(f"🔄 Compiling transcript from: {folder_path}")
    print(f"📝 Output will be saved to: {output_path}\n")
    
    try:
        transcript_data = compile_transcript_from_folder(folder_path, output_path)
        print("\n✅ Transcript compilation complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
