#!/usr/bin/env python3

"""
Setup checker for meeting evaluation system
"""

import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking Environment Setup")
    print("=" * 40)
    
    issues = []
    warnings = []
    
    # Load environment variables
    load_dotenv()
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        issues.append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
    else:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required packages
    required_packages = [
        "deepeval",
        "google.genai",
        "groq",
        "requests",
        "pydantic",
        "python-dotenv"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ Package: {package}")
        except ImportError:
            issues.append(f"Missing package: {package}")
    
    # Check API keys
    api_keys = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY")
    }
    
    available_keys = []
    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"✅ API Key: {key_name} (found)")
            available_keys.append(key_name)
        else:
            print(f"⚠️ API Key: {key_name} (not found)")
    
    if not available_keys:
        issues.append("No API keys found. Set at least one: GEMINI_API_KEY, GOOGLE_API_KEY, or GROQ_API_KEY")
    elif len(available_keys) == 1:
        warnings.append("Only one API key found. Consider adding a backup evaluator.")
    
    # Check file permissions
    try:
        with open("test_permissions.tmp", "w") as f:
            f.write("test")
        os.remove("test_permissions.tmp")
        print("✅ File permissions: OK")
    except Exception as e:
        issues.append(f"File permission issue: {e}")
    
    # Summary
    print("\n" + "=" * 40)
    if issues:
        print("❌ Setup Issues Found:")
        for issue in issues:
            print(f"  • {issue}")
    
    if warnings:
        print("\n⚠️ Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
    
    if not issues:
        print("✅ Environment setup looks good!")
        return True
    else:
        print(f"\n💡 Fix {len(issues)} issue(s) before running evaluations.")
        return False

def show_setup_instructions():
    """Show setup instructions"""
    print("\n📋 Setup Instructions:")
    print("=" * 40)
    
    print("\n1. Install required packages:")
    print("   pip install deepeval google-genai groq requests pydantic python-dotenv")
    
    print("\n2. Set up API keys in .env file:")
    print("   GEMINI_API_KEY=your_gemini_key_here")
    print("   GROQ_API_KEY=your_groq_key_here")
    
    print("\n3. Get API keys:")
    print("   • Gemini: https://aistudio.google.com/app/apikey")
    print("   • Groq: https://console.groq.com/keys")
    
    print("\n4. Test the setup:")
    print("   python check_setup.py")
    print("   python test_evaluation.py")

if __name__ == "__main__":
    success = check_environment()
    
    if not success:
        show_setup_instructions()
    else:
        print("\n🚀 Ready to run evaluations!")
        print("Next: python test_evaluation.py")