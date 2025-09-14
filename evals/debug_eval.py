#!/usr/bin/env python3
"""
Debug script to identify and fix evaluation issues
Run this to diagnose problems with individual evaluation scripts
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import traceback

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'HuggingFace Datasets'),
        ('vllm', 'vLLM'),
        ('flask', 'Flask'),
        ('requests', 'Requests'),
        ('tqdm', 'TQDM'),
        ('numpy', 'NumPy'),
        ('bert_score', 'BERTScore'),
        ('sacrebleu', 'SacreBLEU'),
        ('rouge_score', 'ROUGE Score'),
        ('spacy', 'spaCy')
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True

def check_file_structure():
    """Check if all required files and directories exist"""
    print("\nChecking file structure...")
    
    required_files = [
        "run_all_evals.py",
        "evals/ragtruth_hf_eval.py",
        "evals/squad_v2_official_eval.py", 
        "evals/hotpot_eval.py",
        "evals/ms_marco_eval.py",
        "evals/natural_questions_official_eval.py",
        "evals/triviaqa_eval.py",
        "evals/eval_utils.py",
        "evals/utils/dataset_utils.py",
        "evals/utils/utils.py"
    ]
    
    required_dirs = [
        "evals",
        "evals/utils",
        "outputs",
        "utils"
    ]
    
    missing = []
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"✗ Directory missing: {dir_path}")
            missing.append(dir_path)
        else:
            print(f"✓ Directory: {dir_path}")
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"✗ File missing: {file_path}")
            missing.append(file_path)
        else:
            print(f"✓ File: {file_path}")
    
    return len(missing) == 0

def test_individual_script(script_path, test_args=None):
    """Test an individual evaluation script"""
    print(f"\nTesting script: {script_path}")
    
    if not Path(script_path).exists():
        print(f"✗ Script not found: {script_path}")
        return False
    
    try:
        # Test import
        script_dir = Path(script_path).parent
        script_name = Path(script_path).stem
        
        sys.path.insert(0, str(script_dir))
        
        try:
            module = __import__(script_name)
            print(f"✓ Script imports successfully")
        except Exception as e:
            print(f"✗ Import error: {e}")
            traceback.print_exc()
            return False
        
        # Try to run with --help if no test args provided
        if not test_args:
            try:
                result = subprocess.run([sys.executable, script_path, "--help"], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0 or "usage:" in result.stdout.lower():
                    print("✓ Script shows help correctly")
                else:
                    print(f"⚠ Script help may have issues: {result.stderr}")
            except Exception as e:
                print(f"⚠ Could not test help: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing script: {e}")
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test loading small samples from each dataset"""
    print("\nTesting dataset loading...")
    
    try:
        from datasets import load_dataset
        
        datasets_to_test = [
            ("rajpurkar/squad_v2", None, "validation", "SQuAD v2"),
            ("hotpotqa/hotpot_qa", "distractor", "validation", "HotPotQA"),
            ("microsoft/ms_marco", "v2.1", "test", "MS MARCO"),
            ("google-research-datasets/natural_questions", "default", "validation", "Natural Questions"),
            ("mandarjoshi/trivia_qa", "rc", "test", "TriviaQA")
        ]
        
        for dataset_name, config, split, display_name in datasets_to_test:
            try:
                print(f"Testing {display_name}...")
                if config:
                    ds = load_dataset(dataset_name, config, split=split, streaming=True)
                else:
                    ds = load_dataset(dataset_name, split=split, streaming=True)
                
                # Try to get first example
                first_example = next(iter(ds.take(1)))
                print(f"✓ {display_name} - columns: {list(first_example.keys())}")
                
            except Exception as e:
                print(f"✗ {display_name} failed: {e}")
        
    except ImportError:
        print("✗ datasets library not available")

def create_minimal_test_files():
    """Create minimal test files for debugging"""
    print("\nCreating test files...")
    
    # Create minimal SQuAD test data
    squad_data = {
        "version": "v2.0",
        "data": [{
            "title": "Test",
            "paragraphs": [{
                "context": "This is a test context.",
                "qas": [{
                    "id": "test_q1",
                    "question": "What is this?",
                    "answers": [{"text": "test", "answer_start": 10}]
                }]
            }]
        }]
    }
    
    os.makedirs("test_data", exist_ok=True)
    
    with open("test_data/squad_test.json", "w") as f:
        json.dump(squad_data, f)
    
    # Create minimal predictions
    squad_preds = {"test_q1": "test"}
    with open("test_data/squad_preds.json", "w") as f:
        json.dump(squad_preds, f)
    
    print("✓ Created test_data/squad_test.json")
    print("✓ Created test_data/squad_preds.json")

def test_squad_script():
    """Test SQuAD evaluation script with minimal data"""
    print("\nTesting SQuAD evaluation...")
    
    create_minimal_test_files()
    
    try:
        result = subprocess.run([
            sys.executable, 
            "evals/squad_v2_official_eval.py",
            "test_data/squad_test.json",
            "test_data/squad_preds.json"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ SQuAD script runs successfully")
            print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print("✗ SQuAD script failed")
            print("Error:", result.stderr)
        
    except Exception as e:
        print(f"✗ Error testing SQuAD script: {e}")

def main():
    print("=== Self-RAG Evaluation Debug Tool ===\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check file structure  
    files_ok = check_file_structure()
    
    # Test dataset loading
    test_dataset_loading()
    
    # Test individual scripts
    scripts_to_test = [
        "evals/squad_v2_official_eval.py",
        "evals/ragtruth_hf_eval.py", 
        "evals/hotpot_eval.py",
        "evals/ms_marco_eval.py",
        "evals/triviaqa_eval.py"
    ]
    
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL SCRIPTS")
    print("="*50)
    
    for script in scripts_to_test:
        test_individual_script(script)
    
    # Test SQuAD with minimal data
    test_squad_script()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if deps_ok and files_ok:
        print("✓ Basic setup appears correct")
        print("Try running with minimal examples:")
        print("  python run_all_evals.py --max-examples 2 --run squad")
    else:
        print("✗ Setup issues found - fix the above errors first")
    
    print("\nFor detailed debugging:")
    print("1. Check individual script outputs in outputs/<task>/*_stderr.txt")
    print("2. Run scripts directly: python evals/squad_v2_official_eval.py --help")
    print("3. Check model loading with: python -c 'from run_all_evals import load_model_and_tokenizer; load_model_and_tokenizer()'")

if __name__ == "__main__":
    main()
