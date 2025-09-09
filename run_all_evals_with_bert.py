import json
import os
import sys
import subprocess
from pathlib import Path
import logging

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "evals"))

# Import the original runner
from run_all_evals import *

def run_with_bert_scoring():
    """Run all evaluations with BERT scoring enabled"""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run Self-RAG evaluations with BERT scoring")
    parser.add_argument("--max-examples", type=int, default=200, help="Max examples per dataset")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max generation tokens")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated task names or 'all'")
    parser.add_argument("--output-dir", type=str, default="outputs_with_bert", help="Output directory")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep intermediate files")
    args = parser.parse_args()
    
    # Set up environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Update globals
    global MAX_SAMPLES, OUT_ROOT, KEEP_INTERMEDIATES
    MAX_SAMPLES = args.max_examples
    OUT_ROOT = Path(args.output_dir)
    KEEP_INTERMEDIATES = args.keep_intermediates
    
    # Create output directory
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUT_ROOT / "evaluation.log")
        ]
    )
    
    logging.info("="*60)
    logging.info("Starting Self-RAG Evaluation with BERT Scoring")
    logging.info("="*60)
    logging.info(f"Max examples: {args.max_examples}")
    logging.info(f"Max tokens: {args.max_tokens}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Output directory: {OUT_ROOT}")
    
    # Load model
    logging.info("Loading Self-RAG 7B model...")
    model, tokenizer, sampling_params, format_prompt = load_model_and_tokenizer()
    global GLOBAL_TOKENIZER
    GLOBAL_TOKENIZER = tokenizer
    
    # Determine tasks to run
    tasks = args.tasks.split(",") if args.tasks != "all" else ["ragtruth", "squad", "hotpot", "msmarco", "nq", "trivia"]
    
    # Run each task
    all_results = {}
    
    for task in tasks:
        logging.info(f"\n{'='*40}")
        logging.info(f"Running {task.upper()}")
        logging.info(f"{'='*40}")
        
        try:
            if task == "ragtruth":
                metrics_path = run_ragtruth_adapter_and_eval(
                    "127.0.0.1", 8300, "wandbRAGTruth-processed", None, "test",
                    str(OUT_ROOT / "ragtruth_preds.jsonl"), "",
                    model, sampling_params, KEEP_INTERMEDIATES
                )
            elif task == "squad":
                metrics_path = run_squad_v2(
                    model, sampling_params, format_prompt,
                    "rajpurkar/squad_v2", "validation",
                    OUT_ROOT / "squad_v2", args.batch_size, args.max_tokens,
                    0.0, 1.0, KEEP_INTERMEDIATES
                )
            elif task == "hotpot":
                metrics_path = run_hotpot(
                    model, sampling_params, format_prompt,
                    "hotpotqa/hotpot_qa", "validation",
                    OUT_ROOT / "hotpot", args.batch_size, args.max_tokens,
                    0.0, 1.0, KEEP_INTERMEDIATES
                )
            elif task == "msmarco":
                metrics_path = run_ms_marco(
                    model, sampling_params, format_prompt,
                    "microsoft/ms_marco", "v2.1", "test",
                    OUT_ROOT / "ms_marco", args.batch_size, args.max_tokens,
                    0.0, 1.0, KEEP_INTERMEDIATES
                )
            elif task == "nq":
                metrics_path = run_natural_questions(
                    OUT_ROOT / "nq", None, "validation", None, True, KEEP_INTERMEDIATES
                )
            elif task == "trivia":
                metrics_path = run_triviaqa(
                    model, sampling_params, format_prompt,
                    "mandarjoshi/trivia_qa", "test",
                    OUT_ROOT / "trivia", args.batch_size, args.max_tokens,
                    0.0, 1.0, None, KEEP_INTERMEDIATES
                )
            
            # Load and store results
            if metrics_path and os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                all_results[task] = metrics
                logging.info(f"✓ {task} completed successfully")
                
                # Log key metrics
                for key in ['exact_match', 'f1', 'accuracy', 'bert_f1', 'bert_precision', 'bert_recall']:
                    if key in metrics:
                        logging.info(f"  {key}: {metrics[key]:.4f}")
            else:
                logging.warning(f"✗ {task} failed to produce metrics")
                all_results[task] = {"error": "No metrics file produced"}
                
        except Exception as e:
            logging.error(f"✗ {task} failed with error: {e}")
            all_results[task] = {"error": str(e)}
    
    # Save aggregated results
    final_results_path = OUT_ROOT / "all_results_with_bert.json"
    with open(final_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE - SUMMARY")
    print("="*60)
    
    for task, metrics in all_results.items():
        print(f"\n{task.upper()}:")
        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
        else:
            # Print regular metrics
            reg_metrics = ['exact_match', 'f1', 'accuracy', 'precision', 'recall']
            bert_metrics = ['bert_f1', 'bert_precision', 'bert_recall', 
                          'bertscore_f1', 'codebert_f1', 'bert_exact_f1']
            
            print("  Regular Metrics:")
            for m in reg_metrics:
                if m in metrics:
                    print(f"    {m}: {metrics[m]:.4f}")
            
            print("  BERT Metrics:")
            found_bert = False
            for m in bert_metrics:
                if m in metrics:
                    print(f"    {m}: {metrics[m]:.4f}")
                    found_bert = True
            if not found_bert:
                print("    (No BERT scores available)")
    
    print(f"\nFull results saved to: {final_results_path}")
    print(f"Logs saved to: {OUT_ROOT / 'evaluation.log'}")

if __name__ == "__main__":
    run_with_bert_scoring()
