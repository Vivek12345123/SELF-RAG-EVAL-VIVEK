# run_crag_eval.py
import argparse
import pprint
from dataset_utils import load_from_cache_or_hf
from crag.metrics import calculate_metrics

def parse_args():
    p = argparse.ArgumentParser(description="Run CRAG calculate_metrics on a dataset")
    p.add_argument("--dataset-key", type=str, required=True,
                   help="Either the on-disk dataset key under ./datasets_cache/ or a HF dataset id (e.g., mwong/fever-evidence-related)")
    p.add_argument("--split", type=str, default=None, help="Optional split name (train/validation/test)")
    p.add_argument("--pred-adherence", type=str, default=None, help="Column name containing predicted adherence scores/probabilities")
    p.add_argument("--pred-context-relevance", type=str, default=None, help="Column name containing predicted context relevance (float)")
    p.add_argument("--pred-context-utilization", type=str, default=None, help="Column name containing predicted context utilization (float)")
    p.add_argument("--gt-adherence", type=str, default=None, help="Ground-truth adherence column (defaults to RAGBenchFields.SUPPORTED if available)")
    p.add_argument("--gt-relevance", type=str, default=None, help="Ground-truth relevance column")
    p.add_argument("--gt-utilization", type=str, default=None, help="Ground-truth utilization column")
    return p.parse_args()

def main():
    args = parse_args()
    ds = load_from_cache_or_hf(args.dataset_key, split=args.split)
    # Pass column names through to the evaluator. If None, evaluator will use RAGBenchFields defaults.
    metrics = calculate_metrics(
        annotated_dataset=ds,
        pred_adherence=args.pred_adherence,
        pred_context_relevance=args.pred_context_relevance,
        pred_context_utilization=args.pred_context_utilization,
        ground_truth_adherence=args.gt_adherence,
        ground_truth_context_relevance=args.gt_relevance,
        ground_truth_context_utilization=args.gt_utilization,
    )
    print("=== CRAG metrics ===")
    pprint.pprint(metrics)

if __name__ == "__main__":
    main()
