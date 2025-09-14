#!/usr/bin/env python3
"""
Fixed HotpotQA evaluation script with proper BERTScore implementation
"""

import sys
import json
import re
import string
from collections import Counter
import numpy as np

# Proper BERTScore import
try:
    from bert_score import BERTScorer
    BERT_SCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert-score not installed. Install with: pip install bert-score", file=sys.stderr)
    BERT_SCORE_AVAILABLE = False


def normalize_answer(s):
    """Normalize answer text"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth"""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    """Check if prediction exactly matches ground truth"""
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    """Update metrics with answer scores"""
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def update_sp(metrics, prediction, gold):
    """Update metrics with supporting facts scores"""
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def compute_bert_score(predictions, references, device='cuda', batch_size=64):
    """
    Compute BERTScore using roberta-large model
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for processing
    
    Returns:
        Tuple of (mean_precision, mean_recall, mean_f1, mean_f3)
    """
    if not BERT_SCORE_AVAILABLE:
        print("BERTScore not available, returning zeros", file=sys.stderr)
        return 0.0, 0.0, 0.0, 0.0
    
    try:
        import torch
        # Auto-detect device if needed
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("CUDA not available, using CPU for BERTScore", file=sys.stderr)
        
        # Initialize BERTScorer with roberta-large
        scorer = BERTScorer(
            model_type="roberta-large",
            lang="en",
            idf=False,
            device=device,
            batch_size=batch_size,
            use_fast_tokenizer=True,
            rescale_with_baseline=False
        )
        
        # Compute scores
        P, R, F = scorer.score(predictions, references, verbose=False)
        
        # Calculate means
        mean_p = float(P.mean().item())
        mean_r = float(R.mean().item())
        mean_f1 = float(F.mean().item())
        
        # Calculate F3 score (beta=3, emphasizes recall)
        beta = 3.0
        beta_squared = beta * beta
        mean_f3 = (1.0 + beta_squared) * mean_p * mean_r / (beta_squared * mean_p + mean_r) if (beta_squared * mean_p + mean_r) > 0 else 0.0
        
        return mean_p, mean_r, mean_f1, mean_f3
        
    except Exception as e:
        print(f"Error computing BERTScore: {e}", file=sys.stderr)
        return 0.0, 0.0, 0.0, 0.0


def eval(prediction_file, gold_file, device='cuda'):
    """
    Main evaluation function for HotpotQA
    
    Args:
        prediction_file: Path to predictions JSON file
        gold_file: Path to gold standard JSON file
        device: Device for BERTScore computation
    """
    # Load files
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    # Initialize metrics
    metrics = {
        'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0
    }

    # Lists for BERTScore
    bert_predictions = []
    bert_references = []

    # Process each example
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        
        # Process answer prediction
        if cur_id not in prediction['answer']:
            print(f'Missing answer for {cur_id}', file=sys.stderr)
            can_eval_joint = False
            pred_text = ""
        else:
            pred_text = prediction['answer'][cur_id]
        
        gold_text = dp['answer']
        
        # Collect for BERTScore
        bert_predictions.append(pred_text)
        bert_references.append(gold_text)

        # Update answer metrics
        if cur_id in prediction['answer']:
            em, prec, recall = update_answer(metrics, pred_text, gold_text)
        else:
            em, prec, recall = 0, 0, 0

        # Process supporting facts prediction
        if cur_id not in prediction['sp']:
            print(f'Missing supporting facts for {cur_id}', file=sys.stderr)
            can_eval_joint = False
            sp_em, sp_prec, sp_recall = 0, 0, 0
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, 
                prediction['sp'][cur_id], 
                dp['supporting_facts']
            )

        # Calculate joint metrics
        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    # Average all metrics
    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    # Compute BERTScore
    if bert_predictions and bert_references:
        bert_p, bert_r, bert_f1, bert_f3 = compute_bert_score(
            bert_predictions, 
            bert_references, 
            device=device
        )
        
        # Add BERTScore metrics
        metrics['bert_precision'] = bert_p
        metrics['bert_recall'] = bert_r
        metrics['bert_f1'] = bert_f1
        metrics['bert_f3'] = bert_f3
        
        print(f"BERTScore - P: {bert_p:.4f}, R: {bert_r:.4f}, F1: {bert_f1:.4f}, F3: {bert_f3:.4f}")
    else:
        print("No predictions/references for BERTScore computation")
        metrics['bert_precision'] = 0.0
        metrics['bert_recall'] = 0.0
        metrics['bert_f1'] = 0.0
        metrics['bert_f3'] = 0.0

    # Print final metrics
    print("\nFinal Metrics:")
    print(json.dumps(metrics, indent=2))
    
    return metrics


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python hotpot_eval.py <predictions.json> <gold.json> [device]")
        sys.exit(1)
    
    # Parse arguments
    pred_file = sys.argv[1]
    gold_file = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else 'cuda'
    
    # Run evaluation
    eval(pred_file, gold_file, device=device)
