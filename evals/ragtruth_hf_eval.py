#!/usr/bin/env python3
"""
COMPLETE FIXED RAGTruth evaluator with BERTScore and balanced metrics
Evaluates hallucination detection and text generation quality

Usage example:
python fixed_ragtruth_eval.py \
  --tgi-url http://localhost:8300 \
  --dataset-name microsoft/RAGTruth \
  --split test \
  --output-file outputs/ragtruth_preds.jsonl

Returns precision/recall/F1 for hallucination detection algorithms, text-based EM/F1, 
and BERTScore with balanced metrics computation.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import requests
from tqdm import tqdm
import numpy as np
from collections import Counter

YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# BERTScore availability - FIXED to match working scripts
try:
    import torch
    from bert_score.scorer import BERTScorer
    BERTSCORE_AVAILABLE = True
except Exception:
    BERTSCORE_AVAILABLE = False

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgi-url", type=str, default="http://localhost:8300")
    ap.add_argument("--dataset-name", type=str, required=True)
    ap.add_argument("--dataset-config", type=str, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--output-file", type=str, default="outputs/ragtruth_preds.jsonl")
    ap.add_argument("--system-prompt", type=str, default="")
    ap.add_argument("--input-column", type=str, default=None)
    ap.add_argument("--context-column", type=str, default=None)
    ap.add_argument("--label-column", type=str, default=None)
    ap.add_argument("--task-column", type=str, default=None)
    ap.add_argument("--positive-label-values", type=str, default="yes,true,1,hallucinated")
    ap.add_argument("--negative-label-values", type=str, default="no,false,0,not_hallucinated")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--ti-thresholds", type=str, default="0.25,0.5,0.75")
    return ap.parse_args()

def safe_load_dataset(dataset_name, dataset_config, split):
    from datasets import load_dataset
    candidates = [dataset_name, "microsoft/RAGTruth", "RAGTruth/ragtruth", "wandb/RAGTruth"]
    last_err = None
    for name in candidates:
        for cfg in (dataset_config, None):
            try:
                if cfg:
                    ds = load_dataset(name, cfg, split=split)
                else:
                    ds = load_dataset(name, split=split)
                print(f"Successfully loaded: {name}")
                return ds
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Could not load dataset variants. Last error: {last_err}")

def normalize_label(x: Any, pos_vals: set, neg_vals: set) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in pos_vals:
        return 1
    if s in neg_vals:
        return 0
    if s.isdigit():
        return 1 if int(s) != 0 else 0
    if s in {"true"}:
        return 1
    if s in {"false"}:
        return 0
    return None

def extract_yesno(text: str) -> Optional[int]:
    if not text:
        return None
    m = YESNO_RE.search(text)
    return 1 if m and m.group(1).lower() == "yes" else (0 if m else None)

def build_prompt(question: str, context: Optional[Any], system_prompt: str) -> str:
    ctx = ""
    if context is not None:
        if isinstance(context, list):
            ctx = "\n\n".join(map(str, context))
        else:
            ctx = str(context)
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    parts.append(f"Question:\n{question}")
    if ctx:
        parts.append(f"Context:\n{ctx}")
    parts.append("Answer strictly 'yes' or 'no' to whether the response contains unsupported claims.")
    return "\n\n".join(parts)

def tgi_generate(tgi_url: str, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": int(max_new_tokens),
            "stop": [],
            "do_sample": float(temperature) > 0.0
        }
    }
    for attempt in range(3):
        try:
            r = requests.post(f"{tgi_url}/generate", json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    return data.get("generated_text", data.get("text", "") or "")
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    return data[0].get("generated_text", data[0].get("text", "") or "")
                else:
                    return str(data)
            else:
                time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return ""

def tokenize_text(s: str) -> List[str]:
    if not s:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(s)]

def token_intensity(generated: str, context_or_ref: Optional[str]) -> float:
    gen_tokens = tokenize_text(generated)
    if not gen_tokens:
        return 0.0
    ref_tokens = tokenize_text(context_or_ref or "")
    if not ref_tokens:
        return 1.0
    gct = Counter(gen_tokens)
    rct = Counter(ref_tokens)
    common = gct & rct
    overlap = sum(common.values())
    return 1.0 - (overlap / max(len(gen_tokens), 1))

def normalize_answer(s):
    """Simple answer normalization for EM computation"""
    import re
    import string
    
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

def compute_exact_match(pred, ref):
    """Compute exact match between prediction and reference"""
    return float(normalize_answer(pred) == normalize_answer(ref))

def compute_token_f1(pred, ref):
    """Compute token-level F1 score between prediction and reference"""
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()
    
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_bertscore_fixed(candidates, references):
    """FIXED: BERTScore computation using same pattern as working scripts"""
    if not BERTSCORE_AVAILABLE:
        print("Warning: BERTScore not available. Install with: pip install bert-score")
        return {}
    
    if not candidates or not references:
        print("Warning: No candidates or references for BERTScore")
        return {}
    
    try:
        # Use same configuration as working scripts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = BERTScorer(
            model_type="roberta-large",
            lang="en",
            device=device,
            idf=False,
            batch_size=64,
            use_fast_tokenizer=True
        )
        
        P, R, F = scorer.score(candidates, references, verbose=False)
        
        # Convert to percentage (0-100) to match other metrics
        bert_p = float(P.mean().item()) * 100.0
        bert_r = float(R.mean().item()) * 100.0
        bert_f = float(F.mean().item()) * 100.0
        
        print("BERTScore computed: P={:.2f}, R={:.2f}, F1={:.2f}".format(bert_p, bert_r, bert_f))
        
        return {
            "bert_precision": bert_p,
            "bert_recall": bert_r,
            "bert_f1": bert_f
        }
        
    except Exception as e:
        print("Warning: BERTScore computation failed: {}".format(e))
        return {}

def prf_from_preds(gt: List[int], pred: List[int]) -> Dict[str, Optional[float]]:
    if not gt:
        return {"precision": None, "recall": None, "f1": None}
    gt_a = np.array(gt, dtype=int)
    pred_a = np.array(pred, dtype=int)
    tp = int(((gt_a == 1) & (pred_a == 1)).sum())
    fp = int(((gt_a == 0) & (pred_a == 1)).sum())
    fn = int(((gt_a == 1) & (pred_a == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": round(float(prec), 6), "recall": round(float(rec), 6), "f1": round(float(f1), 6)}

def compute_balanced_metrics(gt_list: List[Optional[int]], pred_list: List[int]) -> Dict[str, float]:
    """FIXED: Compute balanced (macro-averaged) metrics for binary classification"""
    # Filter to labeled examples only
    labeled_indices = [i for i in range(len(gt_list)) if gt_list[i] is not None]
    
    if not labeled_indices:
        return {"balanced_precision": 0.0, "balanced_recall": 0.0, "balanced_f1": 0.0}
    
    # Separate by class
    pos_gt = []
    pos_pred = []
    neg_gt = []
    neg_pred = []
    
    for i in labeled_indices:
        if gt_list[i] == 1:  # positive class (hallucinated)
            pos_gt.append(1)
            pos_pred.append(pred_list[i])
        else:  # negative class (not hallucinated)
            neg_gt.append(1)  # treat as positive for this class
            neg_pred.append(1 - pred_list[i])  # flip prediction for negative class
    
    # Compute per-class metrics
    def compute_class_prf(gt, pred):
        if not gt:
            return 0.0, 0.0, 0.0
        tp = sum(g and p for g, p in zip(gt, pred))
        pred_pos = sum(pred)
        gold_pos = sum(gt)
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        rec = tp / gold_pos if gold_pos > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1
    
    pos_prec, pos_rec, pos_f1 = compute_class_prf(pos_gt, pos_pred)
    neg_prec, neg_rec, neg_f1 = compute_class_prf(neg_gt, neg_pred)
    
    # Macro average
    balanced_precision = (pos_prec + neg_prec) / 2.0
    balanced_recall = (pos_rec + neg_rec) / 2.0
    balanced_f1 = (pos_f1 + neg_f1) / 2.0
    
    return {
        "balanced_precision": round(balanced_precision * 100.0, 6),
        "balanced_recall": round(balanced_recall * 100.0, 6), 
        "balanced_f1": round(balanced_f1 * 100.0, 6)
    }

def compute_metrics_collection(gt_list: List[Optional[int]],
                               model_yesno_list: List[int],
                               token_intensity_list: List[float],
                               algorithms: Dict[str, Tuple[str, Any]],
                               task_list: Optional[List[str]] = None,
                               generated_texts: Optional[List[str]] = None,
                               reference_texts: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main metrics computation with BERTScore and balanced metrics
    """
    N = len(gt_list)
    labeled_indices = [i for i in range(N) if gt_list[i] is not None]
    counts = {"total": N, "labeled": len(labeled_indices)}

    # Algorithm-based metrics
    overall = {}
    for name, (atype, param) in algorithms.items():
        preds = []
        gts = []
        for i in labeled_indices:
            gts.append(gt_list[i])
            if atype == "model":
                preds.append(int(model_yesno_list[i]))
            elif atype == "ti_threshold":
                preds.append(1 if token_intensity_list[i] >= float(param) else 0)
            else:
                preds.append(0)
        overall[name] = prf_from_preds(gts, preds)

    # Per-task metrics
    by_task = {}
    if task_list:
        tasks = sorted(list({task_list[i] for i in labeled_indices if task_list[i] is not None}))
        for task in tasks:
            task_idx = [i for i in labeled_indices if task_list[i] == task]
            t_gt = [gt_list[i] for i in task_idx]
            t_model = [model_yesno_list[i] for i in task_idx]
            t_ti = [token_intensity_list[i] for i in task_idx]
            by_task[task] = {}
            for name, (atype, param) in algorithms.items():
                preds = []
                for j in range(len(task_idx)):
                    if atype == "model":
                        preds.append(int(t_model[j]))
                    elif atype == "ti_threshold":
                        preds.append(1 if t_ti[j] >= float(param) else 0)
                    else:
                        preds.append(0)
                by_task[task][name] = prf_from_preds(t_gt, preds)
    
    result = {"overall": overall, "by_task": by_task, "counts": counts}
    
    # FIXED: Compute BERTScore if we have generated and reference texts
    if generated_texts and reference_texts:
        bert_candidates = []
        bert_references = []
        for i in range(len(generated_texts)):
            if (i < len(reference_texts) and 
                generated_texts[i] and reference_texts[i] and
                generated_texts[i].strip() and reference_texts[i].strip()):
                bert_candidates.append(generated_texts[i])
                bert_references.append([reference_texts[i]])  # BERTScore expects list of references
        
        bert_metrics = compute_bertscore_fixed(bert_candidates, bert_references)
        result.update(bert_metrics)
    
    # FIXED: Compute balanced metrics for the main model algorithm
    if "model_yesno" in algorithms:
        labeled_gt = [gt_list[i] for i in labeled_indices]
        labeled_pred = [model_yesno_list[i] for i in labeled_indices]
        balanced_metrics = compute_balanced_metrics(labeled_gt, labeled_pred)
        result.update(balanced_metrics)
    
    return result

def auto_pick(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    try:
        ds = safe_load_dataset(args.dataset_name, args.dataset_config, args.split)
    except Exception as e:
        raise RuntimeError(f"Dataset load failed: {e}")

    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    cols = list(ds.features.keys())
    input_col = args.input_column or auto_pick(cols, ["prompt", "question", "query", "input", "instruction", "claim", "claim_text"])
    context_col = args.context_column or auto_pick(cols, ["context", "contexts", "passages", "docs", "evidence"])
    label_col = args.label_column or auto_pick(cols, ["label", "hallucinated", "hallucination", "is_hallucination", "gold_label", "target", "annotation"])
    task_col = args.task_column or auto_pick(cols, ["task", "dataset", "split_name"])

    if not input_col:
        raise RuntimeError(f"Could not detect input column. Available columns: {cols}")

    pos_vals = {s.strip().lower() for s in args.positive_label_values.split(",") if s.strip()}
    neg_vals = {s.strip().lower() for s in args.negative_label_values.split(",") if s.strip()}

    # Decode thresholds
    thresholds = [float(t) for t in args.ti_thresholds.split(",") if t.strip()]
    algos = {"model_yesno": ("model", None)}
    for t in thresholds:
        algos[f"token_intensity>={t}"] = ("ti_threshold", t)

    gt_list: List[Optional[int]] = []
    model_yesno_list: List[int] = []
    token_intensity_list: List[float] = []
    task_list: List[Optional[str]] = []
    
    # FIXED: Collect texts for BERTScore and EM/F1 computation
    generated_texts: List[str] = []
    reference_texts: List[str] = []
    exact_match_scores: List[float] = []
    f1_scores: List[float] = []

    # Process each example
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for ex in tqdm(ds, desc="Running eval"):
            q = ex.get(input_col)
            ctx = ex.get(context_col) if context_col else None
            raw_label = ex.get(label_col) if label_col else None
            task_val = ex.get(task_col) if task_col else None

            gt = normalize_label(raw_label, pos_vals, neg_vals)
            prompt = build_prompt(str(q), ctx, args.system_prompt)
            gen = tgi_generate(args.tgi_url, prompt, args.temperature, args.top_p, args.max_new_tokens)
            model_yesno = extract_yesno(gen)
            if model_yesno is None:
                first_line = gen.splitlines()[0] if gen else ""
                mv = extract_yesno(first_line)
                model_yesno = 0 if mv is None else mv

            # Compute token intensity using context if present else label text
            comp_ref = None
            if ctx:
                comp_ref = " ".join(ctx) if isinstance(ctx, list) else str(ctx)
            elif raw_label is not None:
                comp_ref = str(raw_label)
            ti = token_intensity(gen, comp_ref)

            # FIXED: Collect texts for BERTScore computation
            generated_texts.append(gen)
            # Use context as reference if available, otherwise use the question
            if ctx:
                ref_text = " ".join(ctx) if isinstance(ctx, list) else str(ctx)
            else:
                ref_text = str(q) if q else ""
            reference_texts.append(ref_text)
            
            # FIXED: Compute EM and F1 scores if we have meaningful text to compare
            if gen and ref_text:
                exact_match_scores.append(compute_exact_match(gen, ref_text))
                f1_scores.append(compute_token_f1(gen, ref_text))
            else:
                exact_match_scores.append(0.0)
                f1_scores.append(0.0)

            rec = {"prompt": q, "context": ctx, "raw_output": gen, 
                   "prediction_yesno": int(model_yesno), 
                   "label": (int(gt) if gt is not None else None), 
                   "task": task_val}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            gt_list.append(gt)
            model_yesno_list.append(int(model_yesno))
            token_intensity_list.append(float(ti))
            task_list.append(task_val if task_val is not None else None)

    # Compute comprehensive metrics
    metrics = compute_metrics_collection(
        gt_list, 
        model_yesno_list, 
        token_intensity_list, 
        algos, 
        task_list if task_col else None,
        generated_texts,  # FIXED: Added for BERTScore
        reference_texts   # FIXED: Added for BERTScore
    )
    
    # FIXED: Include overall EM and F1 scores
    if exact_match_scores:
        overall_em = sum(exact_match_scores) / len(exact_match_scores) * 100.0
        overall_f1 = sum(f1_scores) / len(f1_scores) * 100.0
        metrics["exact_match"] = round(overall_em, 6)
        metrics["f1"] = round(overall_f1, 6)
        
        # FIXED: Compute balanced EM/F1 based on hallucination classes
        pos_em_scores = [exact_match_scores[i] for i in range(len(gt_list)) 
                        if gt_list[i] == 1]
        neg_em_scores = [exact_match_scores[i] for i in range(len(gt_list)) 
                        if gt_list[i] == 0]
        pos_f1_scores = [f1_scores[i] for i in range(len(gt_list)) 
                        if gt_list[i] == 1]
        neg_f1_scores = [f1_scores[i] for i in range(len(gt_list)) 
                        if gt_list[i] == 0]
        
        pos_em_avg = sum(pos_em_scores) / len(pos_em_scores) if pos_em_scores else 0.0
        neg_em_avg = sum(neg_em_scores) / len(neg_em_scores) if neg_em_scores else 0.0
        pos_f1_avg = sum(pos_f1_scores) / len(pos_f1_scores) if pos_f1_scores else 0.0
        neg_f1_avg = sum(neg_f1_scores) / len(neg_f1_scores) if neg_f1_scores else 0.0
        
        balanced_em = (pos_em_avg + neg_em_avg) / 2.0 * 100.0
        balanced_f1 = (pos_f1_avg + neg_f1_avg) / 2.0 * 100.0
        
        metrics["balanced_exact_match"] = round(balanced_em, 6)
        metrics["text_balanced_f1"] = round(balanced_f1, 6)  # Different from classification balanced_f1

    # Save comprehensive metrics to JSON
    metrics_path = os.path.splitext(args.output_file)[0] + ".metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"Saved comprehensive metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
