#!/usr/bin/env python3
"""
RAGTruth evaluator — minimal outputs: precision, recall, F1 for each detection algorithm
and its variants, overall and per-task.

Usage example:
python evals/ragtruth_hf_eval.py \
  --tgi-url http://localhost:8300 \
  --dataset-name microsoft/RAGTruth \
  --split test \
  --output-file outputs/ragtruth_preds.jsonl

Important: this script intentionally returns ONLY precision/recall/F1 for each algorithm and
per-task breakdowns (plus simple counts).
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

YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

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
    ap.add_argument("--task-column", type=str, default=None, help="Optional task column for per-task breakdown")
    ap.add_argument("--positive-label-values", type=str, default="yes,true,1,hallucinated")
    ap.add_argument("--negative-label-values", type=str, default="no,false,0,not_hallucinated")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--ti-thresholds", type=str, default="0.25,0.5,0.75",
                    help="Comma-separated token-intensity thresholds to evaluate (e.g. 0.25,0.5)")
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
                # transient server error
                time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return ""  # fail-safe empty

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
    from collections import Counter
    gct = Counter(gen_tokens)
    rct = Counter(ref_tokens)
    common = gct & rct
    overlap = sum(common.values())
    return 1.0 - (overlap / max(len(gen_tokens), 1))

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

def compute_metrics_collection(gt_list: List[Optional[int]],
                               model_yesno_list: List[int],
                               token_intensity_list: List[float],
                               algorithms: Dict[str, Tuple[str, Any]],
                               task_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    algorithms: dict name -> (type, param)
      type "model" -> param ignored
      type "ti_threshold" -> param is threshold float
    Returns structure:
      {
        "overall": {algo_name: {"precision":..,"recall":..,"f1":..}, ...},
        "by_task": {task_name: {algo_name: {...}}, ...},
        "counts": {"total":N,"labeled":L}
      }
    """
    N = len(gt_list)
    indices = list(range(N))
    # build lists only for labeled indices
    labeled_indices = [i for i in indices if gt_list[i] is not None]
    counts = {"total": N, "labeled": len(labeled_indices)}

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

    by_task = {}
    if task_list:
        # collect unique tasks among labeled examples
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
    return {"overall": overall, "by_task": by_task, "counts": counts}

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

    # decode thresholds
    thresholds = [float(t) for t in args.ti_thresholds.split(",") if t.strip()]
    algos = {"model_yesno": ("model", None)}
    for t in thresholds:
        algos[f"token_intensity>={t}"] = ("ti_threshold", t)

    gt_list: List[Optional[int]] = []
    model_yesno_list: List[int] = []
    token_intensity_list: List[float] = []
    task_list: List[Optional[str]] = []

    # write predictions jsonl
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
                # fallback interpret first line or default no
                first_line = gen.splitlines()[0] if gen else ""
                mv = extract_yesno(first_line)
                model_yesno = 0 if mv is None else mv

            # compute token intensity using context if present else label text
            comp_ref = None
            if ctx:
                comp_ref = " ".join(ctx) if isinstance(ctx, list) else str(ctx)
            elif raw_label is not None:
                comp_ref = str(raw_label)
            ti = token_intensity(gen, comp_ref)

            rec = {"prompt": q, "context": ctx, "raw_output": gen, "prediction_yesno": int(model_yesno), "label": (int(gt) if gt is not None else None), "task": task_val}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            gt_list.append(gt)
            model_yesno_list.append(int(model_yesno))
            token_intensity_list.append(float(ti))
            task_list.append(task_val if task_val is not None else None)

    # compute PRF collections
    metrics = compute_metrics_collection(gt_list, model_yesno_list, token_intensity_list, algos, task_list if task_col else None)

    # Save only PRF metrics (and counts) to metrics JSON
    metrics_path = os.path.splitext(args.output_file)[0] + ".metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"Saved minimal PRF metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
