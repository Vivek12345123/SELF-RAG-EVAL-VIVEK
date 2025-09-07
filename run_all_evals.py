#!/usr/bin/env python3
"""
run_all_evals.py

One-file orchestrator to run the provided official evaluation scripts unchanged
against the Self-RAG model "selfrag/selfrag_llama2_13b" using vLLM.

What it does:
 - Loads model via vLLM (half precision) from MODEL_NAME.
 - Starts a minimal TGI-compatible adapter (Flask) on --tgi-host:--tgi-port so
   the original RAGTruth script can call /generate unchanged.
 - For each task it can:
     * Create prediction files in the format the official eval expects.
     * Create gold/reference files from HF dataset splits if needed.
     * Call the official eval script under evals/<script>.py as a subprocess.
 - DOES NOT modify the original eval scripts. They must exist under evals/.

IMPORTANT:
 - Put original, untouched eval scripts here:
     evals/ragtruth_hf_eval.py            (RAGTruth)
     evals/squad_v2_official_eval.py      (SQuAD v2 official)
     evals/hotpot_eval.py                 (HotPotQA official)
     evals/ms_marco_eval.py               (MS MARCO official)
     evals/natural_questions_official_eval.py   (Natural Questions official)
     evals/triviaqa_eval.py               (TriviaQA official)
 - For MS MARCO you must have the MS MARCO bleu/rouge modules and spaCy model
   ('en_core_web_lg') available per the original toolkit.
 - Natural Questions official eval expects complex prediction objects. This
   script can create an *empty* predictions file for smoke testing (--nq-empty)
   or you can provide a real predictions path via --nq-predictions.
"""

import argparse
import json
import os
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import threading
import logging

# Model / environment defaults - edit only if needed
MODEL_NAME = "selfrag/selfrag_llama2_13b"
MODEL_CACHE_DIR = "/gscratch/h2lab/akari/model_cache"  # adjust if necessary
DEFAULT_TGI_HOST = "127.0.0.1"
DEFAULT_TGI_PORT = 8300
DATASET_CACHE_ROOT = Path("./datasets_cache")
OUT_ROOT = Path("./outputs")
MIN_FREE_BYTES_WARN = 30 * 1024 ** 3  # warn if < 30 GB free

# Required original eval script paths (unchanged originals must be placed here)
EVALS_DIR = Path("evals")
RAGTRUTH_SCRIPT = EVALS_DIR / "ragtruth_hf_eval.py"
SQUADV2_SCRIPT = EVALS_DIR / "squad_v2_official_eval.py"
HOTPOT_SCRIPT = EVALS_DIR / "hotpot_eval.py"
MSMARCO_SCRIPT = EVALS_DIR / "ms_marco_eval.py"
NQ_SCRIPT = EVALS_DIR / "natural_questions_official_eval.py"
TRIVIA_SCRIPT = EVALS_DIR / "triviaqa_eval.py"
CRAG_SCRIPT = EVALS_DIR / "crag_eval.py"


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- Helper: disk space check ---
def bytes_to_gb(b): return b / (1024 ** 3)
def check_disk_space(path="."):
    usage = shutil.disk_usage(path)
    free = usage.free
    logging.info(f"[disk] total={bytes_to_gb(usage.total):.1f}GB used={bytes_to_gb(usage.used):.1f}GB free={bytes_to_gb(free):.1f}GB")
    if free < MIN_FREE_BYTES_WARN:
        logging.warning(f"Free disk < {bytes_to_gb(MIN_FREE_BYTES_WARN):.1f}GB. Full downloads may fail. Use --max-examples for testing.")
def run_crag():
    run_script(CRAG_SCRIPT, "CRAG")
# --- Minimal TGI-compatible adapter (Flask) that wraps vLLM generate ---
def start_tgi_adapter_in_thread(tgi_host: str, tgi_port: int, model_obj, sampling_params):
    """
    Starts a Flask server in a background thread exposing /generate compatible
    with the ragtruth eval script. Returns the Thread object. Uses the provided
    model and sampling_params (vLLM objects).
    """
    from flask import Flask, request, jsonify
    app = Flask("tgi_adapter_for_ragtruth")

    @app.route("/generate", methods=["POST"])
    def generate_route():
        payload = request.get_json(force=True)
        prompt = payload.get("inputs", "")
        params = payload.get("parameters", {})
        # Map parameters to vLLM SamplingParams if available
        try:
            from vllm import SamplingParams
            temp = params.get("temperature", sampling_params.temperature)
            top_p = params.get("top_p", sampling_params.top_p)
            max_tokens = params.get("max_new_tokens", getattr(sampling_params, "max_tokens", 128))
            sp = SamplingParams(temperature=float(temp), top_p=float(top_p), max_tokens=int(max_tokens), skip_special_tokens=False)
        except Exception:
            sp = sampling_params
        try:
            preds = model_obj.generate([prompt], sp)
            outtext = preds[0].outputs[0].text
            return jsonify({"generated_text": outtext})
        except Exception as e:
            return jsonify({"error": str(e), "generated_text": ""}), 500

    def run_app():
        logging.info(f"[tgi_adapter] Starting adapter at http://{tgi_host}:{tgi_port}")
        app.run(host=tgi_host, port=tgi_port, threaded=True)

    thr = threading.Thread(target=run_app, daemon=True)
    thr.start()
    # wait briefly for server to be ready
    time.sleep(1.5)
    return thr

# --- Model loader matching the code you provided ---
def load_model_and_tokenizer(model_name: str = MODEL_NAME, cache_dir: str = MODEL_CACHE_DIR):
    logging.info(f"[model] Loading model {model_name} (vLLM, dtype=half) with cache_dir={cache_dir}")
    # load tokenizer (some eval scripts might require it)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as e:
        logging.warning(f"[tokenizer] AutoTokenizer load failed: {e}")
        tokenizer = None
    # load model via vLLM
    try:
        from vllm import LLM, SamplingParams
        model = LLM(model_name, download_dir=cache_dir, dtype="half")
        # default sampling params
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128, skip_special_tokens=False)
    except Exception as e:
        logging.error(f"[model] vLLM LLM load failed: {e}")
        raise
    # format_prompt function (Self-RAG style)
    def format_prompt(input_text: str, paragraph: Optional[str] = None):
        prompt = f"### Instruction:\n{input_text}\n\n### Response:\n"
        if paragraph is not None:
            prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
        return prompt
    return model, tokenizer, sampling_params, format_prompt

# --- Utilities to generate predictions and call official eval scripts unchanged ---

def run_subprocess_command(cmd: List[str]) -> int:
    logging.info(f"[run] {cmd}")
    rc = subprocess.call(cmd)
    if rc != 0:
        logging.warning(f"[run] Subprocess returned non-zero exit code {rc}: {' '.join(cmd)}")
    return rc

# SQuAD v2: build SQuAD-style data and preds.json (qid -> answer string) then call official eval
def run_squad_v2(model, sampling_params, format_prompt, dataset_key="rajpurkar/squad_v2", split="validation", out_dir=OUT_ROOT / "squad_v2", batch_size=8, max_tokens=128, temp=0.0, top_p=1.0):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("[squad] Loading dataset from HF: %s split=%s", dataset_key, split)
    from datasets import load_dataset
    ds = load_dataset(dataset_key, split=split)
    # Reconstruct SQuAD JSON structure (best-effort)
    articles = {}
    qlist = []
    for ex in ds:
        qid = ex.get("id") or ex.get("question_id") or ex.get("qid") or str(len(qlist))
        question = ex.get("question") or ex.get("queries") or ex.get("query") or ""
        context = ex.get("context") or ex.get("paragraphs") or ""
        answers = ex.get("answers") or {"text": [], "answer_start": []}
        # group by context text
        key = context
        if key not in articles:
            articles[key] = {"title": ex.get("title") or "", "paragraphs": [{"context": context, "qas": []}]}
        articles[key]["paragraphs"][0]["qas"].append({"id": qid, "question": question, "answers": [{"text": t, "answer_start": s} for t, s in zip(answers.get("text", []), answers.get("answer_start", []))]})
        qlist.append((qid, question, context))
    data = [v for v in articles.values()]
    data_json = {"version": "v2.0", "data": data}
    data_file = out_dir / "squad_v2_data.json"
    with open(data_file, "w", encoding="utf-8") as fh:
        json.dump(data_json, fh)
    # generate preds
    pred_file = out_dir / "preds.json"
    from vllm import SamplingParams as VSamplingParams
    sp = VSamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=True)
    preds = {}
    logging.info(f"[squad] Generating answers for {len(qlist)} examples (batch {batch_size})")
    for i in range(0, len(qlist), batch_size):
        batch = qlist[i:i+batch_size]
        prompts = [format_prompt(f"Answer the question using the context. Question: {q}\nContext: {c}") for (_id,q,c) in batch]
        outs = model.generate(prompts, sp)
        for j, out in enumerate(outs):
            text = out.outputs[0].text.strip()
            qid = batch[j][0]
            preds[qid] = text
    with open(pred_file, "w", encoding="utf-8") as fh:
        json.dump(preds, fh)
    logging.info("[squad] Wrote preds -> %s", pred_file)
    # Call official eval unchanged
    if not SQUADV2_SCRIPT.exists():
        logging.warning(f"[squad] Official SQuAD eval script not found at {SQUADV2_SCRIPT}; skip evaluation.")
        return
    cmd = ["python", str(SQUADV2_SCRIPT), str(data_file), str(pred_file), "--out-file", str(out_dir / "eval.json")]
    run_subprocess_command(cmd)

# HotPotQA: create preds structure {'answer': {id: ans}, 'sp': {id: []}} and gold json list, call original eval
def run_hotpot(model, sampling_params, format_prompt, dataset_key="hotpotqa/hotpot_qa", split="validation", out_dir=OUT_ROOT / "hotpot", batch_size=8, max_tokens=128, temp=0.0, top_p=1.0):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("[hotpot] Loading dataset %s split=%s", dataset_key, split)
    from datasets import load_dataset
    ds = load_dataset(dataset_key, split=split)
    # generate predictions
    preds = {"answer": {}, "sp": {}}
    from vllm import SamplingParams as VSamplingParams
    sp = VSamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=True)
    qlist = []
    for ex in ds:
        qid = ex.get("_id") or ex.get("id") or str(len(qlist))
        question = ex.get("question", "")
        # flatten context list into text if present
        context = ""
        if "context" in ex and ex["context"]:
            parts = []
            for p in ex["context"]:
                title = p[0] if len(p) > 0 else ""
                sents = " ".join(p[1]) if len(p) > 1 else ""
                parts.append(f"{title}: {sents}")
            context = "\n".join(parts)
        qlist.append((qid, question, context))
    logging.info(f"[hotpot] Generating answers for {len(qlist)} items")
    for i in range(0, len(qlist), batch_size):
        batch = qlist[i:i+batch_size]
        prompts = [format_prompt(f"Answer the question using the context. Question: {q}\nContext: {c}") for (_id,q,c) in batch]
        outs = model.generate(prompts, sp)
        for j, out in enumerate(outs):
            ans = out.outputs[0].text.strip()
            qid = batch[j][0]
            preds["answer"][qid] = ans
            # supporting facts left empty (safe default). If you want model-generated SP, request it separately.
            preds["sp"][qid] = []
    pred_file = out_dir / "preds_hotpot.json"
    gold_file = out_dir / "gold_hotpot.json"
    with open(pred_file, "w", encoding="utf-8") as fh:
        json.dump(preds, fh)
    # export gold list from HF split (best-effort; the official script will iterate the list)
    gold_list = [ex for ex in ds]
    with open(gold_file, "w", encoding="utf-8") as fh:
        json.dump(gold_list, fh)
    logging.info("[hotpot] Wrote preds %s and gold %s", pred_file, gold_file)
    if not HOTPOT_SCRIPT.exists():
        logging.warning(f"[hotpot] Official HotPot eval script not found at {HOTPOT_SCRIPT}; skip evaluation.")
        return
    cmd = ["python", str(HOTPOT_SCRIPT), str(pred_file), str(gold_file)]
    run_subprocess_command(cmd)

# RAGTruth: start adapter and call the original ragtruth eval unchanged (it loads HF dataset internally)
def run_ragtruth_adapter_and_eval(tgi_host: str, tgi_port: int, dataset_name: str, dataset_config: Optional[str], split: str, output_file: str, system_prompt: str, model_obj, sampling_params):
    # Start adapter thread (Flask) wrapping model_obj
    thr = start_tgi_adapter_in_thread(tgi_host, tgi_port, model_obj, sampling_params)
    if not RAGTRUTH_SCRIPT.exists():
        logging.warning(f"[ragtruth] Official RAGTruth eval script not found at {RAGTRUTH_SCRIPT}; skip.")
        return
    cmd = ["python", str(RAGTRUTH_SCRIPT), "--tgi-url", f"http://{tgi_host}:{tgi_port}", "--dataset-name", dataset_name, "--split", split, "--output-file", output_file]
    if dataset_config:
        cmd += ["--dataset-config", dataset_config]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]
    logging.info("[ragtruth] Running RAGTruth eval against adapter")
    try:
        run_subprocess_command(cmd)
    finally:
        logging.info("[ragtruth] Adapter is running in background thread (daemon). It will exit when this process exits.")

# MS MARCO: build reference and candidate jsonl files and call official ms_marco_eval.py
def run_ms_marco(model, sampling_params, format_prompt, dataset_key="microsoft/ms_marco", dataset_config="v2.1", split="train", out_dir=OUT_ROOT / "ms_marco", batch_size=8, max_tokens=64, temp=0.0, top_p=1.0):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    logging.info("[msmarco] Loading dataset %s config=%s split=%s", dataset_key, dataset_config, split)
    ds = load_dataset(dataset_key, dataset_config, split=split)
    references = []
    queries = []
    for ex in ds:
        qid = ex.get("query_id") or ex.get("id") or str(len(queries))
        query = ex.get("query") or ex.get("question") or ex.get("query_text") or ""
        answers = ex.get("answers") or []
        if not isinstance(answers, list):
            answers = [str(answers)]
        references.append({"query_id": qid, "answers": answers if answers else [""]})
        queries.append((qid, query))
    ref_path = out_dir / "msmarco_references.jsonl"
    cand_path = out_dir / "msmarco_candidates.jsonl"
    with open(ref_path, "w", encoding="utf-8") as fh:
        for rec in references:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info("[msmarco] Wrote references to %s", ref_path)
    # generate candidates
    from vllm import SamplingParams as VSamplingParams
    sp = VSamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=True)
    candidates = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        prompts = [format_prompt(f"Answer the query: {q}", paragraph=None) for (_id, q) in batch]
        outs = model.generate(prompts, sp)
        for j, out in enumerate(outs):
            text = out.outputs[0].text.strip()
            qid = batch[j][0]
            candidates.append({"query_id": qid, "answers": [text if text else ""]})
    with open(cand_path, "w", encoding="utf-8") as fh:
        for rec in candidates:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info("[msmarco] Wrote candidates to %s", cand_path)
    if not MSMARCO_SCRIPT.exists():
        logging.warning(f"[msmarco] Official MS MARCO eval script not found at {MSMARCO_SCRIPT}; skip.")
        return
    cmd = ["python", str(MSMARCO_SCRIPT), str(ref_path), str(cand_path)]
    run_subprocess_command(cmd)

# Natural Questions: prepare gold, optionally make empty predictions for smoke test or accept a user-provided predictions file.
def run_natural_questions(out_dir=OUT_ROOT / "nq", dataset_key="sentence-transformers/natural-questions", split="validation", predictions_path: Optional[str]=None, make_empty=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    logging.info("[nq] Loading dataset %s split=%s", dataset_key, split)
    ds = load_dataset(dataset_key, split=split)
    gold_path = out_dir / "nq_gold.json"
    # best-effort gold dump (official util.read_annotation may expect gzipped original format; this is a best-effort)
    with open(gold_path, "w", encoding="utf-8") as fh:
        json.dump([ex for ex in ds], fh)
    logging.info("[nq] Wrote gold file to %s (best-effort)", gold_path)
    pred_path = predictions_path
    if pred_path is None and make_empty:
        pred_path = out_dir / "nq_empty_preds.json"
        # Create trivial empty predictions mapping example_id -> {}
        preds = {}
        for ex in ds:
            ex_id = ex.get("example_id") or ex.get("id") or ex.get("query_id") or str(len(preds))
            preds[str(ex_id)] = {}
        with open(pred_path, "w", encoding="utf-8") as fh:
            json.dump(preds, fh)
        logging.info("[nq] Wrote empty predictions to %s", pred_path)
    if pred_path is None:
        logging.info("[nq] No predictions provided. Skipping official NQ eval. Provide --nq-predictions to run official script.")
        return
    if not NQ_SCRIPT.exists():
        logging.warning(f"[nq] Official NQ eval script not found at {NQ_SCRIPT}; skip.")
        return
    cmd = ["python", str(NQ_SCRIPT), f"--gold_path={gold_path}", f"--predictions_path={pred_path}"]
    run_subprocess_command(cmd)

# TriviaQA: generate string predictions mapping qid->answer and optionally run official eval if user provides gold dataset_file path
def run_triviaqa(model, sampling_params, format_prompt, dataset_key="mandarjoshi/trivia_qa", split="rc", out_dir=OUT_ROOT / "trivia", batch_size=8, max_tokens=64, temp=0.0, top_p=1.0, trivia_gold_path: Optional[str]=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    logging.info("[trivia] Loading dataset %s split=%s", dataset_key, split)
    ds = load_dataset(dataset_key, split=split)
    qlist = []
    ids = []
    for ex in ds:
        qid = ex.get("question_id") or ex.get("id") or ex.get("q_id") or ex.get("example_id") or str(len(ids))
        question = ex.get("question") or ex.get("query") or ex.get("text") or ""
        context = ex.get("search_results") or ex.get("context") or None
        prompt_text = f"Answer the trivia question: {question}"
        if context:
            prompt_text += f"\n\nContext: {context}"
        qlist.append((str(qid), prompt_text))
        ids.append(str(qid))
    from vllm import SamplingParams as VSamplingParams
    sp = VSamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=True)
    preds = {}
    logging.info(f"[trivia] Generating {len(qlist)} predictions")
    for i in range(0, len(qlist), batch_size):
        batch = qlist[i:i+batch_size]
        prompts = [format_prompt(t, paragraph=None) for (_id,t) in batch]
        outs = model.generate(prompts, sp)
        for j, out in enumerate(outs):
            ans = out.outputs[0].text.strip()
            qid = batch[j][0]
            preds[qid] = ans
    pred_file = out_dir / "trivia_preds.json"
    with open(pred_file, "w", encoding="utf-8") as fh:
        json.dump(preds, fh)
    logging.info("[trivia] Wrote predictions to %s", pred_file)
    if trivia_gold_path:
        if not TRIVIA_SCRIPT.exists():
            logging.warning(f"[trivia] Official Trivia eval script not found at {TRIVIA_SCRIPT}; skip.")
            return
        cmd = ["python", str(TRIVIA_SCRIPT), "--dataset_file", trivia_gold_path, "--prediction_file", str(pred_file)]
        run_subprocess_command(cmd)
    else:
        logging.info("[trivia] No trivia gold path provided; predictions saved. Provide --trivia-gold to run official eval.")

# --- Main CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Unified runner: generate preds with Self-RAG vLLM and call official eval scripts unchanged.")
    p.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (if needed for private datasets).")
    p.add_argument("--cache-dir", type=str, default=MODEL_CACHE_DIR)
    p.add_argument("--download-datasets", action="store_true", help="If set, attempt to download listed HF dataset splits to ./datasets_cache for repeat runs.")
    p.add_argument("--max-examples", type=int, default=None, help="If set, use streaming subset when downloading / testing to limit examples per split.")
    p.add_argument("--tgi-host", type=str, default=DEFAULT_TGI_HOST)
    p.add_argument("--tgi-port", type=int, default=DEFAULT_TGI_PORT)
    p.add_argument("--run", type=str, default="all", help="Comma-separated tasks to run: ragtruth,squad,hotpot,msmarco,nq,trivia or 'all'")
    # generation controls common to tasks
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    # task-specific options
    p.add_argument("--nq-predictions", type=str, default=None, help="Path to predictions JSON for NQ (preferred).")
    p.add_argument("--nq-empty", action="store_true", help="Create empty predictions for NQ and run official eval (smoke test).")
    p.add_argument("--trivia-gold", type=str, default=None, help="Path to TriviaQA gold dataset for official eval (optional).")
    p.add_argument("--msmarco-config", type=str, default="v2.1")
    p.add_argument("--squad-split", type=str, default="validation")
    p.add_argument("--hotpot-split", type=str, default="validation")
    p.add_argument("--msmarco-split", type=str, default="train")
    p.add_argument("--trivia-split", type=str, default="rc")
    p.add_argument("--nq-split", type=str, default="validation")
    return p.parse_args()

def main():
    args = parse_args()
    check_disk_space(".")
    tasks = [t.strip() for t in args.run.split(",")] if args.run else ["all"]
    if "all" in tasks:
        tasks = ["ragtruth", "squad", "hotpot", "msmarco", "nq", "trivia"]
    # load model
    model, tokenizer, sampling_params, format_prompt = load_model_and_tokenizer(MODEL_NAME, args.cache_dir)
    # optionally download datasets (best-effort simple approach)
    if args.download_datasets:
        logging.info("[dl] Downloading HF datasets to ./datasets_cache (may be large). Use --max-examples to test small subsets.")
        from datasets import load_dataset
        # list of HF datasets we may use
        hf_tasks = {
            "ragtruth": ("wandb/RAGTruth-processed", None, args.squad_split),
            "squad": ("rajpurkar/squad_v2", None, args.squad_split),
            "hotpot": ("hotpotqa/hotpot_qa", "distractor", args.hotpot_split),
            "msmarco": ("microsoft/ms_marco", args.msmarco_config, args.msmarco_split),
            "nq": ("sentence-transformers/natural-questions", None, args.nq_split),
            "trivia": ("mandarjoshi/trivia_qa", "rc", args.trivia_split),
            "fever": ("mwong/fever-evidence-related", None, "train"),
            "crag": ("Quivr/CRAG", None, "train"),
        }
        for k, (dsid, cfg, split) in hf_tasks.items():
            try:
                if args.max_examples:
                    # streaming partial download: write a jsonl with first N examples to datasets_cache/<k>/
                    outdir = DATASET_CACHE_ROOT / k
                    outdir.mkdir(parents=True, exist_ok=True)
                    logging.info(f"[dl] Streaming {args.max_examples} examples for {dsid} -> {outdir}")
                    stream = load_dataset(dsid, cfg, split=split, streaming=True, use_auth_token=args.hf_token) if cfg else load_dataset(dsid, split=split, streaming=True, use_auth_token=args.hf_token)
                    count = 0
                    with open(outdir / "stream.jsonl", "w", encoding="utf-8") as fh:
                        for ex in stream:
                            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                            count += 1
                            if count >= args.max_examples:
                                break
                    logging.info(f"[dl] Wrote {count} examples for {k}")
                else:
                    logging.info(f"[dl] Full download: {dsid} (this may be very large)")
                    _ = load_dataset(dsid, cfg, split=split, use_auth_token=args.hf_token) if cfg else load_dataset(dsid, split=split, use_auth_token=args.hf_token)
                    logging.info(f"[dl] Downloaded {dsid}")
            except Exception as e:
                logging.warning(f"[dl] Failed to download {dsid}: {e}")
    # run requested tasks
    if "ragtruth" in tasks:
        logging.info("[main] Running RAGTruth (adapter + official eval)")
        out_file = OUT_ROOT / "ragtruth_preds.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        run_ragtruth_adapter_and_eval(args.tgi_host, args.tgi_port, "wandb/RAGTruth-processed", None, "validation", str(out_file), "", model, sampling_params)
    if "squad" in tasks:
        logging.info("[main] Running SQuAD v2 eval")
        run_squad_v2(model, sampling_params, format_prompt, dataset_key="rajpurkar/squad_v2", split=args.squad_split, out_dir=OUT_ROOT / "squad_v2", batch_size=args.batch_size, max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p)
    if "hotpot" in tasks:
        logging.info("[main] Running HotPotQA eval")
        run_hotpot(model, sampling_params, format_prompt, dataset_key="hotpotqa/hotpot_qa", split=args.hotpot_split, out_dir=OUT_ROOT / "hotpot", batch_size=args.batch_size, max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p)
    if "msmarco" in tasks:
        logging.info("[main] Running MS MARCO eval")
        run_ms_marco(model, sampling_params, format_prompt, dataset_key="microsoft/ms_marco", dataset_config=args.msmarco_config, split=args.msmarco_split, out_dir=OUT_ROOT / "ms_marco", batch_size=args.batch_size, max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p)
    if "nq" in tasks:
        logging.info("[main] Running Natural Questions (gold prep + optional empty preds)")
        run_natural_questions(out_dir=OUT_ROOT / "nq", dataset_key="sentence-transformers/natural-questions", split=args.nq_split, predictions_path=args.nq_predictions, make_empty=args.nq_empty)
    if "trivia" in tasks:
        logging.info("[main] Running TriviaQA predictions (official eval optional)")
        run_triviaqa(model, sampling_params, format_prompt, dataset_key="mandarjoshi/trivia_qa", split=args.trivia_split, out_dir=OUT_ROOT / "trivia", batch_size=args.batch_size, max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p, trivia_gold_path=args.trivia_gold)
    logging.info("[main] All requested tasks triggered. Wrappers call official eval scripts unchanged where applicable.")
    logging.info("Outputs saved under: %s", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
