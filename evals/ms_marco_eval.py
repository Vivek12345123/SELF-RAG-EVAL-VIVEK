#!/usr/bin/env python3
"""
MS MARCO evaluation wrapper (BLEU / ROUGE / semantic similarity)

Usage:
  python evals/ms_marco_eval.py <path_to_reference_file> <path_to_candidate_file>

Notes:
 - This script tries to compute BLEU-1..4 (using NLTK if available), ROUGE-L, a SpaCy-based
   semantic similarity average, and BERTScore (roberta-large) if installed.
 - If a dependency (nltk, bert-score, large spacy model) is missing it falls back gracefully.
"""

from __future__ import print_function

import json
import sys
import os
import math
import warnings
from typing import Dict, List

import sacrebleu
from rouge_score import rouge_scorer

import spacy
from spacy.lang.en import English  # tokenizer-only pipeline when we don't need vectors

# Try to load the large vectors model for semantic similarity; fall back to smaller model if needed.
try:
    nlp = spacy.load("en_core_web_lg")
    SPACY_LG_AVAILABLE = True
except Exception:
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_LG_AVAILABLE = False
        warnings.warn("en_core_web_lg not available; using en_core_web_sm (lower-quality semantic similarity).")
    except Exception:
        nlp = None
        SPACY_LG_AVAILABLE = False
        warnings.warn("spaCy models not available; semantic similarity will be skipped.")

# NLTK for BLEU-n (optional)
try:
    from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

# BERTScore availability
try:
    from bert_score.scorer import BERTScorer
    import torch
    BERTSCORE_AVAILABLE = True
except Exception:
    BERTSCORE_AVAILABLE = False

# Globals / constants
QUERY_ID_JSON_ID = "query_id"
ANSWERS_JSON_ID = "answers"
MAX_BLEU_ORDER = 4


def normalize_batch(p_iter, p_batch_size=1000, p_thread_count=1):
    """
    Tokenize and lowercase a batch of strings using a tokenizer-only spaCy pipeline.
    This is faster than the full pipeline when only tokenization/lowercasing is needed.
    """
    tokenizer_pipeline = English()  # tokenizer-only pipeline, no parser/NER
    # spaCy v3: use n_process; n_process=1 is safe default (no parallelism)
    output_iter = tokenizer_pipeline.pipe(p_iter, batch_size=p_batch_size, n_process=p_thread_count)
    for doc in output_iter:
        tokens = [str(w).strip().lower() for w in doc]
        yield " ".join(tokens)


def load_file(p_path_to_data: str):
    """
    Load MS MARCO-style JSONL file:
      { "query_id": "...", "answers": ["a1","a2", ...] }
    Returns:
      query_id_to_answers_map: Dict[query_id, List[normalized answers]]
      no_answer_query_ids: set(query_id)
    """
    all_answers = []
    query_ids = []
    no_answer_query_ids = set()
    with open(p_path_to_data, "r", encoding="utf-8") as data_file:
        for line in data_file:
            if not line.strip():
                continue
            try:
                json_object = json.loads(line)
            except json.JSONDecodeError:
                raise Exception('"%s" is not a valid json' % line)

            if QUERY_ID_JSON_ID not in json_object:
                raise AssertionError("json missing query_id")
            query_id = json_object[QUERY_ID_JSON_ID]

            if ANSWERS_JSON_ID not in json_object:
                raise AssertionError("json missing answers")
            answers = json_object[ANSWERS_JSON_ID]
            # MS MARCO uses the sentinel 'No Answer Present.' to indicate no answer
            if isinstance(answers, list) and "No Answer Present." in answers:
                no_answer_query_ids.add(query_id)
                answers = [""]
            # ensure answers is list-like
            if not isinstance(answers, list):
                answers = [str(answers)]
            all_answers.extend(answers)
            query_ids.extend([query_id] * len(answers))

    # Normalize with tokenizer pipeline
    all_normalized_answers = list(normalize_batch(all_answers))

    query_id_to_answers_map = {}
    for i, normalized_answer in enumerate(all_normalized_answers):
        query_id = query_ids[i]
        query_id_to_answers_map.setdefault(query_id, []).append(normalized_answer)

    return query_id_to_answers_map, no_answer_query_ids


def compute_metrics_from_files(p_path_to_reference_file: str,
                               p_path_to_candidate_file: str,
                               p_max_bleu_order: int = MAX_BLEU_ORDER) -> Dict[str, float]:
    # Load references and candidates
    reference_dictionary, reference_no_answer_query_ids = load_file(p_path_to_reference_file)
    candidate_dictionary, candidate_no_answer_query_ids = load_file(p_path_to_candidate_file)

    # Which query ids are answerable (i.e., not marked as 'No Answer Present.' in the references)
    query_id_answerable = set(reference_dictionary.keys()) - reference_no_answer_query_ids
    query_id_answerable_candidate = set(candidate_dictionary.keys()) - candidate_no_answer_query_ids

    # Simple answer/no-answer classification metrics
    true_positives = len(query_id_answerable_candidate.intersection(query_id_answerable))
    false_negatives = len(query_id_answerable) - true_positives
    true_negatives = len(candidate_no_answer_query_ids.intersection(reference_no_answer_query_ids))
    false_positives = len(reference_no_answer_query_ids) - true_negatives

    precision = float(true_positives) / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
    recall = float(true_positives) / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
    F1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Filter out reference queries that are "no answer" for BLEU/ROUGE/semantic metrics:
    filtered_reference_dictionary = {k: v for k, v in reference_dictionary.items() if k not in reference_no_answer_query_ids}
    # Important fix: filter candidates by candidate_no_answer_query_ids (not reference_no_answer_query_ids)
    filtered_candidate_dictionary = {k: v for k, v in candidate_dictionary.items() if k not in candidate_no_answer_query_ids}

    # Sanity checks: candidate should have at most one produced answer per query (MS MARCO-style)
    for query_id, answers in filtered_candidate_dictionary.items():
        if len(answers) > 1:
            raise AssertionError("query_id %s contains more than 1 answer" % str(query_id))

    # Ensure the sets of query ids match after filtering
    reference_query_ids = set(filtered_reference_dictionary.keys())
    candidate_query_ids = set(filtered_candidate_dictionary.keys())
    common_query_ids = reference_query_ids.intersection(candidate_query_ids)
    if not (len(common_query_ids) == len(reference_query_ids) == len(candidate_query_ids)):
        raise AssertionError("Reference and candidate files must share same query ids after filtering no-answer queries")

    all_scores: Dict[str, float] = {}
    ordered_ids = sorted(common_query_ids, key=lambda x: str(x))
    sys_outputs = [filtered_candidate_dictionary[qid][0] for qid in ordered_ids]

    # refs_by_id is a list (per-hypothesis) of lists of reference strings (already normalized)
    refs_by_id: List[List[str]] = [filtered_reference_dictionary[qid] for qid in ordered_ids]

    # ---- BLEU-1..4 ----
    # Prefer NLTK for BLEU-n because it allows custom weights easily.
    if NLTK_AVAILABLE:
        list_of_references_for_nltk = [[ref.split() for ref in ref_list] for ref_list in refs_by_id]
        hypotheses_for_nltk = [hyp.split() for hyp in sys_outputs]
        smooth_fn = SmoothingFunction().method4
        for n in range(1, min(4, p_max_bleu_order) + 1):
            # weights: BLEU-n uses uniform weights across 1..n (commonly BLEU-2 uses (0.5,0.5,0,0))
            weights = tuple((1.0 / n if i < n else 0.0) for i in range(4))
            try:
                bleu_n = nltk_corpus_bleu(list_of_references_for_nltk, hypotheses_for_nltk, weights=weights, smoothing_function=smooth_fn)
                all_scores[f"bleu_{n}"] = float(bleu_n)  # NLTK returns 0..1
            except Exception as e:
                warnings.warn(f"NLTK BLEU-{n} computation failed: {e}")
                all_scores[f"bleu_{n}"] = 0.0
    else:
        # Fallback: compute sacrebleu overall BLEU (standard BLEU-4). Put that in bleu_4 and set others to None.
        try:
            bleu = sacrebleu.corpus_bleu(sys_outputs, refs_by_id, smooth_method="exp", force=True)
            all_scores["bleu_4"] = float(bleu.score) / 100.0
            # mark 1..3 as None since we couldn't compute them accurately
            for n in range(1, 4):
                all_scores[f"bleu_{n}"] = None
        except Exception as e:
            warnings.warn(f"sacrebleu BLEU computation failed: {e}")
            for n in range(1, 5):
                all_scores[f"bleu_{n}"] = None

    # ---- ROUGE-L (token-level, best ref per query) ----
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_sum = 0.0
    for qid in ordered_ids:
        cand = filtered_candidate_dictionary[qid][0]
        best = 0.0
        for ref in filtered_reference_dictionary[qid]:
            score = rscorer.score(ref, cand)["rougeL"].fmeasure
            if score > best:
                best = score
        rouge_sum += best
    all_scores["rouge_l"] = rouge_sum / (len(ordered_ids) or 1)

    # Add the simple binary F1 for presence/no-presence classification
    all_scores["answer_presence_F1"] = F1

    # ---- Semantic similarity (spaCy) ----
    semantic_similarity = 0.0
    if nlp is not None and filtered_reference_dictionary:
        total_sim = 0.0
        for key in filtered_reference_dictionary:
            cand_doc = nlp(filtered_candidate_dictionary[key][0])
            ref_list = filtered_reference_dictionary[key]
            # average similarity against all references for this query
            sum_sim = 0.0
            for ref in ref_list:
                try:
                    sum_sim += cand_doc.similarity(nlp(ref))
                except Exception:
                    # if spaCy small model lacks vectors similarity may be poor or raise; ignore
                    sum_sim += 0.0
            total_sim += (sum_sim / (len(ref_list) or 1))
        semantic_similarity = total_sim / len(filtered_reference_dictionary)
    else:
        semantic_similarity = 0.0
    all_scores["semantic_similarity"] = float(semantic_similarity)

    # ---- BERTScore integration (roberta-large) ----
    if BERTSCORE_AVAILABLE:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # We set idf=True for MS MARCO style scoring; catch exceptions
            scorer = BERTScorer(model_type="roberta-large", lang="en", device=device, idf=True, use_fast_tokenizer=True)
            # Prepare refs in the format accepted by scorer.score (list[list[str]])
            # refs_by_id already matches that shape
            P, R, F = scorer.score(sys_outputs, refs_by_id)
            # mean of tensors or arrays
            import numpy as _np
            all_scores["bertscore_p"] = float(_np.mean(_np.array(P)))
            all_scores["bertscore_r"] = float(_np.mean(_np.array(R)))
            all_scores["bertscore_f1"] = float(_np.mean(_np.array(F)))
        except Exception as e:
            warnings.warn(f"BERTScore computation failed: {e}")
            all_scores["bertscore_p"] = None
            all_scores["bertscore_r"] = None
            all_scores["bertscore_f1"] = None
    else:
        all_scores["bertscore_p"] = None
        all_scores["bertscore_r"] = None
        all_scores["bertscore_f1"] = None

    return all_scores


def main():
    if len(sys.argv) < 3:
        print("Usage: python evals/ms_marco_eval.py <path_to_reference_file> <path_to_candidate_file>", file=sys.stderr)
        sys.exit(1)

    path_to_reference_file = sys.argv[1]
    path_to_candidate_file = sys.argv[2]
    metrics = compute_metrics_from_files(path_to_reference_file, path_to_candidate_file, MAX_BLEU_ORDER)
    for metric in sorted(metrics):
        print("%s: %s" % (metric, metrics[metric]))


if __name__ == "__main__":
    main()
