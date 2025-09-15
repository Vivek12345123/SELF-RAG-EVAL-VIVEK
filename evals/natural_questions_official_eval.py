# coding=utf-8
"""
Official evaluation script for Natural Questions (user-provided).

This file preserves the original scoring behavior while fixing a few bugs:
 - pickle read/write in binary mode and using a concrete filename
 - preventing overwrite of identical score keys when computing PR curves
 - robust import of eval_utils
 - added run_eval(...) wrapper for programmatic use (keeps CLI behavior unchanged)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import json
import os
import pickle
import sys
from absl import app
from absl import flags
from absl import logging
import six

# Robust import for eval_utils (try multiple likely locations)
try:
  import eval_utils as util
except Exception:
  try:
    # If this module sits in a parent package, ensure repo root is on sys.path
    repo_root = os.path.dirname(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
    if repo_root not in sys.path:
      sys.path.insert(0, repo_root)
    import eval_utils as util
  except Exception:
    try:
      # Try evals.utils (if evals is a package)
      from evals import utils as util
    except Exception:
      try:
        # Try plain utils (common alternative)
        import utils as util
      except Exception as e:
        raise ImportError("Could not import eval_utils (tried eval_utils, evals.utils, utils). "
                          "Original error: {}".format(e))

flags.DEFINE_string(
    'gold_path', None, 'Path to the gzip JSON data. For '
    'multiple files, should be a glob '
    'pattern (e.g. "/path/to/files-*")')
flags.DEFINE_string('predictions_path', None, 'Path to prediction JSON.')
flags.DEFINE_bool(
    'cache_gold_data', False,
    'Whether to cache gold data in Pickle format to speed up '
    'multiple evaluations.')
flags.DEFINE_integer('num_threads', 10, 'Number of threads for reading.')
flags.DEFINE_bool('pretty_print', False, 'Whether to pretty print output.')

FLAGS = flags.FLAGS


def safe_divide(x, y):
  """Safe divide returning float, 0 on zero denominator."""
  try:
    if y == 0:
      return 0.0
    return float(x) / float(y)
  except Exception:
    return 0.0


def score_long_answer(gold_label_list, pred_label):
  gold_has_answer = util.gold_has_long_answer(gold_label_list)

  pred_has_answer = pred_label and (
      not pred_label.long_answer_span.is_null_span())

  is_correct = False
  score = pred_label.long_score if pred_label is not None else 0.0

  if gold_has_answer and pred_has_answer:
    for gold_label in gold_label_list:
      if gold_label.long_answer_span.is_null_span():
        continue

      if util.nonnull_span_equal(gold_label.long_answer_span,
                                 pred_label.long_answer_span):
        is_correct = True
        break

  return gold_has_answer, pred_has_answer, is_correct, score


def score_short_answer(gold_label_list, pred_label):
  gold_has_answer = util.gold_has_short_answer(gold_label_list)

  pred_has_answer = pred_label and (
      (not util.is_null_span_list(pred_label.short_answer_span_list)) or
      getattr(pred_label, 'yes_no_answer', 'none') != 'none')

  is_correct = False
  score = getattr(pred_label, 'short_score', 0.0)

  if gold_has_answer and pred_has_answer:
    if getattr(pred_label, 'yes_no_answer', 'none') != 'none':
      for gold_label in gold_label_list:
        if pred_label.yes_no_answer == gold_label.yes_no_answer:
          is_correct = True
          break
    else:
      for gold_label in gold_label_list:
        if util.span_set_equal(gold_label.short_answer_span_list,
                               pred_label.short_answer_span_list):
          is_correct = True
          break

  return gold_has_answer, pred_has_answer, is_correct, score


def score_answers(gold_annotation_dict, pred_dict):
  gold_id_set = set(gold_annotation_dict.keys())
  pred_id_set = set(pred_dict.keys())

  if gold_id_set.symmetric_difference(pred_id_set):
    raise ValueError('ERROR: the example ids in gold annotations and example '
                     'ids in the prediction are not equal.')

  long_answer_stats = []
  short_answer_stats = []

  for example_id in gold_id_set:
    gold = gold_annotation_dict[example_id]
    pred = pred_dict[example_id]

    long_answer_stats.append(score_long_answer(gold, pred))
    short_answer_stats.append(score_short_answer(gold, pred))

  # Sort by score descending as before (keeps original behavior)
  long_answer_stats.sort(key=lambda x: x[-1], reverse=True)
  short_answer_stats.sort(key=lambda x: x[-1], reverse=True)

  return long_answer_stats, short_answer_stats


def compute_f1(answer_stats, prefix=''):
  has_gold, has_pred, is_correct, _ = list(zip(*answer_stats))
  precision = safe_divide(sum(is_correct), sum(has_pred))
  recall = safe_divide(sum(is_correct), sum(has_gold))
  f1 = safe_divide(2 * precision * recall, precision + recall)

  return OrderedDict({
      prefix + 'n': len(answer_stats),
      prefix + 'f1': f1,
      prefix + 'precision': precision,
      prefix + 'recall': recall
  })


def compute_final_f1(long_answer_stats, short_answer_stats):
  scores = compute_f1(long_answer_stats, prefix='long-answer-')
  scores.update(compute_f1(short_answer_stats, prefix='short-answer-'))
  return scores


def compute_pr_curves(answer_stats, targets=None):
  """
  Compute PR curves and best-threshold selection.

  Fixed to preserve duplicate scores and ordering (no overwrite of identical float keys).
  """
  total_correct = 0
  total_has_pred = 0
  total_has_gold = 0

  for has_gold, _, _, _ in answer_stats:
    total_has_gold += has_gold

  # Prepare structures for targets
  max_recall = [0 for _ in targets] if targets else []
  max_precision = [0 for _ in targets] if targets else []
  max_scores = [None for _ in targets] if targets else []

  # Build a list of (score, precision, recall) in the order of answer_stats
  scores_list = []
  for has_gold, has_pred, is_correct, score in answer_stats:
    total_correct += is_correct
    total_has_pred += has_pred

    precision = safe_divide(total_correct, total_has_pred)
    recall = safe_divide(total_correct, total_has_gold)

    # append to list (preserves duplicates and order)
    scores_list.append((score, precision, recall))

  # Now compute best F1 and target recalls/precisions using the scores_list
  best_f1 = 0.0
  best_precision = 0.0
  best_recall = 0.0
  best_threshold = 0.0

  for threshold, precision, recall in scores_list:
    # update target tables
    if targets:
      for t, target in enumerate(targets):
        if precision >= target and recall > max_recall[t]:
          max_recall[t] = recall
          max_precision[t] = precision
          max_scores[t] = threshold

    f1 = safe_divide(2 * precision * recall, precision + recall)
    if f1 > best_f1:
      best_f1 = f1
      best_precision = precision
      best_recall = recall
      best_threshold = threshold

  pr_table = list(zip(targets if targets else [], max_recall, max_precision, max_scores))
  return ((best_f1, best_precision, best_recall, best_threshold), pr_table)


def print_r_at_p_table(answer_stats):
  opt_result, pr_table = compute_pr_curves(
      answer_stats, targets=[0.5, 0.75, 0.9])
  f1, precision, recall, threshold = opt_result
  print('Optimal threshold: {:.5}'.format(threshold))
  print(' F1     /  P      /  R')
  print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))
  for target, recall, precision, row in pr_table:
    print('R@P={}: {:.2%} (actual p={:.2%}, score threshold={:.4})'.format(
        target, recall, precision, row))


def get_metrics_as_dict(gold_path, prediction_path, num_threads=10):
  nq_gold_dict = util.read_annotation(gold_path, n_threads=num_threads)
  nq_pred_dict = util.read_prediction_json(prediction_path)
  long_answer_stats, short_answer_stats = score_answers(nq_gold_dict,
                                                        nq_pred_dict)

  return get_metrics_with_answer_stats(long_answer_stats, short_answer_stats)


def get_metrics_with_answer_stats(long_answer_stats, short_answer_stats):
  def _get_metric_dict(answer_stats, prefix=''):
    opt_result, pr_table = compute_pr_curves(
        answer_stats, targets=[0.5, 0.75, 0.9])
    f1, precision, recall, threshold = opt_result
    metrics = OrderedDict({
        'best-threshold-f1': f1,
        'best-threshold-precision': precision,
        'best-threshold-recall': recall,
        'best-threshold': threshold,
    })
    for target, recall, precision, _ in pr_table:
      metrics['recall-at-precision>={:.2}'.format(target)] = recall
      metrics['precision-at-precision>={:.2}'.format(target)] = precision

    return dict([(prefix + k, v) for k, v in six.iteritems(metrics)])

  metrics = _get_metric_dict(long_answer_stats, 'long-')
  metrics.update(_get_metric_dict(short_answer_stats, 'short-'))
  return metrics


def _compute_and_print_bertscore(nq_gold_dict, nq_pred_dict):
  """
  Best-effort extraction of (candidate, references) pairs and BERTScore computation.
  Preserves original heuristics but hardens device selection and exception handling.
  """
  try:
    import torch
    from bert_score.scorer import BERTScorer
  except Exception as e:
    print("Warning: BERTScore imports failed: {}".format(e))
    return None

  device = "cuda" if torch.cuda.is_available() else "cpu"

  candidates = []
  refs = []

  # If util exposes a helper to extract plain text pairs, prefer it.
  if hasattr(util, "get_gold_pred_text_pairs"):
    try:
      candidates, refs = util.get_gold_pred_text_pairs(nq_gold_dict, nq_pred_dict)
    except Exception:
      candidates, refs = [], []

  # Fallback extraction heuristics (preserve original behavior)
  if not candidates:
    for ex_id in sorted(nq_gold_dict.keys()):
      gold_ann = nq_gold_dict[ex_id]
      pred = nq_pred_dict.get(ex_id)
      gold_texts = []

      # try several known patterns for gold annotation structures
      if isinstance(gold_ann, dict) and 'short_answers_text' in gold_ann:
        gold_texts = gold_ann.get('short_answers_text') or []
      elif isinstance(gold_ann, (list, tuple)):
        for g in gold_ann:
          if isinstance(g, dict) and 'short_answers_text' in g:
            gold_texts.extend(g.get('short_answers_text') or [])
      if not gold_texts and isinstance(gold_ann, dict) and 'short_answers' in gold_ann:
        short_list = gold_ann.get('short_answers') or []
        for s in short_list:
          if isinstance(s, dict) and 'text' in s:
            gold_texts.append(s['text'])

      # extract candidate text from pred
      cand_text = None
      if pred:
        if isinstance(pred, dict):
          if 'short_answers_text' in pred and pred['short_answers_text']:
            cand_text = pred['short_answers_text'][0]
          elif 'long_answer_text' in pred:
            cand_text = pred['long_answer_text']
          elif 'short_answers' in pred and isinstance(pred['short_answers'], list) and pred['short_answers']:
            first = pred['short_answers'][0]
            if isinstance(first, dict) and 'text' in first:
              cand_text = first['text']
        else:
          # object-like pred (older format)
          if hasattr(pred, 'short_answers_text') and getattr(pred, 'short_answers_text'):
            cand_text = getattr(pred, 'short_answers_text')[0]
          elif hasattr(pred, 'long_answer_text'):
            cand_text = getattr(pred, 'long_answer_text')

      # only collect when both cand_text and at least one gold_text exist
      if cand_text and gold_texts:
        candidates.append(cand_text)
        refs.append(gold_texts)

  if not candidates:
    print("BERTScore: no suitable text pairs extracted from gold/pred data; skipping BERTScore computation.")
    return None

  try:
    # Build scorer forcing roberta-large as before
    scorer = BERTScorer(model_type="roberta-large", lang='en', device=device, idf=True, use_fast_tokenizer=True)
    flat_refs = []
    for ref_list in refs:
      flat_refs.extend(ref_list)
    if flat_refs:
      try:
        scorer.compute_idf(flat_refs)
      except Exception:
        # compute_idf can fail for odd inputs; ignore and continue
        pass

    (P, R, F) = scorer.score(candidates, refs)
    import numpy as _np
    bert_mean_p = float(_np.mean(_np.array(P)))
    bert_mean_r = float(_np.mean(_np.array(R)))
    bert_mean_f = float(_np.mean(_np.array(F)))
    print("BERTScore (mean P/R/F): {:.6f} / {:.6f} / {:.6f}".format(bert_mean_p, bert_mean_r, bert_mean_f))
    return (bert_mean_p, bert_mean_r, bert_mean_f)
  except Exception as e:
    print("Warning: BERTScore computation failed: {}".format(e))
    return None


def run_eval(gold_path, predictions_path, cache_gold_data=False, num_threads=10, pretty_print=False):
  """
  Programmatic entrypoint for the evaluator. Returns metrics dict (same as CLI JSON output)
  and prints BERTScore & tables if pretty_print=True (CLI-style).
  """
  # Prepare cache path (directory + concrete filename)
  cache_dir = os.path.join(os.path.dirname(gold_path), 'cache')
  os.makedirs(cache_dir, exist_ok=True)
  cache_path = os.path.join(cache_dir, 'nq_gold_cache.pkl')

  if cache_gold_data and os.path.exists(cache_path):
    logging.info('Reading gold data from cache: %s', cache_path)
    with open(cache_path, 'rb') as _f:
      nq_gold_dict = pickle.load(_f)
  else:
    nq_gold_dict = util.read_annotation(gold_path, n_threads=num_threads)
    if cache_gold_data:
      logging.info('Caching gold data for next time to: %s', cache_path)
      with open(cache_path, 'wb') as _f:
        pickle.dump(nq_gold_dict, _f)

  nq_pred_dict = util.read_prediction_json(predictions_path)

  long_answer_stats, short_answer_stats = score_answers(nq_gold_dict, nq_pred_dict)

  # BERTScore best-effort (printing as before)
  try:
    _ = _compute_and_print_bertscore(nq_gold_dict, nq_pred_dict)
  except Exception as e:
    # Do not fail evaluation if BERTScore fails
    print("Warning: BERTScore computation wrapper raised an error: {}".format(e))

  if pretty_print:
    print('*' * 20)
    print('LONG ANSWER R@P TABLE:')
    print_r_at_p_table(long_answer_stats)
    print('*' * 20)
    print('SHORT ANSWER R@P TABLE:')
    print_r_at_p_table(short_answer_stats)

    scores = compute_final_f1(long_answer_stats, short_answer_stats)
    print('*' * 20)
    print('METRICS IGNORING SCORES (n={}):'.format(scores['long-answer-n']))
    print('              F1     /  P      /  R')
    print('Long answer  {: >7.2%} / {: >7.2%} / {: >7.2%}'.format(
        scores['long-answer-f1'], scores['long-answer-precision'],
        scores['long-answer-recall']))
    print('Short answer {: >7.2%} / {: >7.2%} / {: >7.2%}'.format(
        scores['short-answer-f1'], scores['short-answer-precision'],
        scores['short-answer-recall']))
    # For CLI parity, also return the metrics dict
    return compute_final_f1(long_answer_stats, short_answer_stats)
  else:
    metrics = get_metrics_with_answer_stats(long_answer_stats, short_answer_stats)
    # Print JSON exactly as original CLI branch did
    print(json.dumps(metrics))
    return metrics


def main(_):
  # Use absl flags when running as CLI; keep previous CLI semantics.
  metrics = run_eval(FLAGS.gold_path, FLAGS.predictions_path,
                     cache_gold_data=FLAGS.cache_gold_data,
                     num_threads=FLAGS.num_threads,
                     pretty_print=FLAGS.pretty_print)
  # run_eval already prints CLI output where appropriate


if __name__ == '__main__':
  flags.mark_flag_as_required('gold_path')
  flags.mark_flag_as_required('predictions_path')
  app.run(main)
