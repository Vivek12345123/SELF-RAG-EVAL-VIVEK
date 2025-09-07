# coding=utf-8
"""
Official evaluation script for Natural Questions (user-provided).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import json
import os
import pickle
from absl import app
from absl import flags
from absl import logging
import eval_utils as util
import six

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
  if y == 0:
    return 0
  else:
    return x / y


def score_long_answer(gold_label_list, pred_label):
  gold_has_answer = util.gold_has_long_answer(gold_label_list)

  pred_has_answer = pred_label and (
      not pred_label.long_answer_span.is_null_span())

  is_correct = False
  score = pred_label.long_score

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
      pred_label.yes_no_answer != 'none')

  is_correct = False
  score = pred_label.short_score

  if gold_has_answer and pred_has_answer:
    if pred_label.yes_no_answer != 'none':
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
  total_correct = 0
  total_has_pred = 0
  total_has_gold = 0

  for has_gold, _, _, _ in answer_stats:
    total_has_gold += has_gold

  max_recall = [0 for _ in targets]
  max_precision = [0 for _ in targets]
  max_scores = [None for _ in targets]

  scores_to_stats = OrderedDict()

  for has_gold, has_pred, is_correct, score in answer_stats:
    total_correct += is_correct
    total_has_pred += has_pred

    precision = safe_divide(total_correct, total_has_pred)
    recall = safe_divide(total_correct, total_has_gold)

    scores_to_stats[score] = [precision, recall]

  best_f1 = 0.0
  best_precision = 0.0
  best_recall = 0.0
  best_threshold = 0.0

  for threshold, (precision, recall) in six.iteritems(scores_to_stats):
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

  return ((best_f1, best_precision, best_recall, best_threshold),
          list(zip(targets, max_recall, max_precision, max_scores)))


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


def main(_):
  cache_path = os.path.join(os.path.dirname(FLAGS.gold_path), 'cache')
  if FLAGS.cache_gold_data and os.path.exists(cache_path):
    logging.info('Reading from cache: %s', format(cache_path))
    nq_gold_dict = pickle.load(open(cache_path, 'r'))
  else:
    nq_gold_dict = util.read_annotation(
        FLAGS.gold_path, n_threads=FLAGS.num_threads)
    if FLAGS.cache_gold_data:
      logging.info('Caching gold data for next time to: %s', format(cache_path))
      pickle.dump(nq_gold_dict, open(cache_path, 'w'))

  nq_pred_dict = util.read_prediction_json(FLAGS.predictions_path)

  long_answer_stats, short_answer_stats = score_answers(nq_gold_dict,
                                                        nq_pred_dict)

  # ---- BERTScore integration (added) ----
  try:
    import torch
    from bert_score.scorer import BERTScorer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Attempt to assemble candidate/ref pairs from available data.
    # The NQ util may or may not expose helper functions; try some common patterns
    candidates = []
    refs = []

    # If util exposes a helper to extract plain text of gold/pred pairs, prefer it.
    if hasattr(util, "get_gold_pred_text_pairs"):
      try:
        candidates, refs = util.get_gold_pred_text_pairs(nq_gold_dict, nq_pred_dict)
      except Exception:
        candidates, refs = [], []

    # Otherwise, try extracting short-answer text from pred/gold data structures if present.
    if not candidates:
      for ex_id in sorted(nq_gold_dict.keys()):
        gold_ann = nq_gold_dict[ex_id]
        pred = nq_pred_dict.get(ex_id)
        # gather possible gold short-answer strings
        gold_texts = []
        # common patterns (best-effort): look for 'short_answers_text' fields in gold or in a list of labels
        if isinstance(gold_ann, dict) and 'short_answers_text' in gold_ann:
          gold_texts = gold_ann.get('short_answers_text') or []
        elif isinstance(gold_ann, (list, tuple)):
          for g in gold_ann:
            if isinstance(g, dict) and 'short_answers_text' in g:
              gold_texts.extend(g.get('short_answers_text') or [])
        # fallback: try fields named 'short_answers' that may be list of dicts with 'text'
        if not gold_texts and isinstance(gold_ann, dict) and 'short_answers' in gold_ann:
          short_list = gold_ann.get('short_answers') or []
          for s in short_list:
            if isinstance(s, dict) and 'text' in s:
              gold_texts.append(s['text'])
        # obtain candidate text from prediction
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
          # if pred is an object with attributes (older util), try attribute access
          else:
            if hasattr(pred, 'short_answers_text') and getattr(pred, 'short_answers_text'):
              cand_text = getattr(pred, 'short_answers_text')[0]
            elif hasattr(pred, 'long_answer_text'):
              cand_text = getattr(pred, 'long_answer_text')

        # only add examples where both candidate and at least one gold ref text exist
        if cand_text and gold_texts:
          candidates.append(cand_text)
          refs.append(gold_texts)

    # If we accumulated pairs, compute BERTScore (otherwise skip)
    if candidates and refs:
      # Force roberta-large
      scorer = BERTScorer(model_type="roberta-large", lang='en', device=device, idf=True, use_fast_tokenizer=True)
      # compute idf on flattened reference pool
      flat_refs = []
      for ref_list in refs:
        flat_refs.extend(ref_list)
      if flat_refs:
        scorer.compute_idf(flat_refs)
      (P, R, F) = scorer.score(candidates, refs)
      import numpy as _np
      bert_mean_p = float(_np.mean(_np.array(P)))
      bert_mean_r = float(_np.mean(_np.array(R)))
      bert_mean_f = float(_np.mean(_np.array(F)))
      print("BERTScore (mean P/R/F): {:.6f} / {:.6f} / {:.6f}".format(bert_mean_p, bert_mean_r, bert_mean_f))
    else:
      print("BERTScore: no suitable text pairs extracted from gold/pred data; skipping BERTScore computation.")
  except Exception as e:
    # do not fail evaluation if BERTScore isn't available or if extraction failed
    print("Warning: BERTScore could not be computed:", str(e))
  # ---- end BERTScore integration ----

  if FLAGS.pretty_print:
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
  else:
    metrics = get_metrics_with_answer_stats(long_answer_stats,
                                            short_answer_stats)
    print(json.dumps(metrics))


if __name__ == '__main__':
  flags.mark_flag_as_required('gold_path')
  flags.mark_flag_as_required('predictions_path')
  app.run(main)
