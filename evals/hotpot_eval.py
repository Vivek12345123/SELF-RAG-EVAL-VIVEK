import sys
import json
import re
import string
from collections import Counter
import torch
import code_bert_score  # added for BERTScore
import numpy as _np

def normalize_answer(s):

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
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def update_sp(metrics, prediction, gold):
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

def compute_codebert_score(predictions, references, lang='python', device='cuda:0', batch_size=64, idf=True, use_fast_tokenizer=True):
    """
    Compute BERTScore (forced to 'roberta-large') and return mean_precision, mean_recall, mean_f1, mean_f3.
    Attempts canonical bert_score first; falls back to code_bert_score if needed.
    """
    # Accept either "cuda:0" or "cuda"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # try canonical bert_score implementation with explicit model_type
    try:
        from bert_score.scorer import BERTScorer
        # Build scorer forcing roberta-large
        scorer = BERTScorer(
            model_type="roberta-large",
            num_layers=None,
            batch_size=batch_size,
            nthreads=4,
            all_layers=False,
            idf=bool(idf),
            idf_sents=None,
            device=device,
            lang=None,
            rescale_with_baseline=False,
            baseline_path=None,
            use_fast_tokenizer=use_fast_tokenizer,
        )
        # compute idf on flattened refs if idf requested
        flat_refs = []
        if len(references) > 0 and isinstance(references[0], (list, tuple)):
            for rlist in references:
                flat_refs.extend(rlist)
        else:
            flat_refs = list(references)
        if idf and flat_refs:
            scorer.compute_idf(flat_refs)

        P_arr, R_arr, F_arr = scorer.score(predictions, references)
        meanP = float(_np.mean(_np.array(P_arr)))
        meanR = float(_np.mean(_np.array(R_arr)))
        meanF = float(_np.mean(_np.array(F_arr)))
        # F3 (beta=3)
        beta2 = 9.0
        meanF3 = (1.0 + beta2) * meanP * meanR / (beta2 * meanP + meanR) if (beta2 * meanP + meanR) > 0 else 0.0
        return meanP, meanR, meanF, meanF3

    except Exception as e1:
        # fallback: try code_bert_score (attempt to ask it to use roberta-large if supported)
        try:
            # try passing model_type first (some wrappers accept it)
            try:
                res = code_bert_score.score(
                    cands=predictions,
                    refs=references,
                    model_type="roberta-large",
                    device=device,
                    idf=idf,
                    batch_size=batch_size,
                    use_fast_tokenizer=use_fast_tokenizer,
                )
            except TypeError:
                # older wrapper signature: do not pass model_type
                res = code_bert_score.score(cands=predictions, refs=references, device=device)

            # expect at least (P,R,F)
            if len(res) >= 3:
                P_arr, R_arr, F_arr = res[0], res[1], res[2]
                meanP = float(_np.mean(_np.array(P_arr)))
                meanR = float(_np.mean(_np.array(R_arr)))
                meanF = float(_np.mean(_np.array(F_arr)))
                beta2 = 9.0
                meanF3 = (1.0 + beta2) * meanP * meanR / (beta2 * meanP + meanR) if (beta2 * meanP + meanR) > 0 else 0.0
                return meanP, meanR, meanF, meanF3
            else:
                raise RuntimeError("Unexpected return from code_bert_score.score(): %r" % (res,))
        except Exception as e2:
            raise RuntimeError("Failed to compute BERTScore with roberta-large. bert_score error: %s ; code_bert_score error: %s" % (str(e1), str(e2)))

def eval(prediction_file, gold_file, codebert_lang='python', device='cuda:0'):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    # For BERTScore
    codebert_preds, codebert_refs = [], []

    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
            pred_text = ""
        else:
            pred_text = prediction['answer'][cur_id]
        gold_text = dp['answer']
        codebert_preds.append(pred_text)
        codebert_refs.append(gold_text)

        if cur_id in prediction['answer']:
            em, prec, recall = update_answer(metrics, pred_text, gold_text)
        else:
            em, prec, recall = 0, 0, 0

        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(metrics, prediction['sp'][cur_id], dp['supporting_facts'])

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

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    # Compute CodeBERTScore (actually BERTScore with roberta-large)
    cb_prec, cb_recall, cb_f1, cb_f3 = compute_codebert_score(codebert_preds, codebert_refs, lang=codebert_lang, device=device)
    metrics['codebert_prec'] = cb_prec
    metrics['codebert_recall'] = cb_recall
    metrics['codebert_f1'] = cb_f1
    metrics['codebert_f3'] = cb_f3

    print(metrics)


if __name__ == '__main__':
    # Usage: python eval_script.py predictions.json gold.json [lang] [device]
    lang = sys.argv[3] if len(sys.argv) > 3 else 'python'
    device = sys.argv[4] if len(sys.argv) > 4 else 'cuda:0'
    eval(sys.argv[1], sys.argv[2], codebert_lang=lang, device=device)
