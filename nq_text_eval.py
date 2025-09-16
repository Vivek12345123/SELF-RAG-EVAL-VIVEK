#!/usr/bin/env python3
"""
Natural Questions Evaluation Wrapper for Text Generation Models
Provides F1, EM, and BERTScore metrics while handling the format mismatch
"""

import json
import numpy as np
from collections import Counter, namedtuple
from typing import Dict, List, Any, Optional
import re

# Import BERTScore if available
try:
    from bert_score import BERTScorer
    import torch
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

class NQTextGenerationEvaluator:
    """Evaluator for Natural Questions using text generation instead of span prediction"""
    
    def __init__(self, use_bert_score=True):
        self.bert_scorer = None
        if BERT_SCORE_AVAILABLE and use_bert_score:
            try:
                self.bert_scorer = BERTScorer(
                    model_type="roberta-large",
                    lang="en",
                    idf=True,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    use_fast_tokenizer=True
                )
                print("BERTScore initialized with roberta-large")
            except Exception as e:
                print(f"Warning: Could not initialize BERTScorer: {e}")
    
    def normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison (same as SQuAD normalization)"""
        import string
        
        if not text:
            return ""
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_punc(lower(text)))
    
    def compute_f1_score(self, prediction: str, reference: str) -> float:
        """Compute F1 score between prediction and reference"""
        pred_tokens = self.normalize_answer(prediction).split()
        ref_tokens = self.normalize_answer(reference).split()

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_ct = Counter(pred_tokens)
        ref_ct = Counter(ref_tokens)
        common = pred_ct & ref_ct
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def compute_exact_match(self, prediction: str, reference: str) -> bool:
        """Compute exact match between normalized prediction and reference"""
        return self.normalize_answer(prediction) == self.normalize_answer(reference)
    
    def extract_gold_answers(self, nq_example) -> List[str]:
        """Extract all possible gold answers from NQ example"""
        answers = []
        
        if 'annotations' in nq_example and nq_example['annotations']:
            for annotation in nq_example['annotations']:
                # Extract short answers
                if 'short_answers' in annotation and annotation['short_answers']:
                    for short_ans in annotation['short_answers']:
                        if isinstance(short_ans, dict):
                            # Handle different possible formats
                            if 'text' in short_ans and short_ans['text']:
                                if isinstance(short_ans['text'], list):
                                    answers.extend([t for t in short_ans['text'] if t])
                                else:
                                    answers.append(short_ans['text'])
                        elif isinstance(short_ans, str) and short_ans.strip():
                            answers.append(short_ans)
                
                # Extract yes/no answers
                if 'yes_no_answer' in annotation:
                    yn_answer = annotation['yes_no_answer']
                    if yn_answer and yn_answer != 'NONE':
                        answers.append(yn_answer.lower())
                
                # Extract long answers as fallback
                if not answers and 'long_answer' in annotation:
                    long_ans = annotation['long_answer']
                    if isinstance(long_ans, dict) and 'candidate_text' in long_ans:
                        if long_ans['candidate_text']:
                            # Take first sentence of long answer as short answer
                            sentences = long_ans['candidate_text'].split('. ')
                            if sentences:
                                answers.append(sentences[0])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_answers = []
        for ans in answers:
            ans_normalized = self.normalize_answer(ans)
            if ans_normalized and ans_normalized not in seen:
                seen.add(ans_normalized)
                unique_answers.append(ans)
        
        return unique_answers if unique_answers else [""]
    
    def evaluate_predictions(self, predictions: List[str], gold_examples: List[dict]) -> Dict[str, float]:
        """
        Evaluate text predictions against NQ gold examples
        
        Args:
            predictions: List of generated answer strings
            gold_examples: List of NQ dataset examples
            
        Returns:
            Dictionary with metrics: exact_match, f1, long_f1, short_f1, bert_* metrics
        """
        
        if len(predictions) != len(gold_examples):
            raise ValueError(f"Predictions ({len(predictions)}) and gold examples ({len(gold_examples)}) must have same length")
        
        # Extract all gold answers
        all_gold_answers = []
        for example in gold_examples:
            gold_answers = self.extract_gold_answers(example)
            all_gold_answers.append(gold_answers)
        
        # Compute metrics
        exact_matches = []
        f1_scores = []
        has_answer_predictions = []
        has_answer_gold = []
        
        for pred, gold_list in zip(predictions, all_gold_answers):
            # Check if gold has any non-empty answers
            has_gold_answer = any(ans and ans.strip() for ans in gold_list if ans != "")
            has_answer_gold.append(has_gold_answer)
            
            # Check if prediction is non-empty
            has_pred_answer = bool(pred and pred.strip())
            has_answer_predictions.append(has_pred_answer)
            
            if has_gold_answer and gold_list != [""]:
                # Exact match against any gold answer
                em = any(self.compute_exact_match(pred, gold) for gold in gold_list if gold)
                exact_matches.append(1.0 if em else 0.0)
                
                # F1 against best matching gold answer
                best_f1 = max((self.compute_f1_score(pred, gold) for gold in gold_list if gold), default=0.0)
                f1_scores.append(best_f1)
            else:
                # No gold answer - only correct if prediction is also empty
                exact_matches.append(1.0 if not has_pred_answer else 0.0)
                f1_scores.append(1.0 if not has_pred_answer else 0.0)
        
        # Basic metrics
        metrics = {
            'exact_match': np.mean(exact_matches),
            'f1': np.mean(f1_scores),
            'short_answer_exact_match': np.mean(exact_matches),  # Same as overall for text generation
            'short_answer_f1': np.mean(f1_scores),
            'long_answer_exact_match': 0.0,  # Not applicable for text generation
            'long_answer_f1': 0.0,
            'has_answer_accuracy': np.mean([
                1.0 if (hg and hp) or (not hg and not hp) else 0.0 
                for hg, hp in zip(has_answer_gold, has_answer_predictions)
            ])
        }
        
        # Precision/Recall breakdown
        if has_answer_gold:
            answerable_preds = [p for p, h in zip(predictions, has_answer_gold) if h]
            answerable_gold = [g for g, h in zip(all_gold_answers, has_answer_gold) if h]
            unanswerable_preds = [p for p, h in zip(predictions, has_answer_gold) if not h]
            
            if answerable_preds:
                answerable_em = []
                answerable_f1 = []
                for pred, gold_list in zip(answerable_preds, answerable_gold):
                    em = any(self.compute_exact_match(pred, gold) for gold in gold_list if gold)
                    answerable_em.append(1.0 if em else 0.0)
                    best_f1 = max((self.compute_f1_score(pred, gold) for gold in gold_list if gold), default=0.0)
                    answerable_f1.append(best_f1)
                
                metrics['HasAns_exact'] = np.mean(answerable_em)
                metrics['HasAns_f1'] = np.mean(answerable_f1)
            
            if unanswerable_preds:
                # For unanswerable questions, correct if model gives empty answer
                no_ans_accuracy = np.mean([1.0 if not (p and p.strip()) else 0.0 for p in unanswerable_preds])
                metrics['NoAns_exact'] = no_ans_accuracy
                metrics['NoAns_f1'] = no_ans_accuracy
        
        # BERTScore computation
        if self.bert_scorer:
            bert_metrics = self._compute_bert_score(predictions, all_gold_answers)
            metrics.update(bert_metrics)
        
        # Add summary stats
        metrics['total_examples'] = len(predictions)
        metrics['answerable_examples'] = sum(has_answer_gold)
        metrics['answered_examples'] = sum(has_answer_predictions)
        
        return metrics
    
    def _compute_bert_score(self, predictions: List[str], gold_answers_lists: List[List[str]]) -> Dict[str, float]:
        """Compute BERTScore metrics"""
        
        # Prepare candidates and references for BERTScore
        candidates = []
        references = []
        
        for pred, gold_list in zip(predictions, gold_answers_lists):
            # Only include examples with non-empty gold answers
            if gold_list and any(g and g.strip() for g in gold_list if g != ""):
                candidates.append(pred if pred else "")
                # Use all gold answers as references
                valid_golds = [g for g in gold_list if g and g.strip() and g != ""]
                references.append(valid_golds if valid_golds else [""])
        
        if not candidates:
            return {
                'bert_precision': 0.0,
                'bert_recall': 0.0,
                'bert_f1': 0.0
            }
        
        try:
            P, R, F = self.bert_scorer.score(candidates, references)
            return {
                'bert_precision': float(P.mean()),
                'bert_recall': float(R.mean()),
                'bert_f1': float(F.mean())
            }
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            return {
                'bert_precision': 0.0,
                'bert_recall': 0.0,
                'bert_f1': 0.0
            }
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a readable format"""
        print("\n" + "="*50)
        print("Natural Questions Evaluation Results")
        print("="*50)
        
        # Main metrics
        print(f"Overall Exact Match: {metrics['exact_match']:.4f}")
        print(f"Overall F1: {metrics['f1']:.4f}")
        
        if 'HasAns_exact' in metrics:
            print(f"Has Answer EM: {metrics['HasAns_exact']:.4f}")
            print(f"Has Answer F1: {metrics['HasAns_f1']:.4f}")
        
        if 'NoAns_exact' in metrics:
            print(f"No Answer EM: {metrics['NoAns_exact']:.4f}")
            print(f"No Answer F1: {metrics['NoAns_f1']:.4f}")
        
        # BERTScore
        if 'bert_f1' in metrics and metrics['bert_f1'] > 0:
            print(f"BERTScore Precision: {metrics['bert_precision']:.4f}")
            print(f"BERTScore Recall: {metrics['bert_recall']:.4f}")
            print(f"BERTScore F1: {metrics['bert_f1']:.4f}")
        
        # Summary
        print(f"\nTotal Examples: {metrics['total_examples']}")
        print(f"Answerable Examples: {metrics['answerable_examples']}")
        print(f"Model Answered: {metrics['answered_examples']}")


# Integration function for your Self-RAG evaluator
def evaluate_nq_with_official_metrics(predictions: List[str], nq_dataset, use_bert_score=True) -> Dict[str, float]:
    """
    Convenience function to evaluate NQ predictions with official-style metrics
    
    Args:
        predictions: List of generated text answers
        nq_dataset: HuggingFace dataset object with NQ examples
        use_bert_score: Whether to compute BERTScore
    
    Returns:
        Dictionary with all metrics
    """
    evaluator = NQTextGenerationEvaluator(use_bert_score=use_bert_score)
    
    # Convert dataset to list of examples
    gold_examples = [example for example in nq_dataset]
    
    metrics = evaluator.evaluate_predictions(predictions, gold_examples)
    evaluator.print_metrics(metrics)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("Natural Questions Text Generation Evaluator")
    print("Use this with your Self-RAG runner by calling:")
    print("metrics = evaluate_nq_with_official_metrics(predicted_answers, nq_dataset)")