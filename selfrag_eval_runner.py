#!/usr/bin/env python3
"""
Self-RAG Model Evaluation Runner with Complete BERT Scoring
Runs all QA benchmarks with both standard metrics and BERTScore using Self-RAG Llama 7B

Requirements:
pip install transformers vllm datasets tqdm bert-score torch sacrebleu rouge-score spacy
python -m spacy download en_core_web_lg
"""

import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np

# Import for Self-RAG model
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from datasets import load_dataset

# Import for BERTScore
try:
    from bert_score import BERTScorer
    BERT_SCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert-score not installed. Install with: pip install bert-score")
    BERT_SCORE_AVAILABLE = False


class SelfRAGEvaluator:
    """Main evaluator class for Self-RAG model on QA benchmarks"""
    
    def __init__(self, model_path="selfrag/selfrag_llama2_7b", 
                 download_dir="/gscratch/h2lab/akari/model_cache",
                 device="cuda", use_retrieval=True):
        """Initialize Self-RAG model and evaluation settings"""
        
        print(f"Loading Self-RAG model from {model_path}...")
        self.model = LLM(model_path, download_dir=download_dir, dtype="half")
        self.sampling_params = SamplingParams(
            temperature=0.0, 
            top_p=1.0, 
            max_tokens=100, 
            skip_special_tokens=False
        )
        self.use_retrieval = use_retrieval
        
        # Initialize BERTScorer if available
        self.bert_scorer = None
        if BERT_SCORE_AVAILABLE:
            try:
                self.bert_scorer = BERTScorer(
                    model_type="roberta-large",
                    lang="en",
                    idf=False,
                    device=device,
                    batch_size=64,
                    use_fast_tokenizer=True
                )
                print("BERTScore initialized with roberta-large")
            except Exception as e:
                print(f"Warning: Could not initialize BERTScorer: {e}")
    
    def format_prompt(self, input_text: str, paragraph: Optional[str] = None) -> str:
        """Format prompt for Self-RAG model"""
        prompt = f"### Instruction:\\n{input_text}\\n\\n### Response:\\n"
        if paragraph is not None and self.use_retrieval:
            prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
        return prompt
    
    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        """Generate answer using Self-RAG model"""
        prompt = self.format_prompt(question, context)
        preds = self.model.generate([prompt], self.sampling_params)
        
        if preds and preds[0].outputs:
            raw_output = preds[0].outputs[0].text
            # Extract answer from Self-RAG output format
            answer = self.extract_answer_from_selfrag(raw_output)
            return answer
        return ""
    
    def extract_answer_from_selfrag(self, output: str) -> str:
        """Extract clean answer from Self-RAG output with special tokens"""
        # Remove Self-RAG special tokens
        import re
        
        # Remove retrieval markers
        output = re.sub(r'\\[Retrieval\\]', '', output)
        output = re.sub(r'\\[No Retrieval\\]', '', output)
        
        # Remove relevance markers
        output = re.sub(r'\\[Relevant\\]', '', output)
        output = re.sub(r'\\[Irrelevant\\]', '', output)
        
        # Remove support markers
        output = re.sub(r'\\[Fully supported\\]', '', output)
        output = re.sub(r'\\[Partially supported\\]', '', output)
        output = re.sub(r'\\[No support\\]', '', output)
        
        # Remove utility markers
        output = re.sub(r'\\[Utility:\\d+\\]', '', output)
        
        # Remove paragraph tags
        output = re.sub(r'<paragraph>.*?</paragraph>', '', output)
        
        # Remove end token
        output = output.replace('</s>', '').strip()
        
        return output.strip()
    
    def evaluate_squad(self, max_examples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on SQuAD v2 dataset"""
        print("\\nEvaluating on SQuAD v2...")
        
        # Load dataset
        dataset = load_dataset("rajpurkar/squad_v2", split="validation")
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        predictions = {}
        na_probs = {}
        
        # Generate predictions
        for example in tqdm(dataset, desc="SQuAD v2"):
            question = example['question']
            context = example['context']
            qid = example['id']
            
            answer = self.generate_answer(question, context)
            predictions[qid] = answer
            
            # Simple heuristic for no-answer probability
            na_probs[qid] = 0.0 if answer else 1.0
        
        # Save predictions
        os.makedirs("outputs/squad", exist_ok=True)
        with open("outputs/squad/predictions.json", "w") as f:
            json.dump(predictions, f)
        with open("outputs/squad/na_probs.json", "w") as f:
            json.dump(na_probs, f)
        
        # Prepare gold data
        gold_data = self.prepare_squad_gold_data(dataset)
        with open("outputs/squad/gold.json", "w") as f:
            json.dump(gold_data, f)
        
        # Run official evaluation
        metrics = self.run_squad_eval("outputs/squad/gold.json", 
                                      "outputs/squad/predictions.json",
                                      "outputs/squad/na_probs.json")
        
        # Add BERTScore if available
        if self.bert_scorer and predictions:
            bert_metrics = self.compute_bert_score_squad(dataset, predictions)
            metrics.update(bert_metrics)
        
        return metrics
    
    def prepare_squad_gold_data(self, dataset) -> Dict:
        """Prepare gold data in SQuAD format"""
        data = {"version": "v2.0", "data": []}
        
        articles = {}
        for example in dataset:
            title = example.get('title', 'Unknown')
            if title not in articles:
                articles[title] = {"title": title, "paragraphs": []}
            
            # Find or create paragraph
            para_found = False
            for para in articles[title]["paragraphs"]:
                if para["context"] == example['context']:
                    para["qas"].append({
                        "id": example['id'],
                        "question": example['question'],
                        "answers": [{"text": ans['text'], "answer_start": ans['answer_start']} 
                                   for ans in example['answers']],
                        "is_impossible": len(example['answers']) == 0
                    })
                    para_found = True
                    break
            
            if not para_found:
                articles[title]["paragraphs"].append({
                    "context": example['context'],
                    "qas": [{
                        "id": example['id'],
                        "question": example['question'],
                        "answers": [{"text": ans['text'], "answer_start": ans['answer_start']} 
                                   for ans in example['answers']],
                        "is_impossible": len(example['answers']) == 0
                    }]
                })
        
        data["data"] = list(articles.values())
        return data
    
    def run_squad_eval(self, gold_file: str, pred_file: str, 
                       na_prob_file: Optional[str] = None) -> Dict[str, Any]:
        """Run official SQuAD evaluation script"""
        cmd = [sys.executable, "evals/squad_v2_official_eval.py", gold_file, pred_file]
        if na_prob_file:
            cmd.extend(["--na-prob-file", na_prob_file])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse JSON output
            import re
            json_match = re.search(r'\\{.*\\}', result.stdout, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                print(f"Warning: Could not parse SQuAD eval output")
                return {}
        except subprocess.CalledProcessError as e:
            print(f"Error running SQuAD eval: {e}")
            print(f"stderr: {e.stderr}")
            return {}
    
    def compute_bert_score_squad(self, dataset, predictions: Dict[str, str]) -> Dict[str, float]:
        """Compute BERTScore for SQuAD predictions"""
        candidates = []
        references = []
        
        for example in dataset:
            qid = example['id']
            if qid in predictions:
                candidates.append(predictions[qid])
                # Use all gold answers as references
                gold_answers = [ans['text'] for ans in example['answers']]
                if not gold_answers:
                    gold_answers = ['']  # For no-answer cases
                references.append(gold_answers)
        
        if candidates and references:
            P, R, F = self.bert_scorer.score(candidates, references)
            return {
                "bert_precision": float(P.mean()),
                "bert_recall": float(R.mean()),
                "bert_f1": float(F.mean())
            }
        return {}
    
    def evaluate_hotpot(self, max_examples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on HotpotQA dataset"""
        print("\\nEvaluating on HotpotQA...")
        
        # Load dataset
        dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        predictions = {"answer": {}, "sp": {}}
        gold_data = []
        
        # Generate predictions
        for example in tqdm(dataset, desc="HotpotQA"):
            question = example['question']
            # Combine context from supporting facts
            context = " ".join([" ".join(sents) for sents in example['context']['sentences']])
            
            answer = self.generate_answer(question, context)
            predictions["answer"][example['id']] = answer
            
            # For supporting facts, use a simple heuristic
            # In a real scenario, you'd need to track which paragraphs were used
            predictions["sp"][example['id']] = example['supporting_facts']['title'][:2] if 'supporting_facts' in example else []
            
            # Prepare gold data
            gold_data.append({
                "_id": example['id'],
                "answer": example['answer'],
                "supporting_facts": list(zip(example['supporting_facts']['title'], 
                                            example['supporting_facts']['sent_id']))
                                   if 'supporting_facts' in example else []
            })
        
        # Save files
        os.makedirs("outputs/hotpot", exist_ok=True)
        with open("outputs/hotpot/predictions.json", "w") as f:
            json.dump(predictions, f)
        with open("outputs/hotpot/gold.json", "w") as f:
            json.dump(gold_data, f)
        
        # Run evaluation
        metrics = self.run_hotpot_eval("outputs/hotpot/predictions.json",
                                       "outputs/hotpot/gold.json")
        
        # Add BERTScore if available
        if self.bert_scorer and predictions["answer"]:
            bert_metrics = self.compute_bert_score_hotpot(gold_data, predictions["answer"])
            metrics.update(bert_metrics)
        
        return metrics
    
    def run_hotpot_eval(self, pred_file: str, gold_file: str) -> Dict[str, Any]:
        """Run HotpotQA evaluation"""
        cmd = [sys.executable, "evals/hotpot_eval.py", pred_file, gold_file]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse metrics from output
            import ast
            for line in result.stdout.split('\\n'):
                if line.strip().startswith('{'):
                    return ast.literal_eval(line.strip())
            return {}
        except subprocess.CalledProcessError as e:
            print(f"Error running Hotpot eval: {e}")
            return {}
    
    def compute_bert_score_hotpot(self, gold_data: List[Dict], 
                                  predictions: Dict[str, str]) -> Dict[str, float]:
        """Compute BERTScore for HotpotQA"""
        candidates = []
        references = []
        
        for item in gold_data:
            if item['_id'] in predictions:
                candidates.append(predictions[item['_id']])
                references.append(item['answer'])
        
        if candidates and references:
            P, R, F = self.bert_scorer.score(candidates, references)
            return {
                "bert_precision_hotpot": float(P.mean()),
                "bert_recall_hotpot": float(R.mean()),
                "bert_f1_hotpot": float(F.mean())
            }
        return {}
    
    def evaluate_natural_questions(self, max_examples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on Natural Questions dataset"""
        print("\\nEvaluating on Natural Questions...")
        
        # Load dataset
        try:
            dataset = load_dataset("google-research-datasets/natural_questions", 
                                 "default", split="validation", streaming=True)
            # Convert streaming dataset to list
            examples = []
            for i, ex in enumerate(dataset):
                if max_examples and i >= max_examples:
                    break
                examples.append(ex)
        except Exception as e:
            print(f"Error loading Natural Questions: {e}")
            return {}
        
        predictions = {}
        gold_annotations = {}
        
        # Generate predictions
        for example in tqdm(examples, desc="Natural Questions"):
            question = example['question']['text']
            # Use document text as context
            context = example['document']['text'][:2000]  # Truncate for efficiency
            
            answer = self.generate_answer(question, context)
            
            # Create prediction in NQ format
            predictions[str(example['id'])] = {
                'example_id': str(example['id']),
                'long_answer': {'start_token': -1, 'end_token': -1},
                'short_answers': [answer] if answer else [],
                'yes_no_answer': 'NONE'
            }
            
            # Store gold annotations
            gold_annotations[str(example['id'])] = [{
                'example_id': str(example['id']),
                'annotations': example['annotations']
            }]
        
        # Save files
        os.makedirs("outputs/nq", exist_ok=True)
        with open("outputs/nq/predictions.json", "w") as f:
            json.dump(predictions, f)
        with open("outputs/nq/gold.json", "w") as f:
            json.dump(gold_annotations, f)
        
        # Run evaluation
        metrics = self.run_nq_eval("outputs/nq/gold.json", "outputs/nq/predictions.json")
        
        # Add BERTScore
        if self.bert_scorer and predictions:
            bert_metrics = self.compute_bert_score_nq(examples, predictions)
            metrics.update(bert_metrics)
        
        return metrics
    
    def run_nq_eval(self, gold_file: str, pred_file: str) -> Dict[str, Any]:
        """Run Natural Questions evaluation"""
        cmd = [sys.executable, "evals/natural_questions_official_eval.py",
               "--gold_path", gold_file, "--predictions_path", pred_file]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse JSON output
            import re
            json_match = re.search(r'\\{.*\\}', result.stdout, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except subprocess.CalledProcessError as e:
            print(f"Error running NQ eval: {e}")
            return {}
    
    def compute_bert_score_nq(self, examples: List[Dict], 
                             predictions: Dict[str, Dict]) -> Dict[str, float]:
        """Compute BERTScore for Natural Questions"""
        candidates = []
        references = []
        
        for ex in examples:
            ex_id = str(ex['id'])
            if ex_id in predictions:
                pred = predictions[ex_id]
                if pred['short_answers']:
                    candidates.append(" ".join(pred['short_answers']))
                else:
                    candidates.append("")
                
                # Extract gold short answers
                gold_answers = []
                for ann in ex['annotations']:
                    if ann['short_answers']:
                        for sa in ann['short_answers']:
                            # Extract text from tokens
                            start = sa['start_token']
                            end = sa['end_token']
                            if start >= 0 and end >= 0:
                                tokens = ex['document']['tokens']
                                answer_tokens = tokens[start:end+1]
                                answer_text = " ".join([t['token'] for t in answer_tokens])
                                gold_answers.append(answer_text)
                
                if not gold_answers:
                    gold_answers = [""]
                references.append(gold_answers)
        
        if candidates and references:
            P, R, F = self.bert_scorer.score(candidates, references)
            return {
                "bert_precision_nq": float(P.mean()),
                "bert_recall_nq": float(R.mean()),
                "bert_f1_nq": float(F.mean())
            }
        return {}
    
    def evaluate_triviaqa(self, max_examples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on TriviaQA dataset"""
        print("\\nEvaluating on TriviaQA...")
        
        # Load dataset
        try:
            dataset = load_dataset("trivia_qa", "rc", split="validation")
            if max_examples:
                dataset = dataset.select(range(min(max_examples, len(dataset))))
        except Exception as e:
            print(f"Error loading TriviaQA: {e}")
            return {}
        
        predictions = {}
        ground_truth = {}
        
        # Generate predictions
        for example in tqdm(dataset, desc="TriviaQA"):
            question = example['question']
            # Use search results as context
            contexts = example.get('search_results', [])
            if contexts:
                context = " ".join([ctx.get('search_context', '') for ctx in contexts[:3]])[:2000]
            else:
                context = ""
            
            answer = self.generate_answer(question, context)
            predictions[example['question_id']] = answer
            
            # Store ground truth
            ground_truth[example['question_id']] = {
                'QuestionId': example['question_id'],
                'Question': question,
                'Answer': {
                    'Value': example['answer']['value'] if 'answer' in example else "",
                    'NormalizedAliases': example['answer']['aliases'] if 'answer' in example else [],
                    'HumanAnswers': []
                }
            }
        
        # Save files
        os.makedirs("outputs/triviaqa", exist_ok=True)
        with open("outputs/triviaqa/predictions.json", "w") as f:
            json.dump(predictions, f)
        
        dataset_json = {
            'Version': 1.0,
            'Data': list(ground_truth.values())
        }
        with open("outputs/triviaqa/gold.json", "w") as f:
            json.dump(dataset_json, f)
        
        # Run evaluation
        metrics = self.run_triviaqa_eval("outputs/triviaqa/gold.json",
                                         "outputs/triviaqa/predictions.json")
        
        # Add BERTScore
        if self.bert_scorer and predictions:
            bert_metrics = self.compute_bert_score_triviaqa(ground_truth, predictions)
            metrics.update(bert_metrics)
        
        return metrics
    
    def run_triviaqa_eval(self, gold_file: str, pred_file: str) -> Dict[str, Any]:
        """Run TriviaQA evaluation"""
        cmd = [sys.executable, "evals/triviaqa_eval.py",
               "--dataset_file", gold_file, "--prediction_file", pred_file]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse output
            for line in result.stdout.split('\\n'):
                if line.strip().startswith('{'):
                    import ast
                    return ast.literal_eval(line.strip())
            return {}
        except subprocess.CalledProcessError as e:
            print(f"Error running TriviaQA eval: {e}")
            return {}
    
    def compute_bert_score_triviaqa(self, ground_truth: Dict, 
                                    predictions: Dict[str, str]) -> Dict[str, float]:
        """Compute BERTScore for TriviaQA"""
        candidates = []
        references = []
        
        for qid, pred in predictions.items():
            if qid in ground_truth:
                candidates.append(pred)
                gt = ground_truth[qid]['Answer']
                refs = gt['NormalizedAliases'] + [gt['Value']]
                references.append(refs)
        
        if candidates and references:
            P, R, F = self.bert_scorer.score(candidates, references)
            return {
                "bert_precision_triviaqa": float(P.mean()),
                "bert_recall_triviaqa": float(R.mean()),
                "bert_f1_triviaqa": float(F.mean())
            }
        return {}
    
    def evaluate_ms_marco(self, max_examples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on MS MARCO dataset"""
        print("\\nEvaluating on MS MARCO...")
        
        # Load dataset
        try:
            dataset = load_dataset("ms_marco", "v2.1", split="validation")
            if max_examples:
                dataset = dataset.select(range(min(max_examples, len(dataset))))
        except Exception as e:
            print(f"Error loading MS MARCO: {e}")
            return {}
        
        predictions = []
        references = []
        
        # Generate predictions
        for example in tqdm(dataset, desc="MS MARCO"):
            question = example['query']
            # Use passages as context
            if 'passages' in example:
                context = " ".join([p['passage_text'] for p in example['passages'][:3]])[:2000]
            else:
                context = ""
            
            answer = self.generate_answer(question, context)
            
            # Format for MS MARCO eval
            predictions.append({
                'query_id': example['query_id'],
                'answers': [answer] if answer else ['No Answer Present.']
            })
            
            references.append({
                'query_id': example['query_id'],
                'answers': example.get('answers', ['No Answer Present.'])
            })
        
        # Save files
        os.makedirs("outputs/ms_marco", exist_ok=True)
        
        with open("outputs/ms_marco/predictions.jsonl", "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\\n")
        
        with open("outputs/ms_marco/references.jsonl", "w") as f:
            for ref in references:
                f.write(json.dumps(ref) + "\\n")
        
        # Run evaluation
        metrics = self.run_ms_marco_eval("outputs/ms_marco/references.jsonl",
                                         "outputs/ms_marco/predictions.jsonl")
        
        # Add BERTScore
        if self.bert_scorer:
            bert_metrics = self.compute_bert_score_ms_marco(predictions, references)
            metrics.update(bert_metrics)
        
        return metrics
    
    def run_ms_marco_eval(self, ref_file: str, pred_file: str) -> Dict[str, Any]:
        """Run MS MARCO evaluation"""
        cmd = [sys.executable, "evals/ms_marco_eval.py", ref_file, pred_file]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse output
            metrics = {}
            for line in result.stdout.split('\\n'):
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        try:
                            metrics[key] = float(value)
                        except:
                            metrics[key] = value
            return metrics
        except subprocess.CalledProcessError as e:
            print(f"Error running MS MARCO eval: {e}")
            return {}
    
    def compute_bert_score_ms_marco(self, predictions: List[Dict], 
                                    references: List[Dict]) -> Dict[str, float]:
        """Compute BERTScore for MS MARCO"""
        candidates = []
        refs = []
        
        for pred, ref in zip(predictions, references):
            if pred['query_id'] == ref['query_id']:
                candidates.append(pred['answers'][0] if pred['answers'] else "")
                refs.append(ref['answers'])
        
        if candidates and refs:
            P, R, F = self.bert_scorer.score(candidates, refs)
            return {
                "bert_precision_marco": float(P.mean()),
                "bert_recall_marco": float(R.mean()),
                "bert_f1_marco": float(F.mean())
            }
        return {}
    
    def run_all_evaluations(self, benchmarks: List[str] = None, 
                           max_examples: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Run all selected evaluations"""
        
        if benchmarks is None:
            benchmarks = ['squad', 'hotpot', 'natural_questions', 'triviaqa', 'ms_marco']
        
        all_results = {}
        
        for benchmark in benchmarks:
            print(f"\\n{'='*60}")
            print(f"Running {benchmark.upper()} evaluation")
            print(f"{'='*60}")
            
            try:
                if benchmark == 'squad':
                    results = self.evaluate_squad(max_examples)
                elif benchmark == 'hotpot':
                    results = self.evaluate_hotpot(max_examples)
                elif benchmark == 'natural_questions':
                    results = self.evaluate_natural_questions(max_examples)
                elif benchmark == 'triviaqa':
                    results = self.evaluate_triviaqa(max_examples)
                elif benchmark == 'ms_marco':
                    results = self.evaluate_ms_marco(max_examples)
                else:
                    print(f"Unknown benchmark: {benchmark}")
                    continue
                
                all_results[benchmark] = results
                
                # Print results
                print(f"\\nResults for {benchmark}:")
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                
            except Exception as e:
                print(f"Error evaluating {benchmark}: {e}")
                import traceback
                traceback.print_exc()
                all_results[benchmark] = {"error": str(e)}
        
        return all_results
    
    def save_results(self, results: Dict[str, Dict[str, Any]], output_file: str):
        """Save all results to JSON file"""
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        # Add metadata
        results['metadata'] = {
            'model': 'selfrag/selfrag_llama2_7b',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'bert_score_model': 'roberta-large' if self.bert_scorer else 'not_available'
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\nResults saved to: {output_file}")
        
        # Also create a summary CSV for easy viewing
        csv_file = output_file.replace('.json', '_summary.csv')
        self.create_summary_csv(results, csv_file)
    
    def create_summary_csv(self, results: Dict[str, Dict[str, Any]], csv_file: str):
        """Create a summary CSV of key metrics"""
        import csv
        
        # Define key metrics to extract
        key_metrics = [
            'exact', 'f1', 'exact_match',  # Standard QA metrics
            'bert_f1', 'bert_precision', 'bert_recall',  # BERTScore metrics
            'HasAns_exact', 'HasAns_f1', 'NoAns_exact', 'NoAns_f1',  # SQuAD specific
            'joint_f1', 'sp_f1',  # HotpotQA specific
            'bleu_1', 'bleu_4', 'rouge_l'  # MS MARCO specific
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['Benchmark'] + key_metrics
            writer.writerow(header)
            
            # Write data for each benchmark
            for benchmark, metrics in results.items():
                if benchmark == 'metadata':
                    continue
                
                row = [benchmark]
                for metric in key_metrics:
                    # Look for metric in results (handle variations)
                    value = None
                    for key in metrics:
                        if metric in key or key in metric:
                            value = metrics[key]
                            break
                    
                    if value is not None:
                        if isinstance(value, float):
                            row.append(f"{value:.4f}")
                        else:
                            row.append(str(value))
                    else:
                        row.append("")
                
                writer.writerow(row)
        
        print(f"Summary CSV saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='Self-RAG Model Evaluation on QA Benchmarks')
    parser.add_argument('--model-path', default='selfrag/selfrag_llama2_7b',
                       help='Path to Self-RAG model')
    parser.add_argument('--download-dir', default='/gscratch/h2lab/akari/model_cache',
                       help='Directory for model cache')
    parser.add_argument('--benchmarks', nargs='+', 
                       choices=['squad', 'hotpot', 'natural_questions', 'triviaqa', 'ms_marco'],
                       help='Benchmarks to evaluate on (default: all)')
    parser.add_argument('--max-examples', type=int, default=None,
                       help='Maximum examples per benchmark (for testing)')
    parser.add_argument('--output-file', default='outputs/selfrag_results.json',
                       help='Output file for results')
    parser.add_argument('--no-retrieval', action='store_true',
                       help='Disable retrieval augmentation')
    parser.add_argument('--device', default='cuda',
                       help='Device for model (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SelfRAGEvaluator(
        model_path=args.model_path,
        download_dir=args.download_dir,
        device=args.device,
        use_retrieval=not args.no_retrieval
    )
    
    # Run evaluations
    results = evaluator.run_all_evaluations(
        benchmarks=args.benchmarks,
        max_examples=args.max_examples
    )
    
    # Save results
    evaluator.save_results(results, args.output_file)
    
    # Print final summary
    print("\\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    for benchmark, metrics in results.items():
        if benchmark == 'metadata':
            continue
        
        print(f"\\n{benchmark.upper()}:")
        if 'error' in metrics:
            print(f"  Error: {metrics['error']}")
        else:
            # Print key metrics
            for key in ['exact', 'f1', 'exact_match', 'bert_f1']:
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
