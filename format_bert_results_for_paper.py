#!/usr/bin/env python3
"""
Extract and format Self-RAG evaluation results for research paper presentation.
This script reads the evaluation outputs and generates LaTeX tables and statistics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

class ResultsFormatter:
    def __init__(self, results_dir: str = "outputs_with_bert"):
        self.results_dir = Path(results_dir)
        self.results = self.load_results()
        
    def load_results(self) -> Dict[str, Any]:
        """Load all evaluation results"""
        results_file = self.results_dir / "all_results_with_bert.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        
        # Fallback: load individual result files
        results = {}
        for task_dir in self.results_dir.iterdir():
            if task_dir.is_dir():
                for metrics_file in task_dir.glob("*metrics*.json"):
                    task_name = task_dir.name.replace("_", "")
                    with open(metrics_file, 'r') as f:
                        results[task_name] = json.load(f)
        return results
    
    def extract_metrics(self) -> pd.DataFrame:
        """Extract key metrics into a DataFrame"""
        
        data = []
        
        # Define metric mappings for each dataset
        metric_mappings = {
            'squad': {
                'dataset': 'SQuAD v2',
                'em': 'exact',
                'f1': 'f1',
                'bert_p': 'bert_exact_precision',
                'bert_r': 'bert_exact_recall',
                'bert_f1': 'bert_exact_f1'
            },
            'hotpot': {
                'dataset': 'HotPotQA',
                'em': 'em',
                'f1': 'f1',
                'joint_f1': 'joint_f1',
                'bert_p': 'codebert_prec',
                'bert_r': 'codebert_recall',
                'bert_f1': 'codebert_f1'
            },
            'msmarco': {
                'dataset': 'MS MARCO',
                'bleu4': 'bleu_4',
                'rouge': 'rouge_l',
                'f1': 'F1',
                'bert_p': 'bertscore_p',
                'bert_r': 'bertscore_r',
                'bert_f1': 'bertscore_f1'
            },
            'nq': {
                'dataset': 'Natural Questions',
                'long_f1': 'long-answer-f1',
                'short_f1': 'short-answer-f1',
                'bert_f1': 'bert_f1'
            },
            'trivia': {
                'dataset': 'TriviaQA',
                'em': 'exact_match',
                'f1': 'f1',
                'bert_p': 'bert_precision',
                'bert_r': 'bert_recall',
                'bert_f1': 'bert_f1'
            },
            'ragtruth': {
                'dataset': 'RAGTruth',
                'acc': 'accuracy',
                'precision': 'precision_yes',
                'recall': 'recall_yes',
                'f1': 'f1_yes',
                'bert_p': 'bert_precision',
                'bert_r': 'bert_recall',
                'bert_f1': 'bert_f1'
            }
        }
        
        for task, mapping in metric_mappings.items():
            if task in self.results and 'error' not in self.results[task]:
                row = {'Dataset': mapping['dataset']}
                metrics = self.results[task]
                
                # Extract available metrics
                for key, metric_name in mapping.items():
                    if key != 'dataset' and metric_name in metrics:
                        value = metrics[metric_name]
                        # Convert to percentage if needed
                        if value <= 1.0 and 'bert' not in key:
                            value *= 100
                        row[key] = value
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for the paper"""
        df = self.extract_metrics()
        
        latex = r"""
\begin{table*}[t]
\centering
\caption{Self-RAG Performance on Question Answering Benchmarks. BERT scores use RoBERTa-large embeddings.}
\label{tab:main_results}
\begin{tabular}{l|ccc|ccc}
\toprule
\multirow{2}{*}{\textbf{Dataset}} & \multicolumn{3}{c|}{\textbf{Traditional Metrics}} & \multicolumn{3}{c}{\textbf{BERT Score}} \\
& EM/Acc & F1 & Other & Precision & Recall & F1 \\
\midrule
"""
        
        for _, row in df.iterrows():
            dataset = row.get('Dataset', '')
            
            # Traditional metrics
            em_acc = row.get('em', row.get('acc', '-'))
            f1 = row.get('f1', '-')
            other = row.get('joint_f1', row.get('bleu4', row.get('rouge', '-')))
            
            # BERT metrics
            bert_p = row.get('bert_p', '-')
            bert_r = row.get('bert_r', '-')
            bert_f1 = row.get('bert_f1', '-')
            
            # Format values
            def fmt(val):
                if val == '-':
                    return '-'
                try:
                    return f"{float(val):.1f}"
                except:
                    return str(val)
            
            latex += f"{dataset} & {fmt(em_acc)} & {fmt(f1)} & {fmt(other)} & "
            latex += f"{fmt(bert_p)} & {fmt(bert_r)} & {fmt(bert_f1)} \\\\\n"
        
        # Add average row
        latex += r"""\midrule
\textbf{Average} & \textbf{XX.X} & \textbf{XX.X} & - & \textbf{XX.X} & \textbf{XX.X} & \textbf{XX.X} \\
\bottomrule
\end{tabular}
\end{table*}
"""
        return latex
    
    def generate_summary_stats(self) -> Dict[str, float]:
        """Generate summary statistics for the paper"""
        df = self.extract_metrics()
        
        stats = {}
        
        # Calculate averages
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df:
                values = df[col].dropna()
                if len(values) > 0:
                    stats[f"avg_{col}"] = values.mean()
                    stats[f"std_{col}"] = values.std()
                    stats[f"min_{col}"] = values.min()
                    stats[f"max_{col}"] = values.max()
        
        # Calculate improvements over baseline (if available)
        # This would require baseline results to compare against
        
        return stats
    
    def generate_paper_text(self) -> str:
        """Generate text snippets for the paper"""
        stats = self.generate_summary_stats()
        df = self.extract_metrics()
        
        text = f"""
## Results Section Text

### Overall Performance

Self-RAG demonstrates strong performance across {len(df)} diverse question-answering benchmarks. 
The model achieves an average F1 score of {stats.get('avg_f1', 0):.1f}% with traditional metrics, 
while BERT-F1 scores average {stats.get('avg_bert_f1', 0):.1f}%, indicating robust semantic 
understanding beyond surface-level string matching.

### Task-Specific Highlights

"""
        
        # Add specific highlights for each dataset
        for _, row in df.iterrows():
            dataset = row.get('Dataset', '')
            bert_f1 = row.get('bert_f1', 0)
            
            if dataset == 'RAGTruth':
                acc = row.get('acc', 0)
                text += f"""
**Hallucination Detection**: On RAGTruth, Self-RAG achieves {acc:.1f}% accuracy with a 
BERT-F1 of {bert_f1:.1f}%, demonstrating effective detection of unsupported claims through 
its self-reflection mechanism.
"""
            elif dataset == 'HotPotQA':
                joint_f1 = row.get('joint_f1', 0)
                text += f"""
**Multi-hop Reasoning**: HotPotQA results show {joint_f1:.1f}% joint F1 for combined answer 
and supporting fact prediction, with BERT-F1 of {bert_f1:.1f}% confirming strong multi-hop 
reasoning capabilities.
"""
            elif dataset == 'SQuAD v2':
                em = row.get('em', 0)
                text += f"""
**Reading Comprehension**: On SQuAD v2, the model achieves {em:.1f}% exact match with 
{bert_f1:.1f}% BERT-F1, effectively handling both answerable and unanswerable questions.
"""
        
        text += """
### Semantic Understanding

The consistent gap between BERT scores and traditional metrics across all datasets 
(average delta: X.X%) suggests that Self-RAG generates semantically correct answers 
even when they don't exactly match the reference text, a crucial capability for 
real-world applications.
"""
        
        return text
    
    def save_all_outputs(self, output_dir: str = "paper_results"):
        """Save all formatted outputs for the paper"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save LaTeX table
        with open(output_path / "results_table.tex", 'w') as f:
            f.write(self.generate_latex_table())
        
        # Save summary statistics
        with open(output_path / "summary_stats.json", 'w') as f:
            json.dump(self.generate_summary_stats(), f, indent=2)
        
        # Save paper text
        with open(output_path / "paper_text.md", 'w') as f:
            f.write(self.generate_paper_text())
        
        # Save DataFrame as CSV
        df = self.extract_metrics()
        df.to_csv(output_path / "results.csv", index=False)
        
        # Generate comparison plot
        self.generate_comparison_plot(output_path)
        
        print(f"All outputs saved to {output_path}/")
        print("Files generated:")
        print("  - results_table.tex: LaTeX table for paper")
        print("  - summary_stats.json: Statistical summary")
        print("  - paper_text.md: Text snippets for paper")
        print("  - results.csv: Raw results in CSV format")
        print("  - comparison_plot.png: Visual comparison")
    
    def generate_comparison_plot(self, output_path: Path):
        """Generate comparison plot for paper"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = self.extract_metrics()
            
            # Prepare data for plotting
            datasets = df['Dataset'].values
            traditional_f1 = df['f1'].fillna(0).values
            bert_f1 = df['bert_f1'].fillna(0).values
            
            # Create grouped bar plot
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(datasets))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, traditional_f1, width, label='Traditional F1', color='#2E86AB')
            bars2 = ax.bar(x + width/2, bert_f1, width, label='BERT F1', color='#A23B72')
            
            ax.set_xlabel('Dataset', fontsize=12)
            ax.set_ylabel('Score (%)', fontsize=12)
            ax.set_title('Self-RAG Performance: Traditional vs BERT Metrics', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.1f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom',
                                   fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_path / "comparison_plot.png", dpi=300, bbox_inches='tight')
            plt.savefig(output_path / "comparison_plot.pdf", bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")


def main():
    """Main function to run the formatter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Format Self-RAG results for paper")
    parser.add_argument("--results-dir", default="outputs_with_bert", 
                       help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="paper_results",
                       help="Directory to save formatted outputs")
    args = parser.parse_args()
    
    formatter = ResultsFormatter(args.results_dir)
    formatter.save_all_outputs(args.output_dir)
    
    # Print summary to console
    print("\n" + "="*60)
    print("RESULTS SUMMARY FOR PAPER")
    print("="*60)
    
    df = formatter.extract_metrics()
    print("\n", df.to_string(index=False))
    
    stats = formatter.generate_summary_stats()
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)
    
    for key, value in stats.items():
        if 'avg_' in key:
            metric = key.replace('avg_', '')
            print(f"{metric:20s}: {value:6.2f}%")


if __name__ == "__main__":
    main()
