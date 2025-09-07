# crag/metrics.py
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score

from datasets import Dataset
from constants import RAGBenchFields


def rmse(trues: List[float], preds: List[float]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) between input ground truth (`trues`) and predictions (`preds`)
    Returns float (np.nan-safe). If lengths mismatch, returns np.nan.
    """
    if len(trues) != len(preds):
        return float("nan")
    
    trues_arr = np.array(trues, dtype=float)
    preds_arr = np.array(preds, dtype=float)

    # Ignore Nulls in predictions (NaNs)
    eval_idx = ~np.isnan(preds_arr)
    if eval_idx.sum() == 0:
        return float("nan")
    trues_filtered = trues_arr[eval_idx]
    preds_filtered = preds_arr[eval_idx]
    
    return float(np.sqrt(np.mean((preds_filtered - trues_filtered) ** 2)))


def auroc(trues: List[bool], preds: List[float]) -> float:
    """
    Calculate Area Under Receiver Operator Characteristic Curve (AUROC)
    Filters out NaN preds before computing roc_auc_score.
    """
    preds_arr = np.array(preds, dtype=float)
    eval_idx = ~np.isnan(preds_arr)
    if eval_idx.sum() == 0:
        return float("nan")
    return float(roc_auc_score(np.array(trues)[eval_idx], preds_arr[eval_idx]))


def calculate_metrics(
    annotated_dataset: Dataset,
    pred_adherence: Optional[str] = None,
    pred_context_relevance: Optional[str] = None,
    pred_context_utilization: Optional[str] = None,
    ground_truth_adherence: str = RAGBenchFields.SUPPORTED, 
    ground_truth_context_relevance: str = RAGBenchFields.RELEVANCE,
    ground_truth_context_utilization: str = RAGBenchFields.UTILIZATION,
) -> Dict[str, float]:
    """
    annotated_dataset: HuggingFace datasets.Dataset or DatasetDict-like object supporting indexing by column name.
    Column names for predictions (pred_*) and ground truth may be overridden; defaults use RAGBenchFields.
    Returns a dict of computed metrics. Missing inputs lead to missing metrics or NaN values.
    """
    calculated_metrics = {}
    
    # Evaluate Hallucination Detection Task (AUROC)
    if pred_adherence:
        # ground_truth_adherence expected to be boolean-like where SUPPORTED==True means supported
        trues_hallucination = ~np.array(annotated_dataset[ground_truth_adherence])
        preds_hallucination = 1.0 - np.array(annotated_dataset[pred_adherence], dtype=float)
        calculated_metrics["hallucination_auroc"] = auroc(trues_hallucination, preds_hallucination)

    # Evaluate Context Relevance Task (RMSE)
    if pred_context_relevance:
        trues_relevance = np.array(annotated_dataset[ground_truth_context_relevance], dtype=float)
        preds_relevance = np.array(annotated_dataset[pred_context_relevance], dtype=float)
        calculated_metrics["relevance_rmse"] = rmse(trues_relevance.tolist(), preds_relevance.tolist())

    # Evaluate Context Utilization Task (RMSE)
    if pred_context_utilization:
        trues_utilization = np.array(annotated_dataset[ground_truth_context_utilization], dtype=float)
        preds_utilization = np.array(annotated_dataset[pred_context_utilization], dtype=float)
        calculated_metrics["utilization_rmse"] = rmse(trues_utilization.tolist(), preds_utilization.tolist())

    return calculated_metrics
