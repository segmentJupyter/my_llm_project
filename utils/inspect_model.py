#!/usr/bin/env python3
"""
Inspect and display model parameters and metrics for trained LSTM and LLM models.

Usage:
  python3 inspect_model.py                    # List all available models
  python3 inspect_model.py --dataset dataset1 # Show specific model info
  python3 inspect_model.py --all              # Show details for all models
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
    max_error,
)

from models.lstm_model import LSTMForecaster
from utils.dataset_io import ARTIFACTS_DIR
from utils.llm_metrics import check_llm_metrics_availability, calculate_all_text_metrics


def list_available_models() -> list[str]:
    """List all available trained models in artifacts directory."""
    if not ARTIFACTS_DIR.exists():
        print(f"Artifacts directory not found: {ARTIFACTS_DIR}")
        return []
    
    models = set()
    for f in ARTIFACTS_DIR.iterdir():
        if f.suffix == ".pt":
            dataset_id = f.stem
            models.add(dataset_id)
    
    return sorted(list(models))


def load_model_metadata(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Load metadata JSON for a model."""
    meta_path = ARTIFACTS_DIR / f"{dataset_id}.meta.json"
    
    if not meta_path.exists():
        print(f"Metadata file not found: {meta_path}")
        return None
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def load_lstm_model(dataset_id: str, n_features: int) -> Optional[LSTMForecaster]:
    """Load trained LSTM model."""
    model_path = ARTIFACTS_DIR / f"{dataset_id}.pt"
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        model = LSTMForecaster(n_features=n_features)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def count_parameters(model: LSTMForecaster) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_by_layer = {}
    for name, param in model.named_parameters():
        params_by_layer[name] = param.numel()
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "by_layer": params_by_layer,
    }


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive set of regression metrics."""
    metrics = {}
    
    try:
        # Basic errors
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['medae'] = float(median_absolute_error(y_true, y_pred))
        metrics['max_error'] = float(max_error(y_true, y_pred))
        
        # Percentage errors
        try:
            metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
        except:
            metrics['mape'] = None  # Can fail if y_true has zeros
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        denominator = np.abs(y_true) + np.abs(y_pred)
        if denominator.sum() > 0:
            metrics['smape'] = float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator))
        else:
            metrics['smape'] = None
        
        # R² and explained variance
        metrics['r2_score'] = float(r2_score(y_true, y_pred))
        metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
        
        # MASE (Mean Absolute Scaled Error) - using naive forecast as baseline
        if len(y_true) > 1:
            naive_forecast = y_true[:-1]
            naive_mae = np.mean(np.abs(y_true[1:] - naive_forecast))
            if naive_mae > 0:
                metrics['mase'] = float(np.mean(np.abs(y_true - y_pred)) / naive_mae)
            else:
                metrics['mase'] = None
        else:
            metrics['mase'] = None
        
        # Mean Bias Error (MBE)
        metrics['mbe'] = float(np.mean(y_true - y_pred))
        
        # Mean Percentage Error (MPE)
        try:
            metrics['mpe'] = float(np.mean((y_true - y_pred) / y_true)) * 100
        except:
            metrics['mpe'] = None
        
        # Standard deviation of errors
        errors = y_true - y_pred
        metrics['std_error'] = float(np.std(errors))
        
        # Correlation
        try:
            metrics['correlation'] = float(np.corrcoef(y_true, y_pred)[0, 1])
        except:
            metrics['correlation'] = None
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
    
    return metrics


def print_llm_metrics_comparison(reference_text: str, generated_text: str, 
                                 include_bertscore: bool = False) -> None:
    """Print LLM text quality metrics comparison."""
    print("\n--- LLM TEXT QUALITY METRICS ---")
    print(f"\nReference Text:\n  {reference_text[:100]}...")
    print(f"\nGenerated Text:\n  {generated_text[:100]}...")
    
    # Check available metrics
    available = check_llm_metrics_availability()
    print("\n--- Available Metrics ---")
    for metric, available in available.items():
        status = "✓" if available else "✗"
        print(f"  {status} {metric}")
    
    # Calculate metrics
    metrics = calculate_all_text_metrics(reference_text, generated_text, 
                                         include_bertscore=include_bertscore)
    
    if not metrics:
        print("\nNo metrics could be calculated. Install required packages:")
        print("  pip install rouge-score bert-score nltk")
        return
    
    print("\n--- Scores ---")
    print("\nROUGE Scores (overlap-based):")
    rouge_keys = [k for k in metrics.keys() if 'rouge' in k]
    if rouge_keys:
        for key in sorted(rouge_keys):
            val = metrics[key]
            if val is not None:
                print(f"  {key:30s}: {val:.4f}")
    else:
        print("  (Not available)")
    
    print("\nBLEU Scores (n-gram precision):")
    bleu_keys = [k for k in metrics.keys() if 'bleu' in k]
    if bleu_keys:
        for key in sorted(bleu_keys):
            val = metrics[key]
            if val is not None:
                print(f"  {key:30s}: {val:.4f}")
    else:
        print("  (Not available)")
    
    print("\nMETEOR Score (synonyms & word order):")
    if 'meteor' in metrics:
        print(f"  {'meteor':30s}: {metrics['meteor']:.4f}")
    else:
        print("  (Not available)")
    
    print("\nToken-based F1 Score:")
    f1_keys = [k for k in metrics.keys() if 'token_f1' in k]
    if f1_keys:
        for key in sorted(f1_keys):
            val = metrics[key]
            if val is not None:
                print(f"  {key:30s}: {val:.4f}")
    else:
        print("  (Not available)")
    
    if include_bertscore:
        print("\nBERTScore (semantic similarity with embeddings):")
        bert_keys = [k for k in metrics.keys() if 'bertscore' in k]
        if bert_keys:
            for key in sorted(bert_keys):
                val = metrics[key]
                if val is not None:
                    print(f"  {key:30s}: {val:.4f}")
        else:
            print("  (Not available)")


def print_model_info(dataset_id: str) -> None:
    """Print comprehensive model information."""
    print("\n" + "=" * 80)
    print(f"MODEL INFORMATION: {dataset_id}")
    print("=" * 80)
    
    # Load metadata
    meta = load_model_metadata(dataset_id)
    if not meta:
        return
    
    # Extract model configuration from metadata
    feature_cols = meta.get("feature_cols", [])
    n_features = len(feature_cols)
    
    # Print metadata info
    print("\n--- DATASET INFO ---")
    print(f"Dataset Name:       {meta.get('dataset', 'N/A')}")
    print(f"Trained At:         {meta.get('trained_at', 'N/A')}")
    print(f"Lookback Window:    {meta.get('lookback', 'N/A')} steps")
    print(f"Forecast Horizon:   {meta.get('horizon', 'N/A')} steps")
    print(f"Feature Columns:    {len(feature_cols)}")
    if feature_cols:
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i}. {col}")
    
    print(f"\nExtra Features:     {meta.get('extra_cols', [])}")
    
    # Load and display model architecture
    print("\n--- LSTM MODEL ARCHITECTURE ---")
    model = load_lstm_model(dataset_id, n_features)
    if not model:
        return
    
    print(f"Input Features:     {n_features}")
    
    # Extract architecture from model state
    lstm_weight_ih = [k for k in model.state_dict().keys() if 'weight_ih_l0' in k]
    if lstm_weight_ih:
        hidden_size = model.state_dict()['lstm.weight_ih_l0'].shape[0] // 4  # 4 gates in LSTM
        print(f"Hidden Size:        {hidden_size}")
    
    num_layers = len([k for k in model.state_dict().keys() if 'weight_ih' in k])
    print(f"Number of Layers:   {num_layers}")
    
    # Count parameters
    param_info = count_parameters(model)
    print(f"\n--- PARAMETER COUNT ---")
    print(f"Total Parameters:   {param_info['total_parameters']:,}")
    print(f"Trainable:          {param_info['trainable_parameters']:,}")
    print(f"Non-trainable:      {param_info['non_trainable_parameters']:,}")
    
    print(f"\n--- PARAMETERS BY LAYER ---")
    for layer_name, param_count in param_info['by_layer'].items():
        print(f"{layer_name:40s}: {param_count:>10,} params")
    
    # Display model structure
    print(f"\n--- FULL MODEL STRUCTURE ---")
    print(model)
    
    # Display metrics
    metrics = meta.get("metrics", {})
    print(f"\n--- VALIDATION METRICS ---")
    if metrics:
        print("\nBasic Error Metrics:")
        print(f"  MAE (Mean Absolute Error):        {metrics.get('val_mae', 'N/A')}")
        print(f"  RMSE (Root Mean Squared Error):   {metrics.get('val_rmse', 'N/A')}")
        print(f"  MSE (Mean Squared Error):         {metrics.get('mse', 'N/A')}")
        print(f"  MedAE (Median Absolute Error):    {metrics.get('medae', 'N/A')}")
        print(f"  Max Error:                        {metrics.get('max_error', 'N/A')}")
        
        print("\nPercentage Error Metrics:")
        print(f"  MAPE (Mean Absolute % Error):     {metrics.get('mape', 'N/A')}")
        print(f"  SMAPE (Symmetric MAPE):           {metrics.get('smape', 'N/A')}")
        print(f"  MPE (Mean Percentage Error %):    {metrics.get('mpe', 'N/A')}")
        
        print("\nRelative Performance Metrics:")
        print(f"  R² Score:                         {metrics.get('r2_score', 'N/A')}")
        print(f"  Explained Variance:               {metrics.get('explained_variance', 'N/A')}")
        print(f"  MASE (Mean Absolute Scaled Error): {metrics.get('mase', 'N/A')}")
        print(f"  Correlation:                      {metrics.get('correlation', 'N/A')}")
        
        print("\nError Distribution Metrics:")
        print(f"  MBE (Mean Bias Error):            {metrics.get('mbe', 'N/A')}")
        print(f"  Std Error (Std Dev of Errors):    {metrics.get('std_error', 'N/A')}")
        
        print("\nAll Metrics:")
        for metric_name, metric_value in sorted(metrics.items()):
            if not metric_name.startswith('val_'):
                print(f"  {metric_name:30s}: {metric_value}")
    else:
        print("No metrics found in metadata. Stored only: val_mae and val_rmse")
    
    # Training configuration
    print(f"\n--- TRAINING CONFIGURATION ---")
    torch_info = meta.get("torch", {})
    print(f"CUDA Available:     {torch_info.get('cuda_available', False)}")
    print(f"Torch Threads:      {torch_info.get('num_threads', 'N/A')}")
    
    print("\n" + "=" * 80 + "\n")


def print_all_models_summary() -> None:
    """Print summary information for all available models."""
    models = list_available_models()
    
    if not models:
        print("No trained models found.")
        return
    
    print("\n" + "=" * 120)
    print("SUMMARY OF ALL TRAINED MODELS")
    print("=" * 120)
    print(f"\nTotal models found: {len(models)}\n")
    
    for dataset_id in models:
        meta = load_model_metadata(dataset_id)
        if meta:
            metrics = meta.get("metrics", {})
            feature_count = len(meta.get("feature_cols", []))
            
            print(f"Dataset ID:  {dataset_id}")
            print(f"  Dataset:     {meta.get('dataset', 'N/A')}")
            print(f"  Features:    {feature_count}")
            print(f"  Lookback:    {meta.get('lookback', 'N/A')}")
            
            if metrics:
                print(f"  Key Metrics:")
                print(f"    - MAE:           {metrics.get('val_mae', 'N/A')}")
                print(f"    - RMSE:          {metrics.get('val_rmse', 'N/A')}")
                print(f"    - R² Score:      {metrics.get('r2_score', 'N/A')}")
                print(f"    - MAPE:          {metrics.get('mape', 'N/A')}")
                print(f"    - Correlation:   {metrics.get('correlation', 'N/A')}")
            
            print()
    
    print("=" * 120 + "\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect and display LSTM model parameters and metrics."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset ID to inspect (e.g., 'dataset1'). If not provided, shows available models.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show detailed info for all available models.",
    )
    parser.add_argument(
        "--llm-eval",
        action="store_true",
        help="Show available LLM text quality metrics.",
    )
    parser.add_argument(
        "--compare-texts",
        nargs=2,
        metavar=("REFERENCE", "GENERATED"),
        help="Compare two texts and calculate LLM metrics. Use quoted strings.",
    )
    parser.add_argument(
        "--with-bertscore",
        action="store_true",
        help="Include BERTScore calculation (slower, requires model download).",
    )
    
    args = parser.parse_args()
    
    # Show LLM metrics info
    if args.llm_eval:
        print("\n" + "=" * 80)
        print("LLM TEXT QUALITY METRICS")
        print("=" * 80)
        print("\nAvailable Metrics:")
        print("\n1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)")
        print("   - Measures overlap of n-grams between texts")
        print("   - ROUGE-1: Unigram overlap")
        print("   - ROUGE-2: Bigram overlap")
        print("   - ROUGE-L: Longest common subsequence")
        print("   - Install: pip install rouge-score")
        
        print("\n2. BLEU (Bilingual Evaluation Understudy)")
        print("   - Measures n-gram precision")
        print("   - BLEU-1 to BLEU-4: Unigrams through 4-grams")
        print("   - Install: pip install nltk")
        
        print("\n3. METEOR (Metric for Evaluation of Translation with Explicit ORdering)")
        print("   - Considers synonyms and word order")
        print("   - Install: pip install nltk")
        
        print("\n4. F1 Score (Token-based)")
        print("   - Simple overlap-based F1 score")
        print("   - Precision: How many generated tokens are relevant")
        print("   - Recall: How many reference tokens are covered")
        
        print("\n5. BERTScore (Semantic Similarity with Embeddings)")
        print("   - Uses contextual embeddings for semantic matching")
        print("   - More accurate than n-gram methods")
        print("   - Slower and requires model download")
        print("   - Install: pip install bert-score")
        
        print("\n" + "-" * 80)
        print("Current Availability:")
        available = check_llm_metrics_availability()
        for metric, is_available in available.items():
            status = "✓ Available" if is_available else "✗ Not installed"
            print(f"  {metric:15s}: {status}")
        
        print("\n" + "=" * 80 + "\n")
        return 0
    
    # Compare two texts
    if args.compare_texts:
        reference_text = args.compare_texts[0]
        generated_text = args.compare_texts[1]
        print_llm_metrics_comparison(reference_text, generated_text, 
                                     include_bertscore=args.with_bertscore)
        return 0
    
    # List all available models
    models = list_available_models()
    
    if not models:
        print("\nNo trained models found in artifacts directory.")
        print(f"Artifacts path: {ARTIFACTS_DIR}")
        return 1
    
    # Show summary for all models
    if args.all:
        print_all_models_summary()
        print("\nDetailed information for each model:")
        for model_id in models:
            print_model_info(model_id)
        return 0
    
    # Show info for specific dataset
    if args.dataset:
        if args.dataset not in models:
            print(f"\nModel '{args.dataset}' not found.")
            print(f"\nAvailable models: {', '.join(models)}")
            return 1
        print_model_info(args.dataset)
        return 0
    
    # Default: list available models
    print("\n" + "=" * 80)
    print("AVAILABLE TRAINED MODELS")
    print("=" * 80)
    print(f"\nFound {len(models)} trained model(s):\n")
    
    for i, model_id in enumerate(models, 1):
        meta = load_model_metadata(model_id)
        dataset_name = meta.get('dataset', 'N/A') if meta else 'N/A'
        print(f"{i}. {model_id:30s} (from: {dataset_name})")
    
    print("\n" + "=" * 80)
    print("\nUsage:")
    print(f"  python3 inspect_model.py --dataset <dataset_id>       # Show specific model")
    print(f"  python3 inspect_model.py --all                        # Show all models with details")
    print(f"  python3 inspect_model.py --llm-eval                   # Show LLM metrics info")
    print(f"  python3 inspect_model.py --compare-texts \"ref\" \"gen\" # Compare two texts")
    print(f"  python3 inspect_model.py --compare-texts \"ref\" \"gen\" --with-bertscore")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
