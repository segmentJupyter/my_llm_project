"""
LLM Text Quality Evaluation Metrics

Provides functions to calculate NLP/text generation metrics:
  - ROUGE: Recall-Oriented Understudy for Gisting Evaluation
  - BERTScore: Semantic similarity using contextual embeddings
  - BLEU: Bilingual Evaluation Understudy
  - METEOR: Metric for Evaluation of Translation with Explicit ORdering
  - Perplexity: Model confidence measurement
  - F1 Score: Precision & recall balance
"""
from __future__ import annotations

from typing import Dict, Optional
import warnings

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False

try:
    from nltk.translate.meteor_score import meteor_score
    HAS_METEOR = True
except ImportError:
    HAS_METEOR = False

import numpy as np


def calculate_rouge_score(reference: str, generated: str) -> Optional[Dict[str, float]]:
    """
    Calculate ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation).
    
    ROUGE metrics:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    
    Args:
        reference: Ground truth text
        generated: Generated/predicted text
        
    Returns:
        Dictionary with ROUGE scores or None if rouge_score not installed
    """
    if not HAS_ROUGE:
        warnings.warn("rouge_score not installed. Install with: pip install rouge-score")
        return None
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        
        return {
            'rouge1_precision': float(scores['rouge1'].precision),
            'rouge1_recall': float(scores['rouge1'].recall),
            'rouge1_fmeasure': float(scores['rouge1'].fmeasure),
            'rouge2_precision': float(scores['rouge2'].precision),
            'rouge2_recall': float(scores['rouge2'].recall),
            'rouge2_fmeasure': float(scores['rouge2'].fmeasure),
            'rougeL_precision': float(scores['rougeL'].precision),
            'rougeL_recall': float(scores['rougeL'].recall),
            'rougeL_fmeasure': float(scores['rougeL'].fmeasure),
        }
    except Exception as e:
        warnings.warn(f"Error calculating ROUGE: {e}")
        return None


def calculate_bertscore(reference: str, generated: str, model_type: str = "distilbert-base-uncased") -> Optional[Dict[str, float]]:
    """
    Calculate BERTScore - semantic similarity using contextual embeddings.
    
    BERTScore uses contextual word embeddings to measure semantic similarity
    between reference and generated text.
    
    Args:
        reference: Ground truth text
        generated: Generated/predicted text
        model_type: Which BERT model to use (default: distilbert for speed)
        
    Returns:
        Dictionary with precision, recall, F1 scores or None if bert_score not installed
    """
    if not HAS_BERTSCORE:
        warnings.warn("bert_score not installed. Install with: pip install bert-score")
        return None
    
    try:
        precision, recall, f1 = bert_score_fn(
            [generated],
            [reference],
            model_type=model_type,
            lang="en",
            device="cpu",
            verbose=False
        )
        
        return {
            'bertscore_precision': float(precision.mean()),
            'bertscore_recall': float(recall.mean()),
            'bertscore_f1': float(f1.mean()),
        }
    except Exception as e:
        warnings.warn(f"Error calculating BERTScore: {e}")
        return None


def calculate_bleu_score(reference: str, generated: str) -> Optional[Dict[str, float]]:
    """
    Calculate BLEU score - Bilingual Evaluation Understudy.
    
    Measures n-gram precision. BLEU-1 to BLEU-4 measure unigrams through 4-grams.
    
    Args:
        reference: Ground truth text
        generated: Generated/predicted text
        
    Returns:
        Dictionary with BLEU scores or None if nltk not installed
    """
    if not HAS_BLEU:
        warnings.warn("nltk not installed. Install with: pip install nltk")
        return None
    
    try:
        ref_tokens = word_tokenize(reference.lower())
        gen_tokens = word_tokenize(generated.lower())
        
        smooth = SmoothingFunction().method1
        
        scores = {}
        for n in range(1, 5):
            weights = [1/n] * n
            try:
                bleu = sentence_bleu(
                    [ref_tokens],
                    gen_tokens,
                    weights=weights,
                    smoothing_function=smooth
                )
                scores[f'bleu_{n}'] = float(bleu)
            except:
                scores[f'bleu_{n}'] = None
        
        return scores
    except Exception as e:
        warnings.warn(f"Error calculating BLEU: {e}")
        return None


def calculate_meteor_score(reference: str, generated: str) -> Optional[float]:
    """
    Calculate METEOR score - Metric for Evaluation of Translation with Explicit ORdering.
    
    Considers synonyms and word order.
    
    Args:
        reference: Ground truth text
        generated: Generated/predicted text
        
    Returns:
        METEOR score or None if nltk not installed
    """
    if not HAS_METEOR:
        warnings.warn("nltk not installed. Install with: pip install nltk")
        return None
    
    try:
        ref_tokens = word_tokenize(reference.lower())
        gen_tokens = word_tokenize(generated.lower())
        
        score = meteor_score([ref_tokens], gen_tokens)
        return float(score)
    except Exception as e:
        warnings.warn(f"Error calculating METEOR: {e}")
        return None


def calculate_perplexity(probabilities: list[float]) -> Optional[float]:
    """
    Calculate Perplexity - measure of model confidence/uncertainty.
    
    Lower perplexity indicates the model is more confident in its predictions.
    Perplexity = exp(-1/N * sum(log(P(word))))
    
    Args:
        probabilities: List of log probabilities for each token
        
    Returns:
        Perplexity score or None if invalid input
    """
    try:
        if not probabilities or len(probabilities) == 0:
            return None
        
        probs = np.array(probabilities)
        if np.any(probs > 0):
            warnings.warn("Probabilities should be <= 0 (log probabilities expected)")
        
        perplexity = np.exp(-np.mean(probs))
        return float(perplexity)
    except Exception as e:
        warnings.warn(f"Error calculating Perplexity: {e}")
        return None


def calculate_text_f1_score(reference: str, generated: str) -> Optional[Dict[str, float]]:
    """
    Calculate F1 Score for text comparison (simple token-based).
    
    This is a simple token-based F1 (not intended for semantic similarity).
    For better semantic F1, use BERTScore.
    
    Args:
        reference: Ground truth text
        generated: Generated/predicted text
        
    Returns:
        Dictionary with precision, recall, F1 or None if error
    """
    try:
        if HAS_BLEU:
            ref_tokens = set(word_tokenize(reference.lower()))
            gen_tokens = set(word_tokenize(generated.lower()))
        else:
            # Fallback: simple split
            ref_tokens = set(reference.lower().split())
            gen_tokens = set(generated.lower().split())
        
        if len(gen_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        true_positives = len(ref_tokens & gen_tokens)
        false_positives = len(gen_tokens - ref_tokens)
        false_negatives = len(ref_tokens - gen_tokens)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'token_f1_precision': float(precision),
            'token_f1_recall': float(recall),
            'token_f1_score': float(f1),
        }
    except Exception as e:
        warnings.warn(f"Error calculating F1: {e}")
        return None


def calculate_all_text_metrics(reference: str, generated: str, 
                               include_bertscore: bool = False) -> Dict[str, any]:
    """
    Calculate all available text quality metrics.
    
    Args:
        reference: Ground truth text
        generated: Generated/predicted text
        include_bertscore: Whether to calculate BERTScore (slower, requires model download)
        
    Returns:
        Dictionary with all calculated metrics
    """
    all_metrics = {}
    
    # ROUGE scores
    rouge_scores = calculate_rouge_score(reference, generated)
    if rouge_scores:
        all_metrics.update(rouge_scores)
    
    # BLEU scores
    bleu_scores = calculate_bleu_score(reference, generated)
    if bleu_scores:
        all_metrics.update(bleu_scores)
    
    # METEOR score
    meteor = calculate_meteor_score(reference, generated)
    if meteor is not None:
        all_metrics['meteor'] = meteor
    
    # F1 Score
    f1_scores = calculate_text_f1_score(reference, generated)
    if f1_scores:
        all_metrics.update(f1_scores)
    
    # BERTScore (optional - slower)
    if include_bertscore:
        bert_scores = calculate_bertscore(reference, generated)
        if bert_scores:
            all_metrics.update(bert_scores)
    
    return all_metrics


# Convenience function for checking which metrics are available
def check_llm_metrics_availability() -> Dict[str, bool]:
    """Check which metrics libraries are available."""
    return {
        'ROUGE': HAS_ROUGE,
        'BERTScore': HAS_BERTSCORE,
        'BLEU': HAS_BLEU,
        'METEOR': HAS_METEOR,
    }
