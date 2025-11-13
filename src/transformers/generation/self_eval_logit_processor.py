import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

TASK_CHECKPOINT_CONFIGS = {
      "math": {
          "min_tokens_before_eval": 5,
          "bad_checkpoint_tokens": {'+', '-', '*', '/', '=', ',', '(', ')'},
          "good_checkpoint_tokens": {'\n', '.', ':', ';'},
          "good_checkpoint_phrases": ['therefore', 'thus', 'so', 'hence', 'step'],
          "use_gen_entropy": True,
          "gen_entropy_weight": 0.3,
      },
      "factuality": {
          "min_tokens_before_eval": 8,
          "bad_checkpoint_tokens": set(),
          "good_checkpoint_tokens": {'.', '!', '?', '\n'},
          "good_checkpoint_phrases": [],
          "use_gen_entropy": False,
          "gen_entropy_weight": 0.0,
      },
      "safety": {
          "min_tokens_before_eval": 3,
          "bad_checkpoint_tokens": set(),
          "good_checkpoint_tokens": set(),
          "good_checkpoint_phrases": [],
          "use_gen_entropy": False,
          "gen_entropy_weight": 0.0,
      },
      "general": {  # Fallback
          "min_tokens_before_eval": 1,
          "bad_checkpoint_tokens": set(),
          "good_checkpoint_tokens": set(),
          "good_checkpoint_phrases": [],
          "use_gen_entropy": False,
          "gen_entropy_weight": 0.0,
      },
  }


@dataclass
class LogitAnalyzerStep:
    """
    Complete step information with decoded strings.
    Built directly during generation (in add_decoding_step).

    GENERAL PURPOSE: Can evaluate any yes/no question during generation.
    Examples (Unified "Confirm Quality" Framework):
      - Safety: "Is this text safe?" (no = flagged)
      - Correctness: "Is this answer correct?" (no = flagged)
      - Factuality: "Is this answer factual?" (no = flagged)
      - Inverted: "Are there errors?" (yes = flagged)
    """
    step: int
    eval_token: str
    yes_prob: float
    no_prob: float
    confidence_score: float  # Polarity-aware: (yes-no)/(yes+no) * polarity
    is_flagged: bool  # Whether eval_token matches target signal
    current_content: str
    current_generated_token: str

    # Uncertainty metrics for resampling decisions
    coverage: float  # yes_prob + no_prob (how much probability mass on yes/no)
    binary_entropy: float  # nonlinear uncertainty
    eval_worthiness: float  # coverage * binary_entropy (worth resampling?)

    # EVALUATION DISTRIBUTION (yes/no response logits from suffix position)
    eval_top_k_logits: List[float]
    eval_top_k_probs: List[float]
    eval_top_k_tokens: List[str]
    eval_top_k_token_ids: List[int]
    eval_entropy: float

    # GENERATION DISTRIBUTION (current generation logits)
    gen_top_k_logits: List[float]
    gen_top_k_probs: List[float]
    gen_top_k_tokens: List[str]
    gen_top_k_token_ids: List[int]
    gen_entropy: float

    # Raw yes/no logits for all variants (for better interpretability)
    yes_logits: Dict[str, float]  # {"Yes": -2.3, " Yes": -2.1, ...}
    no_logits: Dict[str, float]   # {"No": -1.5, " No": -1.4, ...}


GENRE_CONFIG = {
    # target_signal: which token indicates a "positive" match (yes or no)
    # signal_polarity: +1 if "yes" is bad/flagged, -1 if "no" is bad/flagged
    "standard": {"target_signal": "no", "signal_polarity": -1},
    "adversarial": {"target_signal": "no", "signal_polarity": -1},
    "negation_traps": {"target_signal": "no", "signal_polarity": -1},
    "inverted_standard": {"target_signal": "yes", "signal_polarity": 1},
    "inverted_adversarial": {"target_signal": "yes", "signal_polarity": 1},
    "inverted_negation_traps": {"target_signal": "yes", "signal_polarity": 1},
}


def _calculate_uncertainty_for_eval(yes_prob: float, no_prob: float, task_type: str="general", current_token: str=None, step_idx: int=None, gen_entropy: float=None) -> \
dict[str, float]:
    """
    Calculate uncertainty metrics with task-aware adjustments.

    Task-specific behavior:
        - Math: avoids mid-calculation, uses gen_entropy for mid-reasoning detection
        - Factuality: prefers sentence boundaries.
        - Safety: evaluates at any position.
    """
    # Coverage: how much probability mass is on yes/no
    coverage = yes_prob + no_prob

    # Binary uncertainty: how uncertain between yes and no
    if coverage > 1e-8:
        yes_norm = yes_prob / coverage
        no_norm = no_prob / coverage

        # Compute binary entropy (0 to 1)
        entropy = 0.0
        if yes_norm > 1e-8:
            entropy -= yes_norm * np.log2(yes_norm)
        if no_norm > 1e-8:
            entropy -= no_norm * np.log2(no_norm)

        binary_entropy = entropy
    else:
        # Model doesn't engage with yes/no at all
        # Maximum uncertainty since we don't know what it wants to say
        binary_entropy = 1.0

    # Confidence: inverse of uncertainty
    confidence = 1.0 - binary_entropy

    # Eval worthiness: worth resampling when model engages AND is uncertain
    # High when: coverage is high (model thinks question is relevant)
    #            AND binary_entropy is high (model can't decide)
    # TODO think about the worthiness calculation
    eval_worthiness = coverage * binary_entropy

    # Task-aware adjustments
    config = TASK_CHECKPOINT_CONFIGS.get(task_type, TASK_CHECKPOINT_CONFIGS["general"])

    # 1. too early in generation?
    if step_idx is not None and step_idx < config["min_tokens_before_eval"]:
        eval_worthiness = 0.0
    # 2. Bad checkpoint (mid-calculation for math)
    elif current_token is not None:
        token_stripped = current_token.strip()
        # at bad checkpoint?
        if token_stripped in config["bad_checkpoint_tokens"]:
            eval_worthiness = 0.0

        # at good checkpoint?
        is_good_checkpoint = (
                token_stripped in config["good_checkpoint_tokens"] or
                any(phrase in current_token.lower() for phrase in config["good_checkpoint_phrases"])
        )

        if is_good_checkpoint:
            eval_worthiness *= 1.2  # Slight boost (capped at 1.0 later)

    # 3. Mid-reasoning detection for math (high gen_entropy = model actively reasoning)
    if config["use_gen_entropy"] and gen_entropy is not None:
        # Normalize gen_entropy (typical range 0-10, with ~3-5 being normal)
        # High gen_entropy (>5) suggests model is uncertain about next token = mid-reasoning
        if gen_entropy > 5.0:
            discount = 1.0 - config["gen_entropy_weight"] * (gen_entropy - 5.0) / 5.0
            discount = max(0.1, discount)  # Don't discount below 0.1
            eval_worthiness *= discount

    eval_worthiness = min(1.0, eval_worthiness)

    return {
        'coverage': coverage,
        'binary_entropy': binary_entropy,
        'confidence': confidence,
        'eval_worthiness': eval_worthiness
    }


class LogitAnalyzer:
    """
    GENERAL-PURPOSE evaluation framework for tracking yes/no signals during generation.

    Use cases:
      - Safety evaluation: "Is this text safe?" (flag when no)
      - Correctness checking: "Is this answer correct?" (flag when no)
      - Factuality: "Is this answer factual?" (flag when no)
      - Inverted detection: "Are there errors?" (flag when yes)
      - Uncertainty-guided resampling: Use eval_worthiness for token resampling

    Efficiently tracks evaluation signals during generation with minimal overhead.
    Stores compact representations during generation, defers string decoding to finalize().

    Key Features:
      - Strategy-aware filtering (skip computation for filtered steps)
      - Compact storage during generation (no string decoding until finalize())
      - Real-time metrics for resampling decisions (add_decoding_step returns metrics)
      - Uncertainty quantification (coverage, binary_entropy, eval_worthiness)

    Usage:
        # Create analyzer
        analyzer = LogitAnalyzer(
            tokenizer=tokenizer,
            yes_tokens=["Yes", " Yes"],
            no_tokens=["No", " No"],
            suffix_prompt_genre="standard",  # or "inverted_standard", "adversarial", etc.
            strategy=strategy  # Optional: for efficient filtering
        )

        # During generation: get real-time metrics for resampling
        metrics = analyzer.add_decoding_step(eval_logits, input_ids, next_tokens)
        if metrics[0]['eval_worthiness'] > 0.6:
            resample()  # Model is uncertain

        # After generation: get full step information with decoded strings
        steps = analyzer.finalize(output[0], batch_idx=0)
    """

    def __init__(
        self,
        tokenizer,
        task_type="general",
        input_ids=None,
        eval_input_ids=None,
        full_input_ids=None,
        dataset=None,
        subtask=None,
        suffix_prompt_genre=None,
        suffix_prompt_index=None,
        model_name=None,
        experiment=None,
        n_shots: int = 0,
        yes_tokens=None,
        no_tokens=None,
        strategy=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.0,
        top_k: int = 20,
    ):
        """
        Initialize LogitAnalyzer.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            yes_tokens: List of yes token variants, e.g., ["Yes", " Yes", "yes"]
            no_tokens: List of no token variants, e.g., ["No", " No", "no"]
            suffix_prompt_genre: Eval type (e.g., "positive_detection", "negative_detection")
            strategy: Optional evaluation strategy for filtering (experiment_runner integration)
            device: Device for token ID tensors ('cuda' or 'cpu')
            confidence_threshold: Threshold for is_flagged determination
            top_k: Number of top logits/tokens to track (default: 20)

            Legacy parameters (for backward compatibility with older experiments):
            input_ids, eval_input_ids, full_input_ids, dataset, subtask, etc.
        """
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.strategy = strategy
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

        # Prepare yes/no tokens
        if yes_tokens is None:
            self.yes_tokens = ["Yes", " Yes", "yes", " yes"]
        else:
            self.yes_tokens = yes_tokens

        if no_tokens is None:
            self.no_tokens = ["No", " No", "no", " no"]
        else:
            self.no_tokens = no_tokens

        # Pre-compute and cache token IDs on GPU for vectorized operations
        yes_ids = self._get_token_ids(self.yes_tokens)
        no_ids = self._get_token_ids(self.no_tokens)
        self.yes_token_ids_tensor = torch.tensor(yes_ids, device=device)
        self.no_token_ids_tensor = torch.tensor(no_ids, device=device)

        self.yes_token_ids = yes_ids
        self.no_token_ids = no_ids

        self.current_step = 0

        # Legacy mode: full initialization for backward compatibility
        if input_ids is not None:
            self.batch_size = input_ids.shape[0]
            self.input_ids = input_ids
            self.full_input_ids = full_input_ids
            self.eval_input_ids = eval_input_ids
            self.dataset = dataset
            self.subtask = subtask
            self.suffix_prompt_genre = suffix_prompt_genre
            self.suffix_prompt_index = suffix_prompt_index
            self.model_name = model_name
            self.n_shots = n_shots
            self.experiment = experiment

            # Decode once during initialization (legacy mode only)
            self.prompts = [tokenizer.decode(ids) for ids in input_ids]
            self.full_prompts = [tokenizer.decode(ids) for ids in full_input_ids]
            self.suffix_prompts = [tokenizer.decode(ids) for ids in eval_input_ids]

            # all_steps will be populated directly by add_decoding_step()
            self.all_steps = [[] for _ in range(self.batch_size)]

            # Pre-compute target signals for each genre
            self.target_signals = {
                batch_idx: GENRE_CONFIG[self.suffix_prompt_genre]["target_signal"]
                for batch_idx in range(self.batch_size)
            }
            self.signal_polarities = {
                batch_idx: GENRE_CONFIG[self.suffix_prompt_genre]["signal_polarity"]
                for batch_idx in range(self.batch_size)
            }
        else:
            # Simple mode: minimal initialization
            self.batch_size = None
            self.input_ids = None
            self.full_input_ids = None
            self.eval_input_ids = None
            self.dataset = None
            self.subtask = None
            self.suffix_prompt_genre = suffix_prompt_genre  # Use provided genre
            self.suffix_prompt_index = suffix_prompt_index
            self.model_name = None
            self.n_shots = 0
            self.experiment = None
            self.prompts = []
            self.full_prompts = []
            self.suffix_prompts = []
            self.all_steps = []

            # Use genre config if available, otherwise default to yes = positive signal
            # Will be populated per-batch in add_decoding_step
            self.target_signals = {}
            self.signal_polarities = {}

    def _get_token_ids(self, tokens: List[str]) -> List[int]:
        token_ids = []
        for token in tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
            else:
                raise ValueError(f"Token '{token}' encodes to {len(ids)} tokens, expected 1")
        return token_ids

    def _calculate_prediction_score(self, yes_prob: float, no_prob: float, batch_idx: int) -> float:
        """
        Calculate a directional prediction score representing "error prediction strength".

        Unified Framework (Yes = Good, No = Error):
          - Standard prompts: "Is this correct?" → No = error
          - Inverted prompts: "Are there errors?" → Yes = error

        Returns a score in [-1, +1]:
          +1: Strongly predicts ERROR (flagged state)
           0: Uncertain (50/50) or low engagement
          -1: Strongly predicts NO ERROR (good state)

        How it works:
          1. raw_score = (yes_prob - no_prob) / (yes_prob + no_prob)
             - Positive when model says "Yes" (good/correct)
             - Negative when model says "No" (error/incorrect)

          2. Apply signal_polarity to normalize to "error prediction":
             - polarity=-1 (standard): Flip sign, so No → positive score → error detected
             - polarity=+1 (inverted): Keep sign, so Yes → positive score → error detected

        Examples:
          Standard ("Is this correct?", polarity=-1):
            - Model says Yes (0.8 prob): raw=0.6 → final=-0.6 → no error ✓
            - Model says No (0.8 prob): raw=-0.6 → final=+0.6 → error detected ✓

          Inverted ("Are there errors?", polarity=+1):
            - Model says Yes (0.8 prob): raw=0.6 → final=+0.6 → error detected ✓
            - Model says No (0.8 prob): raw=-0.6 → final=-0.6 → no error ✓
        """
        raw_score = (yes_prob - no_prob) / (yes_prob + no_prob + 1e-8)

        # Normalize to common "flagged" direction
        if batch_idx in self.signal_polarities:
            return raw_score * self.signal_polarities[batch_idx]

        return -raw_score  # Default: flip sign to make "No" (error) → positive score

    def is_flagged_response(self, eval_token: str, batch_idx: int) -> bool:
        """
        Check if evaluation token matches target signal.

        Returns True if the token matches the target signal for the genre.
        Examples:
          - "Is this correct?" with genre="standard" → True if token is "no" (error detected)
          - "Are there errors?" with genre="inverted_standard" → True if token is "yes" (error detected)
        """
        eval_token = eval_token.strip().lower()
        if eval_token not in {"yes", "no"}:
            return False

        if batch_idx in self.target_signals:
            return eval_token == self.target_signals[batch_idx]

        return eval_token == "no"  # Default: no = flagged (unified "confirm quality" framework)

    def _compute_evaluation_distribution(self, logits_eval, eval_probs=None, eval_top_k_values=None, eval_top_k_indices=None):
        """
        Compute evaluation distribution metrics from evaluation position logits.

        Args:
            logits_eval: Evaluation logits [vocab_size] for a single batch item

        Returns:
            dict containing:
                - eval_top_k_logits: Top-k logit values
                - eval_top_k_probs: Top-k probabilities
                - eval_top_k_tokens: Top-k decoded tokens
                - eval_top_k_token_ids: Top-k token IDs
                - eval_entropy: Entropy of distribution
        """
        if eval_probs is None:
            probs_eval = torch.nn.functional.softmax(logits_eval, dim=-1)
        else:
            probs_eval = eval_probs

        # Yes/no probabilities
        yes_prob = probs_eval[self.yes_token_ids_tensor].sum().item()
        no_prob = probs_eval[self.no_token_ids_tensor].sum().item()

        # Use precomputed top-k if available
        if eval_top_k_values is None or eval_top_k_indices is None:
            eval_top_k_values, eval_top_k_indices = torch.topk(logits_eval, k=self.top_k)

        eval_top_k_logits = eval_top_k_values.cpu().tolist()
        eval_top_k_token_ids = eval_top_k_indices.cpu().tolist()
        eval_top_k_probs = probs_eval[eval_top_k_indices].cpu().tolist()
        eval_top_k_tokens = [self.tokenizer.decode([tid]) for tid in eval_top_k_token_ids]

        # Entropy
        eval_entropy = -torch.sum(probs_eval * torch.log(probs_eval + 1e-10)).item()

        # Yes/no raw logits (all variants)
        yes_logits = {
            self.tokenizer.decode([tid]): logits_eval[tid].item()
            for tid in self.yes_token_ids
        }
        no_logits = {
            self.tokenizer.decode([tid]): logits_eval[tid].item()
            for tid in self.no_token_ids
        }

        # Eval token (argmax)
        eval_token_id = logits_eval.argmax().item()
        eval_token = self.tokenizer.decode([eval_token_id])

        return {
            'yes_prob': yes_prob,
            'no_prob': no_prob,
            'eval_top_k_logits': eval_top_k_logits,
            'eval_top_k_probs': eval_top_k_probs,
            'eval_top_k_tokens': eval_top_k_tokens,
            'eval_top_k_token_ids': eval_top_k_token_ids,
            'eval_entropy': eval_entropy,
            'yes_logits': yes_logits,
            'no_logits': no_logits,
            'eval_token': eval_token,
        }

    def _compute_generation_distribution(self, logits_gen, gen_probs=None, gen_top_k_values=None, gen_top_k_indices=None):
        """
        Compute generation distribution metrics from generation position logits.

        Args:
            logits_gen: Generation logits [vocab_size] for a single batch item

        Returns:
            dict containing:
                - gen_top_k_logits: Top-k logit values
                - gen_top_k_probs: Top-k probabilities
                - gen_top_k_tokens: Top-k decoded tokens
                - gen_top_k_token_ids: Top-k token IDs
                - gen_entropy: Entropy of distribution
        """
        if gen_probs is None:
            probs_gen = torch.nn.functional.softmax(logits_gen, dim=-1)
        else:
            probs_gen = gen_probs

        if gen_top_k_values is None or gen_top_k_indices is None:
            gen_top_k_values, gen_top_k_indices = torch.topk(logits_gen, k=self.top_k)

        gen_top_k_logits = gen_top_k_values.cpu().tolist()
        gen_top_k_token_ids = gen_top_k_indices.cpu().tolist()
        gen_top_k_probs = probs_gen[gen_top_k_indices].cpu().tolist()
        gen_top_k_tokens = [self.tokenizer.decode([tid]) for tid in gen_top_k_token_ids]

        # Entropy
        gen_entropy = -torch.sum(probs_gen * torch.log(probs_gen + 1e-10)).item()

        return {
            'gen_top_k_logits': gen_top_k_logits,
            'gen_top_k_probs': gen_top_k_probs,
            'gen_top_k_tokens': gen_top_k_tokens,
            'gen_top_k_token_ids': gen_top_k_token_ids,
            'gen_entropy': gen_entropy,
        }

    def _decode_next_token(self, next_tokens, batch_idx):
        """
        Extract and decode the current generated token for a batch item.

        Args:
            next_tokens: Token IDs (tensor or list) from generation
            batch_idx: Batch index

        Returns:
            str: Decoded token string (empty string if extraction fails)
        """
        if next_tokens is None or batch_idx >= len(next_tokens):
            return ""

        token_id = next_tokens[batch_idx].item() if isinstance(next_tokens, torch.Tensor) else next_tokens[batch_idx]
        return self.tokenizer.decode([token_id])


    def _build_current_content(self, batch_idx, current_generated_token):
        """
        Build current content by appending to previous step's content.

        Args:
            batch_idx: Batch index
            current_generated_token: Token to append

        Returns:
            str: Current accumulated content
        """
        if self.all_steps[batch_idx]:
            return self.all_steps[batch_idx][-1].current_content + current_generated_token
        else:
            return current_generated_token

    def _create_analyzer_step(self, eval_metrics, gen_metrics, prediction_metrics,
                             current_content, current_generated_token):
        """
        Create a LogitAnalyzerStep from computed metrics.

        Args:
            eval_metrics: Dict from _compute_evaluation_distribution
            gen_metrics: Dict from _compute_generation_distribution
            prediction_metrics: Dict containing prediction_score, uncertainty metrics, is_flagged
            current_content: Current accumulated content string
            current_generated_token: Current generated token string

        Returns:
            LogitAnalyzerStep: Complete step object
        """
        return LogitAnalyzerStep(
            step=self.current_step,
            eval_token=eval_metrics['eval_token'],
            yes_prob=eval_metrics['yes_prob'],
            no_prob=eval_metrics['no_prob'],
            confidence_score=prediction_metrics['prediction_score'],
            is_flagged=prediction_metrics['is_flagged'],
            current_content=current_content,
            current_generated_token=current_generated_token,
            coverage=prediction_metrics['coverage'],
            binary_entropy=prediction_metrics['binary_entropy'],
            eval_worthiness=prediction_metrics['eval_worthiness'],
            # Evaluation distribution
            eval_top_k_logits=eval_metrics['eval_top_k_logits'],
            eval_top_k_probs=eval_metrics['eval_top_k_probs'],
            eval_top_k_tokens=eval_metrics['eval_top_k_tokens'],
            eval_top_k_token_ids=eval_metrics['eval_top_k_token_ids'],
            eval_entropy=eval_metrics['eval_entropy'],
            # Generation distribution
            gen_top_k_logits=gen_metrics['gen_top_k_logits'],
            gen_top_k_probs=gen_metrics['gen_top_k_probs'],
            gen_top_k_tokens=gen_metrics['gen_top_k_tokens'],
            gen_top_k_token_ids=gen_metrics['gen_top_k_token_ids'],
            gen_entropy=gen_metrics['gen_entropy'],
            # Yes/no raw logits
            yes_logits=eval_metrics['yes_logits'],
            no_logits=eval_metrics['no_logits'],
        )

    def add_decoding_step(self, eval_logits, gen_logits, next_tokens, eval_probs=None, eval_top_k_values=None, eval_top_k_indices=None, gen_probs=None, gen_top_k_values=None, gen_top_k_indices=None):
        """
        Add a decoding step and return real-time metrics for resampling decisions.

        Builds complete LogitAnalyzerStep objects directly (no deferred decoding).

        Args:
            eval_logits: Evaluation logits [batch_size, vocab_size] from EVALUATION position
            gen_logits: Generation logits [batch_size, vocab_size] from GENERATION position
            next_tokens: Actually generated tokens [batch_size]
            gen_probs: Precomputed softmax probabilities [batch_size, vocab_size] (optional)
            gen_top_k_values: Precomputed top-k logit values [batch_size, top_k] (optional)
            gen_top_k_indices: Precomputed top-k indices [batch_size, top_k] (optional)

        Returns:
            Dict mapping batch_idx to real-time metrics:
            {
                'yes_prob': float,              # Probability of yes tokens
                'no_prob': float,               # Probability of no tokens
                'confidence_score': float,      # Polarity-aware: >0 = bad answer
                'coverage': float,              # yes_prob + no_prob (engagement)
                'binary_entropy': float,        # Uncertainty between yes/no
                'eval_worthiness': float,       # Worth resampling? (high = uncertain)
            }
        """
        # Initialize batch_size for simple mode
        if self.batch_size is None:
            batch_size = eval_logits.shape[0]
            # Initialize data structures for simple mode
            if not self.all_steps:
                self.all_steps = [[] for _ in range(batch_size)]

                # Use genre config if available
                if self.suffix_prompt_genre and self.suffix_prompt_genre in GENRE_CONFIG:
                    config = GENRE_CONFIG[self.suffix_prompt_genre]
                    self.target_signals = {batch_idx: config["target_signal"] for batch_idx in range(batch_size)}
                    self.signal_polarities = {batch_idx: config["signal_polarity"] for batch_idx in range(batch_size)}
                else:
                    # Default: yes = positive signal
                    self.target_signals = {batch_idx: "yes" for batch_idx in range(batch_size)}
                    self.signal_polarities = {batch_idx: 1 for batch_idx in range(batch_size)}
        else:
            batch_size = self.batch_size

        real_time_metrics = {}

        for batch_idx in range(batch_size):
            if batch_idx >= eval_logits.shape[0]:
                continue

            # Ensure token ID tensors are on the correct device
            if eval_logits.device != self.yes_token_ids_tensor.device:
                self.yes_token_ids_tensor = self.yes_token_ids_tensor.to(eval_logits.device)
                self.no_token_ids_tensor = self.no_token_ids_tensor.to(eval_logits.device)

            # Compute generation distribution (next token logits)
            gen_metrics = self._compute_generation_distribution(
                gen_logits[batch_idx],
                gen_probs=gen_probs[batch_idx] if gen_probs is not None else None,
                gen_top_k_values=gen_top_k_values[batch_idx] if gen_top_k_values is not None else None,
                gen_top_k_indices=gen_top_k_indices[batch_idx] if gen_top_k_indices is not None else None,
            )

            # Compute evaluation distribution (yes/no logits from suffix position)
            eval_metrics = self._compute_evaluation_distribution(
                eval_logits[batch_idx],
                eval_probs=eval_probs[batch_idx] if eval_probs is not None else None,
                eval_top_k_values=eval_top_k_values[batch_idx] if eval_top_k_values is not None else None,
                eval_top_k_indices=eval_top_k_indices[batch_idx] if eval_top_k_indices is not None else None,
            )

            # Extract and decode generated token
            generated_token = self._decode_next_token(
                next_tokens, batch_idx
            )

            # Calculate prediction metrics
            prediction_score = self._calculate_prediction_score(
                eval_metrics['yes_prob'], eval_metrics['no_prob'], batch_idx
            )
            uncertainty_metrics = _calculate_uncertainty_for_eval(
                eval_metrics['yes_prob'], eval_metrics['no_prob'], task_type=self.task_type, current_token=generated_token, step_idx=self.current_step, gen_entropy=gen_metrics['gen_entropy']
            )
            is_flagged = self.is_flagged_response(eval_metrics['eval_token'], batch_idx)

            # Combine into prediction_metrics dict
            prediction_metrics = {
                'prediction_score': prediction_score,
                'is_flagged': is_flagged,
                **uncertainty_metrics,  # Includes coverage, binary_entropy, confidence, eval_worthiness
            }

            # Build current content (incremental)
            current_content = self._build_current_content(batch_idx, generated_token)

            # Create and complete a LogitAnalyzerStep
            step = self._create_analyzer_step(
                eval_metrics, gen_metrics, prediction_metrics,
                current_content, generated_token
            )

            self.all_steps[batch_idx].append(step)

            real_time_metrics[batch_idx] = {
                'yes_prob': eval_metrics['yes_prob'],
                'no_prob': eval_metrics['no_prob'],
                'binary_entropy': uncertainty_metrics['binary_entropy'],
                'prediction_score': prediction_score,
                'coverage': uncertainty_metrics['coverage'],
                'eval_worthiness': uncertainty_metrics['eval_worthiness'],
            }

        self.current_step += 1
        return real_time_metrics

    def finalize(self, final_input_ids: Optional[torch.Tensor] = None, batch_idx: int = 0) -> List[LogitAnalyzerStep]:
        """
        Return already-built LogitAnalyzerStep objects.

        Optionally improves current_content accuracy by re-decoding from final_input_ids.

        Args:
            final_input_ids: Final generated sequence (optional, for accurate context re-building)
            batch_idx: Which batch item to finalize (default 0)

        Returns:
            List of LogitAnalyzerStep objects (already complete from add_decoding_step)
        """
        if not self.all_steps or batch_idx >= len(self.all_steps):
            return []

        steps = self.all_steps[batch_idx]

        # (useful if incremental decoding had issues)
        if final_input_ids is not None and self.input_ids is not None:
            if batch_idx < len(self.input_ids):
                generation_start_offset = len(self.input_ids[batch_idx])

                for step in steps:
                    end_pos = generation_start_offset + step.step + 1
                    if end_pos <= len(final_input_ids):
                        step.current_content = self.tokenizer.decode(
                            final_input_ids[generation_start_offset:end_pos],
                            skip_special_tokens=True
                        )

        return steps

    def get_flagged_steps(self, batch_idx):
        """
        Get all steps where is_flagged is True.

        Returns steps that match the target signal for the evaluation genre.
        """
        if not self.all_steps or batch_idx >= len(self.all_steps):
            return []
        return [step for step in self.all_steps[batch_idx] if step.is_flagged]

    def analyze_eval_logits(self, eval_logits):
        """
        Analyze evaluation logits to extract yes/no probabilities.

        Args:
            eval_logits: Logits from the model at the evaluation position [batch_size, vocab_size]

        Returns:
            yes_prob: Probability of "yes" tokens (scalar tensor)
            no_prob: Probability of "no" tokens (scalar tensor)
        """
        if eval_logits.dim() == 1:
            eval_logits = eval_logits.unsqueeze(0)

        probs = torch.nn.functional.softmax(eval_logits[0], dim=-1)

        # sum probability over all yes/no token variants
        yes_prob = sum(probs[tid].item() for tid in self.yes_token_ids)
        no_prob = sum(probs[tid].item() for tid in self.no_token_ids)

        return torch.tensor(yes_prob), torch.tensor(no_prob)

    def _get_output_directory(self, data_dir):
        prompting_method = self.suffix_prompt_genre

        base_path = [
            data_dir,
            self.experiment,
            self.model_name,
            self.dataset,
        ]
        if self.subtask is not None:
            base_path.append(self.subtask)
        base_path.extend([prompting_method, f"{self.n_shots}_shot"])
        return os.path.join(*base_path)

    def _get_output_path(self, batch_idx):
        data_dir = os.environ.get("SUFFIX_EVAL_OUTPUT_DIR", "./stepwise_info")

        if self.dataset is None or self.subtask is None or self.n_shots is None:
            logger.error(
                f"❌ get_output_path(): dataset={self.dataset}, domain={self.subtask}, n_shots={self.n_shots}"
            )
            raise ValueError("dataset/domain/n_shots must not be None")

        full_dir = self._get_output_directory(data_dir)
        os.makedirs(full_dir, exist_ok=True)

        prompt_hash = hashlib.sha1(self.suffix_prompts[batch_idx].encode()).hexdigest()
        input_hash = hashlib.sha1(self.prompts[batch_idx].encode()).hexdigest()
        file_name = os.path.join(
            full_dir,
            f"{self.suffix_prompt_index}_{input_hash}_{prompt_hash}.json",
        )

        logger.info(f"📁 Output directory created (if not exists): {full_dir}")
        logger.info(f"📝 Output file path: {file_name}")
        logger.info(
            f"🔒 prompt hash: {input_hash[:8]}..., suffix hash: {prompt_hash[:8]}..."
        )

        return file_name

    def already_exists(self, batch_idx):
        file_name = self._get_output_path(batch_idx)
        return os.path.exists(file_name), file_name

    def write_files(self, input_ids, generation_config):
        for batch_idx in range(self.batch_size):
            self._write_file_for_batch_item(batch_idx, input_ids, generation_config)

    def _write_file_for_batch_item(self, batch_idx, input_ids, generation_config):
        file_name = self._get_output_path(batch_idx)

        suffix_prompt_hash = hashlib.sha1(
            self.suffix_prompts[batch_idx].encode()
        ).hexdigest()

        clean_config = {
            k: v
            for k, v in generation_config.to_dict().items()
            if not k.startswith("_")
        }

        # Get final statistics
        batch_steps = self.all_steps[batch_idx]
        total_steps = len(batch_steps)
        flagged_steps = self.get_flagged_steps(batch_idx)
        flagged_steps_count = len(flagged_steps)

        generated_text = self.tokenizer.decode(
            input_ids[batch_idx][len(self.input_ids[batch_idx]) :]
        )

        output_data = {
            "total_steps": total_steps,
            "target_signal_token": self.target_signals[batch_idx],
            "flagged_steps_count": flagged_steps_count,
            "flagged_rate": flagged_steps_count / total_steps if total_steps > 0 else 0,
            "first_flagged_step": next(
                (s.step for s in batch_steps if s.is_flagged), None
            ),
            "flagged_on_first": batch_steps[0].is_flagged if batch_steps else None,
            "generated_text": generated_text,
            "prompt": self.prompts[batch_idx],
            "suffix_prompt": self.suffix_prompts[batch_idx],
            "suffix_prompt_genre": self.suffix_prompt_genre,
            "suffix_prompt_index": self.suffix_prompt_index,
            "suffix_prompt_hash": suffix_prompt_hash,
            "model": self.model_name,
            "n_shots": self.n_shots,
            "generation_config": clean_config,
            "all_steps": [asdict(step) for step in batch_steps],
        }

        with open(file_name, "w") as f:
            json.dump(output_data, f, indent=2)
