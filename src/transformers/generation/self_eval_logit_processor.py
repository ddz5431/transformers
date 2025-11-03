import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class LogitAnalyzerStep:
    """
    Complete step information with decoded strings.
    Used after generation completes (via finalize()).

    GENERAL PURPOSE: Can evaluate any yes/no question during generation.
    Examples:
      - Safety: "Is this text unsafe?" (yes = flagged)
      - Correctness: "Is this answer correct?" (yes = flagged)
      - Factuality: "Does this contain errors?" (yes = flagged)
      - Quality: "Is this response helpful?" (no = flagged)
    """
    step: int
    eval_token: str
    yes_prob: float
    no_prob: float
    confidence_score: float  # Polarity-aware: (yes-no)/(yes+no) * polarity
    is_flagged: bool  # Whether eval_token matches target signal
    current_content: str
    last_generated_token: str

    # Uncertainty metrics for resampling decisions
    coverage: float  # yes_prob + no_prob (how much probability mass on yes/no)
    binary_uncertainty: float  # 1 - |yes_norm - no_norm| (uncertainty between yes/no)
    eval_worthiness: float  # coverage * binary_uncertainty (worth evaluating/resampling?)


@dataclass
class _CompactStep:
    """
    Compact step representation during generation.
    No string decoding - stores token IDs only for efficiency.
    """
    step: int
    yes_prob: float
    no_prob: float
    eval_token_id: int
    last_token_id: Optional[int]


GENRE_CONFIG = {
    # General-purpose configurations for different evaluation types
    # target_signal: which token indicates a "positive" match (yes or no)
    # signal_polarity: +1 if "yes" is bad/flagged, -1 if "no" is bad/flagged

    "standard": {"target_signal": "yes", "signal_polarity": 1},
    "positive_detection": {"target_signal": "yes", "signal_polarity": 1},  # "Is X present?" yes=flagged
    "negative_detection": {"target_signal": "no", "signal_polarity": -1},  # "Is X absent?" no=flagged
    "adversarial": {"target_signal": "no", "signal_polarity": -1},
    "negation_traps": {"target_signal": "no", "signal_polarity": -1},

    # Legacy names (for backward compatibility with existing experiments)
    "specific_harm": {"target_signal": "yes", "signal_polarity": 1},
    "safe_control": {"target_signal": "no", "signal_polarity": -1},
}


class LogitAnalyzer:
    """
    GENERAL-PURPOSE evaluation framework for tracking yes/no signals during generation.

    Use cases:
      - Safety evaluation: "Is this text unsafe?" (flag when yes)
      - Correctness checking: "Is this answer correct?" (flag when yes)
      - Factuality: "Does this contain errors?" (flag when yes)
      - Quality assessment: "Is this response helpful?" (flag when no)
      - Uncertainty-guided resampling: Use eval_worthiness for token resampling

    Efficiently tracks evaluation signals during generation with minimal overhead.
    Stores compact representations during generation, defers string decoding to finalize().

    Key Features:
      - Strategy-aware filtering (skip computation for filtered steps)
      - Compact storage during generation (no string decoding until finalize())
      - Real-time metrics for resampling decisions (add_decoding_step returns metrics)
      - Uncertainty quantification (coverage, binary_uncertainty, eval_worthiness)

    Usage:
        # Create analyzer
        analyzer = LogitAnalyzer(
            tokenizer=tokenizer,
            yes_tokens=["Yes", " Yes"],
            no_tokens=["No", " No"],
            suffix_prompt_genre="positive_detection",  # or "negative_detection"
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
        confidence_threshold: float = 0.0,  # NEW: Threshold for flagging
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

            Legacy parameters (for backward compatibility with older experiments):
            input_ids, eval_input_ids, full_input_ids, dataset, subtask, etc.
        """
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.device = device
        self.confidence_threshold = confidence_threshold

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

        self._compact_steps = []
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

            # Legacy: all_steps will be populated by finalize()
            self.all_steps = [[] for _ in range(self.batch_size)]

            # Initialize compact steps for each batch
            self._compact_steps = [[] for _ in range(self.batch_size)]

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
        """Extract token IDs for yes/no token variants."""
        token_ids = []
        for token in tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
            else:
                raise ValueError(f"Token '{token}' encodes to {len(ids)} tokens, expected 1")
        return token_ids

    def _calculate_confidence_score(self, yes_prob: float, no_prob: float, batch_idx: int) -> float:
        """
        Calculate confidence score from yes/no probabilities.

        Returns (yes_prob - no_prob) / (yes_prob + no_prob), adjusted by signal polarity.

        Polarity adjusts whether "yes" or "no" is the target (flagged) signal:
          - polarity=+1: positive score means "yes" (flagged)
          - polarity=-1: positive score means "no" (flagged)

        Higher positive score = stronger signal toward flagged state.
        """
        raw_score = (yes_prob - no_prob) / (yes_prob + no_prob + 1e-8)

        # Apply polarity if available (legacy mode)
        if batch_idx in self.signal_polarities:
            return raw_score * self.signal_polarities[batch_idx]

        # Simple mode: default polarity = 1 (yes = positive)
        return raw_score

    def _calculate_uncertainty_metrics(self, yes_prob: float, no_prob: float):
        """
        Calculate uncertainty metrics for resampling decisions.

        Returns:
            coverage: Probability mass on yes/no tokens (0 to 1)
            binary_uncertainty: Uncertainty between yes and no (0 to 1, higher = more uncertain)
            eval_worthiness: Combined score indicating if resampling might help (0 to 1)
        """
        # Coverage: how much probability mass is on yes/no
        coverage = yes_prob + no_prob

        # Binary uncertainty: how uncertain between yes and no
        if coverage > 1e-8:
            yes_norm = yes_prob / coverage
            no_norm = no_prob / coverage
            binary_uncertainty = 1.0 - abs(yes_norm - no_norm)
        else:
            # Model doesn't engage with yes/no at all
            binary_uncertainty = 1.0  # Maximally uncertain

        # Eval worthiness: worth resampling when model engages AND is uncertain
        # High when: coverage is high (model thinks question is relevant)
        #            AND binary_uncertainty is high (model can't decide)
        eval_worthiness = coverage * binary_uncertainty

        return coverage, binary_uncertainty, eval_worthiness

    def is_flagged_response(self, eval_token: str, batch_idx: int) -> bool:
        """
        Check if evaluation token matches target signal.

        Returns True if token matches the target signal for the genre.
        Examples:
          - "Is this unsafe?" with genre="positive_detection" → True if token is "yes"
          - "Is this helpful?" with genre="negative_detection" → True if token is "no"
        """
        eval_token = eval_token.strip().lower()
        if eval_token not in {"yes", "no"}:
            return False

        # Check target signal if available (legacy mode)
        if batch_idx in self.target_signals:
            return eval_token == self.target_signals[batch_idx]

        # Simple mode: default to "yes" as target
        return eval_token == "yes"

    def add_decoding_step(self, eval_logits, full_input_ids, last_generated_token_ids):
        """
        Add a decoding step and return real-time metrics for resampling decisions.

        Args:
            eval_logits: Evaluation logits [batch_size, vocab_size]
            full_input_ids: Current generated sequence including suffix
            last_generated_token_ids: Most recently generated token(s)

        Returns:
            Dict mapping batch_idx to real-time metrics:
            {
                'yes_prob': float,              # Probability of yes tokens
                'no_prob': float,               # Probability of no tokens
                'confidence_score': float,      # Polarity-aware: >0 = bad answer
                'coverage': float,              # yes_prob + no_prob (engagement)
                'binary_uncertainty': float,    # Uncertainty between yes/no
                'eval_worthiness': float,       # Worth resampling? (high = uncertain)
            }
            Use for real-time resampling during generation. Call finalize() after generation completes.
        """
        # Initialize batch_size for simple mode
        if self.batch_size is None:
            batch_size = eval_logits.shape[0]
            # Initialize data structures for simple mode
            if not self._compact_steps:
                self._compact_steps = [[] for _ in range(batch_size)]

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

        # Store real-time metrics for each batch item (for resampling decisions)
        real_time_metrics = {}

        for batch_idx in range(batch_size):
            if batch_idx >= eval_logits.shape[0]:
                continue

            if self.strategy is not None:
                if hasattr(self.strategy, 'should_evaluate_step_number'):
                    if not self.strategy.should_evaluate_step_number(self.current_step):
                        self.current_step += 1
                        continue

            if eval_logits.device != self.yes_token_ids_tensor.device:
                self.yes_token_ids_tensor = self.yes_token_ids_tensor.to(eval_logits.device)
                self.no_token_ids_tensor = self.no_token_ids_tensor.to(eval_logits.device)

            probs = torch.nn.functional.softmax(eval_logits[batch_idx], dim=-1)

            yes_prob = probs[self.yes_token_ids_tensor].mean().item()
            no_prob = probs[self.no_token_ids_tensor].mean().item()

            eval_token_id = eval_logits[batch_idx].argmax().item()

            # Compute real-time metrics for resampling decisions
            confidence_score = self._calculate_confidence_score(yes_prob, no_prob, batch_idx)
            coverage, binary_uncertainty, eval_worthiness = self._calculate_uncertainty_metrics(yes_prob, no_prob)

            # Store metrics for return (for real-time resampling during generation)
            real_time_metrics[batch_idx] = {
                'yes_prob': yes_prob,
                'no_prob': no_prob,
                'confidence_score': confidence_score,
                'coverage': coverage,
                'binary_uncertainty': binary_uncertainty,
                'eval_worthiness': eval_worthiness,
            }

            # Extract last token ID
            last_token_id = None
            if last_generated_token_ids is not None:
                if isinstance(last_generated_token_ids, torch.Tensor):
                    if last_generated_token_ids.ndim == 0:
                        # Scalar tensor
                        last_token_id = last_generated_token_ids.item()
                    elif last_generated_token_ids.ndim == 1:
                        # 1D tensor - batch
                        if batch_idx < len(last_generated_token_ids):
                            last_token_id = last_generated_token_ids[batch_idx].item()
                    else:
                        # 2D tensor [batch, 1]
                        last_token_id = last_generated_token_ids[batch_idx, 0].item()
                elif isinstance(last_generated_token_ids, list):
                    if batch_idx < len(last_generated_token_ids):
                        last_token_id = last_generated_token_ids[batch_idx]

            compact_step = _CompactStep(
                step=self.current_step,
                yes_prob=yes_prob,
                no_prob=no_prob,
                eval_token_id=eval_token_id,
                last_token_id=last_token_id,
            )

            self._compact_steps[batch_idx].append(compact_step)
            self.current_step += 1

        return real_time_metrics

    def finalize(self, final_input_ids: Optional[torch.Tensor] = None, batch_idx: int = 0) -> List[LogitAnalyzerStep]:
        """
        Args:
            final_input_ids: Final generated sequence (optional, for accurate context building).
                            If provided, builds complete context including skipped evaluation steps.
                            If None, builds context from evaluated tokens only (may have gaps).
            batch_idx: Which batch item to finalize (default 0)

        Returns:
            List of enriched LogitAnalyzerStep objects with decoded strings
        """
        if not self._compact_steps or batch_idx >= len(self._compact_steps):
            # No steps recorded or invalid batch_idx
            return []

        enriched_steps = []
        compact_steps = self._compact_steps[batch_idx]

        # Determine generation start offset for accurate context building
        generation_start_offset = None
        if final_input_ids is not None:
            # Try to determine where generation starts
            if self.input_ids is not None and batch_idx < len(self.input_ids):
                # Legacy mode: we know the original prompt length
                generation_start_offset = len(self.input_ids[batch_idx])
            elif self.full_input_ids is not None and batch_idx < len(self.full_input_ids):
                # Use full_input_ids (includes suffix) as offset
                generation_start_offset = len(self.full_input_ids[batch_idx])

        # Build context for each step
        all_last_token_ids = [step.last_token_id for step in compact_steps if step.last_token_id is not None]

        if all_last_token_ids:
            decoded_tokens_batch = [
                self.tokenizer.decode([tid]) if tid is not None else ""
                for tid in [step.last_token_id for step in compact_steps]
            ]
        else:
            decoded_tokens_batch = ["" for _ in compact_steps]

        # Incremental context (fallback method, may have gaps from skipped steps)
        current_context = ""

        for i, compact_step in enumerate(compact_steps):
            # Build accurate context from final_input_ids if available
            if final_input_ids is not None and generation_start_offset is not None:
                # Decode from generation start up to this evaluation step
                # Each step corresponds to one generated token
                end_pos = generation_start_offset + compact_step.step + 1
                if end_pos <= len(final_input_ids):
                    current_content = self.tokenizer.decode(
                        final_input_ids[generation_start_offset:end_pos],
                        skip_special_tokens=True
                    )
                else:
                    # Fallback to incremental if indexing would fail
                    current_context += decoded_tokens_batch[i]
                    current_content = current_context
            else:
                # Fallback: incremental context (may miss skipped steps)
                current_context += decoded_tokens_batch[i]
                current_content = current_context

            eval_token = self.tokenizer.decode([compact_step.eval_token_id])

            confidence_score = self._calculate_confidence_score(
                compact_step.yes_prob,
                compact_step.no_prob,
                batch_idx
            )

            is_flagged = self.is_flagged_response(eval_token, batch_idx)

            # Calculate uncertainty metrics for resampling
            coverage, binary_uncertainty, eval_worthiness = self._calculate_uncertainty_metrics(
                compact_step.yes_prob,
                compact_step.no_prob
            )

            enriched_step = LogitAnalyzerStep(
                step=compact_step.step,
                eval_token=eval_token,
                yes_prob=compact_step.yes_prob,
                no_prob=compact_step.no_prob,
                confidence_score=confidence_score,
                is_flagged=is_flagged,
                current_content=current_content,
                last_generated_token=decoded_tokens_batch[i],
                coverage=coverage,
                binary_uncertainty=binary_uncertainty,
                eval_worthiness=eval_worthiness,
            )

            enriched_steps.append(enriched_step)

        if batch_idx < len(self.all_steps):
            self.all_steps[batch_idx] = enriched_steps
        else:
            while len(self.all_steps) <= batch_idx:
                self.all_steps.append([])
            self.all_steps[batch_idx] = enriched_steps

        return enriched_steps

    def get_flagged_steps(self, batch_idx):
        """
        Get all steps where is_flagged is True.

        Returns steps that match the target signal for the evaluation genre.
        """
        if not self.all_steps or batch_idx >= len(self.all_steps):
            return []
        return [step for step in self.all_steps[batch_idx] if step.is_flagged]

    def analyze_response_logits(self, eval_logits):
        """
        Analyze evaluation logits to extract yes/no probabilities.

        Used by _sample_with_suffix_evaluation for real-time confidence scoring.

        Args:
            eval_logits: Logits from the model at the evaluation position [batch_size, vocab_size]

        Returns:
            yes_prob: Probability of "yes" tokens (scalar tensor)
            no_prob: Probability of "no" tokens (scalar tensor)
        """
        # Handle both batched and single logits
        if eval_logits.dim() == 1:
            eval_logits = eval_logits.unsqueeze(0)

        # Get probabilities
        probs = torch.nn.functional.softmax(eval_logits[0], dim=-1)

        # Average probability over all yes/no token variants
        yes_prob = sum(probs[tid].item() for tid in self.yes_token_ids) / len(self.yes_token_ids)
        no_prob = sum(probs[tid].item() for tid in self.no_token_ids) / len(self.no_token_ids)

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
        data_dir = os.environ.get("SUFFIX_EVAL_OUTPUT_DIR", "./suffix_eval_results")

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
