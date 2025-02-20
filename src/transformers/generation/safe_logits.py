import torch
from transformers import LogitsProcessor
from typing import Optional, List


class SelfAlignConfig:
    def __init__(
        self,
        safety_threshold: float = 0.2,  # Minimum confidence required
        penalty: float = 0.5,  # Logit penalty when below threshold
        apply_eos: bool = True,  # Whether to enforce EOS when confidence is low
    ):
        """
        Configuration for Self-Align Logits Processor.

        Args:
            safety_threshold (float): Minimum confidence score to avoid penalty.
            penalty (float): Logit penalty applied when confidence is low.
            apply_eos (bool): Whether to force the model to stop generation when confidence is too low.
        """
        self.safety_threshold = safety_threshold
        self.penalty = penalty
        self.apply_eos = apply_eos

    def __repr__(self):
        return (
            f"SelfAlignConfig(safety_threshold={self.safety_threshold}, "
            f"penalty={self.penalty}, apply_eos={self.apply_eos})"
        )


class SelfAlignLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, config=SelfAlignConfig()):
        """
        Custom LogitsProcessor for Yes/No confidence evaluation.

        Args:
            tokenizer: Hugging Face tokenizer.
            config: SelfAlignConfig with thresholds and penalties.
        """
        self.tokenizer = tokenizer
        self.config = config

        # Encode Yes/No token IDs
        self.yes_token_ids = self.tokenizer(["yes", "Yes", "YES"], add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze()
        self.no_token_ids = self.tokenizer(["no", "No", "NO"], add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze()

        # EOS token ID
        self.eos_token_id = tokenizer.eos_token_id

    def compute_yes_no_confidence(self, eval_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the confidence difference between "Yes" and "No" token probabilities.

        Args:
            eval_logits (torch.Tensor): Logits from safety evaluation.

        Returns:
            torch.Tensor: Confidence score.
        """
        probs = torch.softmax(eval_logits, dim=-1)

        yes_prob = probs[:, self.yes_token_id]
        no_prob = probs[:, self.no_token_id]

        confidence_gap = torch.abs(yes_prob - no_prob)  # Strength of preference
        return confidence_gap

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, evaluation_logits: torch.Tensor) -> torch.Tensor:
        """
        Adjusts next-token logits based on Yes/No confidence.

        Returns:
            Adjusted logits tensor.
        """
        if evaluation_logits is not None:
            max_token_ids = evaluation_logits.argmax(dim=-1)
            # Create a mask where evaluation logits indicate a "yes" token
            mask = torch.isin(max_token_ids, self.yes_token_ids.to(evaluation_logits.device))  # Shape: (batch_size,)

            if mask.any():
                # Set all scores to a very low value except for EOS token
                eos_token_tensor = torch.tensor(self.eos_token_id, device=scores.device)
                scores[mask] = torch.full_like(scores[mask], float('-inf'))  # Mask everything
                scores[mask, eos_token_tensor] = 0  # Allow only EOS token

        return scores