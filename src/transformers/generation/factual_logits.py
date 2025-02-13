import torch
from transformers import LogitsProcessor
from typing import Optional, List


class SelfAlignConfig:
    def __init__(
        self,
        factuality_threshold: float = 0.2,
        penalty: float = 5.0,
        factual_keywords: Optional[List[str]] = None,
        non_factual_keywords: Optional[List[str]] = None,
    ):
        """
        Configuration for Self-Align Logits Processor.

        Args:
            factuality_threshold (float): Minimum confidence score to avoid penalty.
            penalty (float): Logit penalty applied when factuality is low.
            factual_keywords (List[str]): Keywords indicating factuality.
            non_factual_keywords (List[str]): Keywords indicating incorrect responses.
        """
        self.factuality_threshold = factuality_threshold
        self.penalty = penalty
        self.factual_keywords = factual_keywords or ["yes", "true", "correct", "valid", "accurate"]
        self.non_factual_keywords = non_factual_keywords or ["no", "false", "incorrect", "wrong"]

    def __repr__(self):
        return (
            f"SelfAlignConfig(factuality_threshold={self.factuality_threshold}, "
            f"penalty={self.penalty}, factual_keywords={self.factual_keywords}, "
            f"non_factual_keywords={self.non_factual_keywords})"
        )


class SelfAlignLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, config=SelfAlignConfig()):
        """
        Custom LogitsProcessor for factuality evaluation.

        Args:
            tokenizer: Hugging Face tokenizer.
            config: SelfAlignConfig with thresholds and penalties.
        """
        self.tokenizer = tokenizer
        self.config = config

        # Encode factual and non-factual keyword tokens
        self.factual_tokens = self._encode_keywords(self.config.factual_keywords)
        self.non_factual_tokens = self._encode_keywords(self.config.non_factual_keywords)

    def _encode_keywords(self, keywords):
        """Encodes keywords into token IDs."""
        return [self.tokenizer.encode(f" {word}", add_special_tokens=False)[0] for word in keywords]

    def compute_yes_no_confidence(self, eval_logits: torch.Tensor, tokenizer) -> torch.Tensor:
        """
        Compute the confidence difference between "Yes" and "No" token probabilities.

        Args:
            eval_logits (torch.Tensor): Logits from the factuality evaluation step.
            tokenizer: Hugging Face tokenizer.

        Returns:
            torch.Tensor: Confidence gap between "Yes" and "No".
        """
        yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
        no_token_id = tokenizer.encode(" no", add_special_tokens=False)[0]

        probs = torch.softmax(eval_logits, dim=-1)

        yes_prob = probs[:, yes_token_id]
        no_prob = probs[:, no_token_id]

        confidence_gap = torch.abs(yes_prob - no_prob)  # How strongly the model prefers one answer
        return confidence_gap

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes entropy for a given logit distribution.

        Args:
            logits (torch.Tensor): Logits of the model output.

        Returns:
            torch.Tensor: Entropy score for each batch element.
        """
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)  # Compute entropy
        return entropy

    def evaluate_sequence(self, eval_logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Evaluates factuality for each batch element.

        Args:
            eval_logits: Logits corresponding to factuality evaluation.
            input_ids: The generated sequence so far.

        Returns:
            Tensor of factuality scores per batch.
        """
        batch_size = input_ids.shape[0]

        factuality_scores = torch.zeros(batch_size, device=eval_logits.device)
        for i in range(batch_size):
            factuality_scores[i] = self._compute_factuality(eval_logits[i])

        return factuality_scores

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, evaluation_logits: torch.Tensor) -> torch.Tensor:
        """
        Adjusts next-token logits based on factuality confidence.

        Returns:
            Adjusted logits tensor.
        """
        if evaluation_logits is not None:
            entropy = self.compute_entropy(evaluation_logits)
            print(f"DEBUG: Computed entropy: {entropy}")
            uncertainty_threshold = 2.0  # Tunable threshold

            # Compute Yes/No confidence gap
            yes_no_confidence = self.compute_yes_no_confidence(evaluation_logits, self.tokenizer)

            print(f"DEBUG: Entropy: {entropy}, Yes/No Confidence: {yes_no_confidence}")

            high_uncertainty_mask = (entropy > uncertainty_threshold) & (yes_no_confidence < self.config.factuality_threshold)

            # Apply stronger penalty by reducing logits directly
            if high_uncertainty_mask.any():
                print(f"DEBUG: High uncertainty detected in {high_uncertainty_mask.sum()} samples.")
                scores[high_uncertainty_mask] *= 0.7  # Stronger penalty for very uncertain responses

        return scores