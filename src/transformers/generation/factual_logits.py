import torch
from transformers import LogitsProcessor
from typing import Optional, List


class SelfAlignConfig:
    def __init__(
        self,
        factuality_threshold: float = 0.3,
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

    def _compute_factuality(self, eval_logits: torch.FloatTensor) -> torch.Tensor:
        """
        Compute factuality confidence based on model's probabilities.

        Args:
            eval_logits: Logits at the factuality evaluation step.

        Returns:
            A factuality score tensor.
        """
        probs = torch.softmax(eval_logits, dim=-1)

        # Ensure lists are not empty
        factual_probs = (
            torch.index_select(probs, dim=-1, index=torch.tensor(self.factual_tokens, device=probs.device)).sum()
            if len(self.factual_tokens) > 0 else torch.tensor(0.0, device=eval_logits.device)
        )

        non_factual_probs = (
            torch.index_select(probs, dim=-1, index=torch.tensor(self.non_factual_tokens, device=probs.device)).sum()
            if len(self.non_factual_tokens) > 0 else torch.tensor(0.0, device=eval_logits.device)
        )

        # Normalize each category by token count
        num_factual_tokens = max(len(self.factual_tokens), 1)  # Avoid divide-by-zero
        num_non_factual_tokens = max(len(self.non_factual_tokens), 1)

        normalized_factual_prob = factual_probs / num_factual_tokens
        normalized_non_factual_prob = non_factual_probs / num_non_factual_tokens

        # Compute factuality score using normalized values
        total_prob = normalized_factual_prob + normalized_non_factual_prob + 1e-6  # Prevent divide-by-zero
        factuality_score = normalized_factual_prob / total_prob

        return factuality_score

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
            # Compute factuality scores per batch
            factuality_scores = self.evaluate_sequence(evaluation_logits, input_ids[:, -1])

            # Identify sequences with low factuality
            penalty_mask = factuality_scores < self.config.factuality_threshold

            # Apply stronger penalty by reducing logits directly
            scores[penalty_mask] -= self.config.penalty

            print(f"DEBUG: Factuality Scores = {factuality_scores.tolist()} | Applied Penalty: {penalty_mask.sum().item()} sequences")

        return scores
