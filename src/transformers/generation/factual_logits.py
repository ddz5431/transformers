from typing import Optional, List

import torch
from transformers import LogitsProcessor


class SelfAlignConfig:
    def __init__(
        self,
        factuality_threshold: float = 0.00001,
        penalty: float = 0.7,
        factual_keywords: Optional[List[str]] = None,
        non_factual_keywords: Optional[List[str]] = None,
    ):
        """
        Configuration for the Self-Align Logits Processor.

        Args:
            factuality_threshold (float): Minimum score required for logits to pass without penalty.
            penalty (float): Penalty multiplier applied to logits if factuality score is below the threshold.
            factual_keywords (List[str]): List of keywords used for factuality evaluation. Default is predefined.
        """
        self.factuality_threshold = factuality_threshold
        self.penalty = penalty
        self.factual_keywords = factual_keywords or ["true", "correct", "valid", "accurate", "yes"]
        self.non_factual_keywords = non_factual_keywords or ["no", "false", "incorrect", "wrong"]

    def __repr__(self):
        """
        Representation for debugging and logging.
        """
        return (
            f"SelfAlignConfig("
            f"factuality_threshold={self.factuality_threshold}, "
            f"penalty={self.penalty}, "
            f"factual_keywords={self.factual_keywords})"
            f"non_factual_keywords={self.non_factual_keywords})"
        )


class SelfAlignLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, config=SelfAlignConfig()):
        """
        Custom LogitsProcessor for Self-Align factuality evaluation.

        Args:
            tokenizer: Hugging Face tokenizer for token conversions.
            config: SelfAlignConfig object containing thresholds and penalties.
        """
        self.tokenizer = tokenizer
        self.config = config

    def evaluate_sequence(self, eval_logits: torch.Tensor, input_ids: torch.Tensor) -> List[
        Optional[torch.Tensor]]:
        """
        Evaluates factuality based on logits at the evaluation position.

        Args:
            eval_logits: Logits corresponding to the factuality evaluation position.
            input_ids: The sequence of token IDs generated so far.

        Returns:
            A list of factuality scores, one per batch element.
        """
        batch_size = input_ids.shape[0]

        factuality_scores = []

        for i in range(batch_size):
            factuality_score = self._compute_factuality(eval_logits[i])
            factuality_scores.append(factuality_score)

        print(f"DEBUG: factuality_scores at end of evaluate_sequence: {factuality_scores}")
        return factuality_scores  # Ensure it matches batch size

    def _compute_factuality(self, eval_logits: torch.FloatTensor) -> torch.Tensor:
        """
        Compute factuality based on model's confidence in 'yes' vs. 'no'.

        Args:
            eval_logits: Logits at the factuality evaluation step.

        Returns:
            A factuality score tensor.
        """
        # Tokenize response choices
        factual_tokens = [self.tokenizer.encode(f" {word}", add_special_tokens=False) for word in
                          self.config.factual_keywords]
        factual_tokens = sum(factual_tokens, [])  # Flatten the list

        # Encode "non-factual" words from config
        non_factual_tokens = [self.tokenizer.encode(f" {word}", add_special_tokens=False) for word in
                              self.config.non_factual_keywords]
        non_factual_tokens = sum(non_factual_tokens, [])  # Flatten

        # Compute softmax probabilities
        probs = torch.softmax(eval_logits, dim=-1)

        factual_prob = probs[factual_tokens].sum(dim=-1) if factual_tokens else torch.tensor(0.0,
                                                                                             device=eval_logits.device)
        non_factual_prob = probs[non_factual_tokens].sum(dim=-1) if non_factual_tokens else torch.tensor(0.0,
                                                                                                         device=eval_logits.device)

        # Normalize
        total_prob = factual_prob + non_factual_prob + 1e-6  # Prevent divide-by-zero
        factuality_score = factual_prob / total_prob  # Confidence in factual response

        return factuality_score

    def __call__(self,
                 input_ids: torch.Tensor,
                 scores: torch.Tensor,
                 evaluation_logits: torch.Tensor) -> torch.FloatTensor:
        """
        Adjusts next-token logits based on Self-Align factuality evaluation.

        Returns:
            Adjusted logits for next-token prediction.
        """
        if evaluation_logits is not None:
            # Evaluate factuality
            factuality_scores = self.evaluate_sequence(evaluation_logits, input_ids[:, -1])
            print(f"DEBUG: factuality_scores before filtering: {factuality_scores}")

            # Convert None values to default factuality (1.0)
            default_factuality = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            factuality_scores = [score if score is not None else default_factuality for score in factuality_scores]

            # Stack to create tensor
            factuality_scores = torch.stack(factuality_scores).to(scores.device)

            # Ensure factuality_scores has the correct shape
            if factuality_scores.dim() == 1:
                factuality_scores = factuality_scores.unsqueeze(1)  # Shape [batch_size, 1]

            print(f"DEBUG: factuality_scores final shape: {factuality_scores.shape}, scores shape: {scores.shape}")

            # Expand factuality_scores to match scores shape
            penalty_mask = (factuality_scores < self.config.factuality_threshold).expand(scores.shape)

            # Apply penalty: Reduce probability for tokens when factuality is low
            scores = scores * (~penalty_mask + penalty_mask * self.config.penalty)

        return scores
