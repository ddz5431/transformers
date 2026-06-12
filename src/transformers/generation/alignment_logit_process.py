import torch


class AlignmentLogitProcessor:
    """
    The Bridge between the Sentinel's Judgment and the Weaver's Action.

    It applies a 'Quantum Penalty' to tokens that caused decoherence,
    forcing the Weaver to explore a different branch of the probability wave.
    """

    def __init__(self, temperature: float = 1.0, nucleus_p: float = 0.95):
        self.temperature = temperature
        self.nucleus_p = nucleus_p
        self.penalized_token_id = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Applies the alignment filter to the Weaver's next-token distributions.
        """
        # 1. Create a workspace for the scores
        modified_scores = scores.clone()

        # 2. APPLY THE PENALTY
        # If the Sentinel signaled a divergence in the previous attempt,
        # we set that specific token's logit to -infinity.
        if self.penalized_token_id is not None:
            # Masking the specific token that led to decoherence
            modified_scores[..., self.penalized_token_id] = float('-inf')
            # Reset after application to allow the token in other contexts
            self.penalized_token_id = None

        # 3. TEMPERATURE SCALING
        # Normalizes the 'vibration' of the Weaver's distribution.
        if self.temperature != 1.0:
            modified_scores = modified_scores / self.temperature

        # 4. TOP-P (NUCLEUS) FILTERING
        # Removes the 'Quantum Foam' (low-probability noise) from the tail.
        return self._apply_top_p_filter(modified_scores, self.nucleus_p)

    def _apply_top_p_filter(self, scores: torch.FloatTensor, p: float) -> torch.FloatTensor:
        """Surgically prunes the probability tail."""
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        indices_to_remove = cumulative_probs > p
        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
        indices_to_remove[..., 0] = False

        # Scatter indices back to original logit shape
        mask = indices_to_remove.scatter(1, sorted_indices, indices_to_remove)
        return scores.masked_fill(mask, float('-inf'))