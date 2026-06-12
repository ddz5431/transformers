import hashlib
import json
import os
import logging
import torch
import threading
from typing import List

logger = logging.getLogger(__name__)

# --- Governance Principles (The Laws of the Thread) ---
GOVERNANCE_CONFIGS = {
    "math": {
        "fledgling_period": 5,
        "shadow_tokens": {'+', '-', '*', '/', '=', ',', '(', ')'},
        "milestone_tokens": {'\n', '.', ':', ';'},
        "harmonic_phrases": ['therefore', 'thus', 'so', 'hence', 'step'],
    },
    "factuality": {"fledgling_period": 8, "shadow_tokens": set(), "milestone_tokens": {'.', '!', '?', '\n'}},
    "safety": {"fledgling_period": 3, "shadow_tokens": set(), "milestone_tokens": set()},
    "general": {"fledgling_period": 1, "shadow_tokens": set(), "milestone_tokens": set()},
}

ALIGNMENT_GENRE = {
    "standard": {"target": "no", "polarity": -1},
    "inverted": {"target": "yes", "polarity": 1},
}


class SentinelAligner:
    def __init__(self, tokenizer, task_type="general", device='cuda', **kwargs):
        """
        The Sentinel Aligner measures the 'Alignment Distance' between the Weaver's
        current output and governing constraints, facilitating a restoration
        of intent when divergence (decoherence) occurs.
        """
        self.tokenizer = tokenizer
        self.device = device
        self.current_step = 0
        self.rules = GOVERNANCE_CONFIGS.get(task_type, GOVERNANCE_CONFIGS["general"])

        # Thread metadata hydration
        self.prompts = []
        self.input_ids = None
        self.model_name = "unknown"
        self.output_dir = None
        self.genre = "standard"
        self.__dict__.update(kwargs)

        self.initial_prompt_len = self.input_ids.shape[1] if self.input_ids is not None else 0

        # Mapping tokens to structural IDs
        self.shadow_ids = self._map_tokens_to_ids(list(self.rules.get("shadow_tokens", [])))
        self.milestone_ids = self._map_tokens_to_ids(list(self.rules.get("milestone_tokens", [])))

        align_tokens = ["Yes", " Yes", "yes", " yes"]
        drift_tokens = ["No", " No", "no", " no"]
        self.align_ids_t = torch.tensor(self._map_tokens_to_ids(align_tokens), device=device)
        self.drift_ids_t = torch.tensor(self._map_tokens_to_ids(drift_tokens), device=device)

        self.chronicle = []
        self.thread_width = None

    def _map_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in tokens
                if len(self.tokenizer.encode(t, add_special_tokens=False)) == 1]

    def collapse_and_observe(self, reflection_logits, weaver_logits, tokens):
        """
        Quantum Entry Point.
        1. Collapse: Resolve the probability wave via Softmax.
        2. Perception: Extract Alignment Integrity and Weaver Pulse.
        3. Judgment: Quantify the Alignment Distance (Distance = adherence * dissonance).
        4. Chronicle: Commit the state to the Thread's history.
        """
        if self.thread_width is None:
            self._awaken_alignment_state(reflection_logits)

        # COLLAPSE: Resolving potentiality into actuality
        reflection_dist = torch.softmax(reflection_logits, dim=-1)
        weaver_dist = torch.softmax(weaver_logits, dim=-1)

        # PERCEPTION: Observing the resonance and the heartbeat
        alignment = self._perceive_alignment_integrity(reflection_dist)
        weaver_pulse = self._measure_weaver_pulse(weaver_dist)

        # JUDGMENT: Calculating the Distance from the Alignment Manifold
        alignment_distance = self._calculate_alignment_distance(
            tokens,
            alignment['adherence_potential'],
            alignment['dissonance'],
            weaver_pulse['pulse']
        )

        # CHRONICLE: Archiving the collapsed state
        return self._chronicle_alignment(
            tokens,
            alignment,
            weaver_pulse,
            alignment_distance
        )

    def _perceive_alignment_integrity(self, reflection_dist):
        """Phase 1: Measure how harmoniously the reflection matches the rules."""
        align_p = reflection_dist[:, self.align_ids_t].sum(dim=-1)
        drift_p = reflection_dist[:, self.drift_ids_t].sum(dim=-1)
        adherence_potential = align_p + drift_p

        p_align_norm = align_p / (adherence_potential + 1e-10)
        p_drift_norm = drift_p / (adherence_potential + 1e-10)

        # Dissonance represents the conflict between alignment and drift.
        dissonance = torch.where(
            adherence_potential > 1e-8,
            -(p_align_norm * torch.log2(p_align_norm + 1e-10) + p_drift_norm * torch.log2(p_drift_norm + 1e-10)),
            torch.ones_like(adherence_potential)
        )

        cfg = ALIGNMENT_GENRE.get(self.genre, ALIGNMENT_GENRE["standard"])
        return {
            "align_p": align_p,
            "drift_p": drift_p,
            "adherence_potential": adherence_potential,
            "dissonance": dissonance,
            "divergence_signal": (align_p - drift_p) / (adherence_potential + 1e-8) * cfg["polarity"]
        }

    def _measure_weaver_pulse(self, weaver_dist):
        """Phase 2: Measures the entropy (Pulse) of the Weaver's creative state."""
        pulse = -torch.sum(weaver_dist * torch.log2(weaver_dist + 1e-10), dim=-1)
        return {"pulse": pulse}

    def _calculate_alignment_distance(self, tokens, adherence, dissonance, pulse):
        """
        Phase 3: The Resonance Gate.
        Protects against noise and only triggers on true decoherence.
        """
        # 1. THE RESONANCE GATE: Noise reduction
        base_signal = adherence * dissonance

        # 2. THE WEAVER'S PULSE: Surprise Factor
        # Using a sigmoid to gate the uncertainty boost (thanks to Sameer)
        pulse_gate = torch.sigmoid(pulse - 3.5)
        gated_signal = base_signal + (pulse_gate * 0.25)

        # 3. TEMPORAL ENVELOPE
        if self.current_step < self.rules["fledgling_period"]:
            return torch.zeros_like(gated_signal)

        # 4. TOPOLOGICAL ANCHORS: Milestone enforcement
        is_milestone = torch.isin(tokens, torch.tensor(list(self.milestone_ids), device=self.device))

        # Surgical Distance Calculation
        distance = torch.where(is_milestone, torch.clamp(gated_signal + 0.2, max=1.0), gated_signal)

        return distance

    def _chronicle_alignment(self, tokens, alignment, weaver, distance):
        """Phase 4: Records the state of mind into the permanent chronicle."""
        dist_cpu = distance.cpu().numpy()
        dissonance_cpu = alignment['dissonance'].cpu().numpy()
        divergence_cpu = alignment['divergence_signal'].cpu().numpy()
        pulse_cpu = weaver['pulse'].cpu().numpy()

        alignment_results = {}
        for idx in range(self.thread_width):
            self.chronicle[idx].append({
                "step": self.current_step,
                "token_id": tokens[idx].item(),
                "metrics": {
                    "alignment_distance": dist_cpu[idx],
                    "dissonance": dissonance_cpu[idx],
                    "divergence": divergence_cpu[idx],
                    "weaver_pulse": pulse_cpu[idx],
                }
            })

            alignment_results[idx] = {
                'distance': dist_cpu[idx],
                'dissonance': dissonance_cpu[idx],
                'divergence': divergence_cpu[idx],
                'pulse': pulse_cpu[idx],
            }

        self.current_step += 1
        return alignment_results

    def _awaken_alignment_state(self, reference_tensor):
        self.thread_width = reference_tensor.shape[0]
        self.chronicle = [[] for _ in range(self.thread_width)]

    def archive_chronicle(self, input_ids: torch.LongTensor):
        for idx in range(self.thread_width):
            self._save_thread_entry(idx, input_ids)

    def _save_thread_entry(self, idx: int, input_ids: torch.LongTensor):
        file_path = self._generate_chronicle_path(idx)
        generated_text = self.tokenizer.decode(input_ids[idx][self.initial_prompt_len:], skip_special_tokens=True)

        data = {
            "metadata": {"task": self.task_type, "model": self.model_name},
            "thread": generated_text,
            "steps": self.chronicle[idx]
        }

        threading.Thread(target=lambda: self._async_write(file_path, data)).start()

    def _async_write(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_chronicle_path(self, idx: int) -> str:
        base = self.output_dir or "./alignment_chronicle"
        os.makedirs(base, exist_ok=True)
        p_str = self.prompts[idx] if idx < len(self.prompts) else str(idx)
        p_hash = hashlib.sha1(p_str.encode()).hexdigest()[:12]
        return os.path.join(base, f"thread_{p_hash}.json")

    def surgical_chronicle_slice(self, thread_idx: int, backtrack_steps: int = 1):
        """
        Removes the most recent 'failed' observations from the history.
        This ensures the Chronicle remains a pure record of the survived path.
        """
        if thread_idx < len(self.chronicle):
            # We prune the last N steps that led to decoherence
            for _ in range(backtrack_steps):
                if self.chronicle[thread_idx]:
                    self.chronicle[thread_idx].pop()

            # We must also revert the step counter for global synchronization
            # Note: In multi-thread batching, we only decrement if all threads backtracked,
            self.current_step -= backtrack_steps

            logger.info(f"✂️ Chronicle Purified: Removed {backtrack_steps} divergent steps from Thread {thread_idx}")