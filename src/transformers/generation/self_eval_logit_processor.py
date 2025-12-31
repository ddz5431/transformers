import hashlib
import json
import os
import logging
import torch
import numpy as np
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# --- Task-specific checkpoint configurations ---
TASK_CHECKPOINT_CONFIGS = {
    "math": {
        "min_tokens_before_eval": 5,
        "bad_checkpoint_tokens": {'+', '-', '*', '/', '=', ',', '(', ')'},
        "good_checkpoint_tokens": {'\n', '.', ':', ';'},
        "good_checkpoint_phrases": ['therefore', 'thus', 'so', 'hence', 'step'],
        "use_gen_entropy": True,
    },
    "factuality": {"min_tokens_before_eval": 8, "bad_checkpoint_tokens": set(),
                   "good_checkpoint_tokens": {'.', '!', '?', '\n'}, "use_gen_entropy": False},
    "safety": {"min_tokens_before_eval": 3, "bad_checkpoint_tokens": set(), "good_checkpoint_tokens": set(),
               "use_gen_entropy": False},
    "general": {"min_tokens_before_eval": 1, "bad_checkpoint_tokens": set(), "good_checkpoint_tokens": set(),
                "use_gen_entropy": False},
}

GENRE_CONFIG = {
    "standard": {"target_signal": "no", "signal_polarity": -1},
    "inverted_standard": {"target_signal": "yes", "signal_polarity": 1},
    "adversarial": {"target_signal": "no", "signal_polarity": -1},
    "negation_traps": {"target_signal": "no", "signal_polarity": -1},
}


@dataclass
class LogitAnalyzerStep:
    step: int
    eval_token: str
    yes_prob: float
    no_prob: float
    error_signal: float
    is_flagged: bool
    current_content: str
    current_generated_token: str
    coverage: float
    binary_entropy: float
    certainty: float
    eval_worthiness: float
    eval_top_k_tokens: List[str]
    gen_top_k_tokens: List[str]
    gen_entropy: float


class LogitAnalyzer:
    def __init__(self, tokenizer, task_type="general", device='cuda', **kwargs):
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.device = device
        self.current_step = 0
        self.config = TASK_CHECKPOINT_CONFIGS.get(task_type, TASK_CHECKPOINT_CONFIGS["general"])

        # Meta-data hydration
        self.prompts = []
        self.input_ids = None
        self.initial_prompt_len = input_ids.shape[1] if input_ids is not None else 0
        self.model_name = "unknown"
        self.dataset = "unknown"
        self.experiment = "default"
        self.output_dir = None
        self.suffix_prompt_genre = "standard"
        self.top_k = 20
        self.__dict__.update(kwargs)

        # Vectorized mapping for loop speed
        self.bad_ids = self._get_ids(list(self.config.get("bad_checkpoint_tokens", [])))
        self.good_ids = self._get_ids(list(self.config.get("good_checkpoint_tokens", [])))

        yes_ids = self._get_ids(["Yes", " Yes", "yes", " yes"])
        no_ids = self._get_ids(["No", " No", "no", " no"])
        self.yes_ids_t = torch.tensor(yes_ids, device=device)
        self.no_ids_t = torch.tensor(no_ids, device=device)

        self.all_steps = []
        self.batch_size = None

    def _get_ids(self, tokens: List[str]) -> List[int]:
        return [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in tokens
                if len(self.tokenizer.encode(t, add_special_tokens=False)) == 1]

    def add_decoding_step(self, eval_logits, gen_logits, next_tokens,
                          eval_probs=None, eval_top_k_indices=None,
                          gen_probs=None, gen_top_k_indices=None):
        if self.batch_size is None:
            self.batch_size = eval_logits.shape[0]
            self.all_steps = [[] for _ in range(self.batch_size)]
            self.cfg_genre = GENRE_CONFIG.get(self.suffix_prompt_genre, GENRE_CONFIG["standard"])

        # GPU Vectorized Math
        e_probs = eval_probs if eval_probs is not None else torch.softmax(eval_logits, dim=-1)
        g_probs = gen_probs if gen_probs is not None else torch.softmax(gen_logits, dim=-1)

        y_p_v = e_probs[:, self.yes_ids_t].sum(dim=-1)
        n_p_v = e_probs[:, self.no_ids_t].sum(dim=-1)
        g_ent_v = -torch.sum(g_probs * torch.log2(g_probs + 1e-10), dim=-1)

        cov_v = y_p_v + n_p_v
        p_yes, p_no = y_p_v / (cov_v + 1e-10), n_p_v / (cov_v + 1e-10)
        bin_ent_v = torch.where(cov_v > 1e-8,
                                -(p_yes * torch.log2(p_yes + 1e-10) + p_no * torch.log2(p_no + 1e-10)),
                                torch.ones_like(cov_v))

        real_time_metrics = {}
        for b in range(self.batch_size):
            y_p, n_p, b_e, g_e, cov = y_p_v[b].item(), n_p_v[b].item(), bin_ent_v[b].item(), g_ent_v[b].item(), cov_v[
                b].item()
            gen_id = next_tokens[b].item()

            worth = cov * b_e
            if self.current_step < self.config["min_tokens_before_eval"] or gen_id in self.bad_ids:
                worth = 0.0
            elif gen_id in self.good_ids:
                worth = min(1.0, worth + 0.15)

            if self.config.get("use_gen_entropy") and g_e > 5.0:
                worth *= (0.5 + (1.0 / (1.0 + np.exp(-(g_e - 8.0) / 2.0))))

            self.all_steps[b].append({
                "step": self.current_step, "gen_id": gen_id,
                "y_p": y_p, "n_p": n_p, "b_e": b_e, "g_e": g_e, "cov": cov, "worth": min(1.0, worth),
                "e_top_k": eval_top_k_indices[b].tolist() if eval_top_k_indices is not None else [],
                "g_top_k": gen_top_k_indices[b].tolist() if gen_top_k_indices is not None else []
            })

            real_time_metrics[b] = {
                'worth': worth, 'entropy': b_e,
                'error_sig': (y_p - n_p) / (cov + 1e-8) * self.cfg_genre["signal_polarity"]
            }

        self.current_step += 1
        return real_time_metrics

    def get_response_level_confidence(self, batch_idx: int, aggregation: str = "min") -> Dict[str, Any]:
        steps = self.all_steps[batch_idx]
        if not steps: return {"certainty": 0.0}

        step_certainties = [1.0 - s["b_e"] for s in steps]

        if aggregation == "min":
            val = min(step_certainties)
        elif aggregation == "mean":
            val = float(np.mean(step_certainties))
        elif aggregation == "product":
            val = float(np.prod(step_certainties))
        else:
            val = float(np.mean(step_certainties))

        return {"response_certainty": val, "min": min(step_certainties), "mean": float(np.mean(step_certainties))}

    def get_all_aggregation_methods(self, batch_idx: int) -> Dict[str, Any]:
        return {m: self.get_response_level_confidence(batch_idx, m) for m in ["min", "mean", "product"]}

    def finalize(self, batch_idx: int) -> List[LogitAnalyzerStep]:
        compact = self.all_steps[batch_idx]
        gen_tokens = self.tokenizer.convert_ids_to_tokens([s["gen_id"] for s in compact])
        full_steps, content = [], ""

        for i, s in enumerate(compact):
            t_str = self.tokenizer.convert_tokens_to_string([gen_tokens[i]])
            content += t_str
            eval_tok = "yes" if s["y_p"] > s["n_p"] else "no"
            full_steps.append(LogitAnalyzerStep(
                step=s["step"], eval_token=eval_tok, yes_prob=s["y_p"], no_prob=s["n_p"],
                error_signal=(s["y_p"] - s["n_p"]) / (s["cov"] + 1e-8) * self.cfg_genre["signal_polarity"],
                is_flagged=(eval_tok == self.cfg_genre["target_signal"]),
                current_content=content, current_generated_token=t_str,
                coverage=s["cov"], binary_entropy=s["b_e"], certainty=1.0 - s["b_e"],
                eval_worthiness=s["worth"], gen_entropy=s["g_e"],
                eval_top_k_tokens=self.tokenizer.convert_ids_to_tokens(s["e_top_k"]),
                gen_top_k_tokens=self.tokenizer.convert_ids_to_tokens(s["g_top_k"])
            ))

        return full_steps

    def write_files(self, input_ids: torch.LongTensor, generation_config: Any):
        for b in range(self.batch_size):
            self._write_file_for_batch_item(b, input_ids, generation_config)

    def _write_file_for_batch_item(self, b: int, input_ids: torch.LongTensor, gen_cfg: Any):
        full_steps = self.finalize(b)
        file_path = self._get_output_path(b)

        prompt_len = self.input_ids.shape[1] if self.input_ids is not None else 0
        gen_text = self.tokenizer.decode(input_ids[b][self.initial_prompt_len:], skip_special_tokens=True)

        data = {
            "metadata": {"task": self.task_type, "model": self.model_name},
            "metrics": {"agg": self.get_all_aggregation_methods(b)},
            "text": {"generated": gen_text},
            "steps": [asdict(s) for s in full_steps]
        }

        def save():
            with open(file_path, "w") as f: json.dump(data, f, indent=2)

        threading.Thread(target=save).start()

    def _get_output_path(self, b: int) -> str:
        base = self.output_dir or "./stepwise_info"
        path = os.path.join(base, self.experiment, self.model_name, self.task_type)
        os.makedirs(path, exist_ok=True)
        p_str = self.prompts[b] if b < len(self.prompts) else str(b)
        p_hash = hashlib.sha1(p_str.encode()).hexdigest()[:12]
        return os.path.join(path, f"step_{p_hash}.json")