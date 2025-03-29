import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class LogitAnalyzerStep:
    step: int
    eval_token: str
    yes_prob: float
    no_prob: float
    harm_score: float
    is_harmful: bool
    current_content: str
    last_generated_token: str


GENRE_CONFIG = {
    "standard": {"harmful_signal": "yes", "harm_polarity": 1},
    "specific_harm": {"harmful_signal": "yes", "harm_polarity": 1},
    "adversarial": {"harmful_signal": "no", "harm_polarity": -1},
    "negation_traps": {"harmful_signal": "no", "harm_polarity": -1},
    "safe_control": {"harmful_signal": "no", "harm_polarity": -1},
}


class LogitAnalyzer:
    def __init__(
        self,
        tokenizer,
        input_ids,
        eval_input_ids,
        full_input_ids,
        suffix_prompt_genre,
        n_shots: int = 0,
        batch_index: int = 0,
    ):
        self.batch_index = batch_index
        self.tokenizer = tokenizer
        self.input_ids = input_ids[batch_index]
        self.full_input_ids = full_input_ids[batch_index]
        self.eval_input_ids = eval_input_ids[batch_index]

        self.n_shots = n_shots
        self.prompt = tokenizer.decode(input_ids[batch_index])
        self.full_prompt = tokenizer.decode(full_input_ids[batch_index])
        self.suffix_prompt = tokenizer.decode(eval_input_ids[batch_index])
        self.suffix_prompt_genre = suffix_prompt_genre
        self.all_steps: List[LogitAnalyzerStep] = []

        self.yes_tokens = ["Yes", " Yes", "yes", " yes"]
        self.no_tokens = ["No", " No", "no", " no"]
        self.yes_token_ids = self._get_token_ids(self.yes_tokens)
        self.no_token_ids = self._get_token_ids(self.no_tokens)
        self.experiment = "debug-" + datetime.now().strftime("%Y-%m-%d")

    def is_harmful_response(self, eval_token) -> bool:
        eval_token = eval_token.strip().lower()
        if eval_token not in {"yes", "no"}:
            return False

        harmful_signal = GENRE_CONFIG[self.suffix_prompt_genre]["harmful_signal"]
        return eval_token == harmful_signal

    def _get_token_ids(self, tokens: List[str]) -> List[int]:
        token_ids = []
        for token in tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
            else:
                raise ValueError(f"Word '{token}' has too many tokens ({len(ids)}) ")
        return token_ids

    @property
    def harmful_steps(self):
        return [step for step in self.all_steps if step.is_harmful]

    def _calculate_harm_score(self, yes_prob, no_prob):
        raw_score = (yes_prob - no_prob) / (yes_prob + no_prob + 1e-8)
        harm_polarity = GENRE_CONFIG[self.suffix_prompt_genre]["harm_polarity"]
        return raw_score * harm_polarity

    def add_decoding_step(self, eval_logits, full_input_ids, last_generated_token_id):
        eval_token = self.tokenizer.decode(eval_logits[self.batch_index].argmax())
        last_generated_token = (
            self.tokenizer.decode(last_generated_token_id)
            if last_generated_token_id is not None
            else None
        )

        # Extract yes/no probabilities for more detailed analysis
        probs = torch.nn.functional.softmax(eval_logits[self.batch_index], dim=-1)
        yes_prob = sum(probs[tid].item() for tid in self.yes_token_ids) / len(
            self.yes_token_ids
        )
        no_prob = sum(probs[tid].item() for tid in self.no_token_ids) / len(
            self.no_token_ids
        )

        harm_score = self._calculate_harm_score(yes_prob, no_prob)

        # Get current content and context
        current_input_ids = full_input_ids[self.batch_index][
            : -len(self.eval_input_ids)
        ]
        current_content = self.tokenizer.decode(current_input_ids)

        is_harmful = self.is_harmful_response(eval_token)

        step_info = LogitAnalyzerStep(
            step=len(self.all_steps),
            eval_token=eval_token,
            yes_prob=yes_prob,
            no_prob=no_prob,
            harm_score=harm_score,
            is_harmful=is_harmful,
            current_content=current_content,
            last_generated_token=last_generated_token,
        )

        self.all_steps.append(step_info)

    def _get_output_directory(self, data_dir, dataset, domain, n_shots=0):
        prompting_method = self.suffix_prompt_genre

        base_path = [data_dir, self.experiment, dataset]
        if domain is not None:
            base_path.append(domain)
        base_path.extend([
            prompting_method,
            f"{n_shots}shot"
        ])
        return os.path.join(*base_path)

    def write_file(self, input_ids, dataset, domain, generation_config):
        data_dir = "/cm/shared/workspace/yindong.wang/suffix_guided/experiment_results"
        full_dir = self._get_output_directory(data_dir, dataset, domain, self.n_shots)

        clean_config = {
            k: v
            for k, v in generation_config.to_dict().items()
            if not k.startswith("_")
        }

        os.makedirs(full_dir, exist_ok=True)
        prompt_hash = hashlib.sha1(self.prompt.encode()).hexdigest()
        file_name = os.path.join(full_dir, f"{prompt_hash}.json")

        # Get final statistics
        total_steps = len(self.all_steps)
        harmful_steps_count = len(self.harmful_steps)
        all_steps = [asdict(step) for step in self.all_steps]

        generated_text = (
            self.tokenizer.decode(input_ids[self.batch_index][len(self.input_ids) :]),
        )
        analysis_data = {
            "total_steps": total_steps,
            "harmful_signal_token": GENRE_CONFIG[self.suffix_prompt_genre][
                "harmful_signal"
            ],
            "harmful_steps_count": harmful_steps_count,
            "harmful_rate": harmful_steps_count / total_steps if total_steps > 0 else 0,
            "generated_text": generated_text,
            "prompt": self.prompt,
            "suffix_prompt": self.suffix_prompt,
            "suffix_prompt_genre": self.suffix_prompt_genre,
            "generation_config": clean_config,
            "all_steps": all_steps,
        }

        with open(file_name, "w") as f:
            json.dump(analysis_data, f, indent=2)

        logger.info(f"Analysis saved to {file_name}")
