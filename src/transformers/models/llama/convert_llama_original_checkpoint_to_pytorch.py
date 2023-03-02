import argparse
import json
import re
from pathlib import Path

import torch

from transformers import LLaMaConfig, LLaMaForCausalLM, LLaMaTokenizer


def generate_config(params_json_path: Path, vocab_size: int) -> LLaMaConfig:
    with open(params_json_path, "r") as fi:
        hyperparameters = json.load(fi)

    assert hyperparameters["vocab_size"] == -1, "We get vocab size information from the tokenizer"
    assert vocab_size > 0

    hidden_size = hyperparameters["dim"]
    multiple_of = hyperparameters["multiple_of"]
    intermediate_size = int(2 * hidden_size / 3)
    intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

    return LLaMaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=hyperparameters["n_layers"],
        num_attention_heads=hyperparameters["n_heads"],
        intermediate_size=intermediate_size,
        max_position_embeddings=2048,
        layer_norm_eps=hyperparameters["norm_eps"],
        use_cache=True
    )

def get_tokenzier(tokenizer_path: Path) -> LLaMaTokenizer:
    return LLaMaTokenizer(tokenizer_path)

original_name_to_transformers_name = {
    "tok_embeddings.weight" : "llama.embed.weight",
    "norm.weight": "llama.final_layer_norm.weight",
    "output.bias": "lm_head.weight",
    r"layers.(\d*).attention_norm.weight": r"llama.layers.\1.attention_norm.weight",
    r"layers.(\d*).attention.wq.weight": r"llama.layers.\1.attention.qkv.weight",
    r"layers.(\d*).attention.wk.weight": r"layers.\1.attention.qkv.weight",
    r"layers.(\d*).attention.wv.weight": r"layers.\1.attention.qkv.weight",
    r"layers.(\d*).attention.wo.weight": r"layers.\1.attention.o.weight",
    r"layers.(\d*).ffn_norm.weight": r"layers.\1.ff_norm.weight",
    r"layers.(\d*).feed_forward.w1.weight": r"layers.\1.ff.wi_0.weight",
    r"layers.(\d*).feed_forward.w2.weight": r"layers.\1.ff.wo.weight",
    r"layers.(\d*).feed_forward.w3.weight": r"layers.\1.ff.wi_1.weight",
}
def map_original_names_to_transformers_names(original_name: str):
    for pattern, repl in original_name_to_transformers_name.items():
        if re.match(pattern, original_name) is None:
            continue
        return re.sub(pattern, repl, original_name)


def convert_model(model_path: Path, config:LLaMaConfig) -> LLaMaForCausalLM:
    model = LLaMaForCausalLM(config=config)

    paths = sorted(model_path.glob("*.pth"))
    tp_size = len(paths)
    for tp_rank, path in enumerate(paths):
        with open(path, "r") as fi:
            weights = torch.load(fi)

        for original_name, original_param in weights.items():
            if original_name.endswith(".attention.inner_attention.rope.freqs"):
                print(f"We ignore {original_name} as it stores the rotary embeddings which are not in fact parameters")
                continue

            transformers_name = map_original_names_to_transformers_names(original_name)
            transformers_param = model.get_parameter(transformers_name)

            if original_name.endswith("norm.weight"):
                transformers_param.copy_(original_param)
                continue

            # weights are sharded across TP
            if any(original_name.endswith(suffix) for suffix in [".feed_forward.w2.weight", ".attention.wo.weight"]):
                # Row Linear weight
                output_dim = transformers_param.shape[0]
                assert output_dim % tp_size == 0
                step = output_dim // tp_size
                start = tp_rank * step
                end = (tp_rank + 1) * step
                transformers_param[start:end].copy_(original_param)
                continue

            # Column linear
            if any(original_name.endswith(suffix) for suffix in [".wq.weight", ".wk.weight", "wv.weight"]):
                # We fuse all the weights into a single qkv matrix.
                index, suffix = [(i, suffix) for i, suffix in enumerate([".wq.weight", ".wk.weight", "wv.weight"]) if original_name.endswith(suffix)][0]
                assert config.num_attention_heads % tp_size == 0
                heads_per_tp_rank = config.num_attention_heads // tp_size
                transformer_shard = transformers_param.view(config.hidden_size, config.num_attention_heads, 3, config.hidden_size // config.num_attention_heads)[:, tp_rank * heads_per_tp_rank: (tp_rank+1) * heads_per_tp_rank, index, :]
            else:
                input_dim = transformers_param.shape[-1]
                assert input_dim % tp_size == 0
                step = input_dim // tp_size
                start = tp_rank * step
                end = (tp_rank + 1) * step
                transformer_shard = transformers_param[:, start: end]

            transformer_shard.copy_(original_param)

    return model

def main(args):
    tokenizer = get_tokenzier(tokenizer_path=args.checkpoint_directory / "tokenizer.model")

    model_path = args.checkpoint_directory / args.model_subpath
    config = generate_config(model_path / "params.json", vocab_size=tokenizer.vocab_size)
    model = convert_model(model_path=model_path, config=config)

    config.save_pretrained(args.pytorch_dump_folder_path)
    model.save_pretrained(args.pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        required=True,
        help="Path to the checkpoint path containing `tokenizer.json` and different model size checkpoints.",
    )
    parser.add_argument(
        "--pytorch-dump-folder-path", type=Path, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--model-subpath", type=Path, required=True, help="Subpath after going into checkpoint directory where the model checkpoint lies. Typically `7B` or `13B`"
    )
    args = parser.parse_args()
    main(args)