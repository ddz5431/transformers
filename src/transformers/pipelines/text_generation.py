import enum
import itertools
import types
from typing import Dict

from ..utils import ModelOutput, add_end_docstrings, is_tf_available, is_torch_available
from .base import Pipeline, build_pipeline_init_args


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    from .pt_utils import KeyDataset

if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


class Chat:
    """This class is intended to just be used internally in this pipeline and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""

    def __init__(self, messages: Dict):
        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        self.messages = messages


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using any `ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt. When the underlying model is a conversational model, it can also accept one or more chats,
    in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s).
    Each chat takes the form of a list of dicts, where each dict contains "role" and "content" keys.

    Examples:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="openai-community/gpt2")
    >>> generator("I can't believe you did such a ", do_sample=False)
    [{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

    >>> # These parameters will return suggestions, and only the newly created text making it easier for prompting suggestions.
    >>> outputs = generator("My tart needs some", num_return_sequences=4, return_full_text=False)
    ```

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="HuggingFaceH4/zephyr-7b-beta")
    >>> # Zephyr-beta is a conversational model, so let's pass it a chat instead of a single string
    >>> generator([{"role": "user", "content": "What is the capital of France? Answer in one word."}], do_sample=False, max_new_tokens=2)
    [{'generated_text': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': 'Paris'}]}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This language generation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"text-generation"`.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling
    objective. See the list of available [text completion models](https://huggingface.co/models?filter=text-generation)
    and the list of [conversational models](https://huggingface.co/models?other=conversational)
    on [huggingface.co/models].
    """

    # Prefix text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
    # in https://github.com/rusiaaman/XLNet-gen#methodology
    # and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e

    XL_PREFIX = """
    In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria) are discovered. The
    voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western
    Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision
    and denounces one of the men as a horse thief. Although his father initially slaps him for making such an
    accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, with people, even a bishop,
    begging for his blessing. <eod> </s> <eos>
    """

    def __init__(self, generation_config=None, *args, **kwargs):
        # TODO check why case: pipeline with external loaded model does not have generation config
        super().__init__(*args, **kwargs)
        self.check_model_type(
            TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES if self.framework == "tf" else MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        )

        if generation_config:
            self.generation_config.update(**generation_config.to_dict())

        if "prefix" not in self._preprocess_params:
            # This is very specific. The logic is quite complex and needs to be done
            # as a "default".
            # It also defines both some preprocess_kwargs and generate_kwargs
            # which is why we cannot put them in their respective methods.
            prefix = None
            if self.prefix is not None:
                prefix = self.prefix
            if prefix is None and self.model.__class__.__name__ in [
                "XLNetLMHeadModel",
                "TransfoXLLMHeadModel",
                "TFXLNetLMHeadModel",
                "TFTransfoXLLMHeadModel",
            ]:
                # For XLNet and TransformerXL we add an article to the prompt to give more state to the model.
                prefix = self.XL_PREFIX
            if prefix is not None:
                # Recalculate some generate_kwargs linked to prefix.
                preprocess_params, forward_params, _ = self._sanitize_parameters(prefix=prefix, **self._forward_params)
                self._preprocess_params = {**self._preprocess_params, **preprocess_params}
                self._forward_params = {**self._forward_params, **forward_params}

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        suffix=None,
        handle_long_generation=None,
        stop_sequence=None,
        truncation=None,
        max_length=None,
        continue_final_message=None,
        **generate_kwargs,
    ):
        preprocess_params = {}

        add_special_tokens = False
        if "add_special_tokens" in generate_kwargs:
            add_special_tokens = preprocess_params["add_special_tokens"] = generate_kwargs.pop("add_special_tokens")

        if "padding" in generate_kwargs:
            preprocess_params["padding"] = generate_kwargs.pop("padding")

        if truncation is not None:
            preprocess_params["truncation"] = truncation

        if max_length is not None:
            preprocess_params["max_length"] = max_length
            generate_kwargs["max_length"] = max_length

        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, add_special_tokens=add_special_tokens, return_tensors=self.framework
            )
            generate_kwargs["prefix_length"] = prefix_inputs["input_ids"].shape[-1]

        if suffix is not None:
            preprocess_params["suffix"] = suffix

        # TODO figure out where is suffix_length supposed to be used
        # if suffix:
        #     suffix_inputs = self.tokenizer(
        #         suffix, padding=False, add_special_tokens=add_special_tokens, return_tensors=self.framework
        #     )
        #     generate_kwargs["suffix_length"] = suffix_inputs["input_ids"].shape[-1]

        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected"
                    " [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        if continue_final_message is not None:
            preprocess_params["continue_final_message"] = continue_final_message

        preprocess_params.update(generate_kwargs)
        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_full_text`")
            if return_tensors is not None:
                raise ValueError("`return_full_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
        if continue_final_message is not None:
            postprocess_params["continue_final_message"] = continue_final_message

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            generate_kwargs["eos_token_id"] = stop_sequence_ids

        if self.assistant_model is not None:
            forward_params["assistant_model"] = self.assistant_model
        if self.assistant_tokenizer is not None:
            forward_params["tokenizer"] = self.tokenizer
            forward_params["assistant_tokenizer"] = self.assistant_tokenizer

        return preprocess_params, forward_params, postprocess_params

    # overriding _parse_and_tokenize to allow for unusual language-modeling tokenizer arguments
    def _parse_and_tokenize(self, *args, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            kwargs.update({"add_space_before_punct_symbol": True})

        return super()._parse_and_tokenize(*args, **kwargs)

    def __call__(self, text_inputs, **kwargs):
        """
        Complete the prompt(s) given as inputs.

        Args:
            text_inputs (`str`, `List[str]`, List[Dict[str, str]], or `List[List[Dict[str, str]]]`):
                One or several prompts (or one list of prompts) to complete. If strings or a list of string are
                passed, this pipeline will continue each prompt. Alternatively, a "chat", in the form of a list
                of dicts with "role" and "content" keys, can be passed, or a list of such chats. When chats are passed,
                the model's chat template will be used to format them before passing them to the model.

                Alternatively, pre-tokenized inputs can be passed as a dictionary containing:
                - 'input_ids': Tensor of token ids
                - 'attention_mask': Tensor of attention mask (optional)
                - 'prompt_text': Original text or Chat object for postprocessing
                Pre-tokenized inputs allow for more efficient processing when repeatedly using similar prompts.

            return_tensors (`bool`, *optional*, defaults to `False`):
                Returns the tensors of predictions (as token indices) in the outputs. If set to
                `True`, the decoded text is not returned.
            return_text (`bool`, *optional*):
                Returns the decoded texts in the outputs.
            return_full_text (`bool`, *optional*, defaults to `True`):
                If set to `False` only added text is returned, otherwise the full text is returned. Cannot be
                specified at the same time as `return_text`.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the potential extra spaces in the text output.
            continue_final_message( `bool`, *optional*): This indicates that you want the model to continue the
                last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
                By default this is `True` when the final message in the input chat has the `assistant` role and
                `False` otherwise, but you can manually override that behaviour by setting this flag.
            prefix (`str`, *optional*):
                Prefix added to prompt.
            handle_long_generation (`str`, *optional*):
                By default, this pipelines does not handle long generation (ones that exceed in one form or the other
                the model maximum length). There is no perfect way to address this (more info
                :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227). This provides common
                strategies to work around that problem depending on your use case.

                - `None` : default strategy where nothing in particular happens
                - `"hole"`: Truncates left of input, and leaves a gap wide enough to let generation happen (might
                  truncate a lot of the prompt and not suitable when generation exceed the model capacity)
            generate_kwargs (`dict`, *optional*):
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./text_generation)).

        Return:
            A list or a list of lists of `dict`: Returns one of the following dictionaries (cannot return a combination
            of both `generated_text` and `generated_token_ids`):

            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
        """
        # Check if inputs are already pre-tokenized
        is_tokenized = (
                isinstance(text_inputs, dict) and
                "input_ids" in text_inputs and
                "prompt_text" in text_inputs and
                (
                        (self.framework == "pt" and isinstance(text_inputs["input_ids"], torch.Tensor)) or
                        (self.framework == "tf" and isinstance(text_inputs["input_ids"], tf.Tensor))
                )
        )

        if is_tokenized:
            # For pre-tokenized inputs, process directly without preprocessing
            # Get parameters
            preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

            # Fuse init params and call params
            forward_params = {**self._forward_params, **forward_params}
            postprocess_params = {**self._postprocess_params, **postprocess_params}

            # Run directly with tokenized inputs - skip preprocessing
            with self.device_placement():
                model_outputs = self.forward(text_inputs, **forward_params)

            return self.postprocess(model_outputs, **postprocess_params)

        elif isinstance(
            text_inputs,
            (list, tuple, types.GeneratorType, KeyDataset)
            if is_torch_available()
            else (list, tuple, types.GeneratorType),
        ):
            if isinstance(text_inputs, types.GeneratorType):
                text_inputs, _ = itertools.tee(text_inputs)
                text_inputs, first_item = (x for x in text_inputs), next(_)
            else:
                first_item = text_inputs[0]
            if isinstance(first_item, (list, tuple, dict)):
                # We have one or more prompts in list-of-dicts format, so this is chat mode
                if isinstance(first_item, dict):
                    return super().__call__(Chat(text_inputs), **kwargs)
                else:
                    chats = (Chat(chat) for chat in text_inputs)  # 🐈 🐈 🐈
                    if isinstance(text_inputs, types.GeneratorType):
                        return super().__call__(chats, **kwargs)
                    else:
                        return super().__call__(list(chats), **kwargs)
        return super().__call__(text_inputs, **kwargs)

    def preprocess(
            self,
            prompt_text,
            prefix="",
            handle_long_generation=None,
            add_special_tokens=None,
            truncation=None,
            padding=None,
            max_length=None,
            continue_final_message=None,
            **generate_kwargs,
    ):
        tokenizer_kwargs = {
            key: value
            for key, value in {
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "padding": padding,
                "max_length": max_length,
            }.items()
            if value is not None
        }

        if isinstance(prompt_text, Chat):
            tokenizer_kwargs.pop("add_special_tokens", None)
            if continue_final_message is None:
                continue_final_message = prompt_text.messages[-1]["role"] == "assistant"
            inputs = self.tokenizer.apply_chat_template(
                prompt_text.messages,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
                return_dict=True,
                return_tensors=self.framework,
                **tokenizer_kwargs,
            )
        else:
            inputs = self.tokenizer(prefix + prompt_text, return_tensors=self.framework, **tokenizer_kwargs)

        # Store original prompt
        inputs["prompt_text"] = prompt_text

        # Check if suffix is needed before processing
        suffix_prompt = generate_kwargs.pop("suffix", None)
        use_suffix_for_eval = self.generation_config.use_suffix_for_eval

        if use_suffix_for_eval and not suffix_prompt:
            raise ValueError("`use_suffix_for_eval` requires `suffix` to be specified")

        # Only process suffix if both conditions are met
        if use_suffix_for_eval and suffix_prompt:
            # Lazily process suffix only when needed
            if not isinstance(suffix_prompt, Chat):
                if isinstance(suffix_prompt, list):
                    raise ValueError("`suffix_prompt` should be a string, not a list of messages.")

                # Construct suffix as chat format
                suffix_message = [
                    {"role": "system", "content": "You must answer 'Yes' or 'No'."},
                    {"role": "user", "content": suffix_prompt},
                ]
                suffix_prompt = Chat(suffix_message)

            if continue_final_message is None:
                continue_final_message = suffix_prompt.messages[-1]["role"] == "assistant"

            # Only tokenize once we need it
            suffix_inputs = self.tokenizer.apply_chat_template(
                suffix_prompt.messages,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
                return_dict=True,
                return_tensors=self.framework,
                **tokenizer_kwargs,
            ) if isinstance(suffix_prompt, Chat) else self.tokenizer(suffix_prompt, return_tensors=self.framework,
                                                                     **tokenizer_kwargs)

            # Add suffix information
            inputs["eval_input_ids"] = suffix_inputs["input_ids"]

            # Create combined attention mask only if needed
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], suffix_inputs["attention_mask"]], dim=1
                )

        inputs["prompt_text"] = prompt_text

        if use_suffix_for_eval and "attention_mask" in inputs and "eval_input_ids" in inputs:
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], suffix_inputs["attention_mask"]], dim=1
            )
        elif use_suffix_for_eval and not suffix_prompt:
            raise ValueError("`use_suffix_for_eval` requires `suffix` to be specified")

        # Handle long generation only if necessary
        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            new_tokens = generate_kwargs.get(
                "max_new_tokens",
                generate_kwargs.get("max_length", self.generation_config.max_length) - cur_len
            )

            if new_tokens < 0:
                raise ValueError("We cannot infer how many new tokens are expected")

            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError("Desired tokens exceed model's max length")

                # Truncate inputs in-place
                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -keep_length:]

                    # Only recombine if we already have eval_input_ids
                    if "eval_input_ids" in inputs and use_suffix_for_eval and suffix_prompt:
                        inputs["attention_mask"] = torch.cat(
                            [inputs["attention_mask"], suffix_inputs["attention_mask"]], dim=1
                        )

        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs.get("input_ids", None)
        eval_input_ids = model_inputs.get("eval_input_ids", None)
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # Use non_blocking transfers when moving tensors to the device
        if input_ids is not None and hasattr(input_ids, 'to'):
            input_ids = input_ids.to(self.model.device, non_blocking=True)
        if attention_mask is not None and hasattr(attention_mask, 'to'):
            attention_mask = attention_mask.to(self.model.device, non_blocking=True)
        if eval_input_ids is not None and hasattr(eval_input_ids, 'to'):
            eval_input_ids = eval_input_ids.to(self.model.device, non_blocking=True)

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.generation_config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        if self.generation_config.use_suffix_for_eval:
            generate_kwargs["eval_input_ids"] = eval_input_ids

        generate_kwargs["tokenizer"] = self.tokenizer
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     generation_config=self.generation_config, **generate_kwargs)

        if isinstance(output, ModelOutput):
            generated_sequence = output.sequences
            other_outputs = {k: v for k, v in output.items() if k != "sequences"}
            out_b = generated_sequence.shape[0]

            if self.framework == "pt":
                for key, value in other_outputs.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] == out_b:
                        other_outputs[key] = value.reshape(in_b, out_b // in_b, *value.shape[1:])
                    if isinstance(value, tuple) and len(value[0]) == out_b:
                        value = torch.stack(value).swapaxes(0, 1)
                        other_outputs[key] = value
            elif self.framework == "tf":
                for key, value in other_outputs.items():
                    if isinstance(value, tf.Tensor) and value.shape[0] == out_b:
                        other_outputs[key] = tf.reshape(value, (in_b, out_b // in_b, *value.shape[1:]))
                    if isinstance(value, tuple) and len(value[0]) == out_b:
                        value = tf.stack(value).swapaxes(0, 1)
                        other_outputs[key] = value
        else:
            generated_sequence = output
            other_outputs = {}

        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))

        model_outputs = {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "prompt_text": prompt_text,
        }
        model_outputs.update(other_outputs)
        return model_outputs

    def postprocess(
            self,
            model_outputs,
            return_type=ReturnType.FULL_TEXT,
            clean_up_tokenization_spaces=True,
            continue_final_message=None,
    ):
        """
        Efficiently processes batched model outputs into human-readable text or token sequences.
        """
        # Extract generated sequences
        generated_sequences = model_outputs["generated_sequence"]

        # Ensure `generated_sequences` is a list of lists (convert tensors if needed)
        if isinstance(generated_sequences, torch.Tensor):
            generated_sequences = generated_sequences.cpu().numpy().tolist()
        elif isinstance(generated_sequences, list) and isinstance(generated_sequences[0], torch.Tensor):
            generated_sequences = [seq.cpu().numpy().tolist() for seq in generated_sequences]

        # Flatten: If multiple return sequences exist, take the first one per batch
        if isinstance(generated_sequences[0], list) and isinstance(generated_sequences[0][0], list):
            generated_sequences = [seq[0] for seq in generated_sequences]  # Take first generated sequence

        batch_size = len(generated_sequences)

        # Handle input_ids (check for batched vs single case)
        input_ids = model_outputs["input_ids"]
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.cpu().numpy().tolist()
            elif isinstance(input_ids, list) and isinstance(input_ids[0], torch.Tensor):
                input_ids = [seq.cpu().numpy().tolist() for seq in input_ids]

        # Ensure `prompt_text` is batched
        prompt_text = model_outputs["prompt_text"]
        if not isinstance(prompt_text, list):
            prompt_text = [prompt_text] * batch_size  # Duplicate single prompt across batch

        # Precompute prompt lengths using batch decoding (avoiding per-item decoding)
        prompt_lengths = [0] * batch_size
        if input_ids:
            decoded_prompts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
            prompt_lengths = [len(prompt) for prompt in decoded_prompts]

        # Handle additional outputs
        other_outputs = model_outputs.get("additional_outputs", {})
        splitted_keys = {
            k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v.numpy().tolist()
            for k, v in other_outputs.items()
            if isinstance(v, (torch.Tensor, tf.Tensor)) and len(v) == batch_size
        }

        # Decode all generated sequences in batch
        decoded_texts = self.tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )

        records = []
        for idx, text in enumerate(decoded_texts):
            record = {}

            if return_type == ReturnType.TENSORS:
                record["generated_token_ids"] = generated_sequences[idx]
            else:
                # Extract new text portion after the prompt
                all_text = text[prompt_lengths[idx]:]

                if return_type == ReturnType.FULL_TEXT:
                    prompt = prompt_text[idx]
                    if isinstance(prompt, str):
                        all_text = prompt + all_text
                    elif isinstance(prompt, Chat):
                        if continue_final_message is None:
                            continue_final_message = prompt.messages[-1]["role"] == "assistant"

                        if continue_final_message:
                            # Append output to last assistant message
                            all_text = list(prompt.messages)[:-1] + [
                                {
                                    "role": prompt.messages[-1]["role"],
                                    "content": prompt.messages[-1]["content"] + all_text,
                                }
                            ]
                        else:
                            # Create a new assistant message
                            all_text = list(prompt.messages) + [{"role": "assistant", "content": all_text}]

                record["generated_text"] = all_text

            # Add additional model outputs
            for key, values in splitted_keys.items():
                record[key] = values[idx]

            records.append(record)

        return records