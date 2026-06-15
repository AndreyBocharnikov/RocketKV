"""OpenCompass wrapper for RocketKV (method='rocket' only).

This bridges RocketKV's validated inference machinery into OpenCompass so it can
be evaluated on the same datasets as ShadowKV / ArkVale / LRQK.

It reuses RocketKV's own, already-validated code paths (the ones exercised by
``scripts/longbench/llama3.1-8b-instruct.sh``):

    * ``initialize_model_tokenizer`` -- load the HF model + tokenizer the way
      RocketKV expects (fp16, flash-attn-2, padding_side='left', pad=eos).
    * ``compress``                   -- compute the rocket hyper-parameters for a
      given sequence length and install the two-stage RocketKV KV-cache
      compression on the model (re-run per sample).
    * ``GreedySearch``               -- RocketKV's single-sequence greedy decoder.

Only Llama and Qwen3 / Qwen3-MoE models and the 'rocket' method are supported on
purpose (no full attention, no rocket-mt, no Mistral/Qwen2).
"""

import gc
import os
import sys
from typing import List, Union

import torch

# Make RocketKV's packages importable regardless of how this module is loaded
# (directly by the launcher, or via OpenCompass ``custom_imports`` inside a
# spawned worker subprocess).
#   _THIS_DIR  -> /workspace/RocketKV            (for ``pipeline.*`` / ``eval.*``)
#   _INNER_DIR -> .../pipeline/inf_stream_llm    (for ``inf_llm`` / ``infllm_utils``)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_INNER_DIR = os.path.join(_THIS_DIR, 'pipeline', 'inf_stream_llm')
for _p in (_THIS_DIR, _INNER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opencompass.models.huggingface_above_v4_33 import (  # noqa: E402
    HuggingFacewithChatTemplate,
    _convert_chat_messages,
)
from opencompass.registry import MODELS  # noqa: E402

# RocketKV internals (same call chain as the author-provided LongBench script).
from inf_llm import GreedySearch, initialize_model_tokenizer  # noqa: E402
from infllm_utils import compress  # noqa: E402


@MODELS.register_module()
class RocketKVChatBot(HuggingFacewithChatTemplate):
    """RocketKV ('rocket' method) wrapped as an OpenCompass chat model.

    Args:
        path: HuggingFace model id / path. Must be a Llama or Qwen3 / Qwen3-MoE
            model (Mistral / Qwen2 are unsupported by design).
        token_budget: RocketKV KV-cache token budget (absolute token count,
            e.g. 1024). Internally split between stage-1 (SnapKV) and stage-2
            (hybrid sparse top-k).
        max_seq_len: maximum allowed prompt length. Prompts longer than this
            raise an assertion (no truncation -- that keeps the behaviour
            transparent, unlike RocketKV's 'middle' truncation hack).
        local_window: number of most-recent tokens always attended during sparse
            decode, as EXTRA budget on top of the top-k token quota (token_budget
            // 2), not carved out of it. Qwen3's per-head q/k-norm flattens the
            query channel magnitudes, which makes RocketKV's coarse top-r-channel
            score approximation drop recent tokens during decode -> the model
            loses local context and stutters (token doubling: "TheThe", "Z
            Zambot"), wrecking exact-match scores at aggressive budgets. A small
            window pins recent tokens in and fixes it. Default: 128 for Qwen3, 0
            for Llama (0 == exactly the validated RocketKV behaviour). Pass an int
            to override.
    """

    def __init__(
        self,
        path: str,
        token_budget: Union[int, float],
        max_seq_len: int = 128 * 1024,
        local_window: int = None,
        **kwargs,
    ):
        # 'qwen3' matches both dense Qwen3 and Qwen3-MoE (e.g. Qwen3-30B-A3B);
        # Qwen2 / Mistral paths don't contain 'qwen3' / 'llama' and are rejected.
        # The loader (initialize_model_tokenizer) re-validates via AutoConfig.
        lower = path.lower()
        if 'llama' not in lower and 'qwen3' not in lower:
            raise ValueError(
                f"RocketKVChatBot supports Llama and Qwen3 / Qwen3-MoE models "
                f"only, got '{path}'.")

        # Auto-enable the local decode window for q/k-norm models (Qwen3); keep
        # Llama on the validated path (0) unless the caller overrides.
        if local_window is None:
            local_window = 128 if 'qwen3' in lower else 0

        self.dynamic_budget = isinstance(token_budget, float)
        self.pipeline_params = dict(
            model_name=path,
            tokenizer_name=path,
            rope_theta_factor=1.0,
            base=None,
            fattn=True,
            method='rocket',
            local_window=local_window,
        )
        if not self.dynamic_budget:
            self.pipeline_params["token_budget"] = token_budget
        else:
            self.token_budget = token_budget

        super().__init__(path=path, max_seq_len=max_seq_len, **kwargs)

    def _load_model(self, path, kwargs, peft_path=None, peft_kwargs=dict()):
        # Returns BOTH model and tokenizer with RocketKV's exact settings. This
        # overwrites the tokenizer the base __init__ loaded a moment earlier.
        self.model, self.tokenizer = initialize_model_tokenizer(
            self.pipeline_params)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        # RocketKV keeps a module-global decode position (rocket.py: kv_pos) and
        # runs a single-sequence greedy loop, so only batch_size == 1 is valid.
        assert len(inputs) == 1, \
            'RocketKVChatBot only supports batch_size == 1'

        messages = _convert_chat_messages(inputs)[0]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        input_ids = self.tokenizer(
            prompt, return_tensors='pt', add_special_tokens=False).input_ids[0]

        # No truncation: require the prompt to fit within max_seq_len.
        assert len(input_ids) <= self.max_seq_len, (
            f'prompt length {len(input_ids)} exceeds max_seq_len '
            f'{self.max_seq_len}; raise max_seq_len or shorten the input.')

        # Stop tokens come from the model itself (generation_config). For
        # Llama-3.1-Instruct this is [128001, 128008, 128009] and includes
        # <|eot_id|> -- the token RocketKV's get_pred adds by hand for llama3.
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            extra_end_token_ids = []
        elif isinstance(eos, int):
            extra_end_token_ids = [eos]
        else:
            extra_end_token_ids = list(eos)

        # ``compress`` expects the *total* sequence length (prompt + generated
        # tokens) to derive the compression ratio. It mutates self.pipeline_params
        # in place and (re-)installs RocketKV on the model for this sample -- the
        # same per-sample re-patching the author's get_pred does.
        if self.dynamic_budget:
            token_budget = int(self.token_budget * len(input_ids)) 
            self.pipeline_params["token_budget"] = token_budget

        total_seq_len = len(input_ids) + max_out_len
        compressed_model = compress({}, self.pipeline_params, total_seq_len,
                                    max_out_len, self.model)

        searcher = GreedySearch(compressed_model, self.tokenizer)
        output = searcher.generate(
            input_ids=input_ids,
            max_length=max_out_len,
            chunk_size=self.pipeline_params.get('chunk_size'),
            extra_end_token_ids=extra_end_token_ids,
        )
        searcher.clear()
        del searcher
        gc.collect()
        torch.cuda.empty_cache()

        return [output[0]]
