# MIT License

# Copyright (c) 2024 jiayi yuan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA License

# =======================================================================

# 1. Definitions

# “Licensor” means any person or entity that distributes its Work.

# “Work” means (a) the original work of authorship made available under
# this license, which may include software, documentation, or other files,
# and (b) any additions to or derivative works thereof that are made
# available under this license.

# The terms “reproduce,” “reproduction,” “derivative works,” and “distribution”
# have the meaning as provided under U.S. copyright law; provided, however,
# that for the purposes of this license, derivative works shall not include works
# that remain separable from, or merely link (or bind by name) to the
# interfaces of, the Work.

# Works are “made available” under this license by including in or with the Work
# either (a) a copyright notice referencing the applicability of
# this license to the Work, or (b) a copy of this license.

# 2. License Grant

# 2.1 Copyright Grant. Subject to the terms and conditions of this license, each
# Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free,
# copyright license to use, reproduce, prepare derivative works of, publicly display,
# publicly perform, sublicense and distribute its Work and any resulting derivative
# works in any form.

# 3. Limitations

# 3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under
# this license, (b) you include a complete copy of this license with your distribution,
# and (c) you retain without modification any copyright, patent, trademark, or
# attribution notices that are present in the Work.

# 3.2 Derivative Works. You may specify that additional or different terms apply to the use,
# reproduction, and distribution of your derivative works of the Work (“Your Terms”) only
# if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative
# works, and (b) you identify the specific derivative works that are subject to Your Terms.
# Notwithstanding Your Terms, this license (including the redistribution requirements in
# Section 3.1) will continue to apply to the Work itself.

# 3.3 Use Limitation. The Work and any derivative works thereof only may be used or
# intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation
# and its affiliates may use the Work and any derivative works commercially.
# As used herein, “non-commercially” means for research or evaluation purposes only.

# 3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor
# (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that
# you allege are infringed by any Work, then your rights under this license from
# such Licensor (including the grant in Section 2.1) will terminate immediately.

# 3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its
# affiliates’ names, logos, or trademarks, except as necessary to reproduce
# the notices described in this license.

# 3.6 Termination. If you violate any term of this license, then your rights under
# this license (including the grant in Section 2.1) will terminate immediately.

# 4. Disclaimer of Warranty.

# THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT.
# YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE.

# 5. Limitation of Liability.

# EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY,
# WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR
# BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR
# INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS
# INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY
# OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGES.

# =======================================================================

import torch
from ..attention import RotaryEmbeddingESM, ATTN_FORWRAD

def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions
        num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
        num_key_value_heads = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
        head_dim = self.head_dim

        # Qwen3 / Qwen3-MoE apply a per-head RMSNorm (q_norm / k_norm) to the
        # queries and keys before RoPE. Llama has neither, so for Llama these
        # wrappers collapse to the plain projection and behaviour is unchanged.
        q_norm = getattr(self, "q_norm", None)
        k_norm = getattr(self, "k_norm", None)
        if q_norm is None and k_norm is None:
            project_q, project_k = self.q_proj, self.k_proj
        else:
            def project_q(x):
                h = self.q_proj(x)
                h = h.view(*h.shape[:-1], num_heads, head_dim)
                if q_norm is not None:
                    h = q_norm(h)
                return h.reshape(*h.shape[:-2], num_heads * head_dim)

            def project_k(x):
                h = self.k_proj(x)
                h = h.view(*h.shape[:-1], num_key_value_heads, head_dim)
                if k_norm is not None:
                    h = k_norm(h)
                return h.reshape(*h.shape[:-2], num_key_value_heads * head_dim)

        ret = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_value,
            project_q, project_k, self.v_proj, self.o_proj,
            head_dim, num_heads, num_key_value_heads,
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None
        self._rocket_cache = pkv
        return o, None

    return hf_forward


def patch_hf(
    model,
    attn_type: str = "inf_llm",
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    rope_theta_factor = None,
    rope_linear_scaling_factor = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    from transformers import (
        LlamaForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM)
    from transformers.models.llama.modeling_llama import BaseModelOutputWithPast

    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        *args,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        if use_cache:
            pkv = tuple()

        else:
            pkv = None

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=self.position_bias,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs

            if use_cache:
                _cache = decoder_layer.self_attn._rocket_cache
                pkv = pkv + (_cache,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    forward = huggingface_forward(ATTN_FORWRAD[attn_type](**attn_kwargs))

    if isinstance(model, (LlamaForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM)):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    else:
        raise ValueError("Only supports llama, qwen3 and qwen3_moe models.")
    
    # Build the replacement RoPE from model.config for every supported model.
    # In transformers 5.x the rotary module is config-driven (no .dim / .base
    # attrs), and rope_scaling / rope_parameters both carry rope_theta, so a
    # single path covers Llama and Qwen3 / Qwen3-MoE. We hand the real config to
    # RotaryEmbeddingESM unconditionally: its llama3 frequency rescaling
    # self-gates on rope_type == "llama3", so it fires for Llama-3.1 and is a
    # no-op for Qwen3's default RoPE.
    cfg = model.config
    if base is not None and rope_theta_factor is not None:
        base = base * rope_theta_factor
    else:
        base = cfg.rope_parameters["rope_theta"]
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    distance_scale = distance_scale if distance_scale is not None else 1.0
    rope = RotaryEmbeddingESM(
        head_dim,
        base,
        distance_scale,
        rope_linear_scaling_factor,
        cfg,
    )
    
    model.model.position_bias = rope
    
    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(set_forward)

    model.model._old_forward = model.model.forward
    model.model.forward = model_forward.__get__(model.model, Model)

    return model

