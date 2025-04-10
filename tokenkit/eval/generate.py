import copy
import itertools
import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from jax import lax
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import FlaxTopPLogitsWarper
from transformers.generation.flax_logits_process import FlaxNoRepeatNGramLogitsProcessor

from tokenkit import utils
from tokenkit.constants import EXPAND_INPUT_IDS_MAX_LENGTH
from tokenkit.models import param, sharding
from tokenkit.utils import tqdm


@partial(jax.jit, static_argnums=(1,))
def pad_cache(cache, n_pad):
    flat_cache = traverse_util.flatten_dict(cache)
    for key in flat_cache.keys():
        if key[-1] in ("cached_key", "cached_value"):
            flat_cache[key] = jnp.pad(
                flat_cache[key],
                ((0, 0), (0, n_pad), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
    return traverse_util.unflatten_dict(flat_cache)


def get_lowest_upper_bound(target, lengths):
    return min(x for x in lengths if x >= target)


@dataclass(frozen=True)
class SamplingConfig:
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    no_repeat_ngram_size: int | None = None


@flax.struct.dataclass
class State:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    prng: jnp.ndarray
    model_kwargs: dict[str, jnp.ndarray]
    params: flax.struct.PyTreeNode
    logits: jnp.ndarray | None = None
    hidden_states: jnp.ndarray | None = None


class Generator:
    def prefill(
        self,
        model_fn,
        input_ids,
        expanded_input_ids,
        attention_mask,
        params,
        init_cache,
    ):
        position_ids = (jnp.cumsum(attention_mask, axis=1) - 1) * attention_mask

        inputs_embeds = self.compute_inputs_embeds(input_ids, expanded_input_ids)

        out = model_fn(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=init_cache,
            params=params,
        )
        logits = out.logits.astype(jnp.float32)

        shift_logits = logits[:, :-1]
        shift_labels = input_ids[:, 1:]

        shift_logprobs = jax.nn.log_softmax(shift_logits, axis=-1)
        sequence_logprobs = jnp.take_along_axis(
            shift_logprobs, shift_labels[:, :, None], axis=-1
        ).squeeze(-1)

        return out.past_key_values, sequence_logprobs

    def __init__(
        self,
        mesh,
        model,
        params,
        tokenizer,
        sampling_config,
        batch_size,
        max_new_tokens,
        dtype=None,
        logit_mask=None,
        precompile=True,
        until: list[str] | None = None,
        lengths=[1024, 2048, 4096],
        eos_strategy="stop",  # 'forbid', 'stop', 'ignore', or 'restart'
        pad_to_multiple_of=128,
        expand_input_ids=False,
        expand_input_ids_vocab=None,
        expand_input_ids_embeddings=None,
    ):
        self.mesh = mesh
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.sampling_config = sampling_config
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.precompile = precompile
        self.config = model.config

        self.expand_input_ids = expand_input_ids
        if expand_input_ids:
            self.expand_input_ids_matrix = utils.get_expand_input_ids_matrix(
                tokenizer, expand_input_ids_vocab
            )
            self.expand_input_ids_dict = utils.get_expand_input_ids_dict(
                tokenizer, expand_input_ids_vocab
            )
            self.expand_input_ids_embeddings = expand_input_ids_embeddings

        self.until_tokens = []
        if until is not None:
            for stop_sequence_str in until:
                # with/without prefix space
                self.until_tokens.append(
                    tokenizer.encode(stop_sequence_str, add_special_tokens=False)
                )
                self.until_tokens.append(
                    tokenizer.encode(" " + stop_sequence_str, add_special_tokens=False)
                )

        if until is None or tokenizer.eos_token not in until:
            self.until_tokens.append([tokenizer.eos_token_id])

        if (
            until is None
            or self.tokenizer.model_kind_cls.replacements["<|<eot>|>"][0] not in until
        ):
            self.until_tokens.append(
                [
                    self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.model_kind_cls.replacements["<|<eot>|>"][0]
                    )
                ]
            )

        if (
            until is None
            or self.tokenizer.model_kind_cls.replacements["<|<eos>|>"][0] not in until
        ):
            self.until_tokens.append(
                [
                    self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.model_kind_cls.replacements["<|<eos>|>"][0]
                    )
                ]
            )

        self.until_tokens = [
            jnp.array(x, dtype=jnp.int32, device=NamedSharding(mesh, P()))
            for x in self.until_tokens
        ]
        print("Will stop generation at the following stop sequences:")
        for until_sequence in self.until_tokens:
            print(tokenizer.convert_ids_to_tokens(until_sequence))

        self.lengths = lengths

        self.eos_strategy = eos_strategy
        assert self.eos_strategy == "stop"

        # this is a bit hacky, if dtype is None assume already sharded / padded
        if dtype is None:
            assert logit_mask is not None

            if isinstance(logit_mask, jax.Array):
                self.logit_mask = logit_mask
            else:
                self.logit_mask = sharding.to_global_array(
                    logit_mask, NamedSharding(mesh, P())
                )
            self.params = params
            self.param_shardings = jax.tree.map(lambda x: x.sharding, self.params)
        else:
            embeddings, params = param.stack_embeddings(
                params, model.config, pop_embeddings=True
            )
            n_pad = utils.get_n_pad(embeddings.shape[0], pad_to_multiple_of)
            self.logit_mask = jnp.pad(
                jnp.ones(embeddings.shape[0], dtype=bool),
                (0, n_pad),
                mode="constant",
                constant_values=0,
            )
            embeddings = np.pad(
                embeddings, ((0, n_pad), (0, 0), (0, 0)), mode="constant"
            )
            params = param.assign_embeddings(params, embeddings, model.config)

            self.param_shardings = sharding.get_sharding_fn(
                sharding.get_shard_patterns(model.config.model_type), mesh
            )({"params": params})["params"]

            def to_dtype(pytree):
                return jax.tree.map(lambda x: x.astype(dtype), pytree)

            self.params = jax.jit(to_dtype, out_shardings=self.param_shardings)(params)

            model.config.vocab_size = embeddings.shape[0]

        del model.config.mesh  # needed for deepcopy
        self.prefill_model = copy.deepcopy(model)
        model.config.mesh = mesh
        self.prefill_model.config.max_length = max(lengths)

        if self.precompile:
            self.prefill_model.config._attn_implementation = "pallas_flash_attention"

        del model.config.mesh  # needed for deepcopy
        self.generate_model = copy.deepcopy(model)
        model.config.mesh = mesh
        self.generate_model.config.max_length = max(lengths)

        self.generate_model.config._attn_implementation = (
            "eager"  # for seq_length=1 in generation
        )

        self.prefill_model.config.mesh = mesh
        self.generate_model.config.mesh = mesh

        self.prefill_fn = self.prefill_model.__call__
        self.generate_fn = self.generate_model.__call__

        self.pad_token_id = sharding.to_global_array(
            np.array(tokenizer.pad_token_id), NamedSharding(mesh, P())
        )
        self.eos_token_id = sharding.to_global_array(
            np.array(tokenizer.eos_token_id), NamedSharding(mesh, P())
        )
        self.bos_token_id = (
            sharding.to_global_array(
                np.array(tokenizer.bos_token_id), NamedSharding(mesh, P())
            )
            if tokenizer.bos_token_id is not None
            else None
        )

        def get_cache(length):
            cache = self.generate_model.init_cache(batch_size, 1)
            flat_cache = traverse_util.flatten_dict(cache)
            for key in flat_cache.keys():
                if key[-1] in ("cached_key", "cached_value"):
                    flat_cache[key] = flat_cache[key].astype(
                        dtype or self.generate_model.module.dtype
                    )

            cache = traverse_util.unflatten_dict(flat_cache)
            return pad_cache(cache, length - 1)

        def set_cache_length(cache, length):
            flat_cache = traverse_util.flatten_dict(cache)
            for key in flat_cache.keys():
                if key[-1] == "cache_index":
                    flat_cache[key] = sharding.to_global_array(
                        np.array(length),
                        NamedSharding(self.mesh, P()),
                    )
            return traverse_util.unflatten_dict(flat_cache)

        cache_shape = jax.eval_shape(partial(get_cache, length=lengths[0]))

        self.cache_shardings = sharding.get_sharding_fn(
            sharding.get_shard_patterns(model.config.model_type), mesh
        )(cache_shape)
        self.get_cache = jax.jit(
            get_cache, static_argnums=(0,), out_shardings=self.cache_shardings
        )
        self.set_cache_length = set_cache_length

        self.state_shardings = State(
            cur_len=NamedSharding(mesh, P()),
            sequences=NamedSharding(mesh, P(None, None)),
            running_token=NamedSharding(mesh, P(None, None)),
            is_sent_finished=NamedSharding(mesh, P(None)),
            prng=NamedSharding(mesh, P(None)),
            model_kwargs={
                "position_ids": NamedSharding(mesh, P(None, None)),
                "attention_mask": NamedSharding(mesh, P(None, None)),
                "past_key_values": self.cache_shardings,
            },
            params=self.param_shardings,
            logits=NamedSharding(mesh, P(None, None)),
            hidden_states=NamedSharding(mesh, P(None, None)),
        )

    def compute_inputs_embeds(
        self, input_ids, expanded_input_ids=None, last_only=False
    ):
        embedding_matrix = param.get(
            self.params,
            param.get_input_embedding_path(self.config.model_type),
        )

        if last_only:
            inputs_embeds = jnp.take(embedding_matrix, input_ids[:, -1:], axis=0)
        else:
            inputs_embeds = jnp.take(embedding_matrix, input_ids, axis=0)

        if self.expand_input_ids:
            if expanded_input_ids is None:
                expanded_input_ids = utils.jax_expand_input_ids(
                    input_ids,
                    self.expand_input_ids_matrix,
                    last_only=last_only,
                )

            expanded_inputs_embeds = jnp.take(
                self.expand_input_ids_embeddings, expanded_input_ids, axis=0
            )

            inputs_embeds = inputs_embeds + expanded_inputs_embeds

        return inputs_embeds

    def generate_step(
        self,
        state: State,
    ):
        inputs_embeds = self.compute_inputs_embeds(state.running_token, last_only=True)

        out = self.generate_fn(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            params=state.params,
            **state.model_kwargs,
            output_hidden_states=True,
        )
        raw_logits = out.logits[:, -1].astype(jnp.float32)
        raw_logits = jnp.where(
            self.logit_mask[None],
            raw_logits,
            utils.get_large_negative_number(jnp.float32),
        )

        if self.sampling_config.no_repeat_ngram_size is not None:
            raw_logits = FlaxNoRepeatNGramLogitsProcessor(
                self.sampling_config.no_repeat_ngram_size
            )(state.sequences, raw_logits, state.cur_len)

        if not self.sampling_config.do_sample:
            next_token = raw_logits.argmax(-1)
            next_prng = state.prng
        else:
            assert self.sampling_config.top_k == 0, "Top-k sampling not supported yet"

            if self.sampling_config.top_p < 1:
                warper = FlaxTopPLogitsWarper(
                    self.sampling_config.top_p,
                    filter_value=utils.get_large_negative_number(raw_logits.dtype),
                )

                logits = warper(None, raw_logits, None)
            else:
                logits = raw_logits

            probs = jax.nn.softmax(logits / self.sampling_config.temperature, -1)

            def choose_next(prng, p):
                prng, next_prng = jax.random.split(prng, 2)
                return jax.random.choice(prng, probs.shape[1], p=p), next_prng

            next_token, next_prng = jax.vmap(choose_next)(state.prng, probs)

        next_token = (
            next_token * ~state.is_sent_finished
            + self.pad_token_id * state.is_sent_finished
        )
        next_token = next_token[:, None]

        next_sequences = lax.dynamic_update_slice(
            state.sequences, next_token, (0, state.cur_len)
        )
        next_is_sent_finished = state.is_sent_finished
        for until_token in self.until_tokens:
            next_is_sent_finished = next_is_sent_finished | (
                lax.dynamic_slice(
                    next_sequences,
                    (0, state.cur_len + 1 - until_token.shape[0]),
                    (next_sequences.shape[0], until_token.shape[0]),
                )
                == until_token[None, :]
            ).all(axis=-1)

        next_model_kwargs = self.generate_model.update_inputs_for_generation(
            out, state.model_kwargs
        )

        if state.logits is not None:
            next_logits = raw_logits
        else:
            next_logits = None

        if state.hidden_states is not None:
            next_hidden_states = out.hidden_states[-1][:, -1]
        else:
            next_hidden_states = None

        next_running_token = jnp.concatenate(
            [
                state.running_token[:, 1:],
                next_token,
            ],
            axis=1,
        )

        return State(
            cur_len=state.cur_len + 1,
            sequences=next_sequences,
            running_token=next_running_token,
            is_sent_finished=next_is_sent_finished,
            prng=next_prng,
            model_kwargs=next_model_kwargs,
            params=state.params,
            logits=next_logits,
            hidden_states=next_hidden_states,
        )

    def generate_condition(self, state: State):
        has_reached_max_length = state.cur_len == state.sequences.shape[1]
        all_sequence_finished = jnp.all(state.is_sent_finished)
        finish_generation = jnp.logical_or(
            has_reached_max_length, all_sequence_finished
        )
        return ~finish_generation

    def get_zero_state(self, length):
        return State(
            cur_len=sharding.to_global_array(
                np.array(0, dtype=np.int32), self.state_shardings.cur_len
            ),
            sequences=sharding.to_global_array(
                np.full((self.batch_size, length), self.pad_token_id, dtype=np.int32),
                self.state_shardings.sequences,
            ),
            running_token=np.full(
                (self.batch_size, EXPAND_INPUT_IDS_MAX_LENGTH),
                self.pad_token_id,
                dtype=np.int32,
            ),
            is_sent_finished=sharding.to_global_array(
                np.zeros(self.batch_size, dtype=bool),
                self.state_shardings.is_sent_finished,
            ),
            prng=sharding.to_global_array(
                jax.device_get(jax.random.split(jax.random.key(0), self.batch_size)),
                self.state_shardings.prng,
            ),
            model_kwargs={
                "position_ids": sharding.to_global_array(
                    np.zeros((self.batch_size, 1), dtype=np.int32),
                    self.state_shardings.model_kwargs["position_ids"],
                ),
                "attention_mask": sharding.to_global_array(
                    np.zeros((self.batch_size, length), dtype=np.int32),
                    self.state_shardings.model_kwargs["attention_mask"],
                ),
                "past_key_values": self.get_cache(length),
            },
            params=self.params,
        )

    def _compile_fns(
        self, compile_prefill=True, compile_generate=True, tqdm_kwargs=None
    ):
        compiled_prefill_fns = {}

        if compile_prefill and self.precompile:
            for length in tqdm(
                self.lengths, desc="Compiling prefill fns...", **(tqdm_kwargs or {})
            ):
                compile_args = (
                    jnp.zeros((self.batch_size, length), dtype=jnp.int32),
                    jnp.zeros((self.batch_size, length), dtype=jnp.int32),
                    jnp.zeros((self.batch_size, length), dtype=jnp.int32),
                    self.params,
                    self.get_cache(length),
                )
                compiled_prefill_fns[length] = (
                    jax.jit(
                        lambda input_ids, expanded_input_ids, attention_mask, params, init_cache: self.prefill(
                            self.prefill_fn,
                            input_ids,
                            expanded_input_ids,
                            attention_mask,
                            params,
                            init_cache,
                        ),
                        donate_argnums=(3,),
                        in_shardings=(
                            NamedSharding(self.mesh, P()),
                            NamedSharding(self.mesh, P()),
                            NamedSharding(self.mesh, P()),
                            self.param_shardings,
                            self.cache_shardings,
                        ),
                        out_shardings=(self.cache_shardings, None),
                    )
                    .lower(*compile_args)
                    .compile()
                )
        else:
            compiled_prefill_fns = {
                length: lambda input_ids, expanded_input_ids, attention_mask, params, init_cache: self.prefill(
                    self.prefill_fn,
                    input_ids,
                    expanded_input_ids,
                    attention_mask,
                    params,
                    init_cache,
                )
                for length in self.lengths
            }

        if compile_generate and self.precompile:
            compiled_generate_fns = {}
            for length in tqdm(
                self.lengths, desc="Compiling generate fns...", **(tqdm_kwargs or {})
            ):
                compiled_generate_fns[length] = (
                    jax.jit(
                        lambda state: lax.while_loop(
                            self.generate_condition, self.generate_step, state
                        ),
                        donate_argnums=(0,),
                        in_shardings=(self.state_shardings,),
                        out_shardings=self.state_shardings,
                    )
                    .lower(self.get_zero_state(length))
                    .compile()
                )
        else:
            compiled_generate_fns = {
                length: lambda state: lax.while_loop(
                    self.generate_condition, self.generate_step, state
                )
                for length in self.lengths
            }

        return compiled_prefill_fns, compiled_generate_fns

    def generate(self, prompts, seed=1234, tqdm_kwargs=None):
        all_prefill_tokens = []
        all_running_tokens = []

        for prompt in tqdm(prompts, desc="Encoding prompts...", **(tqdm_kwargs or {})):
            prompt_tokens = [
                self.vocab[x] for x in utils.encode_prompt(prompt, self.tokenizer)[0]
            ]

            all_prefill_tokens.append(prompt_tokens[:-1])
            all_running_tokens.append(prompt_tokens[-EXPAND_INPUT_IDS_MAX_LENGTH:])

        # longest prompts first to overestimate (rather than underestimate) time it takes to generate
        permutation_indices = np.argsort([len(x) for x in all_prefill_tokens])[::-1]
        generations = [None] * len(prompts)

        compiled_prefill_fns, compiled_generate_fns = self._compile_fns(
            tqdm_kwargs=tqdm_kwargs
        )

        n_batches = math.ceil(len(prompts) / self.batch_size)
        prngs = jax.random.split(jax.random.key(seed), n_batches)

        init_cache = self.get_cache(self.lengths[0])
        init_cache_length = self.lengths[0]

        for batch_idx in tqdm(
            range(n_batches), desc="Generating...", **(tqdm_kwargs or {})
        ):
            (start, end) = (
                batch_idx * self.batch_size,
                (batch_idx + 1) * self.batch_size,
            )

            batch_indices = permutation_indices[start:end]
            batch_indices = np.pad(
                batch_indices, (0, self.batch_size - len(batch_indices)), mode="edge"
            )

            unpadded_prefill_input_ids = [all_prefill_tokens[i] for i in batch_indices]
            max_new_tokens = min(
                self.max_new_tokens,
                self.lengths[-1] - max(len(x) for x in unpadded_prefill_input_ids),
            )
            if max_new_tokens < self.max_new_tokens:
                print(
                    f"Warning: max_new_tokens reduced from {self.max_new_tokens} to {max_new_tokens}"
                )
            padded_prefill_length = get_lowest_upper_bound(
                max(len(x) for x in unpadded_prefill_input_ids) + max_new_tokens,
                self.lengths,
            )

            prefill_input_ids = np.full(
                (self.batch_size, padded_prefill_length),
                fill_value=self.tokenizer.pad_token_id,
                dtype=np.int32,
            )
            attention_mask = np.zeros(
                (self.batch_size, padded_prefill_length), dtype=np.int32
            )
            running_tokens = jnp.array(
                [all_running_tokens[i] for i in batch_indices],
                dtype=jnp.int32,
            )

            for i, input_ids in enumerate(unpadded_prefill_input_ids):
                prefill_input_ids[i, : len(input_ids)] = input_ids
                attention_mask[i, : len(input_ids)] = 1
                attention_mask[i, padded_prefill_length - max_new_tokens :] = 1

            if padded_prefill_length > init_cache_length:
                init_cache = self.get_cache(padded_prefill_length)
                init_cache_length = padded_prefill_length

            init_cache = self.set_cache_length(init_cache, 0)

            if padded_prefill_length - max_new_tokens == 0:
                cache = init_cache
            else:
                if self.expand_input_ids:
                    prefill_expanded_input_ids = utils.np_expand_input_ids(
                        prefill_input_ids,
                        self.expand_input_ids_dict,
                    )
                else:
                    prefill_expanded_input_ids = np.zeros_like(prefill_input_ids)

                cache = compiled_prefill_fns[padded_prefill_length](
                    prefill_input_ids,
                    prefill_expanded_input_ids,
                    attention_mask,
                    self.params,
                    init_cache,
                )[0]

                cache = self.set_cache_length(
                    cache, padded_prefill_length - max_new_tokens
                )

            position_ids = jnp.array(
                [[len(x)] for x in unpadded_prefill_input_ids], dtype=jnp.int32
            )

            state = State(
                cur_len=jnp.array(
                    padded_prefill_length - max_new_tokens, dtype=jnp.int32
                ),
                sequences=sharding.to_global_array(
                    prefill_input_ids, NamedSharding(self.mesh, P())
                ),
                running_token=running_tokens,
                is_sent_finished=jnp.zeros(self.batch_size, dtype=bool),
                prng=jax.random.split(prngs[batch_idx], self.batch_size),
                model_kwargs={
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": cache,
                },
                params=self.params,
            )

            state = compiled_generate_fns[padded_prefill_length](state)
            sequences = process_allgather(
                state.sequences[:, -max_new_tokens:], tiled=True
            )

            special_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.model_kind_cls.special_tokens
            )
            for i, idx in enumerate(batch_indices):
                generations[idx] = [
                    token_id for token_id in sequences[i] if token_id not in special_ids
                ]

            # recycle out state buffers
            self.params = state.params
            init_cache = state.model_kwargs["past_key_values"]

        return generations


@flax.struct.dataclass
class LockstepState:
    states: list[State]
    combine_params: Any
    prng: jnp.ndarray
    is_sent_finished: jnp.ndarray


class LockstepGenerator:
    def __init__(
        self,
        mesh,
        all_models,
        all_configs,
        all_params,
        all_tokenizers,
        sampling_config,
        batch_size,
        max_new_tokens,
        combine_fn,
        combine_params,
        dtype=None,
        all_logit_masks=None,
        precompile=True,
        until: list[str] | None = None,
        lengths=[4096],
        pad_to_multiple_of=128,
    ):
        regular_token_intersection = set.intersection(
            *[set(x.get_vocab().keys()) for x in all_tokenizers]
        )
        for tokenizer in all_tokenizers:
            regular_token_intersection -= set(tokenizer.all_special_tokens)
        regular_token_intersection = sorted(regular_token_intersection)

        self.precompile = precompile

        self.generators = [
            Generator(
                mesh=mesh,
                model=all_models[model_idx],
                params=all_params[model_idx],
                tokenizer=all_tokenizers[model_idx],
                sampling_config=sampling_config,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                dtype=dtype,
                logit_mask=all_logit_masks[model_idx] if all_logit_masks else None,
                precompile=precompile,
                until=until,
                lengths=lengths,
                eos_strategy="stop",
                pad_to_multiple_of=pad_to_multiple_of,
            )
            for model_idx in range(len(all_models))
        ]
        self.sampling_config = sampling_config

        # disallow next tokens besides eot
        self.unified_tokens = [
            regular_token_intersection + [generator.replacements["<|<eot>|>"][0]]
            for generator in self.generators
        ]

        min_vocab_size = min(config.vocab_size for config in all_configs)

        self.unified_indices = [
            sharding.to_global_array(
                np.pad(
                    np.array(
                        tokenizer.convert_tokens_to_ids(unified_tokens), dtype=np.int32
                    ),
                    ((0, min_vocab_size - len(unified_tokens))),
                    mode="constant",
                    constant_values=0,
                ),
                NamedSharding(mesh, P()),
            )
            for tokenizer, unified_tokens in zip(all_tokenizers, self.unified_tokens)
        ]
        self.unified_indices_mask = np.concatenate(
            [
                np.ones(len(self.unified_tokens[0]), dtype=bool),
                np.zeros(min_vocab_size - len(self.unified_tokens[0]), dtype=bool),
            ]
        )

        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.combine_fn = combine_fn
        self.combine_params = combine_params
        self.lengths = lengths

        self.state_shardings = LockstepState(
            states=[generator.state_shardings for generator in self.generators],
            combine_params=NamedSharding(mesh, P()),
            prng=NamedSharding(mesh, P(None)),
            is_sent_finished=NamedSharding(mesh, P(None)),
        )

    def generate_condition(self, state: LockstepState):
        cond = None

        for i, generator in enumerate(self.generators):
            if cond is None:
                cond = generator.generate_condition(state.states[i])
            else:
                cond = jnp.logical_and(
                    cond, generator.generate_condition(state.states[i])
                )

        return cond

    def generate_step(
        self,
        state: LockstepState,
    ):
        next_states = [
            generator.generate_step(state)
            for generator, state in zip(self.generators, state.states)
        ]

        logits = [
            state.logits[:, unified_indices]
            for state, unified_indices in zip(next_states, self.unified_indices)
        ]
        hidden_states = [state.hidden_states for state in next_states]
        output_embeddings = []
        for i, generator in enumerate(self.generators):
            if generator.config.tie_word_embeddings:
                output_embeddings.append(
                    param.get(
                        state.states[i].params,
                        param.get_input_embedding_path(generator.config.model_type),
                    )[self.unified_indices[i]]
                )
            else:
                output_embeddings.append(
                    param.get(
                        state.states[i].params,
                        param.get_output_embedding_path(generator.config.model_type),
                    ).T[self.unified_indices[i]]
                )

        logits = self.combine_fn(
            hidden_states, logits, self.combine_params, output_embeddings
        )
        logits = jnp.where(
            self.unified_indices_mask[None],
            logits,
            utils.get_large_negative_number(logits.dtype),
        )

        if not self.sampling_config.do_sample:
            next_token = logits.argmax(-1)
            next_prng = state.prng
        else:
            assert self.sampling_config.top_k == 0, "Top-k sampling not supported yet"

            if self.sampling_config.top_p < 1:
                warper = FlaxTopPLogitsWarper(
                    self.sampling_config.top_p,
                    filter_value=utils.get_large_negative_number(logits.dtype),
                )

                logits = warper(None, logits, None)

            probs = jax.nn.softmax(logits / self.sampling_config.temperature, -1)

            def choose_next(prng, p):
                prng, next_prng = jax.random.split(prng, 2)
                return jax.random.choice(prng, probs.shape[1], p=p), next_prng

            next_token, next_prng = jax.vmap(choose_next)(state.prng, probs)

        is_sent_finished = state.is_sent_finished

        for i in range(len(self.generators)):
            next_token_i = self.unified_indices[i][next_token]
            next_token_i = (
                next_token_i * ~state.is_sent_finished
                + self.generators[i].pad_token_id * state.is_sent_finished
            )
            is_sent_finished = is_sent_finished | (
                next_token_i == self.generators[i].eos_token_id
            )

            next_token_i = next_token_i[:, None]
            next_states[i] = next_states[i].replace(
                running_token=next_token_i,
                sequences=lax.dynamic_update_slice(
                    next_states[i].sequences,
                    next_token_i,
                    (0, next_states[i].cur_len - 1),
                ),
                is_sent_finished=is_sent_finished,
            )

        return LockstepState(
            states=next_states,
            combine_params=state.combine_params,
            prng=next_prng,
            is_sent_finished=is_sent_finished,
        )

    def get_zero_state(self, length):
        states = [
            generator.get_zero_state(length).replace(
                logits=jnp.zeros(
                    (self.batch_size, generator.config.vocab_size),
                    dtype=jnp.float32,
                    device=NamedSharding(generator.mesh, P()),
                ),
                hidden_states=jnp.zeros(
                    (
                        self.batch_size,
                        generator.config.hidden_size,
                    ),
                    dtype=jnp.float32,
                    device=NamedSharding(generator.mesh, P()),
                ),
            )
            for generator in self.generators
        ]

        return LockstepState(
            states=states,
            combine_params=self.combine_params,
            prng=jax.random.split(jax.random.key(0), self.batch_size),
            is_sent_finished=jnp.zeros(self.batch_size, dtype=bool),
        )

    def _compile_fns(self, compile_prefill=True, compile_generate=True):
        compiled_prefill_fns = []
        for generator in self.generators:
            compiled_prefill_fns.append(
                generator._compile_fns(
                    compile_prefill=compile_prefill, compile_generate=False
                )[0]
            )

        if compile_generate and self.precompile:
            compiled_generate_fns = {}
            for length in tqdm(self.lengths, desc="Compiling generate fns..."):
                compiled_generate_fns[length] = (
                    jax.jit(
                        lambda state: lax.while_loop(
                            self.generate_condition, self.generate_step, state
                        ),
                        donate_argnums=(0,),
                        out_shardings=self.state_shardings,
                    )
                    .lower(self.get_zero_state(length))
                    .compile()
                )
        else:
            compiled_generate_fns = {
                length: lambda state: lax.while_loop(
                    self.generate_condition, self.generate_step, state
                )
                for length in self.lengths
            }

        return compiled_prefill_fns, compiled_generate_fns

    def generate(self, prompts, seed=1234, verbose=True):
        all_prefill_tokens = [[] for _ in range(len(self.generators))]
        all_running_tokens = [[] for _ in range(len(self.generators))]

        for i, generator in enumerate(self.generators):
            for prompt in tqdm(
                prompts, desc="Encoding prompts...", disable=not verbose
            ):
                prompt_tokens = [
                    generator.vocab[x]
                    for x in utils.encode_prompt(prompt, generator.tokenizer)[0]
                ]

                all_prefill_tokens[i].append(prompt_tokens[:-1])
                all_running_tokens[i].append(prompt_tokens[-1:])

        # longest prompts first to overestimate (rather than underestimate) time it takes to generate
        # inconsequential which model we use for permutation_indices
        permutation_indices = np.argsort([len(x) for x in all_prefill_tokens[0]])[::-1]
        generations = [None] * len(prompts)

        compiled_prefill_fns, compiled_generate_fns = self._compile_fns()

        n_batches = math.ceil(len(prompts) / self.batch_size)
        prngs = jax.random.split(jax.random.key(seed), n_batches)

        init_caches = [
            generator.get_cache(self.lengths[0]) for generator in self.generators
        ]
        init_cache_length = self.lengths[0]

        for batch_idx in tqdm(
            range(n_batches), desc="Generating...", disable=not verbose
        ):
            (start, end) = (
                batch_idx * self.batch_size,
                (batch_idx + 1) * self.batch_size,
            )

            batch_indices = permutation_indices[start:end]
            batch_indices = np.pad(
                batch_indices, (0, self.batch_size - len(batch_indices)), mode="edge"
            )

            unpadded_prefill_input_ids = [
                [all_prefill_tokens[i][j] for j in batch_indices]
                for i in range(len(self.generators))
            ]
            max_new_tokens = min(
                self.max_new_tokens,
                self.lengths[-1]
                - max(len(x) for x in (itertools.chain(*unpadded_prefill_input_ids))),
            )
            if max_new_tokens < self.max_new_tokens:
                print(
                    f"Warning: max_new_tokens reduced from {self.max_new_tokens} to {max_new_tokens}"
                )
            padded_prefill_length = get_lowest_upper_bound(
                max(len(x) for x in (itertools.chain(*unpadded_prefill_input_ids)))
                + max_new_tokens,
                self.lengths,
            )

            prefill_input_ids = [
                np.full(
                    (self.batch_size, padded_prefill_length),
                    fill_value=generator.tokenizer.pad_token_id,
                    dtype=np.int32,
                )
                for generator in self.generators
            ]
            attention_mask = [
                np.zeros((self.batch_size, padded_prefill_length), dtype=np.int32)
                for _ in self.generators
            ]
            running_tokens = [
                jnp.array(
                    [all_running_tokens[model_idx][i] for i in batch_indices],
                    dtype=jnp.int32,
                )
                for model_idx in range(len(self.generators))
            ]

            for model_idx in range(len(self.generators)):
                for i, input_ids in enumerate(unpadded_prefill_input_ids[model_idx]):
                    prefill_input_ids[model_idx][i, : len(input_ids)] = input_ids
                    attention_mask[model_idx][i, : len(input_ids)] = 1
                    attention_mask[model_idx][
                        i, padded_prefill_length - max_new_tokens :
                    ] = 1

            if padded_prefill_length > init_cache_length:
                init_caches = [
                    generator.get_cache(padded_prefill_length)
                    for generator in self.generators
                ]
                init_cache_length = padded_prefill_length

            init_caches = [
                self.generators[model_idx].set_cache_length(init_caches[model_idx], 0)
                for model_idx in range(len(self.generators))
            ]
            caches = [
                compiled_prefill_fns[model_idx][padded_prefill_length](
                    prefill_input_ids[model_idx],
                    attention_mask[model_idx],
                    self.generators[model_idx].params,
                    init_caches[model_idx],
                )[0]
                for model_idx in range(len(self.generators))
            ]
            caches = [
                self.generators[model_idx].set_cache_length(
                    caches[model_idx], padded_prefill_length - max_new_tokens
                )
                for model_idx in range(len(self.generators))
            ]

            position_ids = [
                jnp.array(
                    [[len(x)] for x in unpadded_prefill_input_ids[model_idx]],
                    dtype=jnp.int32,
                )
                for model_idx in range(len(self.generators))
            ]

            states = []

            for model_idx in range(len(self.generators)):
                states.append(
                    State(
                        cur_len=jnp.array(
                            padded_prefill_length - max_new_tokens, dtype=jnp.int32
                        ),
                        sequences=sharding.to_global_array(
                            prefill_input_ids[model_idx],
                            NamedSharding(self.generators[model_idx].mesh, P()),
                        ),
                        running_token=running_tokens[model_idx],
                        is_sent_finished=jnp.zeros(self.batch_size, dtype=bool),
                        prng=jax.random.split(prngs[batch_idx], self.batch_size),
                        model_kwargs={
                            "position_ids": position_ids[model_idx],
                            "attention_mask": attention_mask[model_idx],
                            "past_key_values": caches[model_idx],
                        },
                        params=self.generators[model_idx].params,
                        logits=jnp.zeros(
                            (
                                self.batch_size,
                                self.generators[model_idx].config.vocab_size,
                            ),
                            device=NamedSharding(self.generators[model_idx].mesh, P()),
                        ),
                        hidden_states=jnp.zeros(
                            (
                                self.batch_size,
                                self.generators[model_idx].config.hidden_size,
                            ),
                            device=NamedSharding(self.generators[model_idx].mesh, P()),
                        ),
                    )
                )

            state = LockstepState(
                states=states,
                combine_params=self.combine_params,
                prng=jax.random.split(prngs[batch_idx], self.batch_size),
                is_sent_finished=jnp.zeros(self.batch_size, dtype=bool),
            )

            state = compiled_generate_fns[padded_prefill_length](state)
            last_model_sequences = process_allgather(
                state.states[-1].sequences[:, -max_new_tokens:], tiled=True
            )

            last_model_special_ids = self.generators[
                -1
            ].tokenizer.convert_tokens_to_ids(
                self.generators[-1].tokenizer.model_kind_cls.special_tokens
            )
            for i, idx in enumerate(batch_indices):
                generations[idx] = [
                    token_id
                    for token_id in last_model_sequences[i]
                    if token_id not in last_model_special_ids
                ]

            # recycle out state buffers
            init_caches = []
            for model_idx in range(len(self.generators)):
                init_caches.append(
                    state.states[model_idx].model_kwargs["past_key_values"]
                )
                self.generators[model_idx].params = state.states[model_idx].params

            self.combine_params = state.combine_params

        return generations