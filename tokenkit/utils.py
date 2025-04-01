import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from pprint import pformat

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import regex as re
from flax import traverse_util
from scipy import sparse
from tqdm.auto import tqdm as raw_tqdm
from transformers import AutoTokenizer

import wandb
from tokenkit import constants
from tokenkit.byteify import ByteifyTokenizer, load_byteify_tokenizer

tqdm = partial(raw_tqdm, dynamic_ncols=True, disable=jax.process_index() != 0)

logger = logging.getLogger(__name__)


def log(data, step, **kwargs):
    logger.info(pformat({**data, "_step": step}))
    if jax.process_index() == 0:
        wandb.log(data, step=step, **kwargs)


def keystr(x):
    if hasattr(x, "name"):
        return x.name
    elif hasattr(x, "key"):
        return x.key
    elif hasattr(x, "idx"):
        return x.idx

    assert isinstance(x, str)
    return x


# from praxis
def get_large_negative_number(dtype: jnp.dtype | np.dtype, module=jnp) -> jax.Array:
    """Returns a large negative value for the given dtype."""
    # -0.7 is a float64 in Jax. Explicit cast output to target dtype.
    if module.issubdtype(dtype, module.inexact):
        dtype_max = module.finfo(dtype).max
    elif module.issubdtype(dtype, module.integer):
        dtype_max = module.iinfo(dtype).max
    else:
        raise ValueError("Unsupported dtype for inputs.")

    return module.asarray(-0.7 * dtype_max, dtype=dtype)


def get_space_mask(tokenizer):
    space_mask = np.zeros(len(tokenizer), dtype=bool)
    tokens = tokenizer.convert_ids_to_tokens(np.arange(len(tokenizer)))
    special_token_ids = set(tokenizer.all_special_ids)

    for i, token in enumerate(tokens):
        if (
            len(token) > 0 and (token[0] == "Ġ" or token[0] == "Ċ" or token[0] == "ĉ")
        ) or i in special_token_ids:
            space_mask[i] = True

    return space_mask


def expand_input_ids(
    input_ids_new,
    tokenizer,
    original_vocab,
    use_heuristic=False,
    maxlen=constants.EXPAND_INPUT_IDS_MAX_LENGTH,
):
    expanded_input_ids = np.zeros_like(input_ids_new)

    for example_index in range(len(input_ids_new)):
        tokens_new = tokenizer.convert_ids_to_tokens(input_ids_new[example_index])

        if use_heuristic:
            # use a heuristic for pretokenization with 100% recall to narrow down possible candidates
            starts_with_space = [
                token == tokenizer.pad_token or (len(token) > 0 and token[0] == "Ġ")
                for token in tokens_new
            ]
        else:
            starts_with_space = None

        for token_idx in range(len(tokens_new)):
            expanded_token_id = None

            if use_heuristic:
                prefix_start = token_idx

                while (
                    prefix_start > 0
                    and (maxlen is None or token_idx + 1 - prefix_start < maxlen)
                    and not starts_with_space[prefix_start]
                ):
                    prefix_start -= 1
            else:
                prefix_start = 0

            for prefix_idx in range(prefix_start, token_idx + 1):
                expanded_token_id = original_vocab.get(
                    "".join(tokens_new[prefix_idx : token_idx + 1])
                )
                if expanded_token_id is not None:
                    break

            expanded_input_ids[example_index, token_idx] = expanded_token_id

    return expanded_input_ids


def fvt(
    source_tokenizer: ByteifyTokenizer,
    target_tokenizer: ByteifyTokenizer,
    source_embeddings,
    fallback_mode="random",
    verbose=True,
    allow_exact_match=True,
):
    # assumes both tokenizers are byte-level
    source_vocab = source_tokenizer.get_vocab()

    original_to_new_indices = np.zeros(len(target_tokenizer), dtype=int)
    diff_indices = []
    diff_embeddings = []

    stats = {
        "exact_match": 0,
        "averaged": 0,
        "fallback": 0,
    }

    source_mean = source_embeddings.mean(0)
    source_std = source_embeddings.std(0)

    for i in tqdm(
        range(len(target_tokenizer)), desc="Applying FVT..", disable=not verbose
    ):
        token = target_tokenizer.convert_ids_to_tokens(i)

        if (
            token in source_vocab
            and source_vocab[token] < len(source_embeddings)
            and allow_exact_match
        ):
            stats["exact_match"] += 1
            original_to_new_indices[i] = source_vocab[token]
        else:
            original_to_new_indices[i] = (
                0  # will be overwritten by setting diff_indices
            )
            diff_indices.append(i)

            if token in source_vocab:
                if source_vocab[token] < len(source_embeddings):
                    constituent_idx = np.array([source_vocab[token]])
                else:
                    constituent_idx = np.array([])
            else:
                try:
                    decomposed = source_tokenizer.convert_tokens_to_ids(
                        source_tokenizer.backend_tokenize(token)
                    )
                except UnicodeDecodeError:
                    decomposed = []
                constituent_idx = np.array(
                    [x for x in decomposed if x < len(source_embeddings)]
                )

            if len(constituent_idx) > 0:
                diff_embeddings.append(source_embeddings[constituent_idx].mean(0))
                stats["averaged"] += 1
            else:
                if fallback_mode == "random":
                    fallback_embedding = np.random.normal(
                        loc=source_mean,
                        scale=source_std,
                    )
                else:
                    fallback_embedding = source_embeddings[
                        source_tokenizer.unk_token_id
                    ]

                diff_embeddings.append(fallback_embedding)
                stats["fallback"] += 1

    logger.info(f"FVT exact match: {stats['exact_match']}")
    logger.info(f"FVT averaged: {stats['averaged']}")
    logger.info(f"FVT fallback: {stats['fallback']}")

    diff_indices = np.array(diff_indices, dtype=int)
    diff_embeddings = np.array(diff_embeddings, dtype=np.float32)

    return diff_embeddings, original_to_new_indices, diff_indices


def label_by_prefix(pytree, label_maps, default=None):
    flat_pytree = traverse_util.flatten_dict(pytree)
    labels = {}

    for k in flat_pytree:
        for prefix, label in label_maps:
            is_match = (
                isinstance(prefix, str)
                and re.match(prefix, ".".join(k))
                or isinstance(prefix, tuple)
                and k[: len(prefix)] == prefix
            )

            if is_match:
                labels[k] = label
                break

        if k not in labels:
            if default is None:
                raise ValueError(f"No label found for key: {k}")
            else:
                labels[k] = default

    return traverse_util.unflatten_dict(labels)


def get_n_pad(n, pad_to_multiple_of):
    n_overflow = n % pad_to_multiple_of
    if n_overflow > 0:
        n_pad = pad_to_multiple_of - n_overflow
    else:
        n_pad = 0

    return n_pad


def param_report(params, train_mask):
    # TODO: update with LoRA support
    return

    for key, value in params.items():

        @dataclass
        class ParamInfo:
            size: int
            trainable: bool

        def count_params(acc, info):
            total_count, trainable_count = acc

            return (
                total_count + info.size,
                trainable_count + info.size if info.trainable else trainable_count,
            )

        param_info = jax.tree.map(
            lambda x, trainable: ParamInfo(size=x.size, trainable=trainable),
            value,
            train_mask[key],
        )
        if not isinstance(param_info, dict):
            # make sure reduce works
            param_info = {"dummy": param_info}

        num_params, num_trainable_params = jax.tree.reduce(
            count_params,
            param_info,
            initializer=(0, 0),
        )

        # TODO: get rid of prints, and probably make return arg instead
        print(f"Num {key} params: {num_params}")
        print(f"Num {key} trainable params: {num_trainable_params}")


def get_surface_form_matrix(
    tokenizer_or_tokens, maxlen, hn_tokenizer=None, padding=0, verbose=False
):
    # tokens are expected to be byte encoded
    if isinstance(tokenizer_or_tokens, list):
        tokens = tokenizer_or_tokens
    else:
        tokenizer = tokenizer_or_tokens
        tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))

    vocab_size = len(tokens)
    surface_form_matrix = np.full(
        (vocab_size + padding, maxlen),
        hn_tokenizer.pad_token_id if hn_tokenizer is not None else 0,
        dtype=np.int32,
    )

    n_truncated = 0

    for i, token in tqdm(enumerate(tokens), total=vocab_size, disable=not verbose):
        if token in hn_tokenizer.all_special_tokens:
            surface_form_matrix[i, 0] = hn_tokenizer.convert_tokens_to_ids(token)
            continue

        # assume hn tokenizer uses byte pretokenization
        ids = [x.id for x in hn_tokenizer._tokenizer.model.tokenize(token)]

        if len(ids) > maxlen:
            ids = ids[:maxlen]
            n_truncated += 1

        surface_form_matrix[i, : len(ids)] = ids

    return surface_form_matrix, n_truncated


def preprocess_messages(messages):
    # convert messages format to prompt with chat template
    prompt = "<|<bos>|>"
    for message in messages:
        role_tag = {
            "user": "<|<user_name>|>",
            "assistant": "<|<assistant_name>|>",
            "system": "<|<system_name>|>",
        }[message["role"]]

        prompt += (
            f"<|<start_header>|>{role_tag}<|<end_header>|>{message['content']}<|<eot>|>"
        )
    return prompt


def preprocess_prompt(prompt, chat_template_mode):
    if chat_template_mode == "surround_instruct":
        prompt = f"<|<bos>|><|<start_header>|><|<user_name>|><|<end_header>|>{prompt}<|<eot>|><|<start_header>|><|<assistant_name>|><|<end_header>|>"
    elif chat_template_mode == "direct_encode":
        if not prompt.startswith("<|<bos>|>"):
            prompt = "<|<bos>|>" + prompt
        if not (prompt.endswith("<|<eot>|>") or prompt.endswith("<|<eos>|>")):
            prompt = prompt + "<|<eos>|>"
    elif chat_template_mode == "direct_encode_no_force_eos":
        if not prompt.startswith("<|<bos>|>"):
            prompt = "<|<bos>|>" + prompt
    elif chat_template_mode == "direct_encode_no_force_bos":
        if not (prompt.endswith("<|<eot>|>") or prompt.endswith("<|<eos>|>")):
            prompt = prompt + "<|<eos>|>"
    elif chat_template_mode == "direct_encode_no_force_bos_no_force_eos":
        pass
    else:
        raise ValueError(f"Unknown chat template mode: {chat_template_mode}")

    return prompt


def encode_prompt(prompt, tokenizer, max_length=None):
    tokens = []
    regular_token_indices = []

    if max_length is not None:
        prompt = prompt[: constants.MAX_CHARS_PER_TOKEN * max_length]

    added_token_starts = set(x[0] for x in tokenizer.added_tokens_encoder.keys())

    def process_chunk(chunk):
        if chunk in tokenizer.added_tokens_encoder:
            tokens.append(tokenizer.model_kind_cls.byte_fallback_fn(chunk))
            regular_token_indices.append(-1)
        elif chunk in tokenizer.model_kind_cls.replacements:
            if tokenizer.model_kind_cls.replacements[chunk] is not None:
                tokens.extend(tokenizer.model_kind_cls.replacements[chunk])
                regular_token_indices.extend(
                    [-1] * len(tokenizer.model_kind_cls.replacements[chunk])
                )
        else:
            chunk_tokens = tokenizer.convert_ids_to_tokens(
                tokenizer(chunk, add_special_tokens=False)["input_ids"]
            )
            tokens.extend(chunk_tokens)

            try:
                regular_token_start = next(
                    i for i in regular_token_indices[::-1] if i != -1
                )
            except StopIteration:
                regular_token_start = -1

            regular_token_indices.extend(
                [regular_token_start + 1 + i for i in range(len(chunk_tokens))]
            )

    start_i = 0
    i = 0

    while i < len(prompt):
        try:
            key = next(
                key
                for key in tokenizer.model_kind_cls.replacements.keys()
                if prompt[i:].startswith(key)
            )
        except StopIteration:
            key = None

        if key is None:
            if prompt[i] in added_token_starts:
                try:
                    key = next(
                        key
                        for key in tokenizer.added_tokens_encoder.keys()
                        if prompt[i:].startswith(key)
                    )
                except StopIteration:
                    key = None

        if key is not None:
            if start_i < i:
                chunk = prompt[start_i:i]
                process_chunk(chunk)
                start_i = i

            chunk = prompt[start_i : i + len(key)]
            process_chunk(chunk)
            start_i = i + len(key)
            i = start_i

            if max_length is not None and len(tokens) >= max_length:
                return tokens[:max_length], regular_token_indices[:max_length]
        else:
            i += 1

    if start_i < len(prompt):
        chunk = prompt[start_i:]
        process_chunk(chunk)

    if max_length is not None:
        return tokens[:max_length], regular_token_indices[:max_length]
    else:
        return tokens, regular_token_indices


def make_hashable(obj):
    """Recursively convert lists to tuples so they become hashable."""
    if isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(
            (k, make_hashable(v)) for k, v in sorted(obj.items())
        )  # Sort keys for consistency
    return obj


def init_linear(seed, in_shape, out_shape, dtype, **kwargs):
    return jax.device_get(
        flax.linen.Dense(features=out_shape, dtype=dtype, **kwargs).init(
            jax.random.PRNGKey(seed),
            jnp.ones((1, in_shape), dtype=dtype, device=jax.devices("cpu")[0]),
        )["params"]
    )


def compute_unigram_probabilities(tokenizer, counts, additive_smoothing_constant=1e-9):
    counts_sum = sum(counts.values())
    probs = np.array(
        [
            counts.get(token_id, 0) + additive_smoothing_constant * counts_sum
            for token_id in range(len(tokenizer))
        ],
        dtype=np.float32,
    )
    probs /= probs.sum()

    return probs


def get_side_path_mappings(
    teacher_tokenizer,
    student_tokenizer,
    mode,
    tokenizer_pair_data_path=None,
    tokenizer_pair_bias_threshold=None,
    mined_mapping=None,
):
    if mode == "mined":
        assert mined_mapping is not None
        student_mapping = np.arange(len(student_tokenizer))
        teacher_mapping = mined_mapping
    elif mode in {"exact_match", "bias_threshold"}:
        student_tokens = student_tokenizer.convert_ids_to_tokens(
            np.arange(len(student_tokenizer))
        )
        teacher_vocab = teacher_tokenizer.get_vocab()

        student_mapping = []
        teacher_mapping = []

        for student_idx, token in enumerate(student_tokens):
            if token in teacher_vocab:
                student_mapping.append(student_idx)
                teacher_mapping.append(teacher_vocab[token])

        if mode == "bias_threshold":
            assert tokenizer_pair_data_path is not None
            assert tokenizer_pair_bias_threshold is not None
            bias1_matrix = sparse.load_npz(
                Path(tokenizer_pair_data_path) / "bias1_matrix.npz"
            ).todok()
            bias2_matrix = sparse.load_npz(
                Path(tokenizer_pair_data_path) / "bias2_matrix.npz"
            ).todok()

            teacher_length, student_length = bias1_matrix.shape

            for student_idx, teacher_idx in zip(
                student_mapping.copy(), teacher_mapping.copy()
            ):
                if (
                    student_idx >= student_length
                    or teacher_idx >= teacher_length
                    or (
                        (
                            bias1_matrix[teacher_idx, student_idx]
                            <= tokenizer_pair_bias_threshold
                        )
                        and (
                            bias2_matrix[teacher_idx, student_idx]
                            <= tokenizer_pair_bias_threshold
                        )
                    )
                ):
                    continue

                student_mapping.remove(student_idx)
                teacher_mapping.remove(teacher_idx)

        student_mapping = np.array(student_mapping, dtype=np.int32)
        teacher_mapping = np.array(teacher_mapping, dtype=np.int32)
    else:
        raise ValueError(f"Unknown side path mapping mode: {mode}")

    return student_mapping, teacher_mapping


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "google/gemma-2-2b-it:source=Gemma2",
        "Qwen/Qwen2-1.5B-Instruct:source=Qwen2",
        "meta-llama/Llama-3.1-8B-Instruct:source=Llama3",
    ],
)
def test_encode_prompt(tokenizer_name):
    tokenizer = load_byteify_tokenizer(tokenizer_name)
    comparison_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name.split(":")[0])

    messages = [
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hello, user!"},
    ]

    tokens, _ = encode_prompt(preprocess_messages(messages), tokenizer)
    comparison_token_ids = comparison_tokenizer.apply_chat_template(
        messages, use_system_prompt=False
    )

    tokens = comparison_tokenizer.convert_ids_to_tokens(
        tokenizer.convert_tokens_to_ids(tokens)
    )
    comparison_tokens = comparison_tokenizer.convert_ids_to_tokens(comparison_token_ids)

    # apply_chat_template may inject an (undesired) system prompt, so the best we can do is to check the suffix (and skip first token since it may be bos)
    assert " ".join(comparison_tokens).endswith(" ".join(tokens[1:]))
