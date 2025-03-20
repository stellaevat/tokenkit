from tqdm.auto import tqdm as raw_tqdm
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import copy
import tokenizers
from dataclasses import dataclass
from pathlib import Path
from flax import traverse_util
import regex as re
import logging
import json
from tokenizers import Tokenizer
from tempfile import NamedTemporaryFile
import wandb
from pprint import pprint
from google.cloud import storage
import flax
from scipy import sparse


tqdm = partial(raw_tqdm, dynamic_ncols=True, disable=jax.process_index() != 0)

logger = logging.getLogger(__name__)


def log(data, step, **kwargs):
    pprint({**data, "_step": step})
    if jax.process_index() == 0:
        wandb.log(data, step=step, **kwargs)


def ensure_pad_token_set(tokenizer):
    pad_token = next(
        token
        for token in [
            tokenizer.pad_token,
            tokenizer.unk_token,
            tokenizer.eos_token,
            tokenizer.bos_token,
        ]
        if token is not None
    )
    tokenizer.pad_token = pad_token
    tokenizer.padding_side = "right"


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


def normalize_special_tokens(
    tokens, tokenizer, attention_mask, model_kind, special_tokens_mode
):
    normalized_tokens = []
    special_tokens_mask = []

    first_regular_token = True

    special_tokens = set(get_special_tokens(model_kind)) | set(tokenizer.added_tokens_encoder.keys())

    if "+" in special_tokens_mode:
        special_tokens_mode, normalize_policy = special_tokens_mode.split("+")
    else:
        normalize_policy = None

    if special_tokens_mode == "ignore_pad_bos":
        bos_tokens = get_replacements(model_kind)["<|<bos>|>"]
        assert bos_tokens is None or len(bos_tokens) == 1
        bos_token = bos_tokens[0] if bos_tokens is not None else None

    for token, m in zip(tokens, attention_mask):
        if token in special_tokens:
            # TODO: probably always ignore masked tokens and remove/refactor special_tokens_mode
            if (
                special_tokens_mode == "ignore"
                or (special_tokens_mode == "ignore_pad" and not m)
                or (
                    special_tokens_mode == "ignore_pad_bos"
                    and (not m or token == bos_token)
                )
            ):
                normalized_token = ""
                is_special = True
            else:
                if normalize_policy == "keep_literal":
                    normalized_token = token
                else:
                    normalized_token = "<|<special_token>|>"
                is_special = False
        else:
            # TODO: does this work?
            if first_regular_token:
                normalized_token = token.lstrip(
                    "Ġ"
                )  # to align tokenizers with/without prefix space
                first_regular_token = False
            else:
                normalized_token = token
            is_special = False

        normalized_tokens.append(normalized_token)
        special_tokens_mask.append(is_special)

    return normalized_tokens, np.array(special_tokens_mask)


def get_min_length_alignments(
    input_ids_original,
    input_ids_new,
    tokenizer,
    attention_mask_original=None,
    attention_mask_new=None,
    compute_alignment_matrices=False,
    original_tokenizer=None,
    model_kind_original=None,
    model_kind_new=None,
    special_tokens_mode="use",
):
    # TODO: potentially cleaner to have this required
    if original_tokenizer is None:
        original_tokenizer = tokenizer

    if attention_mask_new is not None:
        attention_mask_new = attention_mask_new.astype(bool)
    if attention_mask_original is not None:
        attention_mask_original = attention_mask_original.astype(bool)

    alignments_mask = np.zeros(input_ids_new.shape, dtype=bool)
    nonalignments_mask = np.zeros(input_ids_new.shape, dtype=bool)
    all_cum_lengths_original = np.zeros(input_ids_original.shape, dtype=np.int32)
    all_cum_lengths_new = np.zeros(input_ids_new.shape, dtype=np.int32)
    mismatches = np.zeros(len(input_ids_new), dtype=bool)

    if compute_alignment_matrices:
        batch_size = input_ids_new.shape[0]
        shared_length = min(input_ids_original.shape[1], input_ids_new.shape[1])
        alignment_matrix_a = np.zeros(
            (batch_size, input_ids_new.shape[1], shared_length), dtype=bool
        )
        alignment_matrix_b = np.zeros(
            (batch_size, input_ids_original.shape[1], shared_length), dtype=bool
        )
    else:
        alignment_matrix_a = alignment_matrix_b = None

    for example_index in range(len(input_ids_original)):
        tokens_original = original_tokenizer.convert_ids_to_tokens(
            input_ids_original[example_index]
        )
        tokens_new = tokenizer.convert_ids_to_tokens(input_ids_new[example_index])
        tokens_original, special_tokens_mask_original = normalize_special_tokens(
            tokens_original,
            original_tokenizer,
            attention_mask_original[example_index],
            model_kind=model_kind_original,
            special_tokens_mode=special_tokens_mode,
        )
        tokens_new, special_tokens_mask_new = normalize_special_tokens(
            tokens_new,
            tokenizer,
            attention_mask_new[example_index],
            model_kind=model_kind_new,
            special_tokens_mode=special_tokens_mode,
        )

        assert tokens_original[0].startswith(tokens_new[0]) or tokens_new[0].startswith(
            tokens_original[0]
        )

        cum_lengths_original = np.cumsum([len(token) for token in tokens_original])
        cum_lengths_new = np.cumsum([len(token) for token in tokens_new])

        all_cum_lengths_original[example_index] = cum_lengths_original
        all_cum_lengths_new[example_index] = cum_lengths_new

        shared_length = min(cum_lengths_original[-1], cum_lengths_new[-1])

        # find alignment of new tokens to original tokens
        alignment_chunk_idx = 0
        prev_i = i = 0
        prev_j = j = 0

        joined_original = "".join(tokens_original)
        joined_new = "".join(tokens_new)

        if not (joined_original.startswith(joined_new) or joined_new.startswith(
            joined_original
        )):
            mismatches[example_index] = True

        while i < len(cum_lengths_new) and j < len(cum_lengths_original):
            if cum_lengths_new[i] == cum_lengths_original[j]:
                i += 1
                j += 1

                if compute_alignment_matrices:
                    alignment_matrix_a[example_index, prev_i:i, alignment_chunk_idx] = (
                        True
                    )
                    alignment_matrix_b[example_index, prev_j:j, alignment_chunk_idx] = (
                        True
                    )

                    alignment_chunk_idx += 1

                if i - prev_i > 1:
                    nonalignments_mask[example_index, prev_i:i] = True
                else:
                    alignments_mask[example_index, prev_i:i] = True

                prev_i = i
                prev_j = j
            elif cum_lengths_new[i] < cum_lengths_original[j]:
                i += 1
            else:
                j += 1

        alignments_mask[example_index, special_tokens_mask_new] = False
        nonalignments_mask[example_index, special_tokens_mask_new] = False

        if compute_alignment_matrices:
            alignment_matrix_a[example_index, special_tokens_mask_new, :] = False
            alignment_matrix_b[example_index, special_tokens_mask_original, :] = False

    return (
        alignments_mask,
        nonalignments_mask,
        all_cum_lengths_original,
        all_cum_lengths_new,
        alignment_matrix_a,
        alignment_matrix_b,
        mismatches,
    )


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


def get_space_alignments(
    input_ids_original,
    input_ids_new,
    tokenizer,
    attention_mask_original=None,
    attention_mask_new=None,
    compute_alignment_matrices=False,
    original_tokenizer=None,
    model_kind_original=None,
    model_kind_new=None,
    special_tokens_mode="use",
):
    # TODO: potentially cleaner to have this required
    if original_tokenizer is None:
        original_tokenizer = tokenizer

    shared_length = min(input_ids_original.shape[1], input_ids_new.shape[1])

    alignments_mask = np.zeros(input_ids_new.shape, dtype=bool)
    nonalignments_mask = np.zeros(input_ids_new.shape, dtype=bool)
    chunk_starts_with_space = np.zeros((len(input_ids_new), shared_length), dtype=bool)

    if compute_alignment_matrices:
        batch_size = input_ids_new.shape[0]
        alignment_matrix_a = np.zeros(
            (batch_size, input_ids_new.shape[1], shared_length), dtype=bool
        )
        alignment_matrix_b = np.zeros(
            (batch_size, input_ids_original.shape[1], shared_length), dtype=bool
        )
    else:
        alignment_matrix_a = alignment_matrix_b = None

    for example_index in range(len(input_ids_original)):
        tokens_original = original_tokenizer.convert_ids_to_tokens(
            input_ids_original[example_index]
        )
        tokens_new = tokenizer.convert_ids_to_tokens(input_ids_new[example_index])
        tokens_original, special_tokens_mask_original = normalize_special_tokens(
            tokens_original,
            original_tokenizer,
            attention_mask_original[example_index],
            model_kind=model_kind_original,
            special_tokens_mode=special_tokens_mode,
        )
        tokens_new, special_tokens_mask_new = normalize_special_tokens(
            tokens_new,
            tokenizer,
            attention_mask_new[example_index],
            model_kind=model_kind_new,
            special_tokens_mode=special_tokens_mode,
        )

        assert tokens_original[0].startswith(tokens_new[0]) or tokens_new[0].startswith(
            tokens_original[0]
        )

        cum_lengths_original = np.cumsum([len(token) for token in tokens_original])
        cum_lengths_new = np.cumsum([len(token) for token in tokens_new])

        original_starts_with_space = np.array(
            [len(token) > 0 and token[0] == "Ġ" for token in tokens_original]
        )
        new_starts_with_space = np.array(
            [len(token) > 0 and token[0] == "Ġ" for token in tokens_new]
        )

        # find alignment of new tokens to original tokens
        alignment_chunk_idx = 0
        prev_i = i = 0
        prev_j = j = 0

        while i < len(cum_lengths_new) and j < len(cum_lengths_original):
            if cum_lengths_new[i] == cum_lengths_original[j] and (
                special_tokens_mask_new[i]
                or (
                    (
                        i < len(new_starts_with_space) - 1
                        and new_starts_with_space[i + 1]
                    )
                    and (
                        j < len(original_starts_with_space) - 1
                        and original_starts_with_space[j + 1]
                    )
                )
            ):
                i += 1
                j += 1

                assert (
                    new_starts_with_space[prev_i] == original_starts_with_space[prev_j]
                )
                chunk_starts_with_space[example_index, alignment_chunk_idx] = (
                    new_starts_with_space[prev_i]
                )

                if compute_alignment_matrices:
                    alignment_matrix_a[example_index, prev_i:i, alignment_chunk_idx] = (
                        True
                    )
                    alignment_matrix_b[example_index, prev_j:j, alignment_chunk_idx] = (
                        True
                    )

                if i - prev_i > 1:
                    nonalignments_mask[example_index, prev_i:i] = True
                else:
                    alignments_mask[example_index, prev_i:i] = True

                alignment_chunk_idx += 1
                prev_i = i
                prev_j = j
            elif cum_lengths_new[i] == cum_lengths_original[j]:
                i += 1
                j += 1
            elif cum_lengths_new[i] < cum_lengths_original[j]:
                i += 1
            else:
                j += 1

        alignments_mask[example_index, special_tokens_mask_new] = False
        nonalignments_mask[example_index, special_tokens_mask_new] = False

        if compute_alignment_matrices:
            alignment_matrix_a[example_index, special_tokens_mask_new, :] = False
            alignment_matrix_b[example_index, special_tokens_mask_original, :] = False

    return (
        alignments_mask,
        nonalignments_mask,
        alignment_matrix_a,
        alignment_matrix_b,
        chunk_starts_with_space,
    )


def get_unbiased_alignments(
    input_ids_original,
    input_ids_new,
    tokenizer,
    pair_data,
    bias_threshold,
    attention_mask_original=None,
    attention_mask_new=None,
    compute_alignment_matrices=False,
    original_tokenizer=None,
    model_kind_original=None,
    model_kind_new=None,
    special_tokens_mode="use",
):
    (bias1_matrix, bias2_matrix, _, _) = pair_data

    # TODO: potentially cleaner to have this required
    if original_tokenizer is None:
        original_tokenizer = tokenizer

    shared_length = min(input_ids_original.shape[1], input_ids_new.shape[1])

    alignments_mask = np.zeros(input_ids_new.shape, dtype=bool)
    nonalignments_mask = np.zeros(input_ids_new.shape, dtype=bool)

    if compute_alignment_matrices:
        batch_size = input_ids_new.shape[0]
        alignment_matrix_a = np.zeros(
            (batch_size, input_ids_new.shape[1], shared_length), dtype=bool
        )
        alignment_matrix_b = np.zeros(
            (batch_size, input_ids_original.shape[1], shared_length), dtype=bool
        )
    else:
        alignment_matrix_a = alignment_matrix_b = None

    teacher_length, student_length = bias1_matrix.shape

    def is_unbiased(original_token_id, new_token_id):
        # hacky, to handle <|<special_token>|> case
        if original_token_id is None or new_token_id is None:
            return True

        return (
            original_token_id >= teacher_length or new_token_id >= student_length
        ) or (
            bias1_matrix[original_token_id, new_token_id] <= bias_threshold
            and bias2_matrix[original_token_id, new_token_id] <= bias_threshold
        )

    for example_index in range(len(input_ids_original)):
        tokens_original = original_tokenizer.convert_ids_to_tokens(
            input_ids_original[example_index]
        )
        tokens_new = tokenizer.convert_ids_to_tokens(input_ids_new[example_index])
        tokens_original, special_tokens_mask_original = normalize_special_tokens(
            tokens_original,
            original_tokenizer,
            attention_mask_original[example_index],
            model_kind=model_kind_original,
            special_tokens_mode=special_tokens_mode,
        )
        tokens_new, special_tokens_mask_new = normalize_special_tokens(
            tokens_new,
            tokenizer,
            attention_mask_new[example_index],
            model_kind=model_kind_new,
            special_tokens_mode=special_tokens_mode,
        )

        assert tokens_original[0].startswith(tokens_new[0]) or tokens_new[0].startswith(
            tokens_original[0]
        )

        cum_lengths_original = np.cumsum([len(token) for token in tokens_original])
        cum_lengths_new = np.cumsum([len(token) for token in tokens_new])

        # find alignment of new tokens to original tokens
        alignment_chunk_idx = 0
        prev_i = i = 0
        prev_j = j = 0

        while i < len(cum_lengths_new) and j < len(cum_lengths_original):
            if cum_lengths_new[i] == cum_lengths_original[j] and (
                special_tokens_mask_new[i]
                or is_unbiased(
                    original_tokenizer.convert_tokens_to_ids(tokens_original[j]),
                    tokenizer.convert_tokens_to_ids(tokens_new[i]),
                )
            ):
                i += 1
                j += 1

                if compute_alignment_matrices:
                    alignment_matrix_a[example_index, prev_i:i, alignment_chunk_idx] = (
                        True
                    )
                    alignment_matrix_b[example_index, prev_j:j, alignment_chunk_idx] = (
                        True
                    )

                if i - prev_i > 1:
                    nonalignments_mask[example_index, prev_i:i] = True
                else:
                    alignments_mask[example_index, prev_i:i] = True

                alignment_chunk_idx += 1
                prev_i = i
                prev_j = j
            elif cum_lengths_new[i] == cum_lengths_original[j]:
                i += 1
                j += 1
            elif cum_lengths_new[i] < cum_lengths_original[j]:
                i += 1
            else:
                j += 1

        alignments_mask[example_index, special_tokens_mask_new] = False
        nonalignments_mask[example_index, special_tokens_mask_new] = False

        if compute_alignment_matrices:
            alignment_matrix_a[example_index, special_tokens_mask_new, :] = False
            alignment_matrix_b[example_index, special_tokens_mask_original, :] = False

    return (
        alignments_mask,
        nonalignments_mask,
        alignment_matrix_a,
        alignment_matrix_b,
    )


# NOTE: assumes tokenizer is byte converted
def to_longest_prefix_tokenizer(tokenizer, inplace=False):
    if not inplace:
        lp_tokenizer = copy.deepcopy(tokenizer)
    else:
        lp_tokenizer = tokenizer

    unk_token = (
        lp_tokenizer.unk_token
        if lp_tokenizer.unk_token is not None
        else lp_tokenizer.eos_token
    )

    # use WordPiece without prefix to achieve longest-prefix tokenization
    lp_tokenizer._tokenizer.model = tokenizers.models.WordPiece(
        lp_tokenizer.get_vocab(),
        unk_token=unk_token,
        max_input_chars_per_word=1_000_000,  # effectively disable limit on input chars
    )
    lp_tokenizer._tokenizer.model.continuing_subword_prefix = ""

    return lp_tokenizer


def fix_postprocessor_data(data, vocab):
    if data["type"] == "TemplateProcessing":
        for k in data["special_tokens"].keys():
            tokens = data["special_tokens"][k]["tokens"]
            ids = [vocab[t] for t in tokens]
            data["special_tokens"][k]["ids"] = ids
    elif data["type"] == "RobertaProcessing":
        data["sep"][1] = vocab[data["sep"][0]]
        data["cls"][1] = vocab[data["cls"][0]]
    elif data["type"] == "Sequence":
        for postprocessor in data["processors"]:
            fix_postprocessor_data(postprocessor, vocab)


def expand_input_ids(input_ids_new, tokenizer, original_vocab, use_heuristic=False):
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

                while prefix_start > 0 and not starts_with_space[prefix_start]:
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


def to_byte_tokenizer(tokenizer, inplace=False):
    if not inplace:
        byte_tokenizer = copy.deepcopy(tokenizer)
    else:
        byte_tokenizer = tokenizer

    # keep special/added tokens
    tokens_to_keep_ids = sorted(
        set(
            tokenizer.all_special_ids
            + list(tokenizer.added_tokens_encoder.values())
            + tokenizer.convert_tokens_to_ids(tokenizer.added_tokens_encoder.keys())
        )
    )
    tokens_to_keep = tokenizer.convert_ids_to_tokens(tokens_to_keep_ids)
    byte_tokens_in_vocab = [
        token
        for token in tokenizer.get_vocab()
        if token in CHARS_TO_BYTES.keys() and token not in tokens_to_keep
    ]
    byte_tokens_not_in_vocab = [
        token for token in CHARS_TO_BYTES.keys() if token not in byte_tokens_in_vocab
    ]

    unk_token = (
        tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
    )

    if len(byte_tokens_not_in_vocab) > 0:
        logger.warning(
            f"Some byte are tokens not in vocab: {byte_tokens_not_in_vocab}. Adding these to the vocab. They will not have a good init."
        )

    tokens = list(CHARS_TO_BYTES.keys()) + tokens_to_keep
    byte_vocab = {token: i for i, token in enumerate(tokens)}

    # use ByteLevel tokenizer to achieve byte tokenization
    byte_tokenizer._tokenizer.model = tokenizers.models.WordPiece(
        byte_vocab,
        unk_token=unk_token,
        max_input_chars_per_word=1_000_000,  # effectively disable limit on input chars
    )

    # remove added tokens, they would persist to the old vocabulary id
    f = NamedTemporaryFile()
    byte_tokenizer._tokenizer.save(f.name)
    tokenizer_data = json.load(open(f.name, "r"))
    if "added_tokens" in tokenizer_data:
        del tokenizer_data["added_tokens"]
    if "post_processor" in tokenizer_data:
        fix_postprocessor_data(tokenizer_data["post_processor"], byte_vocab)

    json.dump(tokenizer_data, open(f.name, "w"))

    byte_tokenizer._tokenizer = Tokenizer.from_file(f.name)
    byte_tokenizer._tokenizer.model.continuing_subword_prefix = ""

    return byte_tokenizer


def fvt(
    source_tokenizer,
    target_tokenizer,
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
                decomposed = source_tokenizer._tokenizer.model.tokenize(token)
                constituent_idx = np.array(
                    [x.id for x in decomposed if x.id < len(source_embeddings)]
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


DEFAULT_SPLIT_REGEX = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
MAX_CHARS_PER_TOKEN = 16
# assumes byte pretokenization
CHARS_TO_BYTES = {
    "Ā": 0,
    "ā": 1,
    "Ă": 2,
    "ă": 3,
    "Ą": 4,
    "ą": 5,
    "Ć": 6,
    "ć": 7,
    "Ĉ": 8,
    "ĉ": 9,
    "Ċ": 10,
    "ċ": 11,
    "Č": 12,
    "č": 13,
    "Ď": 14,
    "ď": 15,
    "Đ": 16,
    "đ": 17,
    "Ē": 18,
    "ē": 19,
    "Ĕ": 20,
    "ĕ": 21,
    "Ė": 22,
    "ė": 23,
    "Ę": 24,
    "ę": 25,
    "Ě": 26,
    "ě": 27,
    "Ĝ": 28,
    "ĝ": 29,
    "Ğ": 30,
    "ğ": 31,
    "Ġ": 32,
    "!": 33,
    '"': 34,
    "#": 35,
    "$": 36,
    "%": 37,
    "&": 38,
    "'": 39,
    "(": 40,
    ")": 41,
    "*": 42,
    "+": 43,
    ",": 44,
    "-": 45,
    ".": 46,
    "/": 47,
    "0": 48,
    "1": 49,
    "2": 50,
    "3": 51,
    "4": 52,
    "5": 53,
    "6": 54,
    "7": 55,
    "8": 56,
    "9": 57,
    ":": 58,
    ";": 59,
    "<": 60,
    "=": 61,
    ">": 62,
    "?": 63,
    "@": 64,
    "A": 65,
    "B": 66,
    "C": 67,
    "D": 68,
    "E": 69,
    "F": 70,
    "G": 71,
    "H": 72,
    "I": 73,
    "J": 74,
    "K": 75,
    "L": 76,
    "M": 77,
    "N": 78,
    "O": 79,
    "P": 80,
    "Q": 81,
    "R": 82,
    "S": 83,
    "T": 84,
    "U": 85,
    "V": 86,
    "W": 87,
    "X": 88,
    "Y": 89,
    "Z": 90,
    "[": 91,
    "\\": 92,
    "]": 93,
    "^": 94,
    "_": 95,
    "`": 96,
    "a": 97,
    "b": 98,
    "c": 99,
    "d": 100,
    "e": 101,
    "f": 102,
    "g": 103,
    "h": 104,
    "i": 105,
    "j": 106,
    "k": 107,
    "l": 108,
    "m": 109,
    "n": 110,
    "o": 111,
    "p": 112,
    "q": 113,
    "r": 114,
    "s": 115,
    "t": 116,
    "u": 117,
    "v": 118,
    "w": 119,
    "x": 120,
    "y": 121,
    "z": 122,
    "{": 123,
    "|": 124,
    "}": 125,
    "~": 126,
    "ġ": 127,
    "Ģ": 128,
    "ģ": 129,
    "Ĥ": 130,
    "ĥ": 131,
    "Ħ": 132,
    "ħ": 133,
    "Ĩ": 134,
    "ĩ": 135,
    "Ī": 136,
    "ī": 137,
    "Ĭ": 138,
    "ĭ": 139,
    "Į": 140,
    "į": 141,
    "İ": 142,
    "ı": 143,
    "Ĳ": 144,
    "ĳ": 145,
    "Ĵ": 146,
    "ĵ": 147,
    "Ķ": 148,
    "ķ": 149,
    "ĸ": 150,
    "Ĺ": 151,
    "ĺ": 152,
    "Ļ": 153,
    "ļ": 154,
    "Ľ": 155,
    "ľ": 156,
    "Ŀ": 157,
    "ŀ": 158,
    "Ł": 159,
    "ł": 160,
    "¡": 161,
    "¢": 162,
    "£": 163,
    "¤": 164,
    "¥": 165,
    "¦": 166,
    "§": 167,
    "¨": 168,
    "©": 169,
    "ª": 170,
    "«": 171,
    "¬": 172,
    "Ń": 173,
    "®": 174,
    "¯": 175,
    "°": 176,
    "±": 177,
    "²": 178,
    "³": 179,
    "´": 180,
    "µ": 181,
    "¶": 182,
    "·": 183,
    "¸": 184,
    "¹": 185,
    "º": 186,
    "»": 187,
    "¼": 188,
    "½": 189,
    "¾": 190,
    "¿": 191,
    "À": 192,
    "Á": 193,
    "Â": 194,
    "Ã": 195,
    "Ä": 196,
    "Å": 197,
    "Æ": 198,
    "Ç": 199,
    "È": 200,
    "É": 201,
    "Ê": 202,
    "Ë": 203,
    "Ì": 204,
    "Í": 205,
    "Î": 206,
    "Ï": 207,
    "Ð": 208,
    "Ñ": 209,
    "Ò": 210,
    "Ó": 211,
    "Ô": 212,
    "Õ": 213,
    "Ö": 214,
    "×": 215,
    "Ø": 216,
    "Ù": 217,
    "Ú": 218,
    "Û": 219,
    "Ü": 220,
    "Ý": 221,
    "Þ": 222,
    "ß": 223,
    "à": 224,
    "á": 225,
    "â": 226,
    "ã": 227,
    "ä": 228,
    "å": 229,
    "æ": 230,
    "ç": 231,
    "è": 232,
    "é": 233,
    "ê": 234,
    "ë": 235,
    "ì": 236,
    "í": 237,
    "î": 238,
    "ï": 239,
    "ð": 240,
    "ñ": 241,
    "ò": 242,
    "ó": 243,
    "ô": 244,
    "õ": 245,
    "ö": 246,
    "÷": 247,
    "ø": 248,
    "ù": 249,
    "ú": 250,
    "û": 251,
    "ü": 252,
    "ý": 253,
    "þ": 254,
    "ÿ": 255,
}
BYTES_TO_CHARS = {v: k for k, v in CHARS_TO_BYTES.items()}


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}"
    )


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(
        f"Uploaded {source_file_name} to bucket {bucket_name} as {destination_blob_name}"
    )


def is_gcs_path(path):
    return path.startswith("gs://")


def parse_gcs_path(gcs_path):
    path_parts = gcs_path[len("gs://") :].split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    return bucket_name, blob_name


# generation / chat utils

SPECIAL_KEYS = [
    "<|<bos>|>",
    "<|<start_header>|>",
    "<|<end_header>|>",
    "<|<eot>|>",
    "<|<user_name>|>",
    "<|<assistant_name>|>",
]
ALL_SPECIAL_TOKENS = {
    "Qwen2": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
    "Llama3": [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|end_of_text|>",
    ],
    "Gemma2": ["<bos>", "<start_of_turn>", "<end_of_turn>", "<eos>", "<pad>"],
    "Phi3": ["<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"],
    "GPT2": ["<|endoftext|>"],
}
REPLACEMENTS = {
    "Qwen2": {
        "<|<bos>|>": None,
        "<|<start_header>|>": ["<|im_start|>"],
        "<|<end_header>|>": ["Ċ"],
        "<|<eot>|>": ["<|im_end|>", "Ċ"],
        "<|<eos>|>": ["<|endoftext|>"],
        "<|<user_name>|>": ["user"],
        "<|<assistant_name>|>": ["assistant"],
        "<|<system_name>|>": ["system"],
    },
    "Llama3": {
        "<|<bos>|>": ["<|begin_of_text|>"],
        "<|<start_header>|>": ["<|start_header_id|>"],
        "<|<end_header>|>": ["<|end_header_id|>", "ĊĊ"],
        "<|<eot>|>": ["<|eot_id|>"],
        "<|<eos>|>": ["<|eot_id|>"],
        "<|<user_name>|>": ["user"],
        "<|<assistant_name>|>": ["assistant"],
        "<|<system_name>|>": ["system"],
    },
    "Gemma2": {
        "<|<bos>|>": ["<bos>"],
        "<|<start_header>|>": ["<start_of_turn>"],
        "<|<end_header>|>": ["Ċ"],
        "<|<eot>|>": ["<end_of_turn>", "Ċ"],
        "<|<eos>|>": ["<eos>"],
        "<|<user_name>|>": ["user"],
        "<|<assistant_name>|>": ["model"],
        "<|<system_name>|>": ["user"],
    },
    "Phi3": {
        "<|<bos>|>": None,
        "<|<start_header>|>": None,
        "<|<end_header>|>": ["Ċ"],
        "<|<eot>|>": ["<|end|>", "Ċ"],
        "<|<eos>|>": ["<|endoftext|>"],
        "<|<user_name>|>": ["<|user|>"],
        "<|<assistant_name>|>": ["<|assistant|>"],
    },
    "GPT2": {
        "<|<bos>|>": None,
        "<|<start_header>|>": None,
        "<|<end_header>|>": None,
        "<|<eot>|>": ["<|endoftext|>"],
        "<|<eos>|>": ["<|endoftext|>"],
        "<|<user_name>|>": None,
        "<|<assistant_name>|>": None,
    },
}


def get_special_tokens(model_kind):
    return ALL_SPECIAL_TOKENS[model_kind]


def get_replacements(model_kind):
    return REPLACEMENTS[model_kind]


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


def encode_prompt(prompt, tokenizer, replacements, max_length=None):
    tokens = []
    regular_token_indices = []

    if max_length is not None:
        prompt = prompt[: MAX_CHARS_PER_TOKEN * max_length]

    added_token_starts = set(x[0] for x in tokenizer.added_tokens_encoder.keys())

    def process_chunk(chunk):
        if chunk in tokenizer.added_tokens_encoder:
            tokens.append(chunk)
            regular_token_indices.append(-1)
        elif chunk in replacements:
            if replacements[chunk] is not None:
                tokens.extend(replacements[chunk])
                regular_token_indices.extend([-1] * len(replacements[chunk]))
        else:
            chunk_pretokens = [
                x[0]
                for x in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
                    chunk
                )
            ]
            chunk_tokens = [
                t.value
                for pretoken in chunk_pretokens
                for t in tokenizer.backend_tokenizer.model.tokenize(pretoken)
            ]

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
            key = next(key for key in replacements.keys() if prompt[i:].startswith(key))
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

            chunk = prompt[start_i:i + len(key)]
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


def get_num_layers(config):
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):  # gpt2
        return config.n_layer
    else:
        raise ValueError("Could not determine number of layers from config")


def set_num_layers(config, num_layers):
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_layers
    elif hasattr(config, "n_layer"):  # gpt2
        config.n_layer = num_layers
    else:
        raise ValueError("Could not determine number of layers from config")


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

            for student_idx, teacher_idx in zip(student_mapping.copy(), teacher_mapping.copy()):
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
