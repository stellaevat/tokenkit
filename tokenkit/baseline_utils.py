import math
import multiprocessing
from functools import partial

import editdistance
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


def _compute_edit_distance(token, sorted_original_vocab):
    min_edit_distance = math.inf
    best_match = None

    closer_to_start = len(token) < len(
        sorted_original_vocab[int(len(sorted_original_vocab) / 2)]
    )

    if closer_to_start:
        candidates = sorted_original_vocab
    else:
        candidates = reversed(sorted_original_vocab)

    for original_token in candidates:
        if closer_to_start:
            # tokens only get longer
            if len(original_token) - len(token) >= min_edit_distance:
                break
            if len(token) - len(original_token) >= min_edit_distance:
                continue
        else:
            # tokens only get shorter
            if len(token) - len(original_token) >= min_edit_distance:
                break
            if len(original_token) - len(token) >= min_edit_distance:
                continue

        edit_distance = editdistance.eval(token, original_token)
        if edit_distance < min_edit_distance:
            min_edit_distance = edit_distance
            best_match = original_token

    return token, best_match, min_edit_distance


def compute_mined_mapping(
    tokenizer_original, tokenizer_new, num_workers=1, chunksize=500
):
    original_vocab = tokenizer_original.get_vocab()
    new_vocab = tokenizer_new.get_vocab()

    mapping = np.zeros(len(tokenizer_new), dtype=np.int32)
    edit_distances = {}

    intersection = [token for token in new_vocab.keys() if token in original_vocab]
    completion = [token for token in new_vocab.keys() if token not in original_vocab]
    sorted_completion = sorted(completion, key=lambda x: len(x))
    sorted_original_vocab = sorted(original_vocab.keys(), key=lambda x: len(x))

    for token in intersection:
        mapping[new_vocab[token]] = original_vocab[token]
        edit_distances[token] = 0

    with multiprocessing.Pool(max(num_workers, 1)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    partial(
                        _compute_edit_distance,
                        sorted_original_vocab=sorted_original_vocab,
                    ),
                    sorted_completion,
                    chunksize=chunksize,
                ),
                desc="Computing MinED mapping",
                total=len(sorted_completion),
            )
        )

    for token, best_match, min_edit_distance in results:
        mapping[new_vocab[token]] = original_vocab[best_match]
        edit_distances[token] = min_edit_distance

    return mapping, edit_distances


def compute_forward_kl_divergence(
    logits,
    teacher_logits,
    target,
    kd_temp,
    padding_id,
    tea_temp=None,
    reduction="sum",
    log=None,
    use_tea_temp=False,
):
    logits = logits / kd_temp
    teacher_logits = teacher_logits / kd_temp
    teacher_logits = teacher_logits / tea_temp if use_tea_temp else teacher_logits

    lprobs = jax.nn.log_softmax(logits, -1)
    teacher_probs = jax.nn.softmax(teacher_logits, -1)
    teacher_lprobs = jax.nn.log_softmax(teacher_logits, -1)
    kld = teacher_probs * (teacher_lprobs - lprobs)
    inf_mask = jnp.isinf(logits)
    kld = jnp.where(inf_mask, 0.0, kld).sum(-1)

    if reduction == "sum":
        pad_mask = target == padding_id
        kld = jnp.where(pad_mask, 0.0, kld)
        kld = kld.sum()

        if log is not None:
            log["forward_kl"] = kld

    return kld


def compute_reverse_kl_divergence(
    logits,
    teacher_logits,
    target,
    kd_temp,
    padding_id,
    tea_temp=None,
    reduction="sum",
    log=None,
    use_tea_temp=False,
):
    logits = logits / kd_temp
    teacher_logits = teacher_logits / kd_temp
    teacher_logits = teacher_logits / tea_temp if use_tea_temp else teacher_logits

    probs = jax.nn.softmax(logits, -1)
    lprobs = jax.nn.log_softmax(logits, -1)
    teacher_lprobs = jax.nn.log_softmax(teacher_logits, -1)
    kld = probs * (lprobs - teacher_lprobs)
    inf_mask = jnp.isinf(logits) | jnp.isinf(teacher_logits)
    kld = jnp.where(inf_mask, 0.0, kld).sum(-1)

    if reduction == "sum":
        pad_mask = target == padding_id
        kld = jnp.where(pad_mask, 0.0, kld)
        kld = kld.sum()

        if log is not None:
            log["reverse_kl"] = kld

    return kld


def compute_adaptive_kl_divergence(
    logits,
    teacher_logits,
    target,
    kd_temp,
    padding_id,
    alpha,
    tea_temp=None,
    reduction="sum",
    log=None,
    use_tea_temp=False,
):
    probs = jax.nn.softmax(logits / kd_temp, axis=-1).astype(jnp.float32)
    if use_tea_temp:
        teacher_probs = jax.nn.softmax(
            teacher_logits / tea_temp / kd_temp, axis=-1
        ).astype(jnp.float32)
    else:
        teacher_probs = jax.nn.softmax(teacher_logits / kd_temp, axis=-1).astype(
            jnp.float32
        )

    sorted_teacher_probs = jnp.sort(teacher_probs, axis=-1)
    sorted_idx = jnp.argsort(teacher_probs, axis=-1)
    sorted_probs = jnp.take_along_axis(
        probs, sorted_idx, axis=-1
    )  # TODO: check if we need [..., None]?
    gap = jnp.abs(sorted_teacher_probs - sorted_probs)
    cum_teacher_probs = jnp.cumsum(sorted_teacher_probs, axis=-1)
    tail_mask = (cum_teacher_probs < alpha).astype(jnp.float32)
    g_head = jax.lax.stop_gradient(jnp.sum(gap * (1 - tail_mask), axis=-1))
    g_tail = jax.lax.stop_gradient(jnp.sum(gap * tail_mask, axis=-1))

    fkl = compute_forward_kl_divergence(
        logits,
        teacher_logits,
        target,
        kd_temp,
        padding_id,
        tea_temp=tea_temp,
        reduction="none",
        use_tea_temp=use_tea_temp,
    )
    rkl = compute_reverse_kl_divergence(
        logits,
        teacher_logits,
        target,
        kd_temp,
        padding_id,
        tea_temp=tea_temp,
        reduction="none",
        use_tea_temp=use_tea_temp,
    )

    akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl

    if reduction == "sum":
        pad_mask = target == padding_id
        akl = jnp.where(pad_mask, 0.0, akl)
        akl = akl.sum()

        if log is not None:
            log["adaptive_kl"] = akl

    return akl


def compute_skewed_forward_kl_divergence(
    logits,
    teacher_logits,
    target,
    kd_temp,
    padding_id,
    skew_lambda,
    tea_temp=None,
    reduction="sum",
    log=None,
    use_tea_temp=False,
    epsilon=1e-9,
):
    logits = logits / kd_temp
    teacher_logits = teacher_logits / kd_temp
    teacher_logits = teacher_logits / tea_temp if use_tea_temp else teacher_logits

    student_probs = jax.nn.softmax(logits, -1).astype(jnp.float32)
    teacher_probs = jax.nn.softmax(teacher_logits, -1).astype(jnp.float32)
    mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * student_probs
    mixed_lprobs = jnp.log(mixed_probs + epsilon)
    teacher_lprobs = jax.nn.log_softmax(teacher_logits, -1).astype(jnp.float32)
    kld = teacher_probs * (teacher_lprobs - mixed_lprobs)
    inf_mask = jnp.isinf(logits) | jnp.isinf(teacher_logits)
    kld = jnp.where(inf_mask, 0.0, kld).sum(-1)

    if reduction == "sum":
        pad_mask = target == padding_id
        kld = jnp.where(pad_mask, 0.0, kld)
        kld = kld.sum()

        if log is not None:
            log["skewed_forward_kl"] = kld

    return kld


def compute_skewed_reverse_kl_divergence(
    logits,
    teacher_logits,
    target,
    kd_temp,
    padding_id,
    skew_lambda,
    tea_temp=None,
    reduction="sum",
    log=None,
    use_tea_temp=False,
    epsilon=1e-9,
):
    logits = logits / kd_temp
    teacher_logits = teacher_logits / kd_temp
    teacher_logits = teacher_logits / tea_temp if use_tea_temp else teacher_logits

    student_probs = jax.nn.softmax(logits, -1).astype(jnp.float32)
    teacher_probs = jax.nn.softmax(teacher_logits, -1).astype(jnp.float32)
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * student_probs
    mixed_lprobs = jnp.log(mixed_probs + epsilon)
    student_lprobs = jax.nn.log_softmax(logits, -1).astype(jnp.float32)
    kld = student_probs * (student_lprobs - mixed_lprobs)
    inf_mask = jnp.isinf(logits) | jnp.isinf(teacher_logits)
    kld = jnp.where(inf_mask, 0.0, kld).sum(-1)

    if reduction == "sum":
        pad_mask = target == padding_id
        kld = jnp.where(pad_mask, 0.0, kld)
        kld = kld.sum()

        if log is not None:
            log["skewed_reverse_kl"] = kld

    return kld


def compute_js_divergence(
    logits,
    teacher_logits,
    target,
    kd_temp,
    tea_temp,
    padding_id,
    reduction="sum",
    log=None,
    use_tea_temp=False,
    epsilon=1e-9,
):
    logits = logits / kd_temp
    teacher_logits = teacher_logits / kd_temp
    teacher_logits = teacher_logits / tea_temp if use_tea_temp else teacher_logits

    probs = jax.nn.softmax(logits, -1).astype(jnp.float32)
    teacher_probs = jax.nn.softmax(teacher_logits, -1).astype(jnp.float32)
    m_probs = (probs + teacher_probs) / 2

    lprobs = jnp.log(probs + epsilon)
    teacher_lprobs = jnp.log(teacher_probs + epsilon)
    m_lprobs = jnp.log(m_probs + epsilon)

    kld1 = teacher_probs * (teacher_lprobs - m_lprobs)
    kld2 = probs * (lprobs - m_lprobs)
    kld = (kld1 + kld2) / 2

    if reduction == "sum":
        pad_mask = target == padding_id
        kld = jnp.where(pad_mask, 0.0, kld)
        kld = kld.sum()

        if log is not None:
            log["js_div"] = kld

    return kld
