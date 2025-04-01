"""
Example Usage:

ipython --pdb scripts/compute_tokenizer_info.py -- \
    teacher_tokenizer_name=google/gemma-2-2b-it:source=Gemma2 \
    target_tokenizer_name=Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2 \
    output='outputs/tokenizer_data/gemma2_to_qwen2_new'
"""

import json
import os
import pickle
from collections import Counter
from functools import partial
from pathlib import Path

import hydra
import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tokenkit import data
from tokenkit.byteify import load_byteify_tokenizer


def compute_prefix_map(tokenizer):
    prefix_map = {}

    for token in tokenizer.get_vocab().keys():
        for i in range(1, len(token) + 1):
            if token[:i] in prefix_map:
                prefix_map[token[:i]].append(token)
            else:
                prefix_map[token[:i]] = [token]

    return prefix_map


def is_valid(tokens, tokenizer):
    try:
        return tokenizer.backend_tokenize("".join(tokens)) == tokens
    except UnicodeDecodeError:
        return False


def compute_cover_set(pretoken, tokenizer, prefix_map):
    cover_set = []
    for i in range(len(pretoken) - 1, -1, -1):
        B = prefix_map.get(pretoken[i:], [])
        try:
            tcur = tokenizer.backend_tokenize(pretoken[:i])
        except UnicodeDecodeError:
            continue

        for b in B:
            if is_valid(tcur + [b], tokenizer):
                cover_set.append(tcur + [b])

    return cover_set


def compute_cover_dict(pretoken, tokenizer, prefix_map):
    cover_set = compute_cover_set(pretoken, tokenizer, prefix_map)
    cover_dict = {}

    for seq in cover_set:
        joined_seq = "".join(seq)[len(pretoken) :]
        if len(joined_seq) == 0:
            continue

        cover_dict[joined_seq] = tokenizer.convert_tokens_to_ids(seq)

    return cover_dict


def compute_pair_bias(
    pretoken1,
    pretoken2,
    tokenizer1,
    tokenizer2,
    prefix_map1,
    prefix_map2,
    probs1,
    probs2,
    return_diff_cover_dicts=False,
):
    cover_dict1 = compute_cover_dict(pretoken1, tokenizer1, prefix_map1)
    cover_dict2 = compute_cover_dict(pretoken2, tokenizer2, prefix_map2)

    diff_keys1 = set(cover_dict1.keys()) - set(cover_dict2.keys())
    diff_keys2 = set(cover_dict2.keys()) - set(cover_dict1.keys())

    bias1 = 0.0
    for key in diff_keys1:
        bias1 += probs1[cover_dict1[key][-1]]

    bias2 = 0.0
    for key in diff_keys2:
        bias2 += probs2[cover_dict2[key][-1]]

    if return_diff_cover_dicts:
        diff_cover_set1 = {key: probs1[cover_dict1[key][-1]] for key in diff_keys1}
        diff_cover_set2 = {key: probs2[cover_dict2[key][-1]] for key in diff_keys2}
        return bias1, bias2, diff_cover_set1, diff_cover_set2
    else:
        return bias1, bias2


def count_tokens_map(examples, tokenizer):
    flat_input_ids = [
        input_id
        for input_ids in tokenizer(examples["text"], add_special_tokens=False)[
            "input_ids"
        ]
        for input_id in input_ids
    ]
    return {
        "counter": pickle.dumps(Counter(flat_input_ids)),
    }


def count_tokens(dset, tokenizer, num_workers):
    token_counters_dset = dset.map(
        partial(count_tokens_map, tokenizer=tokenizer),
        batched=False,  # already batched
        num_proc=num_workers if num_workers > 0 else None,
        remove_columns=dset.column_names,
        desc="Counting tokens",
    )

    global_token_counter = Counter()
    for i in tqdm(range(len(token_counters_dset)), desc="Merging token counters"):
        global_token_counter.update(pickle.loads(token_counters_dset[i]["counter"]))

    return global_token_counter


if __name__ == "__main__":
    args = None

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _parse_args_fn(local_args):
        global args
        args = local_args

    hydra.main(
        version_base=None,
        config_path="../configs",
        config_name="compute_tokenizer_info",
    )(_parse_args_fn)()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer_teacher = load_byteify_tokenizer(args.teacher_tokenizer_name)
    target_tokenizer = load_byteify_tokenizer(args.target_tokenizer_name)

    dset = data.get_dataset(**args.data, seed=args.seed)

    if not (output_dir / "teacher_counts.json").exists():
        assert isinstance(dset, data.HFDataset)
        dset_to_use = dset.stream

        if args.teacher_subsample_percent is not None:
            n_subsample = int(len(dset_to_use) * args.teacher_subsample_percent)
            dset_to_use = dset_to_use.select(np.arange(n_subsample))

        teacher_token_counts = count_tokens(
            dset_to_use, tokenizer_teacher, args.data.num_workers
        )
        json.dump(
            teacher_token_counts,
            open(output_dir / "teacher_counts.json", "w"),
            indent=4,
        )
    else:
        teacher_token_counts = Counter(
            json.load(open(output_dir / "teacher_counts.json"))
        )
    if not (output_dir / "student_counts.json").exists():
        assert isinstance(dset, data.HFDataset)
        dset_to_use = dset.stream

        if args.student_subsample_percent is not None:
            n_subsample = int(len(dset_to_use) * args.student_subsample_percent)
            dset_to_use = dset_to_use.select(np.arange(n_subsample))

        student_token_counts = count_tokens(
            dset_to_use, target_tokenizer, args.data.num_workers
        )
        json.dump(
            student_token_counts,
            open(output_dir / "student_counts.json", "w"),
            indent=4,
        )
    else:
        student_token_counts = Counter(
            json.load(open(output_dir / "student_counts.json"))
        )

    if not (output_dir / "pairs.json").exists():
        teacher_tokens_dict = {}
        for token in sorted(
            tokenizer_teacher.get_vocab().keys(), key=lambda x: x[::-1]
        ):
            if token[-1] not in teacher_tokens_dict:
                teacher_tokens_dict[token[-1]] = []

            teacher_tokens_dict[token[-1]].append(token)

        student_tokens_dict = {}
        for token in sorted(target_tokenizer.get_vocab().keys(), key=lambda x: x[::-1]):
            if token[-1] not in student_tokens_dict:
                student_tokens_dict[token[-1]] = []

            student_tokens_dict[token[-1]].append(token)

        pairs = []
        for last_byte in tqdm(
            set(teacher_tokens_dict.keys()) & set(student_tokens_dict.keys())
        ):
            for teacher_token in teacher_tokens_dict[last_byte]:
                for student_token in student_tokens_dict[last_byte]:
                    if teacher_token.endswith(student_token) or student_token.endswith(
                        teacher_token
                    ):
                        pairs.append((teacher_token, student_token))

        json.dump(pairs, open(output_dir / "pairs.json", "w"), indent=4)
    else:
        pairs = json.load(open(output_dir / "pairs.json"))

    print(f"Found {len(pairs)} pairs")

    prefix_map_teacher = compute_prefix_map(tokenizer_teacher)
    prefix_map_student = compute_prefix_map(target_tokenizer)

    teacher_counts_sum = sum(teacher_token_counts.values())
    teacher_token_probs = np.array(
        [
            teacher_token_counts[token_id]
            + args.additive_smoothing_constant * teacher_counts_sum
            for token_id in range(len(tokenizer_teacher))
        ],
        dtype=np.float32,
    )
    teacher_token_probs /= teacher_token_probs.sum()

    student_counts_sum = sum(student_token_counts.values())
    student_token_probs = np.array(
        [
            student_token_counts[token_id]
            + args.additive_smoothing_constant * student_counts_sum
            for token_id in range(len(target_tokenizer))
        ],
        dtype=np.float32,
    )
    student_token_probs /= student_token_probs.sum()

    is_space_only_teacher = {
        tokenizer_teacher.convert_ids_to_tokens(i): len(
            tokenizer_teacher.decode(i).strip()
        )
        == 0
        for i in range(len(tokenizer_teacher))
    }
    is_space_only_student = {
        target_tokenizer.convert_ids_to_tokens(i): len(
            target_tokenizer.decode(i).strip()
        )
        == 0
        for i in range(len(target_tokenizer))
    }

    def pair_collate(pairs):
        biases1 = []
        biases2 = []
        for pair in pairs:
            if is_space_only_teacher[pair[0]] or is_space_only_student[pair[1]]:
                biases1.append(1.0)  # can take long to compute and likely high
                biases2.append(1.0)
                continue
            bias1, bias2 = compute_pair_bias(
                *pair,
                tokenizer_teacher,
                target_tokenizer,
                prefix_map_teacher,
                prefix_map_student,
                teacher_token_probs,
                student_token_probs,
            )
            biases1.append(bias1)
            biases2.append(bias2)

        return {
            "biases1": biases1,
            "biases2": biases2,
        }

    pair_permutation = np.random.permutation(len(pairs))
    inv_pair_permutation = np.argsort(pair_permutation)

    biases1 = []
    biases2 = []
    pair_data_loader = DataLoader(
        [pairs[i] for i in pair_permutation],
        batch_size=args.data.batch_size,
        num_workers=args.data.num_workers,
        collate_fn=pair_collate,
    )

    for batch in tqdm(pair_data_loader, desc="Computing pair biases"):
        biases1.extend(batch["biases1"])
        biases2.extend(batch["biases2"])

    biases1 = np.array(biases1)[inv_pair_permutation]
    biases2 = np.array(biases2)[inv_pair_permutation]

    bias1_matrix = sparse.coo_matrix(
        (
            biases1,
            (
                np.array(tokenizer_teacher.convert_tokens_to_ids(x[0] for x in pairs)),
                np.array(target_tokenizer.convert_tokens_to_ids(x[1] for x in pairs)),
            ),
        ),
        shape=(len(tokenizer_teacher), len(target_tokenizer)),
    )
    bias2_matrix = sparse.coo_matrix(
        (
            biases2,
            (
                np.array(tokenizer_teacher.convert_tokens_to_ids(x[0] for x in pairs)),
                np.array(target_tokenizer.convert_tokens_to_ids(x[1] for x in pairs)),
            ),
        ),
        shape=(len(tokenizer_teacher), len(target_tokenizer)),
    )

    sparse.save_npz(output_dir / "bias1_matrix.npz", bias1_matrix)
    sparse.save_npz(output_dir / "bias2_matrix.npz", bias2_matrix)
