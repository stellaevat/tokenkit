import numpy as np

from tokenkit.byteify import ByteifyTokenizer, load_byteify_tokenizer


def get_alignment_indices(
    tokens_teacher: list[str],
    tokens_student: list[str],
    attention_mask_teacher: np.ndarray,
    attention_mask_student: np.ndarray,
    tokenizer_teacher: ByteifyTokenizer,
    tokenizer_student: ByteifyTokenizer,
):
    replacements_teacher = {}
    for k, v in tokenizer_teacher.model_kind_cls.replacements.items():
        if v is None:
            continue

        if tokenizer_student.model_kind_cls.replacements[k] is not None:
            replacements_teacher[tuple(v)] = [
                "S" * len(tokenizer_student.model_kind_cls.replacements[k])
            ] * len(v)
        else:
            replacements_teacher[tuple(v)] = ""

    replacements_student = {}
    for k, v in tokenizer_student.model_kind_cls.replacements.items():
        if v is None:
            continue

        if tokenizer_teacher.model_kind_cls.replacements[k] is not None:
            replacements_student[tuple(v)] = [
                "S" * len(tokenizer_teacher.model_kind_cls.replacements[k])
            ] * len(v)
        else:
            replacements_student[tuple(v)] = ""

    def get_replacement(tokens, index, replacements):
        for k, v in replacements.items():
            if tuple(tokens[index : index + len(k)]) == k:
                return k, v

        return None

    normalized_tokens_teacher = []
    normalized_tokens_student = []
    special_tokens_mask_teacher = []
    special_tokens_mask_student = []

    start_i = 0
    start_j = 0
    i = 0
    j = 0

    cum_length_teacher = 0
    cum_length_student = 0
    cum_lengths_teacher_dict = {}
    cum_lengths_student_dict = {}

    alignment_indices = []

    while i < len(tokens_teacher) or j < len(tokens_student):
        if i < len(tokens_teacher) and not attention_mask_teacher[i]:
            i += 1
            continue
        if j < len(tokens_student) and not attention_mask_student[j]:
            j += 1
            continue

        r_teacher = get_replacement(tokens_teacher, i, replacements_teacher)
        r_student = get_replacement(tokens_student, j, replacements_student)

        skipped_align = False

        if r_teacher is not None and r_teacher[1] == "":
            normalized_tokens_teacher.append("")
            special_tokens_mask_teacher.append(True)
            i += 1
            skipped_align = True
        if r_student is not None and r_student[1] == "":
            normalized_tokens_student.append("")
            special_tokens_mask_student.append(True)
            j += 1
            skipped_align = True
        if r_teacher is not None and r_student is not None and not skipped_align:
            normalized_tokens_teacher.extend(r_teacher[1])
            normalized_tokens_student.extend(r_student[1])
            special_tokens_mask_teacher.extend([True] * len(r_teacher[1]))
            special_tokens_mask_student.extend([True] * len(r_student[1]))
            i += len(r_teacher[0])
            j += len(r_student[0])
            skipped_align = True
            alignment_indices.append((start_i, i, start_j, j))
            start_i = i
            start_j = j
        if not skipped_align:
            cum_length_teacher = cum_lengths_teacher_dict.get(i - 1, 0) + len(
                tokens_teacher[i]
            )
            cum_lengths_teacher_dict[i] = cum_length_teacher
            cum_length_student = cum_lengths_student_dict.get(j - 1, 0) + len(
                tokens_student[j]
            )
            cum_lengths_student_dict[j] = cum_length_student

            print(tokens_teacher[i], tokens_student[j])
            print(cum_length_teacher, cum_length_student)

            if cum_length_teacher == cum_length_student:
                normalized_tokens_teacher.append(tokens_teacher[i])
                normalized_tokens_student.append(tokens_student[j])
                special_tokens_mask_teacher.append(False)
                special_tokens_mask_student.append(False)
                i += 1
                j += 1
                alignment_indices.append((start_i, i, start_j, j))
                start_i = i
                start_j = j
            elif cum_length_teacher < cum_length_student:
                normalized_tokens_teacher.append(tokens_teacher[i])
                special_tokens_mask_teacher.append(False)
                cum_length_teacher += len(tokens_teacher[i])
                i += 1
            elif cum_length_teacher > cum_length_student:
                normalized_tokens_student.append(tokens_student[j])
                special_tokens_mask_student.append(False)
                cum_length_student += len(tokens_student[j])
                j += 1

    return (
        alignment_indices,
        normalized_tokens_teacher,
        normalized_tokens_student,
        special_tokens_mask_teacher,
        special_tokens_mask_student,
    )


def get_unconstrained_alignments(
    input_ids_teacher,
    input_ids_student,
    attention_mask_teacher,
    attention_mask_student,
    tokenizer_teacher,
    tokenizer_student,
):
    if attention_mask_teacher is not None:
        attention_mask_teacher = attention_mask_teacher.astype(bool)
    if attention_mask_student is not None:
        attention_mask_student = attention_mask_student.astype(bool)

    batch_size = input_ids_teacher.shape[0]
    shared_length = min(input_ids_teacher.shape[1], input_ids_student.shape[1])
    alignment_matrix_teacher = np.zeros(
        (batch_size, input_ids_teacher.shape[1], shared_length), dtype=bool
    )
    alignment_matrix_student = np.zeros(
        (batch_size, input_ids_student.shape[1], shared_length), dtype=bool
    )

    for example_index in range(len(input_ids_teacher)):
        tokens_teacher = tokenizer_teacher.convert_ids_to_tokens(
            input_ids_teacher[example_index]
        )
        tokens_student = tokenizer_student.convert_ids_to_tokens(
            input_ids_student[example_index]
        )

        (
            alignment_indices,
            normalized_tokens_teacher,
            normalized_tokens_student,
            _,
            _,
        ) = get_alignment_indices(
            tokens_teacher,
            tokens_student,
            attention_mask_teacher[example_index],
            attention_mask_student[example_index],
            tokenizer_teacher,
            tokenizer_student,
        )
        teacher_mask = np.array([len(token) > 0 for token in normalized_tokens_teacher])
        student_mask = np.array([len(token) > 0 for token in normalized_tokens_student])

        chunk_idx = 0
        for start_i, end_i, start_j, end_j in alignment_indices:
            alignment_matrix_teacher[example_index, start_i:end_i, chunk_idx] = True
            alignment_matrix_student[example_index, start_j:end_j, chunk_idx] = True
            chunk_idx += 1

        alignment_matrix_teacher[example_index, ~teacher_mask, :] = False
        alignment_matrix_student[example_index, ~student_mask, :] = False

    return alignment_matrix_teacher, alignment_matrix_student


def get_space_alignments(
    input_ids_teacher,
    input_ids_student,
    attention_mask_teacher,
    attention_mask_student,
    tokenizer_teacher,
    tokenizer_student,
):
    if attention_mask_teacher is not None:
        attention_mask_teacher = attention_mask_teacher.astype(bool)
    if attention_mask_student is not None:
        attention_mask_student = attention_mask_student.astype(bool)

    batch_size = input_ids_teacher.shape[0]
    shared_length = min(input_ids_teacher.shape[1], input_ids_student.shape[1])
    alignment_matrix_teacher = np.zeros(
        (batch_size, input_ids_teacher.shape[1], shared_length), dtype=bool
    )
    alignment_matrix_student = np.zeros(
        (batch_size, input_ids_student.shape[1], shared_length), dtype=bool
    )

    for example_index in range(len(input_ids_teacher)):
        tokens_teacher = tokenizer_teacher.convert_ids_to_tokens(
            input_ids_teacher[example_index]
        )
        tokens_student = tokenizer_student.convert_ids_to_tokens(
            input_ids_student[example_index]
        )

        (
            alignment_indices,
            normalized_tokens_teacher,
            normalized_tokens_student,
            special_tokens_mask_teacher,
            special_tokens_mask_student,
        ) = get_alignment_indices(
            tokens_teacher,
            tokens_student,
            attention_mask_teacher[example_index],
            attention_mask_student[example_index],
            tokenizer_teacher,
            tokenizer_student,
        )
        teacher_mask = np.array([len(token) > 0 for token in normalized_tokens_teacher])
        student_mask = np.array([len(token) > 0 for token in normalized_tokens_student])

        teacher_starts_with_space = np.array(
            [len(token) > 0 and token[0] == "Ġ" for token in normalized_tokens_teacher]
        )
        student_starts_with_space = np.array(
            [len(token) > 0 and token[0] == "Ġ" for token in normalized_tokens_student]
        )

        chunk_idx = 0
        for start_i, end_i, start_j, end_j in alignment_indices:
            alignment_matrix_teacher[example_index, start_i:end_i, chunk_idx] = True
            alignment_matrix_student[example_index, start_j:end_j, chunk_idx] = True

            if (
                end_i < len(normalized_tokens_teacher)
                and teacher_starts_with_space[end_i]
                and end_j < len(normalized_tokens_student)
                and student_starts_with_space[end_j]
            ) or (
                special_tokens_mask_teacher[end_i - 1]
                or special_tokens_mask_student[end_j - 1]
            ):
                assert (
                    teacher_starts_with_space[end_i] == student_starts_with_space[end_j]
                )
                assert (
                    special_tokens_mask_student[end_j - 1]
                    == special_tokens_mask_teacher[end_i - 1]
                )
                chunk_idx += 1

        alignment_matrix_teacher[example_index, ~teacher_mask, :] = False
        alignment_matrix_student[example_index, ~student_mask, :] = False

    return alignment_matrix_teacher, alignment_matrix_student


def get_unbiased_alignments(
    input_ids_teacher,
    input_ids_student,
    attention_mask_teacher,
    attention_mask_student,
    tokenizer_teacher,
    tokenizer_student,
    pair_data,
    bias_threshold,
):
    (bias1_matrix, bias2_matrix, _, _) = pair_data

    if attention_mask_teacher is not None:
        attention_mask_teacher = attention_mask_teacher.astype(bool)
    if attention_mask_student is not None:
        attention_mask_student = attention_mask_student.astype(bool)

    batch_size = input_ids_teacher.shape[0]
    shared_length = min(input_ids_teacher.shape[1], input_ids_student.shape[1])
    alignment_matrix_teacher = np.zeros(
        (batch_size, input_ids_teacher.shape[1], shared_length), dtype=bool
    )
    alignment_matrix_student = np.zeros(
        (batch_size, input_ids_student.shape[1], shared_length), dtype=bool
    )

    teacher_length, student_length = bias1_matrix.shape

    def is_unbiased(original_token_id, new_token_id):
        return (
            original_token_id >= teacher_length or new_token_id >= student_length
        ) or (
            bias1_matrix[original_token_id, new_token_id] <= bias_threshold
            and bias2_matrix[original_token_id, new_token_id] <= bias_threshold
        )

    for example_index in range(len(input_ids_teacher)):
        tokens_teacher = tokenizer_teacher.convert_ids_to_tokens(
            input_ids_teacher[example_index]
        )
        tokens_student = tokenizer_student.convert_ids_to_tokens(
            input_ids_student[example_index]
        )

        (
            alignment_indices,
            normalized_tokens_teacher,
            normalized_tokens_student,
            special_tokens_mask_teacher,
            special_tokens_mask_student,
        ) = get_alignment_indices(
            tokens_teacher,
            tokens_student,
            attention_mask_teacher[example_index],
            attention_mask_student[example_index],
            tokenizer_teacher,
            tokenizer_student,
        )
        teacher_mask = np.array([len(token) > 0 for token in normalized_tokens_teacher])
        student_mask = np.array([len(token) > 0 for token in normalized_tokens_student])

        chunk_idx = 0
        for start_i, end_i, start_j, end_j in alignment_indices:
            alignment_matrix_teacher[example_index, start_i:end_i, chunk_idx] = True
            alignment_matrix_student[example_index, start_j:end_j, chunk_idx] = True

            if (
                special_tokens_mask_teacher[end_i - 1]
                or special_tokens_mask_student[end_j - 1]
                and is_unbiased(
                    input_ids_teacher[example_index][end_i - 1],
                    input_ids_student[example_index][end_j - 1],
                )
            ):
                chunk_idx += 1

        alignment_matrix_teacher[example_index, ~teacher_mask, :] = False
        alignment_matrix_student[example_index, ~student_mask, :] = False

    return alignment_matrix_teacher, alignment_matrix_student


def test_get_alignment_indices():
    teacher_tokenizer = load_byteify_tokenizer("Qwen/Qwen2.5-1.5B:source=Qwen2")
    student_tokenizer = load_byteify_tokenizer(
        "meta-llama/Meta-Llama-3-8B-Instruct:source=Llama3"
    )

    tokens_teacher = ["<|im_start|>", "Hel", "lo", "?", "Ċ", "Ċ"]
    attention_mask_teacher = np.ones(len(tokens_teacher), dtype=bool)
    tokens_student = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "Hello",
        "?",
        "<|end_header_id|>",
        "ĊĊ",
        "Ċ",
    ]
    attention_mask_student = np.ones(len(tokens_student), dtype=bool)
    idx, _, _, _, _ = get_alignment_indices(
        tokens_teacher,
        tokens_student,
        attention_mask_teacher,
        attention_mask_student,
        teacher_tokenizer,
        student_tokenizer,
    )

    assert tokens_teacher[idx[0][0] : idx[0][1]] == ["<|im_start|>"]
    assert tokens_student[idx[0][2] : idx[0][3]] == [
        "<|begin_of_text|>",
        "<|start_header_id|>",
    ]

    assert tokens_teacher[idx[1][0] : idx[1][1]] == ["Hel", "lo"]
    assert tokens_student[idx[1][2] : idx[1][3]] == ["Hello"]

    assert tokens_teacher[idx[2][0] : idx[2][1]] == ["?"]
    assert tokens_student[idx[2][2] : idx[2][3]] == ["?"]

    assert tokens_teacher[idx[3][0] : idx[3][1]] == ["Ċ"]
    assert tokens_student[idx[3][2] : idx[3][3]] == ["<|end_header_id|>", "ĊĊ"]

    assert tokens_teacher[idx[4][0] : idx[4][1]] == ["Ċ"]
    assert tokens_student[idx[4][2] : idx[4][3]] == ["Ċ"]
