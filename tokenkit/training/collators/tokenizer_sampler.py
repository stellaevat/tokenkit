import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import tokenizers
from tokenizers import models, pre_tokenizers, decoders, normalizers, Tokenizer
import copy

from tokenkit import utils, constants, align
from tokenkit.byteify import ByteifyTokenizer
from tokenkit.utils import tqdm


class TokenizerSamplerCollator:
    def _get_sampler(self, initial_texts):
        texts = []
        max_length = constants.MAX_CHARS_PER_TOKEN * self.collator_args.block_size

        for text in initial_texts:
            if self.collator_args.sample_text_span:
                start = np.random.randint(0, max(len(text) - max_length, 0) + 1)
            else:
                start = 0

            end = start + max_length
            texts.append(text[start:end])

        samplers = []

        try:
            import rust_utils
        except ImportError:
            raise ImportError("rust_utils is required for `TokenizerSamplerCollator` but not installed. Please install following the instructions in https://github.com/bminixhofer/tokenkit/blob/main/README.md#Installation.")

        for _ in range(self.collator_args.n_pools):
            sampler = rust_utils.TokenizerSampler()

            for start in tqdm(range(0, len(texts), self.batch_size), desc="Populating sampler"):
                end = start + self.batch_size

                sampler.sample_tokenizer(
                    {text: 1 for text in texts[start:end]},
                    30_000,
                    16,
                    4,
                    0.0,
                    False,
                )

            samplers.append(sampler)

        return samplers

    def __init__(
        self,
        hn_tokenizer,
        collator_args,
        batch_size=None,
        tokenizer_name=None,
        initial_texts=None,
        inner_collator=None,
        is_validation=False,
        with_consistent_whitespace=True,
        with_alignments=False,
        original_tokenizer=None,
        space_mask_mode="space+tab+newline+special",
    ):
        self.tokenizer_name = tokenizer_name
        self.hn_tokenizer = hn_tokenizer
        self.collator_args = collator_args
        self.batch_size = batch_size
        self.inner_collator = inner_collator
        self.is_validation = is_validation
        self.with_consistent_whitespace = with_consistent_whitespace
        self.with_alignments = with_alignments
        self.original_tokenizer = original_tokenizer
        self.space_mask_mode = space_mask_mode

        if self.with_consistent_whitespace:
            raise NotImplementedError("TokenizerSamplerCollator does not support `with_consistent_whitespace` at the moment.")

        assert (tokenizer_name is None) == self.collator_args.do_tokenizer_sampling

        if not collator_args.do_tokenizer_sampling:
            raise NotImplementedError()
        else:
            self.inv_ids_to_embed = (
                np.zeros(self.collator_args.tokenizer_sample_max + 256, dtype=np.int32)
                if self.collator_args.n_token_subsample is not None
                else None
            )

        if initial_texts is not None:
            if isinstance(initial_texts, list):
                self.samplers = self._get_sampler(initial_texts)
            else:
                for lang in initial_texts.keys():
                    self.samplers[lang] = self._get_sampler(initial_texts[lang])

    def encode(
        self,
        tokenizer,
        texts,
        target_surface_form_matrix_to_use,
        target_priors_to_use,
        space_mask,
        special_ids_map=None,
        metrics_data=None,
    ):
        assert len(target_priors_to_use) == len(target_surface_form_matrix_to_use)

        encodings = dict(
            tokenizer(
                texts,
                max_length=self.collator_args.block_size,
                truncation=True,
                padding="max_length",
                return_tensors="np",
                add_special_tokens=True,
            )
        )

        for key, value in (special_ids_map or {}).items():
            encodings["input_ids"][encodings["input_ids"] == key] = value

        if self.inner_collator is not None:
            updates = self.inner_collator(tokenizer, return_tensors="np")(
                encodings["input_ids"]
            )
            encodings.update(updates)
        else:
            if self.with_alignments:
                assert self.original_tokenizer is not None

                original_encodings = dict(
                    self.original_tokenizer(
                        texts,
                        max_length=self.collator_args.block_size,
                        truncation=True,
                        padding="max_length",
                        return_tensors="np",
                        add_special_tokens=True,
                    )
                )

                (
                    alignment_matrix_a,
                    alignment_matrix_b,
                ) = align.get_unconstrained_alignments(
                    original_encodings["input_ids"],
                    encodings["input_ids"],
                    attention_mask_teacher=original_encodings["attention_mask"],
                    attention_mask_student=encodings["attention_mask"],
                    tokenizer_teacher=self.hn_tokenizer,
                    tokenizer_student=tokenizer,
                )

                (
                    alignment_matrix_a_space,
                    alignment_matrix_b_space,
                ) = align.get_space_alignments(
                    original_encodings["input_ids"],
                    encodings["input_ids"],
                    attention_mask_teacher=original_encodings["attention_mask"],
                    attention_mask_student=encodings["attention_mask"],
                    tokenizer_teacher=self.hn_tokenizer,
                    tokenizer_student=tokenizer,
                )

                encodings["alignment_matrix_a"] = alignment_matrix_a
                encodings["alignment_matrix_b"] = alignment_matrix_b
                encodings["alignment_matrix_a_space"] = alignment_matrix_a_space
                encodings["alignment_matrix_b_space"] = alignment_matrix_b_space
                encodings["input_ids_original"] = original_encodings["input_ids"]
                encodings["attention_mask_original"] = original_encodings["attention_mask"]
                encodings["loss_mask_original"] = original_encodings["attention_mask"].astype(bool)
                encodings["labels_original"] = original_encodings["input_ids"].copy()

        input_ids = encodings["input_ids"]

        positive_indices, positive_counts = np.unique(input_ids, return_counts=True)

        if metrics_data is not None:
            byte_lengths_per_token = metrics_data[0]

            non_special_tokens_mask = np.isin(
                input_ids, tokenizer.all_special_ids, invert=True
            )
            byte_lengths = byte_lengths_per_token[input_ids]

            encodings["metrics"] = {
                "avg_byte_length": byte_lengths[non_special_tokens_mask].mean(),
                # "unk_ratio": (input_ids == tokenizer.tokenizer.unk_token_id).mean(), TODO: what to do with this?
            }
            encodings["byte_lengths"] = byte_lengths

        if self.collator_args.n_token_subsample is not None:
            assert (
                self.collator_args.n_token_subsample % self.collator_args.pad_to_multiple_of
                == 0
            )

            tokens_in_batch = np.concatenate(
                [
                    np.array(tokenizer.all_special_ids),
                    np.setdiff1d(
                        np.unique(np.concatenate([input_ids, encodings["labels"]])),
                        np.array(tokenizer.all_special_ids),
                    ),
                ]
            )
            assert len(tokens_in_batch) <= self.collator_args.n_token_subsample

            if self.collator_args.subsample_mode == "positives_only":
                negatives_to_embed = np.zeros(
                    self.collator_args.n_token_subsample - len(tokens_in_batch),
                    dtype=np.int32,
                )
            elif self.collator_args.subsample_mode == "random":
                # random sampling makes sense because the tokenizer is already sampled according to unigram probabilities
                # so if we do unigram sampling again here we would have a sort of "squared sampling"
                negatives_to_embed = np.setdiff1d(
                    np.arange(len(tokenizer)), positive_indices
                )
                assert len(
                    negatives_to_embed
                ) >= self.collator_args.n_token_subsample - len(tokens_in_batch)
                np.random.shuffle(negatives_to_embed)
                negatives_to_embed = negatives_to_embed[
                    : self.collator_args.n_token_subsample - len(tokens_in_batch)
                ]
            elif self.collator_args.subsample_mode == "highest_scores":
                raise NotImplementedError()

            ids_to_embed = np.concatenate([tokens_in_batch, negatives_to_embed])
            ids_to_embed_list = list(ids_to_embed)

            # try to preserve special token indices
            # because e.g. the model might have a hardcoded padding id for which the embedding is ignored
            # we can't always preserve it because special tokens may be at the end of the vocabulary (e.g. GPT2 <endoftext> token)
            for special_token in sorted(tokenizer.all_special_ids):
                del ids_to_embed_list[ids_to_embed_list.index(special_token)]

                ids_to_embed_list.insert(special_token, special_token)

            ids_to_embed = np.array(ids_to_embed_list)

            self.inv_ids_to_embed[ids_to_embed] = np.arange(len(ids_to_embed))
            encodings["input_ids"] = self.inv_ids_to_embed[encodings["input_ids"]]

            active_labels = encodings["labels"] != -100
            encodings["labels"] = np.where(
                active_labels, self.inv_ids_to_embed[encodings["labels"]], -100
            )

            encodings["target_priors"] = target_priors_to_use[ids_to_embed]
            encodings["target_surface_forms"] = target_surface_form_matrix_to_use[
                ids_to_embed
            ]
            encodings["mask"] = np.ones(len(ids_to_embed), dtype=bool)
            encodings["ids_to_embed"] = ids_to_embed
            encodings["space_mask"] = space_mask[ids_to_embed]

            assert tokenizer.all_special_tokens == self.hn_tokenizer.all_special_tokens
            encodings["special_indices"] = np.array(
                [ids_to_embed_list.index(x) for x in tokenizer.all_special_ids]
            )
            encodings["special_indices_in_reference"] = np.array(
                [
                    self.hn_tokenizer.convert_tokens_to_ids(token)
                    for token in tokenizer.all_special_tokens
                ]
            )
        else:
            length = len(target_priors_to_use)
            if self.collator_args.do_tokenizer_sampling:
                # need consistent size
                # + pad_to_multiple_of to take into account potential special tokens
                assert (
                    self.collator_args.tokenizer_sample_max
                    % self.collator_args.pad_to_multiple_of
                    == 0
                )
                n_pad = (
                    self.collator_args.tokenizer_sample_max
                    + self.collator_args.pad_to_multiple_of
                    - length
                )
            elif length % self.collator_args.pad_to_multiple_of != 0:
                n_pad = self.collator_args.pad_to_multiple_of - (
                    length % self.collator_args.pad_to_multiple_of
                )
            else:
                n_pad = 0

            target_priors_to_use = np.pad(
                target_priors_to_use,
                (0, n_pad),
                # NOTE: must not use jax in multithreaded code, leads to a deadlock, so use np here
                constant_values=utils.get_large_negative_number(target_priors_to_use.dtype, module=np),
            )
            target_surface_form_matrix_to_use = np.pad(
                target_surface_form_matrix_to_use,
                ((0, n_pad), (0, 0)),
                constant_values=0,
            )

            encodings["target_priors"] = target_priors_to_use
            encodings["target_surface_forms"] = target_surface_form_matrix_to_use
            encodings["mask"] = np.concatenate(
                [
                    np.ones(length, dtype=bool),
                    np.zeros(n_pad, dtype=bool),
                ]
            )
            encodings["ids_to_embed"] = np.concatenate(
                [
                    np.arange(length),
                    np.zeros(n_pad, dtype=np.int32),
                ]
            )
            encodings["space_mask"] = np.concatenate(
                [
                    space_mask,
                    np.zeros(n_pad, dtype=bool),
                ]
            )
            assert tokenizer.all_special_tokens == self.hn_tokenizer.all_special_tokens
            encodings["special_indices"] = np.array(tokenizer.all_special_ids)
            encodings["special_indices_in_reference"] = np.array(
                [
                    self.hn_tokenizer.convert_tokens_to_ids(token)
                    for token in tokenizer.all_special_tokens
                ]
            )

        # match naming convention
        encodings["input_ids_new"] = encodings.pop("input_ids")
        encodings["attention_mask_new"] = encodings.pop("attention_mask")
        encodings["loss_mask_new"] = encodings["attention_mask_new"].astype(bool)
        if "token_type_ids" in encodings:
            encodings["token_type_ids_new"] = encodings.pop("token_type_ids")

        return encodings

    def sample_tokenizer(self, texts, sampler):
        n_total = int(
            np.random.normal(
                self.collator_args.tokenizer_sample_mean,
                self.collator_args.tokenizer_sample_std,
            )
        )
        n_total = max(self.collator_args.tokenizer_sample_min, n_total)
        n_total = min(self.collator_args.tokenizer_sample_max, n_total)

        pretoken_counts = {}
        for text in texts:
            pretoken_counts[text] = 1

        if self.collator_args.tokenizer_noise_mean > 0:
            noise_std = np.random.lognormal(
                mean=np.log(self.collator_args.tokenizer_noise_mean),
                sigma=self.collator_args.tokenizer_noise_std,
            )
        else:
            noise_std = 0

        pieces, scores = zip(
            *sampler.sample_tokenizer(
                pretoken_counts, n_total, 16, 4, noise_std, True, not self.is_validation
            )
        )
        pieces = list(pieces)
        scores = list(scores)

        piece_set = set(pieces)

        unknown_chars = set(constants.CHARS_TO_BYTES.keys()) - piece_set
        min_score = min(scores)
        pieces = sorted(unknown_chars) + pieces
        scores = [min_score] * len(unknown_chars) + scores

        special_tokens_to_remove = set(self.hn_tokenizer.all_special_tokens).intersection(
            piece_set
        )
        for token in special_tokens_to_remove:
            idx = pieces.index(token)
            pieces.remove(token)
            scores.pop(idx)

        special_ids_map = {}

        for i in np.argsort(self.hn_tokenizer.all_special_ids):
            pieces.insert(
                self.hn_tokenizer.all_special_ids[i], self.hn_tokenizer.all_special_tokens[i]
            )
            scores.insert(self.hn_tokenizer.all_special_ids[i], 0.0)

            if (
                pieces.index(self.hn_tokenizer.all_special_tokens[i])
                != self.hn_tokenizer.all_special_ids[i]
            ):
                special_ids_map[self.hn_tokenizer.all_special_ids[i]] = pieces.index(
                    self.hn_tokenizer.all_special_tokens[i]
                )

        scores = np.array(scores, dtype=np.float32)

        tokenizer = Tokenizer(models.Unigram([(piece, score) for piece, score in zip(pieces, scores)]))

        if self.collator_args.add_prefix_space:
            tokenizer.normalizer = normalizers.Prepend(" ")

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(tokenizers.Regex(constants.DEFAULT_SPLIT_REGEX), "removed", invert=True),
            pre_tokenizers.ByteLevel(False, False),
        ])
        tokenizer.decoder = decoders.ByteLevel()

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, clean_up_tokenization_spaces=False
        )

        if self.hn_tokenizer.tokenizer.backend_tokenizer.post_processor is not None:
            tokenizer.backend_tokenizer.post_processor = (
                self.hn_tokenizer.tokenizer.backend_tokenizer.post_processor
            )

        # TODO: fix, esp for BERT-style models where we need [CLS], [MASK], etc.
        # tokenizer.eos_token = self.hn_tokenizer.eos_token
        # tokenizer.pad_token = self.hn_tokenizer.pad_token
        # tokenizer.sep_token = self.hn_tokenizer.sep_token
        # tokenizer.unk_token = self.hn_tokenizer.unk_token
        # tokenizer.bos_token = self.hn_tokenizer.bos_token
        # tokenizer.cls_token = self.hn_tokenizer.cls_token
        # tokenizer.mask_token = self.hn_tokenizer.mask_token
        # tokenizer.unk_token = self.hn_tokenizer.unk_token

        model_kind_cls = copy.deepcopy(self.hn_tokenizer.model_kind_cls)
        model_kind_cls.byte_fallback_fn = lambda x: x
        model_kind_cls.byte_fallback_precedence_fn = lambda x: 0
        tokenizer = ByteifyTokenizer(tokenizer, model_kind_cls)

        tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))
        byte_lengths = np.array([len(token) for token in tokens])

        if self.hn_tokenizer is not None:
            target_surface_form_matrix_to_use = utils.get_surface_form_matrix(
                tokens,
                self.collator_args.hn_surface_maxlen,
                self.hn_tokenizer,
                verbose=False,
            )[0]
        else:
            target_surface_form_matrix_to_use = None
        target_priors_to_use = scores

        space_mask = utils.get_space_mask(tokenizer, self.space_mask_mode)

        return (
            tokenizer,
            special_ids_map,
            target_surface_form_matrix_to_use,
            target_priors_to_use,
            byte_lengths,
            space_mask,
        )

    def __call__(self, data, for_identity_step=False):
        # data is passed as a batch of size 1
        data = data[0]

        if for_identity_step:
            # choose random uniform
            indices = np.random.choice(
                self.original_length,
                size=self.collator_args.n_token_subsample,
                replace=False,
            )
            target_surface_form_matrix_to_use = self.surface_forms[indices]

            return {
                "target_surface_forms": target_surface_form_matrix_to_use,
                "target_priors": np.zeros(len(indices), dtype=np.float32),
                "ids_to_embed": indices,
                "lang_code": None,
                "lang_index": np.array(0),
            }

        texts = data["text"]
        lang_codes = data["lang_code"]
        if len(set(lang_codes)) > 1:
            lang_code = None
        else:
            lang_code = lang_codes[0]

        max_length = constants.MAX_CHARS_PER_TOKEN * self.collator_args.block_size

        for i in range(len(texts)):
            if self.collator_args.sample_text_span:
                start = np.random.randint(0, max(len(texts[i]) - max_length, 0) + 1)
            else:
                start = 0

            end = start + max_length
            texts[i] = texts[i][start:end]

        if self.collator_args.do_tokenizer_sampling:  # <-> one tokenizer for each language
            if isinstance(self.samplers, dict):
                if lang_code is None:
                    raise ValueError("Language code must be provided for language-specific sampling")
                samplers = self.samplers[lang_code]
            else:
                samplers = self.samplers

            sampler_idx = np.random.randint(0, len(samplers))
            sampler = samplers[sampler_idx]

            (
                tokenizer,
                special_ids_map,
                target_surface_form_matrix_to_use,
                target_priors_to_use,
                byte_lengths,
                space_mask,
            ) = self.sample_tokenizer(texts, sampler)
        else:
            tokenizer = self.tokenizer
            special_ids_map = {}

            target_surface_form_matrix_to_use = self.surface_forms
            target_priors_to_use = self.scores
            byte_lengths = self.byte_lengths

        encodings = self.encode(
            tokenizer,
            texts,
            target_surface_form_matrix_to_use,
            target_priors_to_use,
            space_mask,
            metrics_data=(byte_lengths,),
            special_ids_map=special_ids_map,
        )

        encodings["lang_code"] = lang_code
        encodings["lang_index"] = np.array(0) # TODO: fix

        return encodings