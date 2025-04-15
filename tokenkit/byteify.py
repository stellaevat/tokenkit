import copy
import json
import logging
from tempfile import NamedTemporaryFile
from typing import Dict, List, Union

import tokenizers
import tokenizers.decoders
import tokenizers.normalizers
import tokenizers.pre_tokenizers
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tokenkit.constants import CHARS_TO_BYTES
from tokenkit.model_kinds import BaseModelKind, get_model_kind_cls

logger = logging.getLogger(__name__)


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

def _get_left_count(string, symbol):
    symbol_count = 0
    while symbol_count < len(string) and string[symbol_count] == symbol:
        symbol_count += 1

    return symbol_count

def _make_left_count_match_or_be_zero(string, symbol, count):
    symbol_count = _get_left_count(string, symbol)

    if symbol_count == 0 or symbol_count == count:
        return string

    elif symbol_count > count:
        return string[symbol_count - count:]
    else:
        return symbol * (count - symbol_count) + string

def _get_increasing_offset_pairs(n: int):
    """Generate pairs of (start, end) offsets that sum to strictly increasing values.
    
    For a string of length n, generates offset pairs where:
    - Each pair sums to a value from 0 to n
    - For each sum value, generates all valid (start, end) combinations
    - Pairs are ordered by sum, then by start index
    
    Example for n=2:
    (0,0), (0,1), (1,0), (0,2), (1,1), (2,0), (1,2), (2,1)
    """
    for total in range(n + 1):
        for start in range(total + 1):
            end = total - start
            if end <= n:
                yield (start, end)

def to_byte_level_tokenizer(
    tokenizer, model_kind_cls, tokens_to_keep=None, inplace=False
):
    if not inplace:
        byte_tokenizer = copy.deepcopy(tokenizer)
    else:
        byte_tokenizer = tokenizer

    if tokens_to_keep is None:
        tokens_to_keep = []

    byte_tokens_in_vocab = [
        token for token in tokenizer.get_vocab() if token in CHARS_TO_BYTES.keys()
    ]
    byte_tokens_not_in_vocab = [
        token for token in CHARS_TO_BYTES.keys() if token not in byte_tokens_in_vocab
    ]

    if len(byte_tokens_not_in_vocab) > 0:
        logger.warning(
            f"Some byte tokens not in vocab: {byte_tokens_not_in_vocab}. Adding these to the vocab. They will not have a good init."
        )

    tokens = list(CHARS_TO_BYTES.keys()) + [
        token for token in tokens_to_keep if token not in CHARS_TO_BYTES
    ]
    byte_vocab = {token: i for i, token in enumerate(tokens)}

    # use ByteLevel tokenizer to achieve byte tokenization
    byte_tokenizer.backend_tokenizer.normalizer = None
    byte_tokenizer.backend_tokenizer.pre_tokenizer = (
        tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    )
    byte_tokenizer.backend_tokenizer.model = tokenizers.models.Unigram(
        [(token, 0.0) for token in tokens],
    )
    byte_tokenizer.backend_tokenizer.decoder = tokenizers.decoders.ByteLevel()

    # remove added tokens, they would persist to the old vocabulary id
    f = NamedTemporaryFile()
    byte_tokenizer.backend_tokenizer.save(f.name)
    tokenizer_data = json.load(open(f.name, "r"))
    if "added_tokens" in tokenizer_data:
        del tokenizer_data["added_tokens"]
    if "post_processor" in tokenizer_data:
        fix_postprocessor_data(tokenizer_data["post_processor"], byte_vocab)

    json.dump(tokenizer_data, open(f.name, "w"))

    byte_tokenizer._tokenizer = Tokenizer.from_file(f.name)

    return byte_tokenizer


class ByteifyTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, model_kind_cls: BaseModelKind
    ):
        self.tokenizer = tokenizer
        self.model_kind_cls = model_kind_cls

        if any(
            isinstance(self.tokenizer.backend_tokenizer.normalizer, x)
            for x in [
                tokenizers.normalizers.NFC,
                tokenizers.normalizers.NFD,
                tokenizers.normalizers.NFKC,
                tokenizers.normalizers.NFKD,
            ]
        ):
            logger.warning(
                f"ByteifyTokenizer does not currently support normalizers since they could be different between the teacher and the student. Removing {self.tokenizer.backend_tokenizer.normalizer} normalizer. This could have adverse effects!"
            )
            self.tokenizer.backend_tokenizer.normalizer = None
        elif isinstance(
            self.tokenizer.backend_tokenizer.normalizer, tokenizers.normalizers.Sequence
        ) and any(x in repr(self.tokenizer.backend_tokenizer.normalizer) for x in ["NFC", "NFD", "NFKC", "NFKD"]):
            raise ValueError(
                "ByteifyTokenizer does not currently support sequence normalizers with constituents NFC, NFD, NFKC, NFKD. Please open an issue about this tokenizer / check existing issues."
            )

        for special_token, name in [
            ("pad_token", "<|<pad>|>"),
            ("bos_token", "<|<bos>|>"),
            ("eos_token", "<|<eos>|>"),
        ]:
            token_value = self.model_kind_cls.replacements[name]

            if token_value is not None:
                setattr(self.tokenizer, special_token, token_value[0])

        self.tokenizer.padding_side = "right"

        self.vocab = {}
        self.precedences = {
            v: self.model_kind_cls.byte_fallback_precedence_fn(k)
            for k, v in self.tokenizer.vocab.items()
        }
        self.inv_vocab = {}

        for k, v in self.tokenizer.vocab.items():
            byte_k = self.model_kind_cls.byte_fallback_fn(k)
            # prioritize overlapping byte tokens via precedences (necessary e.g. for SentencePiece byte fallback)
            if (
                byte_k not in self.vocab
                or self.precedences[v] > self.precedences[self.vocab[byte_k]]
            ):
                self.vocab[byte_k] = v

            self.inv_vocab[v] = byte_k

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.inv_vocab[ids]
        else:
            return [self.inv_vocab[int(id)] for id in ids]

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.vocab[tokens]
        else:
            return [self.vocab[token] for token in tokens]

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def __call__(self, *args, **kwargs):
        kwargs.pop("add_special_tokens", None)

        return self.tokenizer(*args, **kwargs, add_special_tokens=False)

    def __len__(self):
        return len(self.tokenizer)

    def add_tokens(self, tokens: List[str]):
        self.tokenizer.add_tokens(tokens)

    def save_pretrained(self, *args, **kwargs):
        self.tokenizer.save_pretrained(*args, **kwargs)

    @property
    def added_tokens_encoder(self):
        return self.tokenizer.added_tokens_encoder

    @property
    def all_special_tokens(self):
        return self.model_kind_cls.special_tokens

    @property
    def all_special_ids(self):
        return self.convert_tokens_to_ids(self.all_special_tokens)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    # TODO: return ids instead of tokens
    def backend_tokenize(self, pretoken: str, unsafe: bool | str = False) -> List[str]:
        if len(pretoken) == 0:
            return []

        is_unsafe = unsafe is True or (unsafe == "auto" and isinstance(self.tokenizer.backend_tokenizer.pre_tokenizer, tokenizers.pre_tokenizers.ByteLevel))

        if is_unsafe:
            return [x.value for x in self.tokenizer.backend_tokenizer.model.tokenize(pretoken)]

        # this is not ideal: needs the pretoken to be a decodable string
        # and needs hacks for handling prefix spaces correctly

        pretoken_bytes = bytes([CHARS_TO_BYTES[c] for c in pretoken])
        pretoken_string = pretoken_bytes.decode("utf-8")

        space_count = _get_left_count(pretoken_string, " ")

        if self.tokenizer.backend_tokenizer.normalizer is not None:
            pretoken_string = self.tokenizer.backend_tokenizer.normalizer.normalize_str(
                pretoken_string
            )
        
        pretoken_string = _make_left_count_match_or_be_zero(pretoken_string, "▁", space_count)
        pretoken_string = _make_left_count_match_or_be_zero(pretoken_string, " ", space_count)

        if self.tokenizer.backend_tokenizer.pre_tokenizer is not None:
                pretoken_string = "".join(
                    [
                        x[0]
                    for x in self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
                        pretoken_string
                    )
                ]
            )

        pretoken_string = _make_left_count_match_or_be_zero(pretoken_string, "Ġ", space_count)

        return self.convert_ids_to_tokens(
            [
                x.id
                for x in self.tokenizer.backend_tokenizer.model.tokenize(
                    pretoken_string
                )
            ]
        )

    def backend_tokenize_with_byte_fallback(self, pretoken: str, unsafe: bool | str = False) -> List[str]:
       tokens = None
       
       for end_offset, start_offset in _get_increasing_offset_pairs(len(pretoken)):
            start_idx = start_offset
            end_idx = len(pretoken) - end_offset

            try:
                tokens = self.convert_tokens_to_ids(self.backend_tokenize(pretoken[start_idx:end_idx], unsafe))
            except UnicodeDecodeError:
                continue

            if tokens is not None:
                prefix = [CHARS_TO_BYTES[x] for x in pretoken[:start_idx]]
                suffix = [CHARS_TO_BYTES[x] for x in pretoken[end_idx:]]
                return prefix + tokens + suffix

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def tokenize(self, *args, **kwargs):
        previous_tokens = self.tokenizer.tokenize(*args, **kwargs)
        return self.convert_ids_to_tokens(
            self.tokenizer.convert_tokens_to_ids(previous_tokens)
        )


def load_byteify_tokenizer(tokenizer_spec: str) -> ByteifyTokenizer:
    spec_parts = tokenizer_spec.split(":")

    tokenizer_name = spec_parts[0]
    kwargs = {}
    for kv in spec_parts[1:]:
        k, v = kv.split("=")
        kwargs[k] = v

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    source_model_kind_cls = get_model_kind_cls(kwargs["source"])
    target_model_kind_cls = get_model_kind_cls(kwargs.get("target", kwargs["source"]))

    tokenizer.add_tokens(target_model_kind_cls.special_tokens)

    tokens_used_in_template = set()

    for values in target_model_kind_cls.replacements.values():
        if values is not None:
            tokens_used_in_template.update(values)

    conversion = kwargs.get("conversion")

    if conversion == "byte":
        tokenizer = to_byte_level_tokenizer(
            tokenizer,
            target_model_kind_cls,
            tokens_to_keep=sorted(tokens_used_in_template),
        )
        target_model_kind_cls.byte_fallback_fn = lambda x: x
    elif conversion == "prebyteified":
        target_model_kind_cls.byte_fallback_fn = lambda x: x
    elif conversion is not None:
        raise ValueError(f"Invalid conversion: {conversion}")
    else:
        target_model_kind_cls.byte_fallback_fn = source_model_kind_cls.byte_fallback_fn

    byteify_tokenizer = ByteifyTokenizer(tokenizer, target_model_kind_cls)
    byteify_vocab = byteify_tokenizer.get_vocab()

    missing_template_tokens = tokens_used_in_template - set(byteify_vocab.keys())
    if len(missing_template_tokens) > 0:
        raise ValueError(
            f"Missing tokens used by tokenization template! {missing_template_tokens}"
        )

    return byteify_tokenizer


def test_byte_level_conversion():
    tok = load_byteify_tokenizer("google/gemma-2-2b:source=Gemma2:conversion=byte")

    assert tok.tokenize("<start_of_turn>Hello?") == [
        "<start_of_turn>",
        "H",
        "e",
        "l",
        "l",
        "o",
        "?",
    ]


def test_special_token_substitution_gemma():
    tok = load_byteify_tokenizer("google/gemma-2-2b:source=Gemma2:target=Qwen2")
    assert tok.tokenize("<|im_start|>Hello?") == ["<|im_start|>", "Hello", "?"]


def test_special_token_substitution_qwen():
    tok = load_byteify_tokenizer("Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2")
    assert tok.tokenize("<start_of_turn>Hello?") == ["<start_of_turn>", "Hello", "?"]
