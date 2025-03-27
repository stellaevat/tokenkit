import copy
import json
import logging
from tempfile import NamedTemporaryFile
from typing import Dict, List, Union

import tokenizers
import tokenizers.decoders
import tokenizers.pre_tokenizers
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tokenkit.constants import CHARS_TO_BYTES
from tokenkit.model_kinds import BaseModelKind, get_model_kind_cls
from tokenkit.utils import fix_postprocessor_data

logger = logging.getLogger(__name__)


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

    unk_token = model_kind_cls.replacements["<|<pad>|>"][0]

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
    byte_tokenizer.backend_tokenizer.model = tokenizers.models.WordPiece(
        byte_vocab,
        unk_token=unk_token,
        max_input_chars_per_word=1_000_000,  # effectively disable limit on input chars
    )
    byte_tokenizer.backend_tokenizer.decoder = tokenizers.decoders.ByteLevel()
    byte_tokenizer.unk_token = unk_token

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
    byte_tokenizer._tokenizer.model.continuing_subword_prefix = ""

    return byte_tokenizer


class ByteifyTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, model_kind_cls: BaseModelKind
    ):
        self.tokenizer = tokenizer
        self.model_kind_cls = model_kind_cls

        self.vocab = {
            self.model_kind_cls.byte_fallback_fn(k): v
            for k, v in self.tokenizer.vocab.items()
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.inv_vocab[ids]
        else:
            return [self.inv_vocab[id] for id in ids]

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

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
