from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from tokenkit.constants import BYTES_TO_CHARS, CHARS_TO_BYTES

BYTE_FALLBACK_MAP = {f"<0x{num:02X}>": num for num in range(256)}
INV_BYTE_FALLBACK_MAP = {v: k for k, v in BYTE_FALLBACK_MAP.items()}


def sentencepiece_byte_fallback_byte_fn(token: str) -> str:
    if token in BYTE_FALLBACK_MAP:
        return BYTES_TO_CHARS[BYTE_FALLBACK_MAP[token]]
    else:
        return "".join(
            BYTES_TO_CHARS[b] for b in token.replace("▁", " ").encode("utf-8")
        )


def sentencepiece_byte_fallback_precedence_fn(token: str) -> int:
    if token in BYTE_FALLBACK_MAP:
        return 0
    else:
        return 1


def identity_byte_fn(token: str) -> str:
    return token


class BaseModelKind(ABC):
    SPECIAL_KEYS = [
        "<|<bos>|>",
        "<|<pad>|>",
        "<|<start_header>|>",
        "<|<end_header>|>",
        "<|<eot>|>",
        "<|<user_name>|>",
        "<|<assistant_name>|>",
        "<|<system_name>|>",
    ]

    def __init__(self):
        self._byte_fallback_fn = identity_byte_fn
        self._byte_fallback_precedence_fn = lambda x: 0

    @property
    @abstractmethod
    def special_tokens(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        pass

    @property
    def byte_fallback_fn(self) -> Callable[[str], str]:
        return self._byte_fallback_fn

    @byte_fallback_fn.setter
    def byte_fallback_fn(self, value: Callable[[str], str]):
        self._byte_fallback_fn = value

    @property
    def byte_fallback_precedence_fn(self) -> Callable[[str], int]:
        return self._byte_fallback_precedence_fn

    @byte_fallback_precedence_fn.setter
    def byte_fallback_precedence_fn(self, value: Callable[[str], int]):
        self._byte_fallback_precedence_fn = value


class Qwen2ModelKind(BaseModelKind):
    @property
    def special_tokens(self) -> List[str]:
        return ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": None,
            "<|<pad>|>": ["<|endoftext|>"],
            "<|<start_header>|>": ["<|im_start|>"],
            "<|<end_header>|>": ["Ċ"],
            "<|<eot>|>": ["<|im_end|>", "Ċ"],
            "<|<eos>|>": ["<|endoftext|>"],
            "<|<user_name>|>": ["user"],
            "<|<assistant_name>|>": ["assistant"],
            "<|<system_name>|>": ["system"],
        }


class Llama3ModelKind(BaseModelKind):
    @property
    def special_tokens(self) -> List[str]:
        return [
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|end_of_text|>",
        ]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": ["<|begin_of_text|>"],
            "<|<pad>|>": ["<|end_of_text|>"],
            "<|<start_header>|>": ["<|start_header_id|>"],
            "<|<end_header>|>": ["<|end_header_id|>", "ĊĊ"],
            "<|<eot>|>": ["<|eot_id|>"],
            "<|<eos>|>": ["<|eot_id|>"],
            "<|<user_name>|>": ["user"],
            "<|<assistant_name>|>": ["assistant"],
            "<|<system_name>|>": ["system"],
        }


class Gemma2ModelKind(BaseModelKind):
    def __init__(self):
        super().__init__()
        self._byte_fallback_fn = sentencepiece_byte_fallback_byte_fn
        self._byte_fallback_precedence_fn = sentencepiece_byte_fallback_precedence_fn

    @property
    def special_tokens(self) -> List[str]:
        return ["<bos>", "<start_of_turn>", "<end_of_turn>", "<eos>", "<pad>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": ["<bos>"],
            "<|<pad>|>": ["<pad>"],
            "<|<start_header>|>": ["<start_of_turn>"],
            "<|<end_header>|>": ["Ċ"],
            "<|<eot>|>": ["<end_of_turn>", "Ċ"],
            "<|<eos>|>": ["<eos>"],
            "<|<user_name>|>": ["user"],
            "<|<assistant_name>|>": ["model"],
            "<|<system_name>|>": ["user"],
        }


class Phi3ModelKind(BaseModelKind):
    @property
    def special_tokens(self) -> List[str]:
        return ["<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": None,
            "<|<pad>|>": ["<|endoftext|>"],
            "<|<start_header>|>": None,
            "<|<end_header>|>": ["Ċ"],
            "<|<eot>|>": ["<|end|>", "Ċ"],
            "<|<eos>|>": ["<|endoftext|>"],
            "<|<user_name>|>": ["<|user|>"],
            "<|<assistant_name>|>": ["<|assistant|>"],
        }


class GPT2ModelKind(BaseModelKind):
    @property
    def special_tokens(self) -> List[str]:
        return ["<|endoftext|>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": None,
            "<|<pad>|>": ["<|endoftext|>"],
            "<|<start_header>|>": None,
            "<|<end_header>|>": None,
            "<|<eot>|>": ["<|endoftext|>"],
            "<|<eos>|>": ["<|endoftext|>"],
            "<|<user_name>|>": None,
            "<|<assistant_name>|>": None,
        }


# Model kind registry
def get_model_kind_cls(model_kind: str) -> BaseModelKind:
    return {
        "Qwen2": Qwen2ModelKind(),
        "Llama3": Llama3ModelKind(),
        "Gemma2": Gemma2ModelKind(),
        "Phi3": Phi3ModelKind(),
        "GPT2": GPT2ModelKind(),
    }[model_kind]
