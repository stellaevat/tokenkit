# Byteification

`tokenkit` uses a unified byte-level interface to tokenizers to prevent issues stemming from tokenizers using different encoding schemes. For example, let's say we want to compute the number of overlapping tokenizers between the Gemma2 and Llama3 tokenizers. Here is the naive approach:

```python
from transformers import AutoTokenizer

tok1 = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
tok2 = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

n_overlap = len(set(tok1.get_vocab().keys()) & set(tok2.get_vocab().keys()))
# 25632 - this is suspiciously low!
```

The two tokenizers use a different encoding, so even if two tokens encode the same UTF-8 bytes, they might look different!

```python
tok1.tokenize(" Café") # ['▁Café']
tok2.tokenize(" Café") # ['ĠCafÃ©']
```

We can fix this by instead using the `tokenkit.byteify.ByteifyTokenizer` interface. 'Byteification' preserves the tokenizers functionality, while providing a unified encoding:

```python
from tokenkit.byteify import load_byteify_tokenizer

tok1 = load_byteify_tokenizer("google/gemma-2-2b-it:source=Gemma2")
tok2 = load_byteify_tokenizer("meta-llama/Llama-3.2-3B-Instruct:source=Llama3")

n_overlap = len(set(tok1.get_vocab().keys()) & set(tok2.get_vocab().keys()))
# 85699 - this is much more reasonable!

tok1.tokenize(" Café") # ['ĠCafÃ©']
tok2.tokenize(" Café") # ['ĠCafÃ©']
```

The API is not 100% compatible with HuggingFace's tokenizers, but most functionality matches (e.g., `convert_ids_to_tokens`, `convert_tokens_to_ids`, `get_vocab`, `tokenize`, `add_tokens`).

This allows us to compute things like lexical overlap and token sequence alignments accurately. `tokenkit` implements an exact alignment algorithm between tokenizers, including tokenizers with different special tokens (e.g., different chat templates).

```python
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit import align

tok1 = load_byteify_tokenizer("google/gemma-2-2b-it:source=Gemma2")
tok2 = load_byteify_tokenizer("meta-llama/Llama-3.2-3B-Instruct:source=Llama3")

# Gemma2 chat template
tokens1 = tok1.tokenize("<bos><start_of_turn>user\nWhat's ultracrepidarianism?<end_of_turn>\n")
# Llama3 chat template
tokens2 = tok2.tokenize("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat's ultracrepidarianism?<|eot_id|>")

alignment_indices = align.get_alignment_indices(tokens1, tokens2, tok1, tok2)[0]

for (start1, end1, start2, end2) in alignment_indices:
    print(tokens1[start1:end1], tokens2[start2:end2])

# ['<bos>'] ['<|begin_of_text|>']
# ['<start_of_turn>'] ['<|start_header_id|>']
# ['user'] ['user']
# ['Ċ'] ['<|end_header_id|>', 'ĊĊ']
# ['What'] ['What']
# ["'", 's'] ["'s"]
# ['Ġultra', 'cre', 'pid'] ['Ġultr', 'ac', 'repid']
# ['arian'] ['arian']
# ['ism'] ['ism']
# ['?'] ['?']
# ['<end_of_turn>', 'Ċ'] ['<|eot_id|>']
```

## Tokenizer Specs

TODO