# Byteification: A Unified Interface to Tokenizers

In this guide, we'll take a look at how `tokenkit` interacts with tokenizers: `tokenkit` uses a unified byte-level interface to tokenizers to prevent issues stemming from tokenizers using different encoding schemes. For example, let's say we want to compute the number of overlapping tokenizers between the Gemma2 and Llama3 tokenizers. Here is the naive approach:

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

We can fix this by instead using the `tokenkit.byteify.ByteifyTokenizer` interface. 'Byteification' preserves the tokenizers functionality, while providing a unified (byte-level) encoding:

```python
from tokenkit.byteify import load_byteify_tokenizer

tok1 = load_byteify_tokenizer("google/gemma-2-2b-it:source=Gemma2")
tok2 = load_byteify_tokenizer("meta-llama/Llama-3.2-3B-Instruct:source=Llama3")

n_overlap = len(set(tok1.get_vocab().keys()) & set(tok2.get_vocab().keys()))
# 85699 - this is much more reasonable!

tok1.tokenize(" Café") # ['ĠCafÃ©']
tok2.tokenize(" Café") # ['ĠCafÃ©']
```

This always 100% preserves the tokenizer functionality (e.g., which tokens any text is encoded as). The API mostly matches the HuggingFace tokenizers API (e.g., `convert_ids_to_tokens`, `convert_tokens_to_ids`, `get_vocab`, `tokenize`, `add_tokens`) but is not exactly the same.

This allows us to compute things like lexical overlap and token sequence alignments accurately. `tokenkit` also implements an exact alignment algorithm between tokenizers, including tokenizers with different special tokens (e.g., different chat templates).

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

As you've seen above, `tokenkit` uses colon-separated *tokenizer spec* to load tokenizers. This gives us a simple way to specify additional arguments and modifications to the tokenizer.

- The `source=<model_family>` argument (the only required argument) enables making sure we set special tokens and the chat template correctly for the given model family. See [tokenkit/model_kinds.py](../tokenkit/model_kinds.py) for supported model families or to add new model families.
- The optional `target=<model_family>` argument enables updating the tokenizer special tokens / chat template to a different model family. E.g. `google/gemma-2-2b-it:source=Gemma2:target=Qwen2` would tokenize all regular text equivalent to the Gemma2 tokenizer, but use the Qwen2 chat template and special tokens. Since Qwen2 does not use a \<bos\> token, it would thus also not use a \<bos\> token.
- The optional `conversion=<conversion_type>` argument enables conversion to a different encoding scheme. `conversion=byte` is the one you are most likely to encounter. This converts the tokenizer to tokenize all regular (non-special-token) bytes as individual tokens i.e. to byte-level tokenization (*this is different and unrelated to byteification!*). Special tokens are kept as-is. For example:

```python
from tokenkit.byteify import load_byteify_tokenizer

tok = load_byteify_tokenizer("google/gemma-2-2b-it:source=Gemma2:conversion=byte")

tok.tokenize("<bos>Hello, world!") # ['<bos>', 'H', 'e', 'l', 'l', 'o', ',', 'Ġ', 'w', 'o', 'r', 'l', 'd', '!']
print(len(tok)) # 256 + some special tokens
```
