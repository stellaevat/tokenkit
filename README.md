# tokenkit

## Converting Tokenizers to the Byte-Level (`byteify.py`)

To convert the Gemma tokenizer:

```bash
ipython --pdb scripts/byteify.py -- \
    --tokenizer_name=google/gemma-2-2b-it \
    --output=outputs/tokenizers/gemma2 \
    --normalizer_strategy="remove" \
    --split_regex='\s*[\p{L}\p{M}]+|\s*\p{N}+|\s*[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+'
```

To check the integrity of a tokenizer:

```bash
ipython -i --pdb scripts/check_tokenizer_integrity.py -- \
    --tokenizer=outputs/tokenizers/gemma2 \
    --reference=google/gemma-2-2b-it \
    --num_workers=0
```