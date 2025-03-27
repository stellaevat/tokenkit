# tokenkit

```bash
ipython --pdb scripts/check_alignments.py -- \
    --reference='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2' \
    --to_check='google/gemma-2-2b-it:source=Gemma2:target=Gemma2' \
    --num_workers=0
```