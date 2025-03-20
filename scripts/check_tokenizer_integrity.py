from transformers import HfArgumentParser, AutoTokenizer
from dataclasses import dataclass
from datasets import load_dataset
import logging
from pprint import pformat
import os

logger = logging.getLogger(__name__)


@dataclass
class Args:
    tokenizer: str
    reference: str
    dataset_name: str = "allenai/tulu-3-sft-mixture"
    dataset_split: str = "train"
    text_column: str = "messages"
    num_workers: int = 0
    batch_size: int = 8192


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(level=logging.INFO)

    logger.info(pformat(args))

    reference_tokenizer = AutoTokenizer.from_pretrained(args.reference)
    tokenizer_to_check = AutoTokenizer.from_pretrained(args.tokenizer)

    dset = load_dataset(args.dataset_name, split=args.dataset_split)

    def tokenize(examples):
        texts = []

        for text_data in examples[args.text_column]:
            if isinstance(text_data, str):
                texts.append(text_data)
            else:
                # assume chat format
                texts.append("\n".join(message["content"] for message in text_data))

        tokens = tokenizer_to_check(texts, add_special_tokens=False)
        reference_tokens = reference_tokenizer(texts, add_special_tokens=False)

        n_mismatch = 0

        for i, (token, reference_token) in enumerate(
            zip(tokens["input_ids"], reference_tokens["input_ids"])
        ):
            if token != reference_token:
                reference_seq = reference_token.copy()
                seq = token.copy()

                while seq[0] == reference_seq[0]:
                    seq = seq[1:]
                    reference_seq = reference_seq[1:]
                
                while seq[-1] == reference_seq[-1]:
                    seq = seq[:-1]
                    reference_seq = reference_seq[:-1]

                logger.error(f"Tokenization mismatch for example {i}")
                logger.error(f"Reference: {[reference_tokenizer.decode(i) for i in reference_seq]}")
                logger.error(f"Got: {[tokenizer_to_check.decode(i) for i in seq]}")

                import ipdb; ipdb.set_trace()

                n_mismatch += 1

        return {
            "n_mismatch": n_mismatch,
            "n_total": len(examples),
        }

    stats = dset.map(
        tokenize,
        num_proc=args.num_workers if args.num_workers > 0 else None,
        batched=True,
        batch_size=args.batch_size,
    )
