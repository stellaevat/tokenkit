import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from tokenkit import baseline_utils, utils
from tokenkit.byteify import load_byteify_tokenizer


@hydra.main(
    version_base=None, config_path="../configs", config_name="compute_mined_mapping"
)
def my_app(args: DictConfig) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer_teacher = load_byteify_tokenizer(args.teacher_tokenizer_name)
    target_tokenizer = load_byteify_tokenizer(args.target_tokenizer_name)

    mined_mapping, mined_distances = baseline_utils.compute_mined_mapping(
        tokenizer_teacher, target_tokenizer, num_workers=args.num_workers
    )

    np.save(output_dir / "mined_mapping.npy", mined_mapping)
    json.dump(
        mined_distances,
        open(output_dir / "mined_distances.json", "w"),
        indent=4,
    )


if __name__ == "__main__":
    my_app()
