import math

import datasets
from datasets import interleave_datasets, load_dataset
from torch.utils.data import Dataset


class JSONLDataset(Dataset):
    def __init__(self, path, lang_code, batch_size, n_subsample=None, seed=None):
        self.path = path
        self.lang_code = lang_code
        self.batch_size = batch_size

        self.raw_dset = load_dataset(
            "json",
            data_files=path,
            split="train" if n_subsample is None else f"train[:{n_subsample}]",
        )
        self.dset = self.raw_dset.map(
            lambda x, idx: {
                "text": [x["text"]],
                "lang_code": [[lang_code] * len(x["text"])],
                "index": [idx],
            },
            batched=True,
            batch_size=batch_size,
            drop_last_batch=True,
            remove_columns=self.raw_dset.column_names,
            with_indices=True,
        )

    # TODO: check consistency between this and iterable data - iterable cycles through the data on consecutive calls
    # (but this does NOT influence the main stream)
    # TODO: return texts directly instead of dicts?
    def get_texts(self, n: int, lang_code: None | str = None):
        if lang_code is not None and lang_code != self.lang_code:
            raise ValueError("Invalid lang_code")

        texts = self.raw_dset[:n]["text"]
        return [{"text": t} for t in texts]

    def get_torch_dataset(self):
        return self.dset


class HFDataset:
    def __init__(
        self,
        dataset_configs,
        mix_languages,
        batch_size,
        n_subsample=None,
        shuffle_buffer_size=None,
        seed=1234,
    ):
        self.dataset_configs = dataset_configs
        self.mix_languages = mix_languages
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size

        if n_subsample is not None:
            raise ValueError("Subsampling not supported for HF datasets")

        self.dset_streams = {}
        self.probs = {}

        for config in dataset_configs:
            stream = load_dataset(
                **config["kwargs"],
                streaming=True,
                trust_remote_code=True,
            )

            if self.shuffle_buffer_size is not None:
                stream = stream.shuffle(buffer_size=self.shuffle_buffer_size, seed=seed)

            self.dset_streams[config["lang_code"]] = stream.map(
                lambda x: {
                    "text": x["text"],
                    "lang_code": config["lang_code"],
                }
            )

            if "p" in config:
                self.probs[config["lang_code"]] = config["p"]

        if 0 < len(self.probs) < len(self.dset_streams):
            raise ValueError(
                "If you provide probabilities, you must provide them for all datasets"
            )

        if len(self.probs) == 0:
            self.probs = {k: 1.0 for k in self.dset_streams.keys()}

        # normalize probabilities
        total = sum(self.probs.values())
        for k in self.probs:
            self.probs[k] /= total

        if self.mix_languages:
            self.stream = interleave_datasets(
                list(self.dset_streams.values()),
                probabilities=list(self.probs.values()),
                seed=seed,
            ).batch(batch_size, drop_last_batch=True)
        else:
            self.stream = interleave_datasets(
                [
                    s.batch(batch_size, drop_last_batch=True)
                    for s in self.dset_streams.values()
                ],
                probabilities=list(self.probs.values()),
                seed=seed,
            )

    def get_texts(self, n: int, lang_code: None | str = None):
        if n == 0:
            return []

        if lang_code is None:
            # unbatch
            batches_to_take = math.ceil(n / self.batch_size)
            batches = list(self.stream.take(batches_to_take))
            out = []
            keys = list(batches[0].keys())

            for batch in batches:
                for i in range(len(batch[keys[0]])):
                    out.append({k: batch[k][i] for k in keys})

            return out

        return list(self.dset_streams[lang_code].take(n))

    def get_torch_dataset(self):
        return self.stream


class HFSavedDataset(Dataset):
    def __init__(
        self, dataset_configs, lang_code, batch_size, n_subsample=None, seed=None
    ):
        self.dataset_configs = dataset_configs
        self.lang_code = lang_code
        self.seed = seed
        self.batch_size = batch_size

        if n_subsample is not None:
            raise ValueError("Subsampling not supported for HF datasets")

        self.dsets = []
        self.probs = []

        for config in dataset_configs:
            self.dsets.append(datasets.load_from_disk(config["path"])["train"])

            if "p" in config:
                self.probs.append(config["p"])

        if 0 < len(self.probs) < len(self.dsets):
            raise ValueError(
                "If you provide probabilities, you must provide them for all datasets"
            )

        if len(self.probs) == 0:
            self.probs = None

        self.dset = interleave_datasets(
            self.dsets,
            probabilities=self.probs,
            seed=seed,
        )

    def __len__(self):
        return len(self.dset) // self.batch_size

    def get_texts(self, n: int, lang_code: None | str = None):
        if lang_code is not None and lang_code != self.lang_code:
            raise ValueError("Invalid lang_code")

        texts = self.dset[:n]["text"]
        return [{"text": t} for t in texts]

    def __getitem__(self, batch_idx):
        start, end = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size

        texts = self.dset[start:end]["text"]

        return {
            "text": texts,
            "lang_code": [self.lang_code] * len(texts),
            "index": list(range(start, start + len(texts))),
        }

    def get_torch_dataset(self):
        return self


def get_dataset(kind, **kwargs):
    if kind == "jsonl":
        return JSONLDataset(**kwargs)
    elif kind == "hf":
        return HFDataset(**kwargs)
    elif kind == "hf_saved":
        return HFSavedDataset(**kwargs)
    else:
        raise ValueError("Invalid dataset kind")
