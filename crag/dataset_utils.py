# dataset_utils.py
from pathlib import Path
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
import json

CACHE_ROOT = Path("./datasets_cache")


def load_from_cache_or_hf(key: str, split: str = None):
    """
    Try to load a dataset from ./datasets_cache/<key>.
    - If a HF id is provided as key (contains '/'), attempt to load from HF via load_dataset.
    - If on-disk cache exists (load_from_disk), load from disk instead.
    - If split is provided, attempt to return that split.
    Returns a datasets.Dataset or DatasetDict.
    """
    disk_path = CACHE_ROOT / key
    if disk_path.exists():
        # If save_to_disk was used, there will be either a dataset folder or split folders
        try:
            ds = load_from_disk(str(disk_path))
            if split is not None:
                if hasattr(ds, "__getitem__") and split in ds:
                    return ds[split]
            return ds
        except Exception:
            # Attempt to load jsonl files (streaming-subset mode)
            jsonl_files = list(disk_path.glob("*.jsonl"))
            if jsonl_files:
                # load first file into a Dataset
                from datasets import load_dataset as hf_load
                return hf_load("json", data_files=str(jsonl_files[0]))["train"]
            raise

    # fallback: treat key as HF id and try to load
    if "/" in key:
        if split:
            return load_dataset(key, split=split)
        return load_dataset(key)
    
    raise FileNotFoundError(f"Neither on-disk dataset at {disk_path} nor HF id '{key}' available.")


def load_jsonl_as_dataset(path: str):
    """
    Utility: load a jsonl file into a datasets.Dataset object.
    """
    from datasets import load_dataset as hf_load
    return hf_load("json", data_files={"train": path})["train"]
