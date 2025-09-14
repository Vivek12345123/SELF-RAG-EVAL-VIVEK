import json

def read_json(file_path):
    """Read JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json(data: Any, filename: str, indent: int = 2) -> None:
    """Write data to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_jsonl(filename: str) -> List[Dict[str, Any]]:
    """Read JSONL file"""
    data = []
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], filename: str) -> None:
    """Write data to JSONL file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_pickle(obj: Any, filename: str) -> None:
    """Save object to pickle file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename: str) -> Any:
    """Load object from pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def ensure_dir(directory: str) -> None:
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """Get file extension"""
    return os.path.splitext(filename)[1]


def is_compressed(filename: str) -> bool:
    """Check if file is compressed"""
    return filename.endswith('.gz')


def safe_divide(numerator: float, denominator: float) -> float:
    """Safe division that returns 0 if denominator is 0"""
    return numerator / denominator if denominator != 0 else 0.0


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text"""
    return ' '.join(text.split()) if text else ''


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + '...'


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Print metrics in a formatted way"""
    print(f"\n{title}:")
    print("-" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value * 100:.2f}%"


def load_predictions(pred_file: str) -> Dict[str, Any]:
    """Load predictions from various file formats"""
    if pred_file.endswith('.jsonl') or pred_file.endswith('.jsonl.gz'):
        # JSONL format - convert to dict
        data = read_jsonl(pred_file)
        predictions = {}
        for item in data:
            if 'id' in item and 'prediction' in item:
                predictions[item['id']] = item['prediction']
            elif 'question_id' in item and 'answer' in item:
                predictions[item['question_id']] = item['answer']
        return predictions
    else:
        # Regular JSON format
        return read_json(pred_file)


def filter_dict(d: Dict[str, Any], keys_to_keep: List[str]) -> Dict[str, Any]:
    """Filter dictionary to keep only specified keys"""
    return {k: v for k, v in d.items() if k in keys_to_keep}


def invert_dict(d: Dict[str, str]) -> Dict[str, str]:
    """Invert dictionary (swap keys and values)"""
    return {v: k for k, v in d.items()}


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list"""
    return [item for sublist in nested_list for item in sublist]


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deduplicate_list(lst: List[Any]) -> List[Any]:
    """Remove duplicates from list while preserving order"""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
