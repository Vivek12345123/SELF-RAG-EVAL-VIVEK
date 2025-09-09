import json
import gzip
from typing import Dict, List, Any

class Span:
    def __init__(self, start=-1, end=-1):
        self.start = start
        self.end = end
    
    def is_null_span(self):
        return self.start == -1 or self.end == -1

class NQLabel:
    def __init__(self, example_id="", long_answer_span=None, short_answer_span_list=None, 
                 yes_no_answer="none", long_score=0.0, short_score=0.0):
        self.example_id = example_id
        self.long_answer_span = long_answer_span or Span()
        self.short_answer_span_list = short_answer_span_list or []
        self.yes_no_answer = yes_no_answer
        self.long_score = long_score
        self.short_score = short_score

def read_annotation(file_path, n_threads=1):
    """Read NQ annotation file"""
    annotations = {}
    
    try:
        # Try to read as JSON first
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Handle different formats
        if isinstance(data, list):
            for item in data:
                example_id = str(item.get('example_id', item.get('id', '')))
                if example_id:
                    annotations[example_id] = [NQLabel(example_id=example_id)]
        elif isinstance(data, dict):
            for key, value in data.items():
                annotations[key] = [NQLabel(example_id=key)]
    except:
        # Fallback: treat as JSONL
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    example_id = str(item.get('example_id', item.get('id', '')))
                    if example_id:
                        annotations[example_id] = [NQLabel(example_id=example_id)]
    
    return annotations

def read_prediction_json(file_path):
    """Read NQ predictions file"""
    with open(file_path, 'r') as f:
        preds = json.load(f)
    
    # Convert to NQLabel format
    result = {}
    for key, value in preds.items():
        # Handle both dict and string predictions
        if isinstance(value, dict):
            result[key] = NQLabel(
                example_id=key,
                long_score=value.get('long_score', 0.0),
                short_score=value.get('short_score', 0.0)
            )
        else:
            # Simple string prediction
            result[key] = NQLabel(example_id=key)
    
    return result

def gold_has_long_answer(gold_label_list):
    """Check if gold has long answer"""
    for label in gold_label_list:
        if hasattr(label, 'long_answer_span') and not label.long_answer_span.is_null_span():
            return True
    return False

def gold_has_short_answer(gold_label_list):
    """Check if gold has short answer"""
    for label in gold_label_list:
        if hasattr(label, 'short_answer_span_list') and label.short_answer_span_list:
            return True
        if hasattr(label, 'yes_no_answer') and label.yes_no_answer != 'none':
            return True
    return False

def nonnull_span_equal(span1, span2):
    """Check if two spans are equal"""
    return span1.start == span2.start and span1.end == span2.end

def is_null_span_list(span_list):
    """Check if span list is null"""
    return not span_list or all(s.is_null_span() for s in span_list)

def span_set_equal(span_list1, span_list2):
    """Check if two span sets are equal"""
    return set((s.start, s.end) for s in span_list1) == set((s.start, s.end) for s in span_list2)
EOF
