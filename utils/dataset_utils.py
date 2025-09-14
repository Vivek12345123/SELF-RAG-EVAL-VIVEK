import json
import gzip

def read_triviaqa_data(file_path):
    """Read TriviaQA dataset file"""
    try:
        # Try gzipped file first
        with gzip.open(file_path, 'rt') as f:
            data = json.load(f)
    except:
        # Try regular JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
    
    # Ensure it has the expected structure
    if 'Data' not in data:
        # Create wrapper if raw data
        data = {'Data': data if isinstance(data, list) else [data], 'Version': 1.0}
    
    return data

def get_key_to_ground_truth(dataset_json):
    """Extract ground truth from TriviaQA dataset"""
    ground_truth = {}
    
    data_items = dataset_json.get('Data', [])
    for item in data_items:
        qid = item.get('QuestionId', item.get('question_id', item.get('id', '')))
        if not qid:
            continue
            
        answer = item.get('Answer', item.get('answer', {}))
        if not isinstance(answer, dict):
            answer = {'Value': str(answer), 'NormalizedAliases': [str(answer).lower()]}
        
        # Ensure required fields
        if 'NormalizedAliases' not in answer:
            answer['NormalizedAliases'] = [answer.get('Value', '').lower()]
        
        ground_truth[str(qid)] = answer
    
    return ground_truth


def read_json_gz(filename):
    """Read compressed JSON file"""
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        return json.load(f)


def read_json(filename):
    """Read JSON file (compressed or uncompressed)"""
    if filename.endswith('.gz'):
        return read_json_gz(filename)
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)


def get_question_doc_string(question, doc):
    """Combine question and document into a single string"""
    return f"Question: {question}\nDocument: {doc}"


def normalize_text(text):
    """Basic text normalization"""
    if not text:
        return ""
    return " ".join(text.strip().split())


def extract_answers_from_item(item):
    """Extract all possible answers from a TriviaQA item"""
    answers = []
    
    # Get from Answer field
    answer_obj = item.get('Answer', {})
    if isinstance(answer_obj, dict):
        # NormalizedAliases
        if 'NormalizedAliases' in answer_obj:
            answers.extend(answer_obj['NormalizedAliases'])
        
        # HumanAnswers
        if 'HumanAnswers' in answer_obj:
            answers.extend(answer_obj['HumanAnswers'])
        
        # Value field
        if 'Value' in answer_obj:
            answers.append(answer_obj['Value'])
    
    return list(set(answers))  # Remove duplicates
