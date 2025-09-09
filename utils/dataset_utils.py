mkdir -p evals/utils
cat > evals/utils/dataset_utils.py << 'EOF'
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
EOF

# Create utils/utils.py
cat > evals/utils/utils.py << 'EOF'
import json

def read_json(file_path):
    """Read JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)
