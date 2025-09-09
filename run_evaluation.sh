MAX_EXAMPLES=200
MAX_TOKENS=100
BATCH_SIZE=8
TASKS="all"
OUTPUT_DIR="outputs_with_bert"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick-test)
            MAX_EXAMPLES=10
            TASKS="squad"
            OUTPUT_DIR="outputs_quick_test"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running Self-RAG evaluation with BERT scoring..."
echo "Max examples: $MAX_EXAMPLES"
echo "Max tokens: $MAX_TOKENS"
echo "Batch size: $BATCH_SIZE"
echo "Tasks: $TASKS"
echo "Output directory: $OUTPUT_DIR"

python run_all_evals_with_bert.py \
    --max-examples $MAX_EXAMPLES \
    --max-tokens $MAX_TOKENS \
    --batch-size $BATCH_SIZE \
    --tasks "$TASKS" \
    --output-dir "$OUTPUT_DIR"
