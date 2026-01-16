#!/bin/bash
#
# End-to-end pipeline execution script
# 
# Usage: ./run_pipeline.sh [baseline|transformer|both]
#

set -e  # Exit on error

MODEL_TYPE=${1:-baseline}  # Default to baseline if not specified

echo "============================================================"
echo "Political Polarity Classification Pipeline"
echo "============================================================"
echo ""
echo "⚠️  ETHICAL USE REMINDER:"
echo "This system is for AGGREGATE-LEVEL analysis only."
echo "Do NOT use for individual profiling or targeting."
echo ""
echo "============================================================"

# Step 1: Generate synthetic data (for testing)
echo ""
echo "[1/7] Generating synthetic test data..."
python -m scripts.generate_synthetic_data

# Step 2: Prepare data
echo ""
echo "[2/7] Preparing data..."
python -m scripts.prepare_data \
  --input_path ./data/raw/synthetic_corpus.jsonl \
  --outlet_labels ./data/raw/outlet_labels.csv \
  --output_dir ./data/processed \
  --test_size 0.15 \
  --val_size 0.15

# Step 3: Train baseline model
if [ "$MODEL_TYPE" = "baseline" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo ""
    echo "[3/7] Training baseline model..."
    python -m scripts.train_baseline \
      --train_path ./data/processed/train.jsonl \
      --val_path ./data/processed/val.jsonl \
      --output_dir ./artifacts/baseline \
      --max_features 5000 \
      --ngram_range 1 3
fi

# Step 4: Train transformer model (optional)
if [ "$MODEL_TYPE" = "transformer" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo ""
    echo "[4/7] Training transformer model (this may take a while)..."
    python -m scripts.train_transformer \
      --train_path ./data/processed/train.jsonl \
      --val_path ./data/processed/val.jsonl \
      --output_dir ./artifacts/transformer \
      --model_name distilbert-base-uncased \
      --epochs 2 \
      --batch_size 16 \
      --learning_rate 2e-5
fi

# Step 5: Evaluate models
echo ""
echo "[5/7] Evaluating models..."

if [ "$MODEL_TYPE" = "baseline" ] || [ "$MODEL_TYPE" = "both" ]; then
    python -m scripts.evaluate \
      --test_path ./data/processed/test.jsonl \
      --model_path ./artifacts/baseline \
      --model_type baseline \
      --output_dir ./artifacts/baseline/evaluation
fi

if [ "$MODEL_TYPE" = "transformer" ] || [ "$MODEL_TYPE" = "both" ]; then
    python -m scripts.evaluate \
      --test_path ./data/processed/test.jsonl \
      --model_path ./artifacts/transformer \
      --model_type transformer \
      --output_dir ./artifacts/transformer/evaluation
fi

# Step 6: Run inference
echo ""
echo "[6/7] Running inference..."

# Use baseline for inference by default, or transformer if that's what was trained
INFER_MODEL="baseline"
if [ "$MODEL_TYPE" = "transformer" ]; then
    INFER_MODEL="transformer"
fi

python -m scripts.infer \
  --input_path ./data/processed/test.jsonl \
  --model_path ./artifacts/$INFER_MODEL \
  --model_type $INFER_MODEL \
  --output_path ./data/predictions.jsonl \
  --batch_size 32 \
  --extract_keywords

# Step 7: Generate subset reports
echo ""
echo "[7/7] Generating privacy-preserving subset reports..."
python -m scripts.report_subsets \
  --predictions_path ./data/predictions.jsonl \
  --subset_key region \
  --output_dir ./artifacts/reports \
  --min_group_size 5

echo ""
echo "============================================================"
echo "✓ Pipeline execution complete!"
echo "============================================================"
echo ""
echo "Results:"
if [ "$MODEL_TYPE" = "baseline" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo "  - Baseline model: ./artifacts/baseline/"
fi
if [ "$MODEL_TYPE" = "transformer" ] || [ "$MODEL_TYPE" = "both" ]; then
    echo "  - Transformer model: ./artifacts/transformer/"
fi
echo "  - Predictions: ./data/predictions.jsonl"
echo "  - Reports: ./artifacts/reports/"
echo ""
echo "Next steps:"
echo "  - Review evaluation metrics in ./artifacts/*/evaluation/"
echo "  - Check subset reports in ./artifacts/reports/"
echo "  - Read MODEL_CARD.md for limitations and ethical use"
echo ""
echo "============================================================"
