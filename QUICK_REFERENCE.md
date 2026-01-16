# Quick Reference Guide

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m scripts.verify_setup
```

### Complete Pipeline (One Command)
```bash
# Baseline model (fast)
./run_pipeline.sh baseline

# Transformer model (better accuracy)
./run_pipeline.sh transformer

# Both models
./run_pipeline.sh both
```

### Step-by-Step

#### 1. Data Preparation
```bash
# Generate synthetic test data
python -m scripts.generate_synthetic_data

# Prepare your own data
python -m scripts.prepare_data \
  --input_path ./data/raw/your_corpus.jsonl \
  --outlet_labels ./data/raw/outlet_labels.csv \
  --output_dir ./data/processed
```

#### 2. Training
```bash
# Baseline (TF-IDF + LogReg)
python -m scripts.train_baseline \
  --train_path ./data/processed/train.jsonl \
  --val_path ./data/processed/val.jsonl \
  --output_dir ./artifacts/baseline

# Transformer (DistilBERT)
python -m scripts.train_transformer \
  --train_path ./data/processed/train.jsonl \
  --val_path ./data/processed/val.jsonl \
  --output_dir ./artifacts/transformer \
  --epochs 3
```

#### 3. Evaluation
```bash
python -m scripts.evaluate \
  --test_path ./data/processed/test.jsonl \
  --model_path ./artifacts/baseline \
  --model_type baseline
```

#### 4. Inference
```bash
python -m scripts.infer \
  --input_path ./data/new_docs.jsonl \
  --model_path ./artifacts/transformer \
  --model_type transformer \
  --output_path ./data/predictions.jsonl \
  --extract_keywords
```

#### 5. Subset Reports
```bash
python -m scripts.report_subsets \
  --predictions_path ./data/predictions.jsonl \
  --subset_key region \
  --output_dir ./artifacts/reports \
  --min_group_size 30
```

## Input Formats

### Document JSONL
```json
{
  "doc_id": "unique_id",
  "text": "Full article text...",
  "title": "Article Title",
  "date": "2024-01-15",
  "source": "outlet_name",
  "url": "https://example.com/article",
  "subset_meta": {
    "region": "Midwest",
    "age_bucket": "25-34"
  }
}
```

### Outlet Labels CSV
```csv
outlet_name,bias_label
NYTimes,left
Reuters,center
FoxNews,right
```

### Manual Labels CSV (Optional)
```csv
doc_id,label
doc_001,0
doc_002,1
doc_003,2
```

## Configuration

Edit `config.yaml` for:
- Data paths
- Model hyperparameters
- K-anonymity threshold (default: 30)
- Random seed (default: 42)

## Key Outputs

### Model Artifacts
- `./artifacts/baseline/model.pkl`
- `./artifacts/transformer/` (HuggingFace checkpoint)

### Evaluation Results
- `./artifacts/*/evaluation/test_metrics.json`
- `./artifacts/*/evaluation/confusion_matrix.png`
- `./artifacts/*/evaluation/predictions.csv`

### Subset Reports
- `./artifacts/reports/polarity_report_*.json`
- `./artifacts/reports/polarity_by_*.png`
- `./artifacts/reports/summary.txt`

## Label Mapping

- **0**: Left/Democrat/Liberal
- **1**: Center/Mixed/Unclear
- **2**: Right/Republican/Conservative

## Polarity Score

- **Range**: -1 (left) to +1 (right)
- **Formula**: P(right) - P(left)
- **Interpretation**:
  - < -0.3: Strong left
  - -0.3 to -0.1: Moderate left
  - -0.1 to 0.1: Center
  - 0.1 to 0.3: Moderate right
  - > 0.3: Strong right

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in transformer training
- Use `max_length=128` instead of 256
- Train baseline model instead

### Slow Training
- Use GPU if available (auto-detected)
- Reduce `max_features` for baseline
- Reduce `epochs` for transformer

### Poor Performance
- Check label distribution (balanced?)
- Increase training data
- Add manual labels
- Review error analysis outputs

## Google Colab

1. Upload `Political_Polarity_Pipeline_Colab.ipynb`
2. Run all cells
3. Download artifacts: `!zip -r artifacts.zip ./artifacts`

## Important Reminders

⚠️ **Ethical Use:**
- AGGREGATE analysis only
- NO individual profiling
- Enforce k-anonymity (default: 30)

⚠️ **Limitations:**
- Outlet labels ≠ article ideology
- Topic ≠ ideology
- Historical bias may be present
- U.S.-centric training data

📚 **Documentation:**
- Full guide: README.md
- Model details: MODEL_CARD.md
- Labeling: LABELING_GUIDELINES.md
- Implementation: IMPLEMENTATION_SUMMARY.md

## Support

- GitHub: [your-repo-url]
- Email: [your-email]
- Issues: [your-issues-url]
