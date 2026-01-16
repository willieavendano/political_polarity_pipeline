# Political Polarity Classification Pipeline

## ⚠️ ETHICAL USE NOTICE

This system is designed for **aggregate-level analysis** of textual political polarity ONLY. 

**PROHIBITED USES:**
- Inferring individual political affiliation or beliefs
- Targeting individuals based on predicted ideology
- Making decisions that affect individual rights or opportunities
- Inferring protected characteristics (race, religion, etc.) from text

**LIMITATIONS:**
- Outlet labels ≠ article ideology (distant supervision has noise)
- Topic correlation ≠ ideological stance
- Historical bias in training data may be reproduced
- Model may conflate style with substance

## Overview

This pipeline classifies text documents into three political polarity categories:
- **Left/Democrat/Liberal** (class 0)
- **Center/Mixed/Unclear** (class 1)  
- **Right/Republican/Conservative** (class 2)

It provides:
- Document-level polarity probabilities
- Interpretable phrase/keyword extraction
- Privacy-preserving aggregate analysis by population subsets
- Both fast baseline (TF-IDF + LogReg) and contextual (DistilBERT) models

## Setup

### Requirements
- Python 3.12+
- 8GB+ RAM (16GB recommended for transformer training)
- GPU optional (CPU mode supported with slower training)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m scripts.verify_setup
```

### Google Colab Setup

```python
# In Colab notebook
!git clone <your-repo-url>
%cd political-polarity-pipeline
!pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Sample Data (for testing)

```bash
# Generate synthetic test dataset
python -m scripts.generate_synthetic_data

# This creates ./data/raw/synthetic_corpus.jsonl
```

### 2. Data Preparation

```bash
# Process raw JSONL → cleaned + split
python -m scripts.prepare_data \
  --input_path ./data/raw/synthetic_corpus.jsonl \
  --outlet_labels ./data/raw/outlet_labels.csv \
  --output_dir ./data/processed \
  --test_size 0.15 \
  --val_size 0.15
```

### 3. Train Baseline Model

```bash
python -m scripts.train_baseline \
  --train_path ./data/processed/train.jsonl \
  --val_path ./data/processed/val.jsonl \
  --output_dir ./artifacts/baseline \
  --max_features 10000 \
  --ngram_range 1 3
```

### 4. Train Transformer Model

```bash
python -m scripts.train_transformer \
  --train_path ./data/processed/train.jsonl \
  --val_path ./data/processed/val.jsonl \
  --output_dir ./artifacts/transformer \
  --model_name distilbert-base-uncased \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5
```

### 5. Evaluate Models

```bash
# Evaluate baseline
python -m scripts.evaluate \
  --test_path ./data/processed/test.jsonl \
  --model_path ./artifacts/baseline/model.pkl \
  --model_type baseline \
  --output_dir ./artifacts/baseline/evaluation

# Evaluate transformer
python -m scripts.evaluate \
  --test_path ./data/processed/test.jsonl \
  --model_path ./artifacts/transformer \
  --model_type transformer \
  --output_dir ./artifacts/transformer/evaluation
```

### 6. Run Inference

```bash
python -m scripts.infer \
  --input_path ./data/new_documents.jsonl \
  --model_path ./artifacts/transformer \
  --model_type transformer \
  --output_path ./data/predictions.jsonl \
  --batch_size 32
```

### 7. Generate Subset Reports

```bash
python -m scripts.report_subsets \
  --predictions_path ./data/predictions.jsonl \
  --model_path ./artifacts/transformer \
  --model_type transformer \
  --subset_key region \
  --output_dir ./artifacts/reports \
  --min_group_size 30
```

## Data Format

### Input JSONL Format

```json
{
  "doc_id": "unique_document_id",
  "text": "Full text content of the document...",
  "title": "Document title",
  "date": "2024-01-15",
  "source": "outlet_name",
  "url": "https://example.com/article",
  "subset_meta": {
    "region": "Midwest",
    "age_bucket": "25-34"
  }
}
```

### Outlet Labels Format (CSV)

```csv
outlet_name,bias_label
NYTimes,left
FoxNews,right
Reuters,center
```

Valid bias labels: `left`, `center`, `right`

### Manual Labels Format (Optional, CSV)

```csv
doc_id,label
doc_001,left
doc_002,center
doc_003,right
```

## Configuration

Edit `config.yaml` to customize:
- Data paths
- Model hyperparameters
- Random seeds
- K-anonymity threshold
- Evaluation metrics

## Project Structure

```
political-polarity-pipeline/
├── README.md
├── requirements.txt
├── config.yaml
├── MODEL_CARD.md
├── data/
│   ├── raw/              # Original JSONL files
│   └── processed/        # Train/val/test splits
├── artifacts/
│   ├── baseline/         # Baseline model artifacts
│   ├── transformer/      # Transformer model artifacts
│   └── reports/          # Subset analysis reports
├── src/
│   ├── __init__.py
│   ├── preprocessing.py  # Text cleaning, deduplication
│   ├── features.py       # TF-IDF, n-grams, keyword extraction
│   ├── models.py         # Baseline and transformer models
│   ├── evaluation.py     # Metrics, calibration, error analysis
│   ├── interpretability.py  # Feature importance, attention
│   └── privacy.py        # K-anonymity enforcement
├── scripts/
│   ├── __init__.py
│   ├── verify_setup.py
│   ├── generate_synthetic_data.py
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   ├── infer.py
│   └── report_subsets.py
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_privacy.py
    └── test_models.py
```

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Model Card

See [MODEL_CARD.md](MODEL_CARD.md) for detailed documentation on:
- Intended use cases
- Training data and labels
- Evaluation metrics
- Limitations and biases
- Ethical considerations

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in transformer training
- Use `max_length=128` instead of 512
- Train baseline model instead

### Slow Training
- Use GPU if available (automatically detected)
- Reduce `max_features` for baseline
- Reduce `epochs` for transformer

### Poor Performance
- Check label distribution (imbalanced classes?)
- Increase training data
- Adjust hyperparameters in `config.yaml`
- Review error analysis outputs

## Citation

If you use this pipeline in research, please cite responsibly and note the ethical limitations.

## License

MIT License - See LICENSE file for details.

## Contact

For questions about ethical use or technical issues, contact [your contact info].
