# Project Index: Political Polarity Classification Pipeline

## 📚 Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| [README.md](README.md) | Main project documentation, setup, and usage | All users |
| [MODEL_CARD.md](MODEL_CARD.md) | Model details, limitations, ethical considerations | Researchers, compliance |
| [LABELING_GUIDELINES.md](LABELING_GUIDELINES.md) | Instructions for manual data labeling | Data annotators |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Complete technical implementation details | Developers, auditors |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Common commands and quick lookup | All users |
| [PROJECT_INDEX.md](PROJECT_INDEX.md) | This file - navigation guide | All users |

## ⚙️ Configuration Files

| File | Purpose |
|------|---------|
| [config.yaml](config.yaml) | Central configuration (paths, hyperparameters, k-anonymity) |
| [requirements.txt](requirements.txt) | Python dependencies with pinned versions |
| [.gitignore](.gitignore) | Git ignore rules |
| [LICENSE](LICENSE) | MIT License with ethical use addendum |

## 🚀 Execution Scripts

| File | Command | Purpose |
|------|---------|---------|
| [run_pipeline.sh](run_pipeline.sh) | `./run_pipeline.sh baseline` | End-to-end pipeline execution |

## 📓 Notebooks

| File | Purpose | Platform |
|------|---------|----------|
| [Political_Polarity_Pipeline_Colab.ipynb](Political_Polarity_Pipeline_Colab.ipynb) | Interactive demo of full pipeline | Google Colab |

## 🐍 Source Code (src/)

### Core Modules

| Module | Key Classes/Functions | Purpose |
|--------|----------------------|---------|
| [preprocessing.py](src/preprocessing.py) | `TextPreprocessor`, `Deduplicator`, `stratified_split` | Text cleaning, deduplication, data splitting |
| [features.py](src/features.py) | `TFIDFFeatureExtractor`, `PhraseExtractor`, `KeywordExtractor` | Feature extraction, phrase identification |
| [models.py](src/models.py) | `BaselineClassifier`, `TransformerClassifier` | Model training and inference |
| [evaluation.py](src/evaluation.py) | `ModelEvaluator`, `compute_polarity_scores` | Metrics, calibration, error analysis |
| [interpretability.py](src/interpretability.py) | `BaselineInterpreter`, `TransformerInterpreter` | Model explanations, feature importance |
| [privacy.py](src/privacy.py) | `KAnonymityEnforcer`, `AggregateReporter` | Privacy-preserving analysis |

## 📜 Scripts (scripts/)

### Executable Python Modules

| Script | Command | Purpose |
|--------|---------|---------|
| [verify_setup.py](scripts/verify_setup.py) | `python -m scripts.verify_setup` | Check dependencies and environment |
| [generate_synthetic_data.py](scripts/generate_synthetic_data.py) | `python -m scripts.generate_synthetic_data` | Create test dataset |
| [prepare_data.py](scripts/prepare_data.py) | `python -m scripts.prepare_data --input_path ... --outlet_labels ...` | Preprocess and split data |
| [train_baseline.py](scripts/train_baseline.py) | `python -m scripts.train_baseline --train_path ... --val_path ...` | Train TF-IDF + LogReg model |
| [train_transformer.py](scripts/train_transformer.py) | `python -m scripts.train_transformer --train_path ... --val_path ...` | Train DistilBERT model |
| [evaluate.py](scripts/evaluate.py) | `python -m scripts.evaluate --test_path ... --model_path ...` | Evaluate trained model |
| [infer.py](scripts/infer.py) | `python -m scripts.infer --input_path ... --model_path ...` | Run predictions on new data |
| [report_subsets.py](scripts/report_subsets.py) | `python -m scripts.report_subsets --predictions_path ... --subset_key ...` | Generate privacy-preserving reports |

## 🧪 Tests (tests/)

| Test File | Covers | Command |
|-----------|--------|---------|
| [test_preprocessing.py](tests/test_preprocessing.py) | Text cleaning, deduplication, splitting | `pytest tests/test_preprocessing.py -v` |
| [test_features.py](tests/test_features.py) | TF-IDF, phrase extraction | `pytest tests/test_features.py -v` |
| [test_privacy.py](tests/test_privacy.py) | K-anonymity enforcement | `pytest tests/test_privacy.py -v` |
| [test_models.py](tests/test_models.py) | Baseline model training/prediction | `pytest tests/test_models.py -v` |

**Run all tests:**
```bash
pytest tests/ -v --cov=src --cov-report=html
```

## 📂 Data Directories

| Directory | Contents | Tracked by Git? |
|-----------|----------|-----------------|
| `data/raw/` | Original JSONL documents, outlet labels | No (gitignored) |
| `data/processed/` | Train/val/test splits | No (gitignored) |
| `artifacts/` | Trained models, evaluations, reports | No (gitignored) |

## 🔄 Typical Workflow

### First-Time Setup
```bash
1. pip install -r requirements.txt
2. python -m scripts.verify_setup
```

### Quick Test with Synthetic Data
```bash
./run_pipeline.sh baseline
```

### Production Pipeline
```bash
# 1. Prepare your data
python -m scripts.prepare_data \
  --input_path ./data/raw/corpus.jsonl \
  --outlet_labels ./data/raw/outlets.csv

# 2. Train model
python -m scripts.train_transformer \
  --train_path ./data/processed/train.jsonl \
  --val_path ./data/processed/val.jsonl \
  --output_dir ./artifacts/transformer

# 3. Evaluate
python -m scripts.evaluate \
  --test_path ./data/processed/test.jsonl \
  --model_path ./artifacts/transformer \
  --model_type transformer

# 4. Run inference
python -m scripts.infer \
  --input_path ./data/new_docs.jsonl \
  --model_path ./artifacts/transformer \
  --model_type transformer \
  --output_path ./data/predictions.jsonl

# 5. Generate reports
python -m scripts.report_subsets \
  --predictions_path ./data/predictions.jsonl \
  --subset_key region \
  --output_dir ./artifacts/reports
```

## 📊 Output Artifacts

### After Training
- `artifacts/baseline/model.pkl` - Trained baseline model
- `artifacts/baseline/vectorizer.pkl` - TF-IDF vectorizer
- `artifacts/baseline/feature_importance.json` - Top features per class
- `artifacts/transformer/` - HuggingFace checkpoint directory

### After Evaluation
- `artifacts/*/evaluation/test_metrics.json` - Performance metrics
- `artifacts/*/evaluation/confusion_matrix.png` - Confusion matrix plot
- `artifacts/*/evaluation/calibration_curve.png` - Calibration analysis
- `artifacts/*/evaluation/error_analysis.csv` - Misclassified samples

### After Inference
- `data/predictions.jsonl` - Document predictions with probabilities
- Each document includes: `predicted_class`, `prob_left`, `prob_center`, `prob_right`, `polarity_score`, `uncertainty`, `keywords`

### After Subset Reporting
- `artifacts/reports/polarity_report_*.json` - Polarity statistics by subset
- `artifacts/reports/polarity_by_*.png` - Polarity visualization
- `artifacts/reports/distribution_by_*.png` - Class distribution plot
- `artifacts/reports/PRIVACY_NOTICE.txt` - Privacy notice

## 🔐 Privacy & Ethics

### Key Files
- [MODEL_CARD.md](MODEL_CARD.md) - Ethical considerations and limitations
- [LICENSE](LICENSE) - Ethical use addendum
- [src/privacy.py](src/privacy.py) - K-anonymity implementation

### Safeguards Implemented
1. **K-anonymity**: Minimum group size enforcement (default: 30)
2. **Aggregate-only reporting**: No individual predictions in reports
3. **Metadata validation**: Whitelist approach for subset keys
4. **Privacy notices**: Attached to all reports
5. **Documentation**: Prominent warnings throughout

### Prohibited Uses
❌ Individual political profiling  
❌ Targeting individuals  
❌ Inferring protected characteristics  
❌ Decisions affecting individual rights  

## 📈 Performance Benchmarks

### Baseline Model
- **Training**: ~1-5 min (10k docs, CPU)
- **Inference**: ~10-50 docs/sec (CPU)
- **Accuracy**: ~70-80%
- **Macro-F1**: ~0.65-0.75

### Transformer Model
- **Training**: ~10-60 min (10k docs, GPU)
- **Inference**: ~50-200 docs/sec (GPU)
- **Accuracy**: ~75-85%
- **Macro-F1**: ~0.70-0.80

## 🐛 Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size`, use baseline model |
| Slow training | Use GPU, reduce epochs/max_features |
| Poor performance | Increase training data, add manual labels |
| Import errors | Run `python -m scripts.verify_setup` |
| K-anonymity suppresses all groups | Reduce `min_group_size` |

## 📝 Customization Points

### Easy Customizations (via config.yaml)
- Random seed
- Data paths
- K-anonymity threshold
- Hyperparameters (learning rate, batch size, etc.)
- Feature extraction parameters

### Moderate Customizations (modify code)
- Add new subset keys: Update `validate_subset_metadata()` in `privacy.py`
- Change label scheme: Update `config.yaml` and `label_map`
- Add new metrics: Extend `ModelEvaluator` in `evaluation.py`

### Advanced Customizations (architectural changes)
- Semi-supervised learning: Extend `prepare_data.py`
- Hierarchical classification: Modify `models.py`
- Multilingual support: Extend `preprocessing.py`
- Alternative privacy: Replace `KAnonymityEnforcer` in `privacy.py`

## 📚 Additional Resources

### External Documentation
- [scikit-learn](https://scikit-learn.org/) - ML algorithms
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - Transformer models
- [ACL Ethics](https://www.aclweb.org/portal/content/acl-code-ethics) - NLP ethics guidelines

### Related Papers
- Preoţiuc-Pietro et al. (2017) - Political ideology prediction
- Baly et al. (2020) - News media profiling
- Card et al. (2015) - Media frames and political bias

## 🔄 Version Control

### Important Files to Track
- ✅ All `.py` source files
- ✅ All `.md` documentation
- ✅ `config.yaml`, `requirements.txt`
- ✅ `run_pipeline.sh`
- ✅ Tests

### Gitignored
- ❌ `data/*` (except .gitkeep)
- ❌ `artifacts/*` (except .gitkeep)
- ❌ `*.pkl`, `*.pt`, `*.pth`
- ❌ `__pycache__/`

## 🤝 Contributing

1. Review [MODEL_CARD.md](MODEL_CARD.md) for ethical guidelines
2. Add tests for new features
3. Update documentation
4. Follow existing code style
5. Ensure reproducibility (fixed seeds)

## 📞 Support

- **Issues**: GitHub Issues (if applicable)
- **Email**: [your-email]
- **Documentation**: This repository

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Production-ready
